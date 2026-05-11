/**
 * Scaffold-ChunGS: Renderer that expands anchors to child Gaussians
 * via MLP decoders and rasterizes via the 3DGS tile-based rasterizer.
 *
 * Two backends:
 *   - HAVE_CUDA_RASTERIZER → INRIA diff_gaussian_rasterization (real-time)
 *   - Fallback → pure-LibTorch GPU compositing (development/CPU-only)
 */

#include "scaffold_chunks/gaussian_renderer.h"
#include "scaffold_chunks/anchor_mlp.h"
#include "scaffold_chunks/chunk_types.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#ifdef HAVE_CUDA_RASTERIZER
#include "diff_gaussian_rasterization/rasterize_points.h"
#endif

namespace scaffold_chungs {

// =============================================================================
// Fallback Rasterizer — Pure LibTorch GPU (no external CUDA dependency)
// =============================================================================
//
// Used when diff_gaussian_rasterization is not available.
// All ops stay on GPU; the only sync point is .any().item<bool>() for early exit.
//
// Approach: depth-sorted alpha compositing on GPU via scatter_add and cumprod.
// All operations stay on GPU; the only synchronization point is the final
// .any().item<bool>() for empty-frame early exit.

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
renderGaussiansGPU(
    const torch::Tensor& means3D,      // [M, 3]
    const torch::Tensor& colors,       // [M, 3]
    const torch::Tensor& opacity,      // [M, 1]
    const torch::Tensor& scaling,      // [M, 3]
    const torch::Tensor& rotation,     // [M, 4]
    const torch::Tensor& world_view,   // [4, 4]
    const torch::Tensor& proj_matrix,  // [4, 4]
    float FoVx, float FoVy,
    int H, int W,
    const torch::Tensor& background) {

  auto device = means3D.device();
  auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  int64_t M = means3D.size(0);

  torch::Tensor bg = background.unsqueeze(1).unsqueeze(2);  // [3, 1, 1]
  torch::Tensor render = bg.expand({3, H, W}).clone();
  torch::Tensor depth_out = torch::full({1, H, W}, 100.0f, opt);
  torch::Tensor radii = torch::zeros({M}, opt);

  if (M == 0) {
    return {render, depth_out, radii};
  }

  // ---- Step 1: Project all points to clip space ----
  torch::Tensor ones = torch::ones({M, 1}, opt);
  torch::Tensor pts4 = torch::cat({means3D, ones}, 1);  // [M, 4]
  torch::Tensor full_proj = torch::matmul(proj_matrix, world_view);  // [4, 4]
  torch::Tensor clip = torch::matmul(pts4, full_proj.t());  // [M, 4]

  torch::Tensor clip_w = clip.index({torch::indexing::Slice(), 3}).clamp_min(1e-8f);
  torch::Tensor ndc_x = clip.index({torch::indexing::Slice(), 0}) / clip_w;
  torch::Tensor ndc_y = clip.index({torch::indexing::Slice(), 1}) / clip_w;
  torch::Tensor ndc_z = clip.index({torch::indexing::Slice(), 2}) / clip_w;

  // NDC -> pixel
  torch::Tensor px = (ndc_x * 0.5f + 0.5f) * static_cast<float>(W);
  torch::Tensor py = (ndc_y * 0.5f + 0.5f) * static_cast<float>(H);

  // ---- Step 2: Filter in-bounds Gaussians ----
  torch::Tensor valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
                       & (clip_w > 0) & (ndc_z > 0) & (ndc_z < 1);

  if (!valid.any().item<bool>()) {
    return {render, depth_out, radii};
  }

  torch::Tensor idx = valid.nonzero().squeeze(1);  // indices of valid Gaussians

  torch::Tensor v_px = px.index({idx}).to(torch::kInt64).clamp(0, W - 1);
  torch::Tensor v_py = py.index({idx}).to(torch::kInt64).clamp(0, H - 1);
  torch::Tensor v_z = ndc_z.index({idx});         // [V]
  torch::Tensor v_color = colors.index({idx});     // [V, 3]
  torch::Tensor v_op = torch::sigmoid(opacity.index({idx}));  // [V, 1]

  // ---- Step 3: Compute approximate screen radii ----
  // For the 2D Gaussian screen radius: use a simplified projection
  torch::Tensor v_scale = scaling.index({idx});  // [V, 3]
  torch::Tensor mean_scale = v_scale.mean(1, true);  // [V, 1]
  // Screen size ~ focal * scale / depth
  float focal = static_cast<float>(W) / (2.0f * std::tan(FoVx * 0.5f) + 1e-8f);
  torch::Tensor pixel_radii = focal * mean_scale / (clip_w.index({idx}) + 1e-8f);
  pixel_radii = pixel_radii.clamp(0.5f, 100.0f);

  // ---- Step 4: Sort by depth (near to far for front-to-back compositing) ----
  auto [sorted_z, sort_idx] = torch::sort(v_z, 0, false);  // ascending depth
  torch::Tensor s_px = v_px.index({sort_idx});
  torch::Tensor s_py = v_py.index({sort_idx});
  torch::Tensor s_color = v_color.index({sort_idx});
  torch::Tensor s_op = v_op.index({sort_idx}).squeeze(1);  // [V]
  torch::Tensor s_radius = pixel_radii.index({sort_idx}).squeeze(1);  // [V]

  // ---- Step 5: Alpha compositing via scatter (pixel-parallel) ----
  // For each pixel, accumulate: C = sum(alpha_i * T_i * color_i)
  // where T_i = product(1 - alpha_j) for j < i
  torch::Tensor pixel_idx = s_py * W + s_px;  // [V] flattened pixel index

  // Compute transmittance via exclusive cumulative product
  torch::Tensor one_minus_alpha = 1.0f - s_op;  // [V]

  // For scattered compositing, use a 2-pass approach:
  // Pass 1: sort by pixel_idx then depth within each pixel group
  auto [sorted_px_idx, px_sort_idx] = torch::sort(pixel_idx, 0, false);
  s_px = s_px.index({px_sort_idx});
  s_py = s_py.index({px_sort_idx});
  s_color = s_color.index({px_sort_idx});
  s_op = s_op.index({px_sort_idx});
  s_radius = s_radius.index({px_sort_idx});
  one_minus_alpha = one_minus_alpha.index({px_sort_idx});
  pixel_idx = pixel_idx.index({px_sort_idx});

  // Compute T_i = cumprod(1-alpha) with shift (exclusive)
  torch::Tensor T = torch::ones({one_minus_alpha.size(0)}, one_minus_alpha.options());
  if (one_minus_alpha.size(0) > 1) {
    // Use rolling product: T[i] = product of one_minus_alpha[0..i-1]
    // Shift: T[1:] = cumprod(1-alpha)[0:-1], T[0] = 1
    torch::Tensor cumprod_oma = torch::cumprod(one_minus_alpha, 0);
    T.index_put_({torch::indexing::Slice(1, torch::indexing::None)},
                 cumprod_oma.index({torch::indexing::Slice(0, -1)}));
  }
  T = T.unsqueeze(1);  // [V, 1]

  // Weight per channel
  torch::Tensor weight = (s_op.unsqueeze(1) * T);  // [V, 1]

  // ---- Step 6: Scatter-add to output buffers ----
  // RGB: 3 channels
  for (int c = 0; c < 3; ++c) {
    render.index_put_({c, s_py, s_px},
        render.index({c, s_py, s_px}) +
        weight.squeeze(1) * s_color.index({torch::indexing::Slice(), c}),
        true);  // accumulate=true
  }

  // Depth: overwrite with nearest
  depth_out.index_put_({0, s_py, s_px}, sorted_z,
      false);  // accumulate=false

  // Radii for valid Gaussians
  radii.index_put_({idx}, pixel_radii.squeeze(1));

  return {render, depth_out, radii};
}

RenderOutput ScaffoldRenderer::render(
    GaussianModel& model,
    torch::Tensor& visible_mask,
    const torch::Tensor& camera_center,
    const torch::Tensor& world_view_transform,
    const torch::Tensor& projection_matrix,
    float FoVx, float FoVy,
    int image_height, int image_width,
    const torch::Tensor& bg_color,
    float scaling_modifier) {

  RenderOutput output;

  // Step 1: Get visible anchor data
  AnchorData anchor_data = model.getVisibleAnchors(visible_mask);

  if (anchor_data.num_anchors == 0) {
    auto device = camera_center.device();
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    output.color = bg_color.unsqueeze(1).unsqueeze(2)
        .expand({3, image_height, image_width}).clone();
    output.depth = torch::full({1, image_height, image_width}, 100.0f, opt);
    output.alpha = torch::zeros({1, image_height, image_width}, opt);
    output.radii = torch::zeros({0}, opt);
    return output;
  }

  // Step 2: Expand anchors to child Gaussians via MLP
  ExpandedGaussians expanded = model.mlp().expandAnchorsToGaussians(
      anchor_data, camera_center, /*camera_uid=*/0, scaling_modifier);

  // Step 3: Filter to active children
  torch::Tensor active_mask = expanded.child_mask;

  torch::Tensor xyz, color, opacity, scaling, rotation;

  if (active_mask.any().item<bool>()) {
    xyz = expanded.xyz.index({active_mask});
    color = expanded.color.index({active_mask});
    opacity = expanded.opacity.index({active_mask});
    scaling = expanded.scaling.index({active_mask});
    rotation = expanded.rotation.index({active_mask});
  } else {
    // No children pass opacity threshold — return background
    auto device = camera_center.device();
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    output.color = bg_color.unsqueeze(1).unsqueeze(2)
        .expand({3, image_height, image_width}).clone();
    output.depth = torch::full({1, image_height, image_width}, 100.0f, opt);
    output.alpha = torch::zeros({1, image_height, image_width}, opt);
    output.radii = torch::zeros({0}, opt);
    return output;
  }

  // Step 4: Rasterize via CUDA kernel (if available) or pure-LibTorch fallback
#ifdef HAVE_CUDA_RASTERIZER
  auto [rendered_color, rendered_depth, radii] = RasterizeGaussians(
      xyz, color, opacity, scaling, rotation,
      world_view_transform, projection_matrix,
      FoVx, FoVy, image_height, image_width, bg_color);
#else
  auto [rendered_color, rendered_depth, radii] = renderGaussiansGPU(
      xyz, color, opacity, scaling, rotation,
      world_view_transform, projection_matrix,
      FoVx, FoVy, image_height, image_width, bg_color);
#endif

  output.color = rendered_color;
  output.depth = rendered_depth;
  output.radii = radii;

  // Alpha from opacity
  output.alpha = (rendered_color != bg_color.unsqueeze(1).unsqueeze(2))
      .to(torch::kFloat32).mean(0, true);

  return output;
}

RenderOutput ScaffoldRenderer::renderInference(
    GaussianModel& model,
    torch::Tensor& visible_mask,
    const torch::Tensor& camera_center,
    const torch::Tensor& world_view_transform,
    const torch::Tensor& projection_matrix,
    float FoVx, float FoVy,
    int image_height, int image_width,
    const torch::Tensor& bg_color,
    float scaling_modifier) {
  torch::NoGradGuard no_grad;
  return render(model, visible_mask, camera_center, world_view_transform,
                projection_matrix, FoVx, FoVy, image_height, image_width,
                bg_color, scaling_modifier);
}

}  // namespace scaffold_chungs
