/**
 * Scaffold-ChunGS: Renderer that expands anchors to child Gaussians
 * via MLP decoders and feeds them to the CUDA 3DGS rasterizer.
 */

#include "scaffold_chunks/gaussian_renderer.h"
#include "scaffold_chunks/anchor_mlp.h"
#include "scaffold_chunks/chunk_types.h"

#include <iostream>

namespace scaffold_chungs {

// =============================================================================
// CUDA Rasterizer Interface (simplified — delegates to torch ops)
// =============================================================================
//
// In production, this would link against the actual CUDA 3DGS rasterizer
// (diff_gaussian_rasterization). Here we implement a minimal placeholder
// that performs the projection math in pure LibTorch. For real deployment,
// replace renderGaussiansCUDA with the actual CUDA rasterizer call.

// =============================================================================
// PLACEHOLDER RASTERIZER — NOT FOR PRODUCTION USE
// =============================================================================
//
// This function uses CPU-side per-Gaussian loops with .item() calls that
// force CUDA synchronization for EVERY Gaussian EVERY frame.
//
// On Jetson Orin (GA10B, 1024 cores), a single 1080p frame with 1.5M Gaussians
// takes MINUTES to render, making real-time operation impossible.
//
// REPLACE THIS with INRIA's diff_gaussian_rasterization CUDA kernel before
// any production deployment. The CUDA tile-based rasterizer delivers
// real-time performance on embedded GPUs.
// =============================================================================

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
renderGaussiansCUDA(
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

  // Pre-allocate reusable buffers (will persist across calls via static)
  // Note: H/W may vary between keyframes; re-alloc only when dimensions change
  static torch::Tensor render, depth;
  static int last_H = 0, last_W = 0;

  if (H != last_H || W != last_W || !render.defined() ||
      render.device() != device) {
    render = background.unsqueeze(1).unsqueeze(2).expand({3, H, W}).clone();
    depth = torch::full({1, H, W}, 100.0f, opt);
    last_H = H; last_W = W;
  } else {
    render.copy_(background.unsqueeze(1).unsqueeze(2).expand({3, H, W}));
    depth.fill_(100.0f);
  }

  torch::Tensor radii = torch::zeros({means3D.size(0)}, opt);

  if (means3D.size(0) == 0) {
    return {render, depth, radii};
  }

  // Project 3D points to 2D
  torch::Tensor ones = torch::ones({means3D.size(0), 1}, opt);
  torch::Tensor pts4 = torch::cat({means3D, ones}, 1);  // [M, 4]

  torch::Tensor full_proj = torch::matmul(proj_matrix, world_view);
  torch::Tensor clip = torch::matmul(full_proj, pts4.t()).t();  // [M, 4]

  // Perspective divide
  torch::Tensor clip_w = clip.index({torch::indexing::Slice(), 3}).clamp_min(1e-6f);
  torch::Tensor ndc_x = clip.index({torch::indexing::Slice(), 0}) / clip_w;
  torch::Tensor ndc_y = clip.index({torch::indexing::Slice(), 1}) / clip_w;
  torch::Tensor ndc_z = clip.index({torch::indexing::Slice(), 2}) / clip_w;

  // NDC -> pixel
  torch::Tensor px = (ndc_x * 0.5f + 0.5f) * W;
  torch::Tensor py = (ndc_y * 0.5f + 0.5f) * H;

  // Filter points in valid pixel range
  torch::Tensor valid = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (clip_w > 0);

  if (!valid.any().item<bool>()) {
    return {render, depth, radii};
  }

  // For valid points: compute screen-space radius from scaling
  torch::Tensor valid_px = px.index({valid}).to(torch::kInt64).clamp(0, W - 1);
  torch::Tensor valid_py = py.index({valid}).to(torch::kInt64).clamp(0, H - 1);
  torch::Tensor valid_color = colors.index({valid});
  torch::Tensor valid_op = torch::sigmoid(opacity.index({valid}));
  torch::Tensor valid_depth = ndc_z.index({valid});

  // NOTE: This CPU loop with .item() calls is the bottleneck.
  // Each .item() forces a full CUDA synchronization. For 1.5M Gaussians
  // this is ~6M synchronizations per frame. Replace with CUDA kernel.
  for (int64_t i = 0; i < valid_px.size(0); ++i) {
    int x = valid_px[i].item<int>();
    int y = valid_py[i].item<int>();
    float op = valid_op[i].item<float>();

    if (op > 0.01f) {
      for (int c = 0; c < 3; ++c) {
        float cur = render[c][y][x].item<float>();
        float col = valid_color[i][c].item<float>();
        render[c][y][x] = cur * (1.0f - op) + col * op;
      }
      if (valid_depth[i].item<float>() < depth[0][y][x].item<float>()) {
        depth[0][y][x] = valid_depth[i];
      }
    }
  }

  radii = torch::ones({means3D.size(0)}, opt) * 2.0f;

  return {render, depth, radii};
}

// =============================================================================
// ScaffoldRenderer (public API)
// =============================================================================

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

  // Step 4: CUDA rasterizer
  auto [rendered_color, rendered_depth, radii] = renderGaussiansCUDA(
      xyz, color, opacity, scaling, rotation,
      world_view_transform, projection_matrix,
      FoVx, FoVy, image_height, image_width, bg_color);

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
