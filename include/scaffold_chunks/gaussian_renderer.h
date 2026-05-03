/**
 * Scaffold-ChunGS: Renderer that expands anchors to child Gaussians
 * and rasterizes via the CUDA 3DGS rasterizer.
 */

#pragma once

#include <torch/torch.h>
#include <memory>

#include "gaussian_model.h"

namespace scaffold_chungs {

struct RenderOutput {
  torch::Tensor color;   // [3, H, W] rendered RGB
  torch::Tensor depth;   // [1, H, W] rendered depth
  torch::Tensor alpha;   // [1, H, W] rendered alpha
  torch::Tensor radii;   // [M] screen-space radii of rendered children
};

/**
 * ScaffoldRenderer: expands anchors via MLP decoders, then rasterizes.
 *
 * Uses the standard CUDA Gaussian rasterizer. The key difference from
 * standard 3DGS is that gaussian parameters are NOT stored directly —
 * they are generated on-the-fly by the AnchorMLP for visible anchors.
 */
class ScaffoldRenderer {
 public:
  ScaffoldRenderer() = default;

  /**
   * Render the scene from a camera viewpoint.
   *
   * @param model         Anchor-based Gaussian model
   * @param visible_mask  [N] bool tensor selecting visible anchors
   * @param camera_center [3] tensor, camera world position
   * @param world_view_transform  [4, 4] world-to-view matrix
   * @param projection_matrix     [4, 4] projection matrix
   * @param FoVx, FoVy    Field of view (radians)
   * @param image_height, image_width  Output resolution
   * @param bg_color      [3] background color tensor
   * @param scaling_modifier  Global scale adjustment
   * @return RenderOutput with color, depth, alpha, radii
   */
  static RenderOutput render(
      GaussianModel& model,
      torch::Tensor& visible_mask,
      const torch::Tensor& camera_center,
      const torch::Tensor& world_view_transform,
      const torch::Tensor& projection_matrix,
      float FoVx, float FoVy,
      int image_height, int image_width,
      const torch::Tensor& bg_color,
      float scaling_modifier = 1.0f);

  /**
   * No-gradient variant for inference-only / visualization.
   */
  static RenderOutput renderInference(
      GaussianModel& model,
      torch::Tensor& visible_mask,
      const torch::Tensor& camera_center,
      const torch::Tensor& world_view_transform,
      const torch::Tensor& projection_matrix,
      float FoVx, float FoVy,
      int image_height, int image_width,
      const torch::Tensor& bg_color,
      float scaling_modifier = 1.0f);
};

}  // namespace scaffold_chungs
