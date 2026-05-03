/**
 * Scaffold-ChunGS: MLP decoders that transform anchor features into
 * child Gaussian parameters (opacity, color, covariance).
 *
 * Based on the Scaffold-GS architecture from Compact_GSSLAM.
 */

#pragma once

#include <torch/torch.h>
#include <memory>

namespace scaffold_chungs {

// =============================================================================
// Anchor Data Pack (for passing through the rendering pipeline)
// =============================================================================

struct AnchorData {
  torch::Tensor anchor;          // [V, 3] anchor positions
  torch::Tensor offset;          // [V, K, 3] per-child offsets
  torch::Tensor anchor_feat;     // [V, D] per-anchor features
  torch::Tensor anchor_scaling;  // [V, 6] base scale ([:3] for offset, [3:] for child)
  torch::Tensor anchor_rotation; // [V, 4] base rotation quaternion
  torch::Tensor anchor_opacity;  // [V, 1] base opacity (inverse-sigmoid)
  int num_anchors = 0;           // V
  int n_offsets = 0;             // K
  int feat_dim = 0;              // D
};

// =============================================================================
// Expanded Gaussian Output
// =============================================================================

struct ExpandedGaussians {
  torch::Tensor xyz;         // [M, 3] child Gaussian world positions
  torch::Tensor color;       // [M, 3] decoded colors (RGB, range [0,1])
  torch::Tensor opacity;     // [M, 1] decoded opacity (pre-sigmoid space)
  torch::Tensor scaling;     // [M, 3] decoded scaling per axis
  torch::Tensor rotation;    // [M, 4] decoded rotation quaternion
  torch::Tensor child_mask;  // [V*K] bool mask of children that pass opacity threshold
  int num_visible_anchors = 0;  // V
  int num_child_gaussians = 0;  // M
};

// =============================================================================
// Anchor MLP Decoders
// =============================================================================

class AnchorMLP {
 public:
  AnchorMLP(int feat_dim = 32, int n_offsets = 5,
            bool use_feat_bank = false,
            int appearance_dim = 0,
            bool add_opacity_dist = true,
            bool add_cov_dist = true,
            bool add_color_dist = true);

  // Discard copy/move (owns LibTorch modules)
  AnchorMLP(const AnchorMLP&) = delete;
  AnchorMLP& operator=(const AnchorMLP&) = delete;

  // Transfer all MLPs to a different device
  void to(torch::Device device);

  // Set up appearance embedding (call once with number of cameras)
  void setAppearanceEmbedding(int num_cameras);

  // ============================================================================
  // Core Expansion: Anchor -> Child Gaussians
  // ============================================================================

  /**
   * Expand visible anchors into child Gaussians via MLP decoding.
   * Gradients flow from child Gaussian losses back through MLPs to anchor features.
   *
   * @param anchors   Visible anchor data (pre-sliced by frustum culling).
   * @param camera_center  [3] tensor of camera world position.
   * @param camera_uid     Camera index for appearance embedding (ignored if appearance_dim=0).
   * @param scaling_modifier  Global scale factor (typically 1.0).
   * @return ExpandedGaussians ready for CUDA rasterization.
   */
  ExpandedGaussians expandAnchorsToGaussians(
      const AnchorData& anchors,
      const torch::Tensor& camera_center,
      int64_t camera_uid = 0,
      float scaling_modifier = 1.0f);

  /**
   * No-gradient variant for inference-only rendering.
   */
  ExpandedGaussians expandAnchorsForInference(
      const AnchorData& anchors,
      const torch::Tensor& camera_center,
      int64_t camera_uid = 0,
      float scaling_modifier = 1.0f);

  // ============================================================================
  // Parameter Access (for optimizer setup)
  // ============================================================================

  std::vector<torch::Tensor> parameters();
  const torch::nn::Embedding& appearanceEmbedding() const {
    return *appearance_embedding_;
  }
  bool hasAppearanceEmbedding() const { return appearance_dim_ > 0; }

 private:
  int feat_dim_;
  int n_offsets_;
  int appearance_dim_;
  bool use_feat_bank_;
  bool add_opacity_dist_;
  bool add_cov_dist_;
  bool add_color_dist_;

  // MLP sub-networks
  // mlp_opacity_: [feat_dim + 3 + dist] -> [feat_dim] -> [n_offsets] -> Tanh
  torch::nn::Sequential mlp_opacity_{nullptr};

  // mlp_cov_: [feat_dim + 3 + dist] -> [feat_dim] -> [7 * n_offsets]
  torch::nn::Sequential mlp_cov_{nullptr};

  // mlp_color_: [feat_dim + 3 + dist + app_dim] -> [feat_dim] -> [3 * n_offsets] -> Sigmoid
  torch::nn::Sequential mlp_color_{nullptr};

  // Optional: view-adaptive feature bank (attention-weighted features)
  torch::nn::Sequential mlp_feature_bank_{nullptr};

  // Optional: per-camera appearance embedding
  torch::nn::Embedding appearance_embedding_{nullptr};
};

}  // namespace scaffold_chungs
