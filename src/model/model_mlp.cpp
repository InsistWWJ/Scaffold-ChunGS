/**
 * Scaffold-ChunGS: MLP decoders for anchor-to-Gaussian expansion.
 *
 * Implements the three sub-MLPs:
 *   mlp_opacity_ — per-child opacity logits
 *   mlp_cov_     — per-child scale residual + rotation quaternion
 *   mlp_color_   — per-child RGB color
 *
 * Together these decode anchor features + view direction into child Gaussian
 * parameters. The architecture follows the Scaffold-GS paper from Compact_GSSLAM.
 */

#include "scaffold_chunks/anchor_mlp.h"
#include "scaffold_chunks/chunk_types.h"

namespace scaffold_chungs {

// =============================================================================
// Construction
// =============================================================================

AnchorMLP::AnchorMLP(int feat_dim, int n_offsets,
                     bool use_feat_bank,
                     int appearance_dim,
                     bool add_opacity_dist,
                     bool add_cov_dist,
                     bool add_color_dist)
    : feat_dim_(feat_dim),
      n_offsets_(n_offsets),
      appearance_dim_(appearance_dim),
      use_feat_bank_(use_feat_bank),
      add_opacity_dist_(add_opacity_dist),
      add_cov_dist_(add_cov_dist),
      add_color_dist_(add_color_dist) {

  // Dimensions for MLP inputs
  // Each MLP takes: [feat_dim + 3 (view_dir) + optional_dist (1)]
  int opacity_dist = add_opacity_dist_ ? 1 : 0;
  int cov_dist = add_cov_dist_ ? 1 : 0;
  int color_dist = add_color_dist_ ? 1 : 0;

  int opacity_input_dim = feat_dim_ + 3 + opacity_dist;
  int cov_input_dim = feat_dim_ + 3 + cov_dist;
  int color_input_dim = feat_dim_ + 3 + color_dist + appearance_dim_;

  // ---- mlp_opacity_: [input] -> [feat_dim] -> [n_offsets] -> Tanh ----
  {
    torch::nn::Sequential seq;
    seq->push_back(torch::nn::Linear(
        torch::nn::LinearOptions(opacity_input_dim, feat_dim_)));
    seq->push_back(torch::nn::Functional(torch::relu));
    seq->push_back(torch::nn::Linear(
        torch::nn::LinearOptions(feat_dim_, n_offsets_)));
    seq->push_back(torch::nn::Functional(torch::tanh));
    mlp_opacity_ = std::move(seq);
    register_module("mlp_opacity", mlp_opacity_);
  }

  // ---- mlp_cov_: [input] -> [feat_dim] -> [7 * n_offsets] ----
  {
    torch::nn::Sequential seq;
    seq->push_back(torch::nn::Linear(
        torch::nn::LinearOptions(cov_input_dim, feat_dim_)));
    seq->push_back(torch::nn::Functional(torch::relu));
    seq->push_back(torch::nn::Linear(
        torch::nn::LinearOptions(feat_dim_, 7 * n_offsets_)));
    mlp_cov_ = std::move(seq);
    register_module("mlp_cov", mlp_cov_);
  }

  // ---- mlp_color_: [input] -> [feat_dim] -> [3 * n_offsets] -> Sigmoid ----
  {
    torch::nn::Sequential seq;
    seq->push_back(torch::nn::Linear(
        torch::nn::LinearOptions(color_input_dim, feat_dim_)));
    seq->push_back(torch::nn::Functional(torch::relu));
    seq->push_back(torch::nn::Linear(
        torch::nn::LinearOptions(feat_dim_, 3 * n_offsets_)));
    seq->push_back(torch::nn::Functional(torch::sigmoid));
    mlp_color_ = std::move(seq);
    register_module("mlp_color", mlp_color_);
  }

  // ---- Optional: mlp_feature_bank_ (view-adaptive attention) ----
  if (use_feat_bank_) {
    torch::nn::Sequential seq;
    seq->push_back(torch::nn::Linear(
        torch::nn::LinearOptions(4, feat_dim_)));  // view_dir[3] + dist[1]
    seq->push_back(torch::nn::Functional(torch::relu));
    seq->push_back(torch::nn::Linear(
        torch::nn::LinearOptions(feat_dim_, 3)));
    seq->push_back(torch::nn::Functional(
        [](const torch::Tensor& x) { return torch::softmax(x, 1); }));
    mlp_feature_bank_ = std::move(seq);
    register_module("mlp_feature_bank", mlp_feature_bank_);
  }
}

void AnchorMLP::to(torch::Device device) {
  mlp_opacity_->to(device);
  mlp_cov_->to(device);
  mlp_color_->to(device);
  if (use_feat_bank_ && mlp_feature_bank_) {
    mlp_feature_bank_->to(device);
  }
  if (appearance_dim_ > 0 && appearance_embedding_) {
    appearance_embedding_->to(device);
  }
}

void AnchorMLP::setAppearanceEmbedding(int num_cameras) {
  if (appearance_dim_ > 0 && num_cameras > 0) {
    appearance_embedding_ = torch::nn::Embedding(
        torch::nn::EmbeddingOptions(num_cameras, appearance_dim_));
    register_module("appearance_embedding", appearance_embedding_);
  }
}

std::vector<torch::Tensor> AnchorMLP::parameters() {
  std::vector<torch::Tensor> params;
  for (auto& p : mlp_opacity_->parameters()) {
    params.push_back(p);
  }
  for (auto& p : mlp_cov_->parameters()) {
    params.push_back(p);
  }
  for (auto& p : mlp_color_->parameters()) {
    params.push_back(p);
  }
  if (use_feat_bank_ && mlp_feature_bank_) {
    for (auto& p : mlp_feature_bank_->parameters()) {
      params.push_back(p);
    }
  }
  if (appearance_embedding_ && appearance_embedding_->options.out_features() > 0) {
    for (auto& p : appearance_embedding_->parameters()) {
      params.push_back(p);
    }
  }
  return params;
}

// =============================================================================
// Core Expansion Logic
// =============================================================================

ExpandedGaussians AnchorMLP::expandAnchorsForInference(
    const AnchorData& anchors,
    const torch::Tensor& camera_center,
    int64_t camera_uid,
    float scaling_modifier) {
  torch::NoGradGuard no_grad;
  return expandAnchorsToGaussians(anchors, camera_center, camera_uid,
                                  scaling_modifier);
}

ExpandedGaussians AnchorMLP::expandAnchorsToGaussians(
    const AnchorData& anchors,
    const torch::Tensor& camera_center,
    int64_t camera_uid,
    float scaling_modifier) {

  int V = anchors.num_anchors;  // number of visible anchors
  int K = anchors.n_offsets;    // children per anchor
  int D = anchors.feat_dim;
  auto device = anchors.anchor.device();

  // ---- Validate inputs ----
  if (V == 0) {
    ExpandedGaussians empty;
    empty.xyz = torch::zeros({0, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    empty.color = torch::zeros({0, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    empty.opacity = torch::zeros({0, 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    empty.scaling = torch::zeros({0, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    empty.rotation = torch::zeros({0, 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    empty.child_mask = torch::zeros({0},
        torch::TensorOptions().dtype(torch::kBool).device(device));
    empty.num_visible_anchors = 0;
    empty.num_child_gaussians = 0;
    return empty;
  }

  auto feat = anchors.anchor_feat;            // [V, D]
  auto anchor_pos = anchors.anchor;           // [V, 3]
  auto offsets = anchors.offset;              // [V, K, 3]
  auto scaling_base = anchors.anchor_scaling; // [V, 6]
  // scaling_base[:, :3]: base scale for offset modulation
  // scaling_base[:, 3:]: extra scale for child Gaussian modulation

  // ---- Step 1: View-dependent computations ----
  torch::Tensor ob_view = anchor_pos - camera_center;  // [V, 3]
  torch::Tensor ob_dist = torch::norm(ob_view, 2, 1, true);  // [V, 1]
  ob_view = ob_view / (ob_dist + 1e-8f);  // normalize

  // ---- Step 2: Build MLP inputs ----
  bool any_dist = add_opacity_dist_ || add_cov_dist_ || add_color_dist_;

  // For opacity MLP
  torch::Tensor opacity_input;
  if (add_opacity_dist_) {
    opacity_input = torch::cat({feat, ob_view, ob_dist}, 1);  // [V, D+3+1]
  } else {
    opacity_input = torch::cat({feat, ob_view}, 1);           // [V, D+3]
  }

  // For cov MLP
  torch::Tensor cov_input;
  if (add_cov_dist_) {
    cov_input = torch::cat({feat, ob_view, ob_dist}, 1);
  } else {
    cov_input = torch::cat({feat, ob_view}, 1);
  }

  // For color MLP
  torch::Tensor color_input;
  if (appearance_dim_ > 0 && appearance_embedding_ &&
      appearance_embedding_->options.out_features() > 0) {
    auto cam_idx = torch::full({V}, camera_uid,
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto app_emb = appearance_embedding_->forward(cam_idx);  // [V, app_dim]
    if (add_color_dist_) {
      color_input = torch::cat({feat, ob_view, ob_dist, app_emb}, 1);
    } else {
      color_input = torch::cat({feat, ob_view, app_emb}, 1);
    }
  } else {
    if (add_color_dist_) {
      color_input = torch::cat({feat, ob_view, ob_dist}, 1);
    } else {
      color_input = torch::cat({feat, ob_view}, 1);
    }
  }

  // ---- Step 3: MLP forward pass ----
  // Opacity: [V, K]
  torch::Tensor opacity_raw = mlp_opacity_->forward(opacity_input);  // [V, K]
  opacity_raw = opacity_raw.reshape({V * K, 1});                     // [V*K, 1]

  // Color: [V, 3*K]
  torch::Tensor color_raw = mlp_color_->forward(color_input);        // [V, 3*K]
  color_raw = color_raw.reshape({V * K, 3});                         // [V*K, 3]

  // Covariance: [V, 7*K]
  torch::Tensor cov_raw = mlp_cov_->forward(cov_input);              // [V, 7*K]
  cov_raw = cov_raw.reshape({V * K, 7});                             // [V*K, 7]

  // ---- Step 4: Child mask from opacity > 0 ----
  torch::Tensor child_mask = opacity_raw.squeeze(1) > 0.0f;  // [V*K] bool

  // ---- Step 5: Expand per-anchor tensors to per-child [V*K] ----
  // scaling_base: [V, 6] -> [V*K, 6]
  auto expand_to_children = [V, K](const torch::Tensor& x, int feat_dim_val) {
    // x: [V, feat_dim_val]
    // returns: [V*K, feat_dim_val]
    auto x_expanded = x.unsqueeze(1).expand({V, K, feat_dim_val});  // [V, K, F]
    return x_expanded.reshape({V * K, feat_dim_val});               // [V*K, F]
  };

  torch::Tensor scaling_repeat = expand_to_children(scaling_base, 6);  // [V*K, 6]
  torch::Tensor anchor_repeat = expand_to_children(anchor_pos, 3);     // [V*K, 3]
  torch::Tensor offsets_flat = offsets.reshape({V * K, 3});             // [V*K, 3]

  // ---- Step 6: Compute child Gaussian parameters ----
  // Scale: scaling_repeat[:, 3:] * sigmoid(cov_raw[:, :3])
  torch::Tensor scale_residual = torch::sigmoid(cov_raw.index(
      {torch::indexing::Slice(), torch::indexing::Slice(0, 3)}));   // [V*K, 3]
  torch::Tensor child_scaling = scaling_repeat.index(
      {torch::indexing::Slice(), torch::indexing::Slice(3, 6)}) * scale_residual;  // [V*K, 3]

  // Rotation: normalize cov_raw[:, 3:7]
  torch::Tensor quat_raw = cov_raw.index(
      {torch::indexing::Slice(), torch::indexing::Slice(3, 7)});  // [V*K, 4]
  torch::Tensor child_rotation = torch::nn::functional::normalize(
      quat_raw, torch::nn::functional::NormalizeFuncOptions().dim(1));  // [V*K, 4]

  // Position: anchor_repeat + offsets_flat * scaling_repeat[:, :3]
  torch::Tensor child_xyz = anchor_repeat +
      offsets_flat * scaling_repeat.index(
          {torch::indexing::Slice(), torch::indexing::Slice(0, 3)});  // [V*K, 3]

  // Scale: apply global scaling modifier
  child_scaling = child_scaling * scaling_modifier;

  // ---- Step 7: Apply child mask to filter active Gaussians ----
  // Return unmasked tensors along with mask - rasterizer will filter
  int total_children = V * K;
  int num_active = child_mask.sum().item<int>();

  ExpandedGaussians result;
  result.xyz = child_xyz;
  result.color = color_raw;
  result.opacity = opacity_raw;
  result.scaling = child_scaling;
  result.rotation = child_rotation;
  result.child_mask = child_mask;
  result.num_visible_anchors = V;
  result.num_child_gaussians = total_children;  // includes inactive; mask filters

  return result;
}

}  // namespace scaffold_chungs
