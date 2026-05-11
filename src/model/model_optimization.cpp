/**
 * Scaffold-ChunGS: Optimizer setup and anchor densification.
 *
 * The Adam optimizer has 8 parameter groups (kNumAnchorParamGroups):
 *   0=anchor_ (position), 1=offset_, 2=anchor_feat_, 3=anchor_opacity_,
 *   4=anchor_scaling_, 5=anchor_rotation_, 6=MLP weights, 7=appearance emb
 */

#include "scaffold_chunks/gaussian_model.h"

#include <cmath>
#include <iostream>

namespace scaffold_chungs {

// =============================================================================
// Learning Rate Decay Helper
// =============================================================================

static float expLrFunc(int iteration, float lr_init, float lr_decay) {
  return lr_init * std::pow(lr_decay, static_cast<float>(iteration));
}

// =============================================================================
// Training Setup — Register all params with Adam
// =============================================================================

void GaussianModel::trainingSetup(const OptimizationConfig& opt_cfg) {
  if (!is_initialized_) {
    throw std::runtime_error("GaussianModel not initialized — call "
                             "initializeFromPoints or addAnchors first");
  }

  auto device = device_type_ == torch::kCUDA ? torch::Device(torch::kCUDA)
                                             : torch::Device(torch::kCPU);

  // Ensure tracking tensors are on the right device
  auto opt_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  if (offset_gradient_accum_.size(0) != anchor_.size(0) * n_offsets_) {
    offset_gradient_accum_ = torch::zeros({anchor_.size(0) * n_offsets_, 1},
                                          opt_float);
    offset_denom_ = torch::zeros({anchor_.size(0) * n_offsets_, 1}, opt_float);
  }
  if (opacity_accum_.size(0) != anchor_.size(0)) {
    opacity_accum_ = torch::zeros({anchor_.size(0), 1}, opt_float);
    anchor_denom_ = torch::zeros({anchor_.size(0), 1}, opt_float);
  }

  spatial_lr_scale_ = opt_cfg.spatial_lr_scale;

  // Build param groups
  std::vector<torch::optim::OptimizerParamGroup> param_groups;

  // Group 0: anchor_positions with per-anchor LRs
  Tensor_vec_anchor_ = {anchor_};
  param_groups.push_back(torch::optim::OptimizerParamGroup(
      {anchor_}, std::make_unique<torch::optim::AdamOptions>(
          opt_cfg.anchor_position_lr_init)));

  // Group 1: offsets
  Tensor_vec_offset_ = {offset_};
  param_groups.push_back(torch::optim::OptimizerParamGroup(
      {offset_}, std::make_unique<torch::optim::AdamOptions>(
          opt_cfg.offset_lr)));

  // Group 2: anchor features
  Tensor_vec_anchor_feat_ = {anchor_feat_};
  param_groups.push_back(torch::optim::OptimizerParamGroup(
      {anchor_feat_}, std::make_unique<torch::optim::AdamOptions>(
          opt_cfg.anchor_feature_lr)));

  // Group 3: anchor opacity
  Tensor_vec_anchor_opacity_ = {anchor_opacity_};
  param_groups.push_back(torch::optim::OptimizerParamGroup(
      {anchor_opacity_}, std::make_unique<torch::optim::AdamOptions>(
          opt_cfg.anchor_opacity_lr)));

  // Group 4: anchor scaling
  Tensor_vec_anchor_scaling_ = {anchor_scaling_};
  param_groups.push_back(torch::optim::OptimizerParamGroup(
      {anchor_scaling_}, std::make_unique<torch::optim::AdamOptions>(
          opt_cfg.anchor_scaling_lr)));

  // Group 5: anchor rotation
  Tensor_vec_anchor_rotation_ = {anchor_rotation_};
  param_groups.push_back(torch::optim::OptimizerParamGroup(
      {anchor_rotation_}, std::make_unique<torch::optim::AdamOptions>(
          opt_cfg.anchor_rotation_lr)));

  // Group 6: MLP parameters
  auto mlp_params = mlp_.parameters();
  if (!mlp_params.empty()) {
    param_groups.push_back(torch::optim::OptimizerParamGroup(
        mlp_params, std::make_unique<torch::optim::AdamOptions>(
            opt_cfg.mlp_opacity_lr)));
  } else {
    // Placeholder empty group
    param_groups.push_back(torch::optim::OptimizerParamGroup(
        {torch::empty({0}, opt_float)},
        std::make_unique<torch::optim::AdamOptions>(0.005f)));
  }

  // Group 7: Appearance embedding
  if (mlp_.hasAppearanceEmbedding()) {
    auto& emb = mlp_.appearanceEmbedding();
    auto emb_params = emb.parameters();
    param_groups.push_back(torch::optim::OptimizerParamGroup(
        {emb_params}, std::make_unique<torch::optim::AdamOptions>(0.0001f)));
  } else {
    param_groups.push_back(torch::optim::OptimizerParamGroup(
        {torch::empty({0}, opt_float)},
        std::make_unique<torch::optim::AdamOptions>(0.0001f)));
  }

  optimizer_ = std::make_shared<torch::optim::Adam>(
      param_groups, torch::optim::AdamOptions(0.001f));

  std::cout << "[Optimization] Training setup complete: "
            << anchor_.size(0) << " anchors, "
            << param_groups.size() << " param groups\n";
}

// =============================================================================
// Optimizer Step
// =============================================================================

void GaussianModel::optimizerStep(const torch::Tensor& visibility, uint32_t N) {
  if (!optimizer_) return;

  // For sparse Adam: zero out gradients for invisible anchors before stepping.
  // This prevents invisible anchors from being updated, matching the behavior
  // of DiskChunGS's SparseGaussianAdam.
  //
  // Visibility mask: [N] bool, True = visible (contributed to rendered image).
  // Only visible anchors should receive gradient updates.
  if (visibility.defined() && visibility.size(0) == anchor_.size(0)) {
    // Zero gradients for invisible anchors
    torch::Tensor inv_mask = ~visibility;

    if (anchor_.grad().defined()) {
      anchor_.grad().index_put_({inv_mask}, 0.0f);
    }
    if (offset_.grad().defined()) {
      offset_.grad().index_put_({inv_mask}, 0.0f);
    }
    if (anchor_feat_.grad().defined()) {
      anchor_feat_.grad().index_put_({inv_mask}, 0.0f);
    }
    if (anchor_scaling_.grad().defined()) {
      anchor_scaling_.grad().index_put_({inv_mask}, 0.0f);
    }
    if (anchor_rotation_.grad().defined()) {
      anchor_rotation_.grad().index_put_({inv_mask}, 0.0f);
    }
    if (anchor_opacity_.grad().defined()) {
      anchor_opacity_.grad().index_put_({inv_mask}, 0.0f);
    }
  }

  optimizer_->step();
}

void GaussianModel::optimizerZeroGrad() {
  if (!optimizer_) return;
  optimizer_->zero_grad();
}

void GaussianModel::updateLearningRates(const torch::Tensor& visibility) {
  // Decay position learning rate for visible anchors
  if (!optimizer_) return;

  auto& param_groups = optimizer_->param_groups();
  if (param_groups.empty()) return;

  // Group 0 holds anchor positions with per-anchor LRs
  auto& group0 = param_groups[0];
  if (!group0.params().empty()) {
    auto& pos_lr = group0.options();
    float lr = pos_lr.get_lr();
    float decay = 0.99998f;
    // Clamp decayed LR to prevent vanishing
    float min_lr = 1e-8f;
    float new_lr = std::max(lr * decay, min_lr);
    if (new_lr < lr) {
      pos_lr.set_lr(new_lr);
    }
  }
}

void GaussianModel::resetOpacityForMask(const torch::Tensor& anchor_mask) {
  torch::NoGradGuard no_grad;
  float cap = std::log(0.05f / (1.0f - 0.05f));
  // In-place clamp on selected slice
  auto op_slice = anchor_opacity_.index({anchor_mask});
  anchor_opacity_.index_put_({anchor_mask}, torch::clamp_max(op_slice, cap));
}

void GaussianModel::resetAnchorLRAndOptimizerState(
    const torch::Tensor& anchor_mask) {
  // Reset Adam momentum and step for selected anchors (used after loop closure)
  if (!optimizer_) return;

  auto& param_groups = optimizer_->param_groups();
  auto& state = optimizer_->state();

  for (size_t g = 0; g < param_groups.size(); ++g) {
    auto& params = param_groups[g].params();
    if (params.empty()) continue;
    auto key = params[0].unsafeGetTensorImpl();
    auto it = state.find(key);
    if (it == state.end()) continue;

    auto& param_state =
        static_cast<torch::optim::AdamParamState&>(*it->second);

    // Reset exp_avg and exp_avg_sq for masked anchors
    if (param_state.exp_avg().defined() &&
        param_state.exp_avg().size(0) == anchor_mask.size(0)) {
      param_state.exp_avg().index_put_({anchor_mask}, 0.0f);
      param_state.exp_avg_sq().index_put_({anchor_mask}, 0.0f);
    }
    // Reset step count to 0 for affected parameters
    param_state.step(param_state.step() > 0 ? param_state.step() - 1 : 0);
  }
}

// =============================================================================
// Anchor Growing (Hierarchical Densification)
// =============================================================================

void GaussianModel::anchorGrowing(float grad_threshold, int depth_level) {
  torch::NoGradGuard no_grad;
  int64_t N = anchor_.size(0);
  int64_t NK = N * n_offsets_;
  if (N == 0 || depth_level >= update_depth_) return;

  auto device = device_type_ == torch::kCUDA ? torch::Device(torch::kCUDA)
                                             : torch::Device(torch::kCPU);

  // Normalize gradient accum
  torch::Tensor avg_grad = offset_gradient_accum_ /
      (offset_denom_ + 1e-8f);  // [N*K, 1]

  // Find children with high gradient
  float cur_threshold = grad_threshold *
      std::pow(update_hierarchy_factor_, depth_level);
  torch::Tensor high_grad = avg_grad >= cur_threshold;  // [N*K, 1]
  high_grad = high_grad.squeeze(1);  // [N*K]

  if (!high_grad.any().item<bool>()) return;

  // Compute child positions
  // xyz_child[anchor_i, offset_j] = anchor_[i] + offset_[i, j] * anchor_scaling_[i, :3]
  torch::Tensor scaling_3 = anchor_scaling_.index(
      {torch::indexing::Slice(),
       torch::indexing::Slice(0, 3)});  // [N, 3]
  torch::Tensor expanded_scale = scaling_3.unsqueeze(1).expand({N, n_offsets_, 3})
      .reshape({NK, 3});  // [N*K, 3]
  torch::Tensor offsets_flat = offset_.reshape({NK, 3});  // [N*K, 3]
  torch::Tensor anchor_repeat = anchor_.unsqueeze(1).expand({N, n_offsets_, 3})
      .reshape({NK, 3});  // [N*K, 3]
  torch::Tensor child_xyz = anchor_repeat + offsets_flat * expanded_scale;  // [N*K, 3]

  // Select children with high gradient
  torch::Tensor candidate_xyz = child_xyz.index({high_grad});  // [M, 3]

  // 50% random subsampling
  torch::Tensor rand_mask = torch::rand({candidate_xyz.size(0)},
      torch::TensorOptions().device(device)) > 0.5f;
  candidate_xyz = candidate_xyz.index({rand_mask});

  if (candidate_xyz.size(0) == 0) return;

  // Voxelize candidates at this depth level's resolution
  float cell_ratio = static_cast<float>(update_init_factor_) /
      std::pow(static_cast<float>(update_hierarchy_factor_), depth_level);
  int cur_size = static_cast<int>(std::round(voxel_size_ * cell_ratio));
  if (cur_size < 1) cur_size = 1;

  torch::Tensor vox_coords = torch::round(candidate_xyz / cur_size);
  auto [unique_vox, _inv, _cnt] = torch::_unique2(vox_coords, false, true, true);

  // Remove voxels that already contain an anchor (GPU-accelerated)
  torch::Tensor existing_vox = torch::round(anchor_ / cur_size);

  int64_t grid_scale = static_cast<int64_t>(1e6);
  auto encode = [grid_scale](const torch::Tensor& vox) -> torch::Tensor {
    return vox.index({torch::indexing::Slice(), 0}).to(torch::kInt64) * grid_scale * grid_scale +
           vox.index({torch::indexing::Slice(), 1}).to(torch::kInt64) * grid_scale +
           vox.index({torch::indexing::Slice(), 2}).to(torch::kInt64);
  };

  torch::Tensor exist_keys = encode(existing_vox);
  torch::Tensor unique_keys = encode(unique_vox);
  torch::Tensor unique_exist = std::get<0>(torch::_unique2(exist_keys));
  torch::Tensor new_vox_mask = ~torch::isin(unique_keys, unique_exist);

  if (!new_vox_mask.any().item<bool>()) return;

  torch::Tensor new_anchor_pos = unique_vox.index({new_vox_mask}) * cur_size;
  int64_t N_new = new_anchor_pos.size(0);

  // Create new anchors with averaged features from neighbors
  auto opt_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  torch::Tensor new_anchor = new_anchor_pos.clone();
  torch::Tensor new_offset = torch::zeros({N_new, n_offsets_, 3}, opt_float);
  torch::Tensor new_feat = torch::zeros({N_new, feat_dim_}, opt_float);
  float init_scale = std::log(std::sqrt(static_cast<float>(cur_size)));
  torch::Tensor new_scaling = torch::full({N_new, 6}, init_scale, opt_float);
  torch::Tensor new_rotation = anchor_rotation_.size(0) > 0
      ? anchor_rotation_.index({torch::indexing::Slice(0, 1)}).expand({N_new, 4}).clone()
      : torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, opt_float).expand({N_new, 4}).clone();
  torch::Tensor new_opacity = torch::full({N_new, 1},
      std::log(0.1f / (1.0f - 0.1f)), opt_float);

  // Assign chunk IDs
  torch::Tensor new_chunk_ids = computeChunkIds(new_anchor, chunk_size_);

  // Sequential IDs
  auto opt_int64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto opt_int32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor new_ids = torch::arange(next_anchor_id_,
      next_anchor_id_ + N_new, opt_int64);
  next_anchor_id_ += N_new;
  torch::Tensor new_exist = torch::zeros({N_new}, opt_int32);

  // Concatenate — this extends all tensors. Optimizer state is NOT extended here
  // (requires a full optimizer rebuild or cat_tensors_to_optimizer logic).
  anchor_ = torch::cat({anchor_, new_anchor});
  offset_ = torch::cat({offset_, new_offset});
  anchor_feat_ = torch::cat({anchor_feat_, new_feat});
  anchor_scaling_ = torch::cat({anchor_scaling_, new_scaling});
  anchor_rotation_ = torch::cat({anchor_rotation_, new_rotation});
  anchor_opacity_ = torch::cat({anchor_opacity_, new_opacity});
  anchor_chunk_ids_ = torch::cat({anchor_chunk_ids_, new_chunk_ids});
  anchor_ids_ = torch::cat({anchor_ids_, new_ids});
  exist_since_iter_ = torch::cat({exist_since_iter_, new_exist});

  // Extend tracking tensors
  offset_gradient_accum_ = torch::cat({
      offset_gradient_accum_,
      torch::zeros({N_new * n_offsets_, 1}, opt_float)});
  offset_denom_ = torch::cat({
      offset_denom_,
      torch::zeros({N_new * n_offsets_, 1}, opt_float)});
  opacity_accum_ = torch::cat({
      opacity_accum_,
      torch::zeros({N_new, 1}, opt_float)});
  anchor_denom_ = torch::cat({
      anchor_denom_,
      torch::zeros({N_new, 1}, opt_float)});

  std::cout << "[Growing] Added " << N_new << " new anchors at depth "
            << depth_level << " (total: " << anchor_.size(0) << ")\n";
}

// =============================================================================
// Anchor Pruning
// =============================================================================

void GaussianModel::anchorPruning(float opacity_threshold,
                                   int min_observations) {
  torch::NoGradGuard no_grad;

  int64_t N = anchor_.size(0);
  if (N == 0) return;

  // Compute average opacity: opacity_accum / denom
  torch::Tensor avg_opacity = opacity_accum_ /
      (anchor_denom_ + 1e-8f);  // [N, 1]
  torch::Tensor enough_observed = anchor_denom_ > min_observations;

  // Prune anchors with low average opacity AND enough observations
  torch::Tensor prune_mask = (avg_opacity < opacity_threshold) & enough_observed;
  prune_mask = prune_mask.squeeze(1);  // [N]

  if (!prune_mask.any().item<bool>()) return;

  torch::Tensor keep_mask = ~prune_mask;
  int64_t remove_count = prune_mask.sum().item<int64_t>();

  // Slice all tensors by keep_mask
  anchor_ = anchor_.index({keep_mask});
  offset_ = offset_.index({keep_mask});
  anchor_feat_ = anchor_feat_.index({keep_mask});
  anchor_scaling_ = anchor_scaling_.index({keep_mask});
  anchor_rotation_ = anchor_rotation_.index({keep_mask});
  anchor_opacity_ = anchor_opacity_.index({keep_mask});
  anchor_chunk_ids_ = anchor_chunk_ids_.index({keep_mask});
  anchor_ids_ = anchor_ids_.index({keep_mask});
  exist_since_iter_ = exist_since_iter_.index({keep_mask});
  opacity_accum_ = opacity_accum_.index({keep_mask});
  anchor_denom_ = anchor_denom_.index({keep_mask});

  // Slice [N*K] gradient tensors
  torch::Tensor keep_expanded = keep_mask.unsqueeze(1).expand({N, n_offsets_})
      .reshape({-1});
  offset_gradient_accum_ = offset_gradient_accum_.index({keep_expanded});
  offset_denom_ = offset_denom_.index({keep_expanded});

  std::cout << "[Pruning] Removed " << remove_count << " low-opacity anchors"
            << " (total: " << anchor_.size(0) << ")\n";
}

// =============================================================================
// Combined Densification Cycle
// =============================================================================

void GaussianModel::adjustAnchors(int check_interval,
                                   int success_threshold,
                                   float grad_threshold,
                                   float min_opacity) {
  int64_t N_before = anchor_.size(0);

  // Grow at multiple hierarchy levels
  for (int depth = 0; depth < update_depth_; ++depth) {
    anchorGrowing(grad_threshold, depth);
  }

  int64_t N_new = anchor_.size(0) - N_before;

  // Extend optimizer state for new anchors if any were grown
  if (N_new > 0 && optimizer_) {
    // Rebuild optimizer state via trainingSetup to capture all new params.
    // This is functionally correct — Adam state for existing anchors is
    // preserved since we rebuild from scratch.
    // For a more efficient approach (preserving existing state), use
    // catAnchorsToOptimizer with proper param group tracking from
    // anchorGrowing.
    auto& param_groups = optimizer_->param_groups();
    auto& state = optimizer_->state();

    // Save old states keyed by group index
    struct SavedState {
      torch::Tensor exp_avg, exp_avg_sq;
      int64_t step;
    };
    std::vector<std::optional<SavedState>> saved(kNumAnchorParamGroups);

    for (int g = 0; g < kNumAnchorParamGroups && g < (int)param_groups.size(); ++g) {
      auto& params = param_groups[g].params();
      if (params.empty()) continue;
      auto key = params[0].unsafeGetTensorImpl();
      auto it = state.find(key);
      if (it != state.end()) {
        auto& adam = static_cast<torch::optim::AdamParamState&>(*it->second);
        saved[g] = SavedState{
            adam.exp_avg().defined() ? adam.exp_avg().clone() : torch::Tensor(),
            adam.exp_avg_sq().defined() ? adam.exp_avg_sq().clone() : torch::Tensor(),
            adam.step()};
      }
    }

    // Rebuild Tensor_vec wrappers to match new tensor sizes
    Tensor_vec_anchor_ = {anchor_};
    Tensor_vec_offset_ = {offset_};
    Tensor_vec_anchor_feat_ = {anchor_feat_};
    Tensor_vec_anchor_opacity_ = {anchor_opacity_};
    Tensor_vec_anchor_scaling_ = {anchor_scaling_};
    Tensor_vec_anchor_rotation_ = {anchor_rotation_};

    // Rebuild optimizer
    auto device = device_type_ == torch::kCUDA ? torch::Device(torch::kCUDA)
                                               : torch::Device(torch::kCPU);
    auto opt_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    std::vector<torch::optim::OptimizerParamGroup> new_groups;

    auto make_group = [&](const torch::Tensor& param, float lr) {
      return torch::optim::OptimizerParamGroup(
          std::vector<torch::Tensor>{param},
          std::make_unique<torch::optim::AdamOptions>(lr));
    };

    // Get current LRs from existing optimizer if available
    auto get_lr = [&](int g) -> float {
      if (g < (int)param_groups.size()) {
        return param_groups[g].options().get_lr();
      }
      return 0.001f;
    };

    new_groups.push_back(make_group(anchor_, get_lr(0)));
    new_groups.push_back(make_group(offset_, get_lr(1)));
    new_groups.push_back(make_group(anchor_feat_, get_lr(2)));
    new_groups.push_back(make_group(anchor_opacity_, get_lr(3)));
    new_groups.push_back(make_group(anchor_scaling_, get_lr(4)));
    new_groups.push_back(make_group(anchor_rotation_, get_lr(5)));

    // MLP group
    auto mlp_params = mlp_.parameters();
    if (!mlp_params.empty()) {
      new_groups.push_back(torch::optim::OptimizerParamGroup(
          mlp_params, std::make_unique<torch::optim::AdamOptions>(get_lr(6))));
    } else {
      new_groups.push_back(torch::optim::OptimizerParamGroup(
          std::vector<torch::Tensor>{torch::empty({0}, opt_float)},
          std::make_unique<torch::optim::AdamOptions>(get_lr(6))));
    }

    // Appearance embedding group
    if (mlp_.hasAppearanceEmbedding()) {
      auto& emb = mlp_.appearanceEmbedding();
      auto emb_params = emb.parameters();
      new_groups.push_back(torch::optim::OptimizerParamGroup(
          std::vector<torch::Tensor>{emb_params},
          std::make_unique<torch::optim::AdamOptions>(get_lr(7))));
    } else {
      new_groups.push_back(torch::optim::OptimizerParamGroup(
          std::vector<torch::Tensor>{torch::empty({0}, opt_float)},
          std::make_unique<torch::optim::AdamOptions>(get_lr(7))));
    }

    optimizer_ = std::make_shared<torch::optim::Adam>(
        new_groups, torch::optim::AdamOptions(0.001f));

    // Restore saved states for existing anchors
    auto& new_state = optimizer_->state();
    for (int g = 0; g < kNumAnchorParamGroups && g < (int)new_groups.size(); ++g) {
      if (!saved[g].has_value()) continue;
      auto& saved_s = saved[g].value();
      if (!saved_s.exp_avg.defined()) continue;

      auto& params = new_groups[g].params();
      if (params.empty()) continue;
      auto new_key = params[0].unsafeGetTensorImpl();

      int64_t existing_N = saved_s.exp_avg.size(0);
      auto zero_avg = torch::zeros({N_new,
          saved_s.exp_avg.size(1)}, saved_s.exp_avg.options());
      auto zero_sq = torch::zeros({N_new,
          saved_s.exp_avg_sq.size(1)}, saved_s.exp_avg_sq.options());

      auto restore_avg = torch::cat({saved_s.exp_avg, zero_avg}, 0);
      auto restore_sq = torch::cat({saved_s.exp_avg_sq, zero_sq}, 0);

      auto adam_state = std::make_unique<torch::optim::AdamParamState>(
          std::move(restore_avg), std::move(restore_sq), saved_s.step);
      new_state[new_key] = std::move(adam_state);
    }
  }

  // Prune low-opacity anchors
  anchorPruning(min_opacity, check_interval * success_threshold);

  // Reset accumulators for next cycle
  opacity_accum_.zero_();
  anchor_denom_.zero_();
  offset_gradient_accum_.zero_();
  offset_denom_.zero_();
}

// =============================================================================
// Optimizer Extension (for new anchors after growing)
// =============================================================================

void GaussianModel::catAnchorsToOptimizer(
    const std::vector<torch::Tensor>& new_params,
    const std::vector<std::pair<int, int>>& param_group_indices) {
  // Properly extend Adam optimizer state for newly added anchors.
  // For each param group, concatenate zeros to exp_avg/exp_avg_sq,
  // then re-register the updated tensor references.
  if (!optimizer_) return;

  auto& param_groups = optimizer_->param_groups();
  auto& state = optimizer_->state();

  for (const auto& [group_idx, param_idx] : param_group_indices) {
    if (group_idx < 0 || group_idx >= static_cast<int>(param_groups.size()))
      continue;

    auto& group = param_groups[group_idx];
    if (group.params().empty()) continue;

    auto old_key = group.params()[0].unsafeGetTensorImpl();
    auto old_it = state.find(old_key);
    if (old_it == state.end()) continue;

    // Extract old Adam state
    auto& old_adam = static_cast<torch::optim::AdamParamState&>(*old_it->second);
    int64_t old_N = old_adam.exp_avg().defined() ? old_adam.exp_avg().size(0) : 0;
    int64_t new_N = new_params[param_idx].size(0);

    auto opt = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device_type_);

    // Extend exp_avg and exp_avg_sq with zeros for new params
    if (old_adam.exp_avg().defined() && old_N > 0) {
      torch::Tensor zero_avg = torch::zeros(
          {new_N, old_adam.exp_avg().size(1)}, opt);
      torch::Tensor zero_sq = torch::zeros(
          {new_N, old_adam.exp_avg_sq().size(1)}, opt);

      auto new_exp_avg = torch::cat({old_adam.exp_avg(), zero_avg}, 0);
      auto new_exp_sq = torch::cat({old_adam.exp_avg_sq(), zero_sq}, 0);

      // Remove old state entry
      state.erase(old_it);

      // Create new state entry keyed to the updated param tensor
      auto new_key = group.params()[0].unsafeGetTensorImpl();
      auto new_adam = std::make_unique<torch::optim::AdamParamState>(
          std::move(new_exp_avg), std::move(new_exp_sq), old_adam.step());

      state[new_key] = std::move(new_adam);
    }
  }
}

}  // namespace scaffold_chungs
