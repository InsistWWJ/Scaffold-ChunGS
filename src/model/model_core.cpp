/**
 * Scaffold-ChunGS: Anchor-based GaussianModel core operations.
 *
 * Anchor creation: input point cloud is voxelized at voxel_size_ resolution.
 * Each non-empty voxel becomes an anchor. The anchor position is the voxel
 * center; initially offsets are zero (children sit at the anchor).
 */

#include "scaffold_chunks/gaussian_model.h"

namespace scaffold_chungs {

// =============================================================================
// Activation Functions
// =============================================================================

static torch::Tensor inverseSigmoid(const torch::Tensor& x) {
  return torch::log(x / (1.0f - x + 1e-8f));
}

// =============================================================================
// Construction
// =============================================================================

GaussianModel::GaussianModel(const AnchorModelConfig& anchor_cfg,
                             const ChunkConfig& chunk_cfg,
                             torch::DeviceType device_type)
    : mlp_(anchor_cfg.feat_dim,
           anchor_cfg.n_offsets,
           anchor_cfg.use_feat_bank,
           anchor_cfg.appearance_dim,
           anchor_cfg.add_opacity_dist,
           anchor_cfg.add_cov_dist,
           anchor_cfg.add_color_dist),
      chunk_size_(chunk_cfg.chunk_size),
      n_offsets_(anchor_cfg.n_offsets),
      feat_dim_(anchor_cfg.feat_dim),
      voxel_size_(anchor_cfg.voxel_size),
      storage_base_path_(chunk_cfg.storage_base_path),
      device_type_(device_type),
      max_anchors_in_memory_(chunk_cfg.max_anchors_in_memory),
      update_depth_(anchor_cfg.update_depth),
      update_init_factor_(anchor_cfg.update_init_factor),
      update_hierarchy_factor_(anchor_cfg.update_hierarchy_factor) {

  // Initialize empty tensors on target device
  auto opt = torch::TensorOptions().device(device_type_);

  anchor_ = torch::empty({0, 3}, opt.dtype(torch::kFloat32)).set_requires_grad(true);
  offset_ = torch::empty({0, n_offsets_, 3}, opt.dtype(torch::kFloat32)).set_requires_grad(true);
  anchor_feat_ = torch::empty({0, feat_dim_}, opt.dtype(torch::kFloat32)).set_requires_grad(true);
  anchor_scaling_ = torch::empty({0, 6}, opt.dtype(torch::kFloat32)).set_requires_grad(true);
  anchor_rotation_ = torch::empty({0, 4}, opt.dtype(torch::kFloat32)).set_requires_grad(false);
  anchor_opacity_ = torch::empty({0, 1}, opt.dtype(torch::kFloat32)).set_requires_grad(false);

  anchor_chunk_ids_ = torch::empty({0}, opt.dtype(torch::kInt64));
  anchor_ids_ = torch::empty({0}, opt.dtype(torch::kInt64));
  exist_since_iter_ = torch::empty({0}, opt.dtype(torch::kInt32));
  offset_gradient_accum_ = torch::empty({0, 1}, opt.dtype(torch::kFloat32));
  offset_denom_ = torch::empty({0, 1}, opt.dtype(torch::kFloat32));
  opacity_accum_ = torch::empty({0, 1}, opt.dtype(torch::kFloat32));
  anchor_denom_ = torch::empty({0, 1}, opt.dtype(torch::kFloat32));

  chunks_loaded_from_disk_ = torch::empty({0}, opt.dtype(torch::kInt64));
  chunks_on_disk_ = torch::empty({0}, opt.dtype(torch::kInt64));
  chunk_anchor_counts_ = torch::empty({0}, opt.dtype(torch::kInt64));

  mlp_.to(device_type_ == torch::kCUDA ? torch::Device(torch::kCUDA)
                                       : torch::Device(torch::kCPU));
}

// =============================================================================
// Voxelization Helper
// =============================================================================

static std::tuple<torch::Tensor, torch::Tensor> voxelizePoints(
    const torch::Tensor& xyz,
    const torch::Tensor& colors,
    float voxel_size) {
  // Quantize coordinates to voxel grid
  torch::Tensor voxel_coords = torch::round(xyz / voxel_size);  // [P, 3]

  auto [unique_coords, inverse_indices, counts] =
      torch::_unique2(voxel_coords, false, true, true);

  int64_t N = unique_coords.size(0);  // number of anchors
  auto device = xyz.device();

  // For each anchor: average color of points in that voxel
  torch::Tensor anchor_pos = unique_coords * voxel_size;  // [N, 3]
  torch::Tensor anchor_colors = torch::zeros({N, 3},
      torch::TensorOptions().dtype(torch::kFloat32).device(device));

  // Accumulate colors per voxel
  for (int64_t i = 0; i < N; ++i) {
    torch::Tensor mask = (inverse_indices == i);
    torch::Tensor pts = colors.index({mask});
    if (pts.size(0) > 0) {
      anchor_colors[i] = pts.mean(0);
    }
  }

  return {anchor_pos, anchor_colors};
}

// =============================================================================
// Anchor Initialization
// =============================================================================

void GaussianModel::initializeFromPoints(const torch::Tensor& xyz,
                                         const torch::Tensor& colors) {
  if (is_initialized_) return;

  auto device = device_type_ == torch::kCUDA ? torch::Device(torch::kCUDA)
                                             : torch::Device(torch::kCPU);

  // Transfer to target device (.to() returns a new tensor, does not modify in-place)
  auto xyz_dev = xyz.to(device);
  auto colors_dev = colors.to(device);

  // Voxelize input
  auto [anchor_positions, anchor_colors] = voxelizePoints(xyz_dev, colors_dev, voxel_size_);
  int64_t N = anchor_positions.size(0);

  if (N == 0) {
    std::cerr << "[GaussianModel] Warning: initializeFromPoints got 0 anchors\n";
    return;
  }

  // Create anchor tensors
  auto opt_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto opt_int64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto opt_int32 = torch::TensorOptions().dtype(torch::kInt32).device(device);

  anchor_ = anchor_positions.clone().set_requires_grad(true);

  // Zeros offset — children initially sit at the anchor position
  offset_ = torch::zeros({N, n_offsets_, 3}, opt_float).set_requires_grad(true);

  // Init anchor features from colors (repeat to feat_dim)
  // Simple approach: pad with zeros beyond RGB
  anchor_feat_ = torch::zeros({N, feat_dim_}, opt_float);
  anchor_feat_.index_put_({torch::indexing::Slice(),
                           torch::indexing::Slice(0, 3)},
                          anchor_colors);
  anchor_feat_.set_requires_grad(true);

  // Init scaling: compute KNN distance for each anchor
  // Simplified: use voxel_size as default scale
  float init_scale = std::log(std::sqrt(voxel_size_));
  torch::Tensor init_scaling = torch::full({N, 6},
      init_scale, opt_float).set_requires_grad(true);
  anchor_scaling_ = init_scaling;

  // Identity rotation
  anchor_rotation_ = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, opt_float)
      .expand({N, 4}).clone();
  anchor_rotation_.set_requires_grad(false);

  // Base opacity: sigmoid^-1(0.1)
  anchor_opacity_ = torch::full({N, 1},
      inverseSigmoid(torch::tensor(0.1f, opt_float)), opt_float);
  anchor_opacity_.set_requires_grad(false);

  // Compute chunk IDs
  anchor_chunk_ids_ = computeChunkIds(anchor_, chunk_size_);

  // Sequential IDs
  anchor_ids_ = torch::arange(next_anchor_id_, next_anchor_id_ + N, opt_int64);
  next_anchor_id_ += N;

  exist_since_iter_ = torch::zeros({N}, opt_int32);

  // Gradient accumulators
  offset_gradient_accum_ = torch::zeros({N * n_offsets_, 1}, opt_float);
  offset_denom_ = torch::zeros({N * n_offsets_, 1}, opt_float);
  opacity_accum_ = torch::zeros({N, 1}, opt_float);
  anchor_denom_ = torch::zeros({N, 1}, opt_float);

  spatial_lr_scale_ = 1.0f;
  is_initialized_ = true;

  std::cout << "[GaussianModel] Initialized " << N << " anchors from "
            << xyz.size(0) << " points (voxel_size=" << voxel_size_ << ")\n";
}

// =============================================================================
// Adding Anchors
// =============================================================================

void GaussianModel::addAnchors(const torch::Tensor& xyz,
                               const torch::Tensor& colors) {
  auto device = device_type_ == torch::kCUDA ? torch::Device(torch::kCUDA)
                                             : torch::Device(torch::kCPU);

  auto [new_positions, new_colors] = voxelizePoints(
      xyz.to(device), colors.to(device), voxel_size_);
  int64_t N_new = new_positions.size(0);

  if (N_new == 0) return;

  auto opt_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto opt_int64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
  auto opt_int32 = torch::TensorOptions().dtype(torch::kInt32).device(device);

  // ---- Filter: remove new anchors that are too close to existing ones ----
  if (anchor_.size(0) > 0) {
    // GPU-accelerated spatial deduplication via hash grid.
    // Quantize existing & new anchors to grid cells, then mask out collisions.
    float dedup_cell = voxel_size_ * 2.0f;
    torch::Tensor existing_vox = torch::round(anchor_.detach() / dedup_cell);
    torch::Tensor new_vox = torch::round(new_positions / dedup_cell);

    // Encode 3D voxel coord to a single int64 key for efficient set difference
    int64_t grid_scale = static_cast<int64_t>(1e6);
    auto encode = [grid_scale](const torch::Tensor& vox) -> torch::Tensor {
      return vox.index({torch::indexing::Slice(), 0}).to(torch::kInt64) * grid_scale * grid_scale +
             vox.index({torch::indexing::Slice(), 1}).to(torch::kInt64) * grid_scale +
             vox.index({torch::indexing::Slice(), 2}).to(torch::kInt64);
    };

    torch::Tensor exist_keys = encode(existing_vox);
    torch::Tensor new_keys = encode(new_vox);

    // Mask: keep new keys not present in existing keys
    torch::Tensor unique_exist = std::get<0>(torch::_unique2(exist_keys));
    torch::Tensor keep_mask = ~torch::isin(new_keys, unique_exist);

    if (!keep_mask.any().item<bool>()) return;

    new_positions = new_positions.index({keep_mask});
    new_colors = new_colors.index({keep_mask});
    N_new = new_positions.size(0);
  }

  if (N_new == 0) return;

  // ---- Build new anchor tensors ----
  torch::Tensor new_anchor = new_positions.clone();
  torch::Tensor new_offset = torch::zeros({N_new, n_offsets_, 3}, opt_float);
  torch::Tensor new_feat = torch::zeros({N_new, feat_dim_}, opt_float);
  new_feat.index_put_({torch::indexing::Slice(),
                       torch::indexing::Slice(0, 3)},
                      new_colors);

  float init_scale = std::log(std::sqrt(voxel_size_));
  torch::Tensor new_scaling = torch::full({N_new, 6}, init_scale, opt_float);
  torch::Tensor new_rotation = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, opt_float)
      .expand({N_new, 4}).clone();
  torch::Tensor new_opacity = torch::full({N_new, 1},
      inverseSigmoid(torch::tensor(0.1f, opt_float)), opt_float);

  torch::Tensor new_chunk_ids = computeChunkIds(new_anchor, chunk_size_);
  torch::Tensor new_ids = torch::arange(next_anchor_id_,
      next_anchor_id_ + N_new, opt_int64);
  next_anchor_id_ += N_new;

  torch::Tensor new_exist = torch::zeros({N_new}, opt_int32);

  // ---- Concatenate to existing ----
  if (anchor_.size(0) == 0) {
    // First anchors
    anchor_ = new_anchor.clone().set_requires_grad(true);
    offset_ = new_offset.clone().set_requires_grad(true);
    anchor_feat_ = new_feat.clone().set_requires_grad(true);
    anchor_scaling_ = new_scaling.clone().set_requires_grad(true);
    anchor_rotation_ = new_rotation.clone().set_requires_grad(false);
    anchor_opacity_ = new_opacity.clone().set_requires_grad(false);
    anchor_chunk_ids_ = new_chunk_ids;
    anchor_ids_ = new_ids;
    exist_since_iter_ = new_exist;
  } else {
    // Concatenate
    anchor_ = torch::cat({anchor_, new_anchor}).set_requires_grad(true);
    offset_ = torch::cat({offset_, new_offset}).set_requires_grad(true);
    anchor_feat_ = torch::cat({anchor_feat_, new_feat}).set_requires_grad(true);
    anchor_scaling_ = torch::cat({anchor_scaling_, new_scaling}).set_requires_grad(true);
    anchor_rotation_ = torch::cat({anchor_rotation_, new_rotation}).set_requires_grad(false);
    anchor_opacity_ = torch::cat({anchor_opacity_, new_opacity}).set_requires_grad(false);
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
  }

  is_initialized_ = true;

  // Load any destination chunks from disk
  auto unique_new_chunks = std::get<0>(torch::_unique2(new_chunk_ids));
  torch::Tensor on_disk_mask = torch::isin(unique_new_chunks, chunks_on_disk_);
  torch::Tensor to_load = unique_new_chunks.index({on_disk_mask});
  if (to_load.size(0) > 0) {
    loadChunks(to_load);
  }

  std::cout << "[GaussianModel] Added " << N_new << " new anchors"
            << " (total: " << anchor_.size(0) << ")\n";
}

// =============================================================================
// Anchor Data Access
// =============================================================================

AnchorData GaussianModel::getVisibleAnchors(const torch::Tensor& visible_mask) const {
  AnchorData data;
  data.n_offsets = n_offsets_;
  data.feat_dim = feat_dim_;

  if (visible_mask.size(0) == 0 || !visible_mask.any().item<bool>()) {
    data.num_anchors = 0;
    data.anchor = torch::empty({0, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    data.offset = torch::empty({0, n_offsets_, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    data.anchor_feat = torch::empty({0, feat_dim_},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    data.anchor_scaling = torch::empty({0, 6},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    data.anchor_rotation = torch::empty({0, 4},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    data.anchor_opacity = torch::empty({0, 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    return data;
  }

  data.anchor = anchor_.index({visible_mask});
  data.offset = offset_.index({visible_mask});
  data.anchor_feat = anchor_feat_.index({visible_mask});
  data.anchor_scaling = anchor_scaling_.index({visible_mask});
  data.anchor_rotation = anchor_rotation_.index({visible_mask});
  data.anchor_opacity = anchor_opacity_.index({visible_mask});
  data.num_anchors = data.anchor.size(0);

  return data;
}

// =============================================================================
// Chunk ID Operations
// =============================================================================

void GaussianModel::updateChunkIDs() {
  anchor_chunk_ids_ = computeChunkIds(anchor_, chunk_size_);
}

torch::Tensor GaussianModel::createAnchorMaskFromChunks(
    const torch::Tensor& chunk_ids) const {
  return torch::isin(anchor_chunk_ids_, chunk_ids);
}

int64_t GaussianModel::countAllAnchors() {
  int64_t in_memory = anchor_.size(0);
  torch::Tensor unloaded_mask = ~torch::isin(chunks_on_disk_,
                                             chunks_loaded_from_disk_);
  int64_t on_disk = torch::sum(
      chunk_anchor_counts_.index({unloaded_mask})).item<int64_t>();
  return in_memory + on_disk;
}

int64_t GaussianModel::countAllGaussians() {
  return countAllAnchors() * n_offsets_;
}

}  // namespace scaffold_chungs
