/**
 * Scaffold-ChunGS: Chunk serialization and deserialization.
 *
 * Each chunk file stores all anchor tensors + tracking tensors + Adam states.
 * File format: [magic(4B)][version(4B)][chunk_id(8B)][num_anchors(4B)]
 *              [TensorHeader+tensor_data]*N_tensors
 *
 * Magic: 0x5343484E ("SCHN") to distinguish from DiskChunGS ("CHNK").
 */

#include "scaffold_chunks/gaussian_model.h"

#include <filesystem>
#include <future>
#include <iostream>

namespace scaffold_chungs {

// File format constants
static constexpr uint32_t kMagic = 0x5343484E;  // "SCHN"
static constexpr uint32_t kVersion = 1;

// =============================================================================
// Binary Tensor Serialization
// =============================================================================

void GaussianModel::saveTensorBinary(const torch::Tensor& tensor,
                                     std::ofstream& file) {
  TensorHeader header = {};
  header.dims = tensor.dim();
  for (int i = 0; i < tensor.dim(); ++i) {
    header.sizes[i] = static_cast<uint32_t>(tensor.size(i));
  }
  header.dtype = static_cast<uint32_t>(tensor.scalar_type());
  header.data_size = tensor.nbytes();

  file.write(reinterpret_cast<const char*>(&header), sizeof(TensorHeader));

  torch::Tensor cpu = tensor.is_cuda() ? tensor.cpu() : tensor;
  file.write(reinterpret_cast<const char*>(cpu.data_ptr()), header.data_size);
}

torch::Tensor GaussianModel::loadTensorBinary(std::ifstream& file) {
  TensorHeader header;
  file.read(reinterpret_cast<char*>(&header), sizeof(TensorHeader));

  std::vector<int64_t> sizes(header.dims);
  for (uint32_t i = 0; i < header.dims; ++i) {
    sizes[i] = header.sizes[i];
  }

  auto options = torch::TensorOptions()
      .dtype(static_cast<torch::ScalarType>(header.dtype))
      .device(torch::kCPU);
  torch::Tensor tensor = torch::empty(sizes, options);
  file.read(reinterpret_cast<char*>(tensor.data_ptr()), header.data_size);

  return tensor.to(device_type_);
}

// =============================================================================
// Chunk Filename Helpers
// =============================================================================

std::string GaussianModel::getChunkFilename(const ChunkCoord& coord) {
  auto sign = [](int64_t v) -> std::string {
    return v >= 0 ? "p" : "n";
  };
  return storage_base_path_ + "/" +
         sign(coord.x) + std::to_string(std::abs(coord.x)) + "_" +
         sign(coord.y) + std::to_string(std::abs(coord.y)) + "_" +
         sign(coord.z) + std::to_string(std::abs(coord.z)) + ".schun";
}

// =============================================================================
// Single-Chunk I/O
// =============================================================================

void GaussianModel::saveSingleChunkToDisk(int64_t chunk_id,
                                           const ChunkData& data) {
  std::string filename = getChunkFilename(decodeChunkCoord(chunk_id));
  std::filesystem::path path(filename);
  std::filesystem::create_directories(path.parent_path());

  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  try {
    // Header
    file.write(reinterpret_cast<const char*>(&kMagic), sizeof(kMagic));
    file.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    file.write(reinterpret_cast<const char*>(&chunk_id), sizeof(chunk_id));
    uint32_t na = static_cast<uint32_t>(data.num_anchors);
    file.write(reinterpret_cast<const char*>(&na), sizeof(na));

    // Anchor parameters (order must match load)
    saveTensorBinary(data.anchor, file);
    saveTensorBinary(data.offset, file);
    saveTensorBinary(data.anchor_feat, file);
    saveTensorBinary(data.anchor_scaling, file);
    saveTensorBinary(data.anchor_rotation, file);
    saveTensorBinary(data.anchor_opacity, file);

    // Tracking tensors
    saveTensorBinary(data.exist_since, file);
    saveTensorBinary(data.anchor_chunk_ids, file);
    saveTensorBinary(data.anchor_ids, file);
    saveTensorBinary(data.offset_gradient_accum, file);
    saveTensorBinary(data.offset_denom, file);
    saveTensorBinary(data.opacity_accum, file);
    saveTensorBinary(data.anchor_denom, file);

    // Optimizer states (one per param group)
    for (int g = 0; g < kNumAnchorParamGroups; ++g) {
      file.write(reinterpret_cast<const char*>(&data.step_counts[g]),
                 sizeof(int64_t));

      if (data.exp_avg_states[g].defined() &&
          data.exp_avg_states[g].numel() > 0) {
        saveTensorBinary(data.exp_avg_states[g], file);
        saveTensorBinary(data.exp_avg_sq_states[g], file);
      } else {
        torch::Tensor empty = torch::empty({0});
        saveTensorBinary(empty, file);
        saveTensorBinary(empty, file);
      }
    }

    file.close();
  } catch (const std::exception& e) {
    file.close();
    std::filesystem::remove(filename);
    throw std::runtime_error("Failed to save chunk " +
                             std::to_string(chunk_id) + ": " + e.what());
  }
}

std::optional<ChunkData> GaussianModel::loadSingleChunkFromDisk(
    int64_t chunk_id) {
  std::string filename = getChunkFilename(decodeChunkCoord(chunk_id));

  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("Chunk file not found: " + filename);
  }

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  // Validate magic and version
  uint32_t magic, version;
  file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (magic != kMagic) {
    throw std::runtime_error("Invalid chunk format (bad magic): " + filename);
  }

  int64_t stored_chunk_id;
  uint32_t stored_num_anchors;
  file.read(reinterpret_cast<char*>(&stored_chunk_id), sizeof(stored_chunk_id));
  file.read(reinterpret_cast<char*>(&stored_num_anchors),
            sizeof(stored_num_anchors));
  if (stored_chunk_id != chunk_id) {
    throw std::runtime_error("Chunk ID mismatch in file: " + filename);
  }

  ChunkData data;
  try {
    data.anchor = loadTensorBinary(file);
    data.offset = loadTensorBinary(file);
    data.anchor_feat = loadTensorBinary(file);
    data.anchor_scaling = loadTensorBinary(file);
    data.anchor_rotation = loadTensorBinary(file);
    data.anchor_opacity = loadTensorBinary(file);

    data.exist_since = loadTensorBinary(file);
    data.anchor_chunk_ids = loadTensorBinary(file);
    data.anchor_ids = loadTensorBinary(file);
    data.offset_gradient_accum = loadTensorBinary(file);
    data.offset_denom = loadTensorBinary(file);
    data.opacity_accum = loadTensorBinary(file);
    data.anchor_denom = loadTensorBinary(file);

    data.exp_avg_states.resize(kNumAnchorParamGroups);
    data.exp_avg_sq_states.resize(kNumAnchorParamGroups);
    data.step_counts.resize(kNumAnchorParamGroups);
    for (int g = 0; g < kNumAnchorParamGroups; ++g) {
      file.read(reinterpret_cast<char*>(&data.step_counts[g]), sizeof(int64_t));
      data.exp_avg_states[g] = loadTensorBinary(file);
      data.exp_avg_sq_states[g] = loadTensorBinary(file);
    }

    data.num_anchors = data.anchor.size(0);
    data.chunk_id = chunk_id;
    file.close();

    if (data.num_anchors != static_cast<int>(stored_num_anchors)) {
      std::cerr << "[Storage] Anchor count mismatch in " << filename << "\n";
      return std::nullopt;
    }

    return data;
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to load chunk " +
                             std::to_string(chunk_id) + ": " + e.what());
  }
}

// =============================================================================
// Chunk Data Extraction (mask-based indexing)
// =============================================================================

ChunkData GaussianModel::extractChunkData(const torch::Tensor& chunk_mask,
                                           int64_t chunk_id) {
  ChunkData data;

  data.anchor = anchor_.index({chunk_mask}).detach().clone();
  data.offset = offset_.index({chunk_mask}).detach().clone();
  data.anchor_feat = anchor_feat_.index({chunk_mask}).detach().clone();
  data.anchor_scaling = anchor_scaling_.index({chunk_mask}).detach().clone();
  data.anchor_rotation = anchor_rotation_.index({chunk_mask}).detach().clone();
  data.anchor_opacity = anchor_opacity_.index({chunk_mask}).detach().clone();
  data.exist_since = exist_since_iter_.index({chunk_mask}).detach().clone();
  data.anchor_chunk_ids = anchor_chunk_ids_.index({chunk_mask}).detach().clone();
  data.anchor_ids = anchor_ids_.index({chunk_mask}).detach().clone();
  // Slice [N*K, 1] gradient tensors by expanded chunk mask
  torch::Tensor offset_mask = chunk_mask.unsqueeze(1)
      .expand({chunk_mask.size(0), n_offsets_}).reshape({-1});
  data.offset_gradient_accum =
      offset_gradient_accum_.index({offset_mask}).detach().clone();
  data.offset_denom =
      offset_denom_.index({offset_mask}).detach().clone();
  data.opacity_accum = opacity_accum_.index({chunk_mask}).detach().clone();
  data.anchor_denom = anchor_denom_.index({chunk_mask}).detach().clone();
  data.num_anchors = data.anchor.size(0);
  data.chunk_id = chunk_id;

  // Extract optimizer states
  data.exp_avg_states.resize(kNumAnchorParamGroups);
  data.exp_avg_sq_states.resize(kNumAnchorParamGroups);
  data.step_counts.resize(kNumAnchorParamGroups);

  if (optimizer_) {
    auto& param_groups = optimizer_->param_groups();
    auto& state = optimizer_->state();

    for (int g = 0; g < kNumAnchorParamGroups && g < (int)param_groups.size(); ++g) {
      auto& params = param_groups[g].params();
      if (params.empty()) {
        data.step_counts[g] = 0;
        continue;
      }
      auto key = params[0].unsafeGetTensorImpl();
      if (state.find(key) != state.end()) {
        auto& param_state =
            static_cast<torch::optim::AdamParamState&>(*state[key]);
        if (param_state.exp_avg().size(0) == anchor_.size(0)) {
          data.exp_avg_states[g] =
              param_state.exp_avg().index({chunk_mask}).detach().clone();
          data.exp_avg_sq_states[g] =
              param_state.exp_avg_sq().index({chunk_mask}).detach().clone();
        }
        data.step_counts[g] = param_state.step();
      }
    }
  }

  return data;
}

// =============================================================================
// Batch Append (after loading from disk)
// =============================================================================

void GaussianModel::appendLoadedChunks(
    const std::vector<ChunkData>& chunks_data,
    const std::vector<int64_t>& chunk_ids) {
  torch::NoGradGuard no_grad;
  if (chunks_data.empty()) return;

  auto device = device_type_ == torch::kCUDA ? torch::Device(torch::kCUDA)
                                              : torch::Device(torch::kCPU);

  std::vector<torch::Tensor> all_anchor, all_offset, all_feat;
  std::vector<torch::Tensor> all_scaling, all_rotation, all_opacity;
  std::vector<torch::Tensor> all_exist, all_chunk_ids, all_anchor_ids;
  std::vector<torch::Tensor> all_grad_accum, all_grad_denom;
  std::vector<torch::Tensor> all_opacity_accum, all_anchor_denom;

  for (const auto& chunk : chunks_data) {
    all_anchor.push_back(chunk.anchor.to(device));
    all_offset.push_back(chunk.offset.to(device));
    all_feat.push_back(chunk.anchor_feat.to(device));
    all_scaling.push_back(chunk.anchor_scaling.to(device));
    all_rotation.push_back(chunk.anchor_rotation.to(device));
    all_opacity.push_back(chunk.anchor_opacity.to(device));
    all_exist.push_back(chunk.exist_since.to(device));
    all_chunk_ids.push_back(chunk.anchor_chunk_ids.to(device));
    all_anchor_ids.push_back(chunk.anchor_ids.to(device));
    all_grad_accum.push_back(chunk.offset_gradient_accum.to(device));
    all_grad_denom.push_back(chunk.offset_denom.to(device));
    all_opacity_accum.push_back(chunk.opacity_accum.to(device));
    all_anchor_denom.push_back(chunk.anchor_denom.to(device));
  }

  // Batch concatenation
  auto cat = [](const std::vector<torch::Tensor>& xs) {
    if (xs.size() == 1) return xs[0];
    return torch::cat(xs, 0);
  };

  auto batch_anchor = cat(all_anchor);
  auto batch_offset = cat(all_offset);
  auto batch_feat = cat(all_feat);
  auto batch_scaling = cat(all_scaling);
  auto batch_rotation = cat(all_rotation);
  auto batch_opacity = cat(all_opacity);
  auto batch_exist = cat(all_exist);
  auto batch_chunk_ids = cat(all_chunk_ids);
  auto batch_anchor_ids = cat(all_anchor_ids);
  auto batch_grad_accum = cat(all_grad_accum);
  auto batch_grad_denom = cat(all_grad_denom);
  auto batch_opacity_accum = cat(all_opacity_accum);
  auto batch_anchor_denom = cat(all_anchor_denom);

  int64_t prev_N = anchor_.size(0);

  // Concatenate to existing
  if (prev_N == 0) {
    anchor_ = batch_anchor.detach().clone().set_requires_grad(true);
    offset_ = batch_offset.detach().clone().set_requires_grad(true);
    anchor_feat_ = batch_feat.detach().clone().set_requires_grad(true);
    anchor_scaling_ = batch_scaling.detach().clone().set_requires_grad(true);
    anchor_rotation_ = batch_rotation.detach().clone().set_requires_grad(false);
    anchor_opacity_ = batch_opacity.detach().clone().set_requires_grad(false);
    anchor_chunk_ids_ = batch_chunk_ids;
    anchor_ids_ = batch_anchor_ids;
    exist_since_iter_ = batch_exist;
    offset_gradient_accum_ = batch_grad_accum;
    offset_denom_ = batch_grad_denom;
    opacity_accum_ = batch_opacity_accum;
    anchor_denom_ = batch_anchor_denom;
  } else {
    anchor_ = torch::cat({anchor_, batch_anchor.detach()}).set_requires_grad(true);
    offset_ = torch::cat({offset_, batch_offset.detach()}).set_requires_grad(true);
    anchor_feat_ = torch::cat({anchor_feat_, batch_feat.detach()}).set_requires_grad(true);
    anchor_scaling_ = torch::cat({anchor_scaling_, batch_scaling.detach()}).set_requires_grad(true);
    anchor_rotation_ = torch::cat({anchor_rotation_, batch_rotation.detach()}).set_requires_grad(false);
    anchor_opacity_ = torch::cat({anchor_opacity_, batch_opacity.detach()}).set_requires_grad(false);
    anchor_chunk_ids_ = torch::cat({anchor_chunk_ids_, batch_chunk_ids});
    anchor_ids_ = torch::cat({anchor_ids_, batch_anchor_ids});
    exist_since_iter_ = torch::cat({exist_since_iter_, batch_exist});
    offset_gradient_accum_ = torch::cat({offset_gradient_accum_, batch_grad_accum});
    offset_denom_ = torch::cat({offset_denom_, batch_grad_denom});
    opacity_accum_ = torch::cat({opacity_accum_, batch_opacity_accum});
    anchor_denom_ = torch::cat({anchor_denom_, batch_anchor_denom});
  }

  is_initialized_ = true;
  int64_t new_N = anchor_.size(0);
  std::cout << "[Storage] Appended " << (new_N - prev_N)
            << " anchors from " << chunks_data.size()
            << " chunks (total: " << new_N << ")\n";
}

// =============================================================================
// Save All Chunks
// =============================================================================

void GaussianModel::saveAllChunks() {
  std::cout << "\n=== SAVING ALL CHUNKS TO DISK ===\n";

  torch::Tensor spatial_chunks =
      std::get<0>(torch::_unique2(anchor_chunk_ids_));

  if (spatial_chunks.size(0) == 0) {
    std::cout << "No chunks to save.\n";
    return;
  }

  torch::Tensor loaded_mask = torch::isin(spatial_chunks, chunks_loaded_from_disk_);
  torch::Tensor on_disk_mask = torch::isin(spatial_chunks, chunks_on_disk_);
  torch::Tensor new_mask = ~on_disk_mask & ~loaded_mask;

  torch::Tensor loaded_chunks = spatial_chunks.index({loaded_mask});
  torch::Tensor new_chunks = spatial_chunks.index({new_mask});

  std::cout << "Loaded chunks: " << loaded_chunks.size(0) << ", "
            << "New chunks: " << new_chunks.size(0) << "\n";

  if (loaded_chunks.size(0) > 0) saveChunks(loaded_chunks);
  if (new_chunks.size(0) > 0) saveChunks(new_chunks);

  std::cout << "=== SAVE COMPLETE ===\n";
}

}  // namespace scaffold_chungs
