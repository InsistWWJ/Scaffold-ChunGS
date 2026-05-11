/**
 * Scaffold-ChunGS: Chunk-based out-of-core memory management.
 *
 * LRU eviction policy: when total anchors exceed max_anchors_in_memory_,
 * the least-recently-accessed chunks are saved to disk and evicted from GPU.
 *
 * Access timestamps are recorded whenever a chunk is visible to a training
 * keyframe (via updateChunkAccess).
 */

#include "scaffold_chunks/gaussian_model.h"

#include <c10/cuda/CUDACachingAllocator.h>

#include <algorithm>
#include <future>
#include <iostream>
#include <thread>

namespace scaffold_chungs {

// =============================================================================
// GPU Memory Query
// =============================================================================

static size_t getGPUMemoryUsage() {
  try {
    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    return stats.allocated_bytes[static_cast<int>(
        c10::cuda::StatType::AGGREGATE)].current;
  } catch (...) {
    return 0;
  }
}

// =============================================================================
// LRU Chunk Selection
// =============================================================================

torch::Tensor GaussianModel::findLRUChunks(
    const torch::Tensor& candidate_chunks,
    int64_t target_anchor_count) {
  int64_t num_candidates = candidate_chunks.size(0);
  if (num_candidates == 0) {
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64)
        .device(candidate_chunks.device()));
  }

  // Batch-compute anchor counts via bincount on flat chunk IDs
  torch::Tensor chunk_ids_flat = anchor_chunk_ids_.contiguous();
  int64_t max_id = chunk_ids_flat.max().item<int64_t>();
  torch::Tensor counts = torch::bincount(
      chunk_ids_flat, torch::Tensor(), max_id + 1);

  // Gather counts for candidate chunks
  torch::Tensor cand_counts = counts.index({candidate_chunks});  // [C]

  // Build access time vector: sort by access time (missing = 0 = oldest)
  auto chunks_cpu = candidate_chunks.cpu();
  auto counts_cpu = cand_counts.cpu();
  auto chunks_acc = chunks_cpu.accessor<int64_t, 1>();
  auto counts_acc = counts_cpu.accessor<int64_t, 1>();

  struct ChunkInfo {
    int64_t id;
    float access_time;
    int64_t anchor_count;
  };
  std::vector<ChunkInfo> infos;
  infos.reserve(num_candidates);

  for (int64_t i = 0; i < num_candidates; ++i) {
    int64_t cid = chunks_acc[i];
    float at = 0.0f;
    auto it = chunk_access_times_.find(cid);
    if (it != chunk_access_times_.end()) at = it->second;
    infos.push_back({cid, at, counts_acc[i]});
  }

  std::sort(infos.begin(), infos.end(),
            [](const ChunkInfo& a, const ChunkInfo& b) {
              return a.access_time < b.access_time;
            });

  std::vector<int64_t> selected;
  int64_t accumulated = 0;
  for (const auto& info : infos) {
    if (accumulated >= target_anchor_count) break;
    selected.push_back(info.id);
    accumulated += info.anchor_count;
  }

  auto opt = torch::TensorOptions().dtype(torch::kInt64).device(
      candidate_chunks.device());
  if (selected.empty()) return torch::empty({0}, opt);
  return torch::tensor(selected, opt);
}

// =============================================================================
// Access Tracking
// =============================================================================

void GaussianModel::updateChunkAccess(const torch::Tensor& accessed_chunk_ids) {
  auto now = std::chrono::duration<float>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  auto cpu = accessed_chunk_ids.cpu();
  auto accessor = cpu.accessor<int64_t, 1>();
  for (int64_t i = 0; i < cpu.size(0); ++i) {
    chunk_access_times_[accessor[i]] = now;
  }
}

// =============================================================================
// Anchor Count Estimation for To-Be-Loaded Chunks
// =============================================================================

int64_t GaussianModel::countAnchorsToLoad(const torch::Tensor& chunk_ids) const {
  auto to_load_cpu = chunk_ids.cpu();
  auto disk_cpu = chunks_on_disk_.cpu();
  auto counts_cpu = chunk_anchor_counts_.cpu();

  auto to_load_acc = to_load_cpu.accessor<int64_t, 1>();
  auto disk_acc = disk_cpu.accessor<int64_t, 1>();
  auto counts_acc = counts_cpu.accessor<int64_t, 1>();

  int64_t total = 0;
  for (int64_t i = 0; i < to_load_acc.size(0); i++) {
    int64_t target = to_load_acc[i];
    for (int64_t j = 0; j < disk_acc.size(0); j++) {
      if (disk_acc[j] == target) {
        total += counts_acc[j];
        break;
      }
    }
  }
  return total;
}

// =============================================================================
// Load Chunks from Disk
// =============================================================================

void GaussianModel::loadChunks(const torch::Tensor& chunk_id_requests) {
  torch::NoGradGuard no_grad;
  if (chunk_id_requests.size(0) == 0) return;

  // Determine which requested chunks are on disk but not yet loaded
  torch::Tensor on_disk_mask = torch::isin(chunk_id_requests, chunks_on_disk_);
  torch::Tensor not_loaded_mask = ~torch::isin(chunk_id_requests,
                                                chunks_loaded_from_disk_);
  torch::Tensor chunks_needing_load =
      chunk_id_requests.index({on_disk_mask & not_loaded_mask});

  if (chunks_needing_load.size(0) == 0) {
    // All requested chunks already in memory — just check pressure
    checkMemoryPressure();
    return;
  }

  // Pre-emptive eviction to make room
  int64_t incoming = countAnchorsToLoad(chunks_needing_load);
  int64_t projected = anchor_.size(0) + incoming;
  int64_t limit = max_anchors_in_memory_;
  if (projected > limit) {
    int64_t excess = projected - limit;
    evictExcessChunks(chunks_needing_load, excess);
  }

  // Parallel load from disk with concurrency cap for embedded (Jetson 6-core)
  auto chunks_cpu = chunks_needing_load.cpu();
  auto accessor = chunks_cpu.accessor<int64_t, 1>();
  int num = static_cast<int>(chunks_cpu.size(0));
  unsigned int max_threads = std::max(1U, std::thread::hardware_concurrency());

  // Load in batches to bound thread count
  std::vector<ChunkData> to_append;
  std::vector<int64_t> loaded_ids;

  for (int batch_start = 0; batch_start < num; batch_start += static_cast<int>(max_threads)) {
    int batch_end = std::min(num, batch_start + static_cast<int>(max_threads));

    std::vector<std::future<std::pair<int64_t, std::optional<ChunkData>>>> futures;
    for (int i = batch_start; i < batch_end; ++i) {
      int64_t cid = accessor[i];
      futures.push_back(std::async(std::launch::async, [this, cid]() {
        return std::make_pair(cid, loadSingleChunkFromDisk(cid));
      }));
    }

    for (auto& f : futures) {
      auto [cid, data_opt] = f.get();
      if (data_opt.has_value()) {
        to_append.push_back(std::move(data_opt.value()));
        loaded_ids.push_back(cid);
      }
    }
  }

  if (!to_append.empty()) {
    appendLoadedChunks(to_append, loaded_ids);
  }

  // Update tracking
  chunks_loaded_from_disk_ = torch::cat({chunks_loaded_from_disk_,
                                         chunks_needing_load}, 0);
  chunks_loaded_from_disk_ = std::get<0>(torch::_unique2(chunks_loaded_from_disk_));
}

// =============================================================================
// Save Chunks to Disk
// =============================================================================

void GaussianModel::saveChunks(const torch::Tensor& chunk_ids_to_save) {
  torch::NoGradGuard no_grad;
  if (chunk_ids_to_save.size(0) == 0) return;

  auto chunks_cpu = chunk_ids_to_save.cpu();
  auto accessor = chunks_cpu.accessor<int64_t, 1>();
  int num = chunks_cpu.size(0);

  // Extract chunk data on main thread (requires GPU tensors)
  std::vector<std::pair<int64_t, ChunkData>> prepared;
  std::unordered_map<int64_t, int64_t> id_to_count;
  for (int i = 0; i < num; ++i) {
    int64_t cid = accessor[i];
    torch::Tensor mask = (anchor_chunk_ids_ == cid);
    ChunkData data = extractChunkData(mask, cid);
    id_to_count[cid] = data.num_anchors;
    prepared.emplace_back(cid, std::move(data));
  }

  // Write in parallel with concurrency cap
  unsigned int max_threads = std::max(1U, std::thread::hardware_concurrency());
  std::vector<int64_t> succeeded;

  for (size_t batch_start = 0; batch_start < prepared.size();
       batch_start += static_cast<size_t>(max_threads)) {
    size_t batch_end = std::min(prepared.size(), batch_start + max_threads);

    std::vector<std::future<std::pair<int64_t, bool>>> futures;
    for (size_t i = batch_start; i < batch_end; ++i) {
      futures.push_back(std::async(std::launch::async,
          [this](int64_t id, ChunkData d) {
            try {
              saveSingleChunkToDisk(id, d);
              return std::make_pair(id, true);
            } catch (const std::exception& e) {
              std::cerr << "[Save] Failed chunk " << id << ": " << e.what() << "\n";
              return std::make_pair(id, false);
            }
          }, prepared[i].first, std::move(prepared[i].second)));
    }

    for (auto& f : futures) {
      auto [cid, ok] = f.get();
      if (ok) succeeded.push_back(cid);
    }
  }

  // Update disk tracking metadata
  for (int64_t cid : succeeded) {
    torch::Tensor cid_t = torch::tensor({cid},
        torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
    torch::Tensor count_t = torch::tensor({id_to_count[cid]},
        torch::TensorOptions().dtype(torch::kInt64).device(device_type_));

    auto mask = torch::eq(chunks_on_disk_, cid_t);
    if (torch::any(mask).item<bool>()) {
      auto indices = torch::where(mask)[0];
      chunk_anchor_counts_[indices[0].item<int64_t>()] = id_to_count[cid];
    } else {
      chunks_on_disk_ = torch::cat({chunks_on_disk_, cid_t}, 0);
      chunk_anchor_counts_ = torch::cat({chunk_anchor_counts_, count_t}, 0);
    }
  }
}

// =============================================================================
// Eviction
// =============================================================================

void GaussianModel::saveAndEvictChunks(const torch::Tensor& chunk_ids) {
  if (chunk_ids.size(0) == 0) return;

  // Categorize
  torch::Tensor loaded_mask = torch::isin(chunk_ids, chunks_loaded_from_disk_);
  torch::Tensor on_disk_mask = torch::isin(chunk_ids, chunks_on_disk_);
  torch::Tensor spillover_mask = on_disk_mask & ~loaded_mask;
  torch::Tensor new_mask = ~on_disk_mask & ~loaded_mask;

  torch::Tensor loaded = chunk_ids.index({loaded_mask});
  torch::Tensor new_chunks = chunk_ids.index({new_mask});

  // Save loaded & new chunks
  if (loaded.size(0) > 0) saveChunks(loaded);
  if (new_chunks.size(0) > 0) saveChunks(new_chunks);
  // Spillover chunks NOT re-saved (disk version is more complete)

  // Remove from GPU
  torch::Tensor remove_mask = torch::isin(anchor_chunk_ids_, chunk_ids);

  if (!remove_mask.any().item<bool>()) {
    // Already evicted somehow
    return;
  }

  torch::Tensor valid_mask = ~remove_mask;

  // Prune evicted anchors from all tensors
  auto prune = [&](torch::Tensor& t) {
    if (t.size(0) == anchor_.size(0) || t.size(0) == anchor_.size(0) * n_offsets_) {
      t = t.index({valid_mask});
    }
  };

  prune(anchor_);
  prune(offset_);
  prune(anchor_feat_);
  prune(anchor_scaling_);
  prune(anchor_rotation_);
  prune(anchor_opacity_);
  prune(anchor_ids_);
  prune(anchor_chunk_ids_);
  prune(exist_since_iter_);
  prune(opacity_accum_);
  prune(anchor_denom_);

  if (offset_gradient_accum_.size(0) > 0) {
    // Build valid mask for [N*K] from valid anchor mask
    torch::Tensor expanded = valid_mask.unsqueeze(1).expand(
        {valid_mask.size(0), n_offsets_}).reshape({-1});
    offset_gradient_accum_ = offset_gradient_accum_.index({expanded});
    offset_denom_ = offset_denom_.index({expanded});
  }

  // Prune optimizer state (Adam momentum) for evicted anchors
  if (optimizer_) {
    auto& param_groups = optimizer_->param_groups();
    auto& state = optimizer_->state();

    for (size_t g = 0; g < param_groups.size(); ++g) {
      auto& params = param_groups[g].params();
      if (params.empty()) continue;
      auto key = params[0].unsafeGetTensorImpl();
      auto it = state.find(key);
      if (it == state.end()) continue;

      auto& adam = static_cast<torch::optim::AdamParamState&>(*it->second);
      if (adam.exp_avg().defined() && adam.exp_avg().size(0) > 0) {
        // Determine whether exp_avg is [N] or [N*K] shaped
        if (adam.exp_avg().size(0) == remove_mask.size(0)) {
          adam.exp_avg(adam.exp_avg().index({valid_mask}));
          adam.exp_avg_sq(adam.exp_avg_sq().index({valid_mask}));
        } else if (adam.exp_avg().size(0) == remove_mask.size(0) * n_offsets_) {
          torch::Tensor expanded = valid_mask.unsqueeze(1)
              .expand({valid_mask.size(0), n_offsets_}).reshape({-1});
          adam.exp_avg(adam.exp_avg().index({expanded}));
          adam.exp_avg_sq(adam.exp_avg_sq().index({expanded}));
        }
      }
    }

    // Update Tensor_vec wrappers to match pruned tensors
    Tensor_vec_anchor_ = {anchor_};
    Tensor_vec_offset_ = {offset_};
    Tensor_vec_anchor_feat_ = {anchor_feat_};
    Tensor_vec_anchor_opacity_ = {anchor_opacity_};
    Tensor_vec_anchor_scaling_ = {anchor_scaling_};
    Tensor_vec_anchor_rotation_ = {anchor_rotation_};

    // Re-register param group 0 tensor (key changed after index())
    if (!param_groups.empty() && !param_groups[0].params().empty()) {
      auto old_key0 = param_groups[0].params()[0].unsafeGetTensorImpl();
      auto it0 = state.find(old_key0);
      if (it0 != state.end()) {
        auto adam_state = std::make_unique<torch::optim::AdamParamState>(
            static_cast<torch::optim::AdamParamState&>(*it0->second));
        state.erase(it0);
        auto new_key = anchor_.unsafeGetTensorImpl();
        state[new_key] = std::move(adam_state);
        param_groups[0].params() = {anchor_};
      }
    }
  }

  // Update loaded tracking
  chunks_loaded_from_disk_ = chunks_loaded_from_disk_.index(
      {~torch::isin(chunks_loaded_from_disk_, chunk_ids)});

  // Remove access time entries
  auto cpu = chunk_ids.cpu();
  auto acc = cpu.accessor<int64_t, 1>();
  for (int64_t i = 0; i < cpu.size(0); ++i) {
    chunk_access_times_.erase(acc[i]);
  }
}

void GaussianModel::evictExcessChunks(const torch::Tensor& protected_chunk_ids,
                                       int64_t excess_anchors) {
  torch::Tensor spatial = std::get<0>(torch::_unique2(anchor_chunk_ids_));
  if (spatial.size(0) == 0) return;

  torch::Tensor evictable_mask = ~torch::isin(spatial, protected_chunk_ids);
  torch::Tensor evictable = spatial.index({evictable_mask});
  if (evictable.size(0) == 0) {
    std::cerr << "[Evict] WARNING: No evictable chunks (all protected)\n";
    return;
  }

  // 5% hysteresis buffer to reduce eviction frequency
  int64_t buffer = static_cast<int64_t>(max_anchors_in_memory_ * 0.05f);
  int64_t target = excess_anchors + buffer;

  torch::Tensor lru = findLRUChunks(evictable, target);
  if (lru.size(0) == 0) {
    std::cerr << "[Evict] No LRU candidates found\n";
    return;
  }

  saveAndEvictChunks(lru);
}

void GaussianModel::checkMemoryPressure() {
  int64_t current = anchor_.size(0);
  int64_t limit = max_anchors_in_memory_;

  if (current > limit) {
    int64_t excess = current - limit;
    std::cout << "[Memory] " << current << " anchors > " << limit
              << " limit, evicting " << excess << "...\n";

    torch::Tensor spatial = std::get<0>(torch::_unique2(anchor_chunk_ids_));
    torch::Tensor empty_protect = torch::empty({0},
        torch::TensorOptions().dtype(torch::kInt64).device(device_type_));
    evictExcessChunks(empty_protect, excess);
  }
}

// =============================================================================
// Visibility-Based Chunk Management
// =============================================================================

torch::Tensor GaussianModel::cullVisibleAnchors(
    const Eigen::Vector3f& camera_center,
    const Eigen::Matrix4f& world_view_proj,
    bool manage_memory) {

  if (!is_initialized_ || anchor_.size(0) == 0) {
    return torch::zeros({0},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  }

  // Step 1: Determine which chunks intersect the frustum
  // For simplicity: use a loose AABB check based on anchor chunk IDs
  auto unique_chunks = std::get<0>(torch::_unique2(anchor_chunk_ids_));

  // Convert camera frustum to an approximate AABB
  // Simplified: use camera position + max_view_dist
  float max_view_dist = 100.0f;  // from camera z_far
  Eigen::Vector3f cam_min = camera_center - Eigen::Vector3f::Constant(max_view_dist);
  Eigen::Vector3f cam_max = camera_center + Eigen::Vector3f::Constant(max_view_dist);

  // Filter chunks that overlap with camera AABB
  std::vector<int64_t> visible_chunk_ids;
  auto chunks_cpu = unique_chunks.cpu();
  auto acc = chunks_cpu.accessor<int64_t, 1>();
  for (int64_t i = 0; i < chunks_cpu.size(0); ++i) {
    ChunkCoord coord = decodeChunkCoord(acc[i]);
    AABB chunk_aabb = getChunkAABB(coord, chunk_size_);
    // Simple overlap test
    if (chunk_aabb.max.x() >= cam_min.x() && chunk_aabb.min.x() <= cam_max.x() &&
        chunk_aabb.max.y() >= cam_min.y() && chunk_aabb.min.y() <= cam_max.y() &&
        chunk_aabb.max.z() >= cam_min.z() && chunk_aabb.min.z() <= cam_max.z()) {
      visible_chunk_ids.push_back(acc[i]);
    }
  }

  if (visible_chunk_ids.empty()) {
    return torch::zeros({anchor_.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
  }

  auto opt = torch::TensorOptions().dtype(torch::kInt64).device(device_type_);
  torch::Tensor visible_chunk_tensor = torch::tensor(visible_chunk_ids, opt);

  // Step 2: Load visible chunks from disk if needed
  if (manage_memory) {
    torch::Tensor not_loaded = ~torch::isin(visible_chunk_tensor,
                                            chunks_loaded_from_disk_);
    // Also check if they're in memory (spillover or new)
    torch::Tensor in_memory = torch::isin(visible_chunk_tensor,
        std::get<0>(torch::_unique2(anchor_chunk_ids_)));

    torch::Tensor to_load = visible_chunk_tensor.index({not_loaded &
        ~in_memory & torch::isin(visible_chunk_tensor, chunks_on_disk_)});
    if (to_load.size(0) > 0) {
      loadChunks(to_load);
    }

    // Track access times for LRU
    updateChunkAccess(visible_chunk_tensor);

    // Check memory pressure after loading
    checkMemoryPressure();
  }

  // Step 3: Return anchor-level visibility mask
  return torch::isin(anchor_chunk_ids_, visible_chunk_tensor);
}

}  // namespace scaffold_chungs
