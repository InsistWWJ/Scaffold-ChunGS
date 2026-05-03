/**
 * Scaffold-ChunGS: Anchor-based Gaussian Model with chunk memory management.
 *
 * Each "anchor" is a voxel center with a learned feature vector. MLP decoders
 * predict child Gaussian parameters (opacity, color, covariance) from the
 * anchor feature + view direction. Total child Gaussians = N_anchors * n_offsets.
 *
 * Chunks partition anchors spatially. Chunk-level LRU eviction swaps inactive
 * anchors to disk, enabling out-of-core operation.
 */

#pragma once

#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "anchor_mlp.h"
#include "chunk_types.h"
#include "config.h"

namespace scaffold_chungs {

// =============================================================================
// ChunkData — Unit of Disk I/O
// =============================================================================

static constexpr int kNumAnchorParamGroups = 8;
// Groups: 0=anchor_pos, 1=offset, 2=anchor_feat, 3=anchor_opacity,
//         4=anchor_scaling, 5=anchor_rotation, 6=mlp_params, 7=appearance_emb

struct ChunkData {
  // Anchor parameters
  torch::Tensor anchor;          // [A, 3]
  torch::Tensor offset;          // [A, K, 3]
  torch::Tensor anchor_feat;     // [A, D]
  torch::Tensor anchor_scaling;  // [A, 6]
  torch::Tensor anchor_rotation; // [A, 4]
  torch::Tensor anchor_opacity;  // [A, 1]

  // Tracking tensors
  torch::Tensor exist_since;        // [A]
  torch::Tensor anchor_chunk_ids;   // [A]
  torch::Tensor anchor_ids;         // [A]
  torch::Tensor offset_gradient_accum;  // [A*K, 1]
  torch::Tensor offset_denom;           // [A*K, 1]
  torch::Tensor opacity_accum;          // [A, 1]
  torch::Tensor anchor_denom;           // [A, 1]

  // Optimizer states (per param group)
  std::vector<torch::Tensor> exp_avg_states;
  std::vector<torch::Tensor> exp_avg_sq_states;
  std::vector<int64_t> step_counts;

  int num_anchors = 0;
  int64_t chunk_id = 0;
};

// =============================================================================
// GaussianModel — Anchor-based 3DGS with Chunk I/O
// =============================================================================

class GaussianModel {
 public:
  GaussianModel(const AnchorModelConfig& anchor_cfg,
                const ChunkConfig& chunk_cfg,
                torch::DeviceType device_type = torch::kCUDA);

  ~GaussianModel() = default;

  // ===========================================================================
  // Initialization
  // ===========================================================================

  /** Create anchors from a 3D point cloud (positions + colors). */
  void initializeFromPoints(const torch::Tensor& xyz, const torch::Tensor& colors);

  /** Add more anchors from new observations. */
  void addAnchors(const torch::Tensor& xyz, const torch::Tensor& colors);

  /** Set up Adam optimizer with proper param groups. */
  void trainingSetup(const OptimizationConfig& opt_cfg);

  // ===========================================================================
  // Anchor-to-Gaussian Expansion (for rendering)
  // ===========================================================================

  /** Slice anchor data for the given visibility mask. */
  AnchorData getVisibleAnchors(const torch::Tensor& visible_mask) const;

  /** Get the MLP module for expansion. */
  AnchorMLP& mlp() { return mlp_; }
  const AnchorMLP& mlp() const { return mlp_; }

  // ===========================================================================
  // Anchor Growing & Pruning (densification)
  // ===========================================================================

  /** Grow new anchors in areas with high gradient. */
  void anchorGrowing(float grad_threshold, int depth_level);

  /** Remove low-opacity anchors. */
  void anchorPruning(float opacity_threshold, int min_observations);

  /** Full densification cycle: grow + prune. */
  void adjustAnchors(int check_interval, int success_threshold,
                    float grad_threshold, float min_opacity);

  /** Concatenate new anchor tensors to existing ones with optimizer extension. */
  void catAnchorsToOptimizer(const std::vector<torch::Tensor>& new_params,
                             const std::vector<std::pair<int, int>>& param_group_indices);

  // ===========================================================================
  // Optimizer & Training
  // ===========================================================================

  void optimizerStep(const torch::Tensor& visibility, uint32_t N);
  void optimizerZeroGrad();
  void updateLearningRates(const torch::Tensor& visibility);
  void resetOpacityForMask(const torch::Tensor& anchor_mask);
  void resetAnchorLRAndOptimizerState(const torch::Tensor& anchor_mask);

  // ===========================================================================
  // Chunk Memory Management
  // ===========================================================================

  void loadChunks(const torch::Tensor& chunk_ids);
  void saveChunks(const torch::Tensor& chunk_ids);
  void saveAndEvictChunks(const torch::Tensor& chunk_ids);
  void saveAllChunks();
  void evictExcessChunks(const torch::Tensor& protected_chunk_ids,
                        int64_t excess_anchors);
  void checkMemoryPressure();
  torch::Tensor cullVisibleAnchors(const Eigen::Vector3f& camera_center,
                                   const Eigen::Matrix4f& world_view_proj,
                                   bool manage_memory = true);
  void updateChunkIDs();
  void updateChunkAccess(const torch::Tensor& accessed_chunk_ids);
  int64_t countAllAnchors();
  int64_t countAllGaussians();

  // ===========================================================================
  // Chunk ID Helpers
  // ===========================================================================

  torch::Tensor getVisibleChunks(const Eigen::Vector3f& camera_center,
                                 const Eigen::Matrix4f& world_view_proj) const;
  torch::Tensor createAnchorMaskFromChunks(const torch::Tensor& chunk_ids) const;

  // ===========================================================================
  // Accessors
  // ===========================================================================

  torch::Tensor& anchor() { return anchor_; }
  torch::Tensor& offset() { return offset_; }
  torch::Tensor& anchorFeat() { return anchor_feat_; }
  torch::Tensor& anchorScaling() { return anchor_scaling_; }
  torch::Tensor& anchorRotation() { return anchor_rotation_; }
  torch::Tensor& anchorOpacity() { return anchor_opacity_; }
  torch::Tensor& anchorChunkIDs() { return anchor_chunk_ids_; }
  torch::Tensor& anchorIDs() { return anchor_ids_; }
  torch::Tensor& existSinceIter() { return exist_since_iter_; }
  torch::Tensor& offsetGradientAccum() { return offset_gradient_accum_; }
  torch::Tensor& offsetDenom() { return offset_denom_; }
  torch::Tensor& opacityAccum() { return opacity_accum_; }
  torch::Tensor& anchorDenom() { return anchor_denom_; }

  const torch::Tensor& anchor() const { return anchor_; }
  void setAnchor(const torch::Tensor& v) { anchor_ = v; }

  bool isInitialized() const { return is_initialized_; }
  int getNumAnchors() const { return anchor_.size(0); }
  int getNumOffsets() const { return n_offsets_; }
  int getFeatDim() const { return feat_dim_; }
  float getChunkSize() const { return chunk_size_; }
  int64_t getMaxAnchorsInMemory() const { return max_anchors_in_memory_; }
  void setMaxAnchorsInMemory(int64_t v) { max_anchors_in_memory_ = v; }
  torch::DeviceType deviceType() const { return device_type_; }
  float spatialLRScale() const { return spatial_lr_scale_; }

  const std::shared_ptr<torch::optim::Adam>& optimizer() const { return optimizer_; }

 private:
  // Serialization helpers
  void saveTensorBinary(const torch::Tensor& tensor, std::ofstream& file);
  torch::Tensor loadTensorBinary(std::ifstream& file);
  std::string getChunkFilename(const ChunkCoord& coord);
  ChunkData extractChunkData(const torch::Tensor& chunk_mask, int64_t chunk_id);
  std::optional<ChunkData> loadSingleChunkFromDisk(int64_t chunk_id);
  void saveSingleChunkToDisk(int64_t chunk_id, const ChunkData& data);
  void appendLoadedChunks(const std::vector<ChunkData>& chunks_data,
                          const std::vector<int64_t>& chunk_ids);
  torch::Tensor findLRUChunks(const torch::Tensor& candidate_chunks,
                              int64_t target_anchor_count);
  int64_t countAnchorsToLoad(const torch::Tensor& chunk_ids) const;
  void optimizerRegisterAnchorParams();

  // MLP sub-network
  AnchorMLP mlp_;

  // ===========================================================================
  // Anchor Parameters (N = number of anchors)
  // ===========================================================================

  torch::Tensor anchor_;          // [N, 3] voxel center positions (trainable)
  torch::Tensor offset_;          // [N, K, 3] per-child offsets (trainable)
  torch::Tensor anchor_feat_;     // [N, D] per-anchor latent features (trainable)
  torch::Tensor anchor_scaling_;  // [N, 6] base scale (trainable)
  torch::Tensor anchor_rotation_; // [N, 4] base rotation quaternion
  torch::Tensor anchor_opacity_;  // [N, 1] base opacity (inverse-sigmoid)

  // ===========================================================================
  // Param Group Wrappers (for Adam optimizer registration)
  // ===========================================================================

  std::vector<torch::Tensor> Tensor_vec_anchor_;
  std::vector<torch::Tensor> Tensor_vec_offset_;
  std::vector<torch::Tensor> Tensor_vec_anchor_feat_;
  std::vector<torch::Tensor> Tensor_vec_anchor_opacity_;
  std::vector<torch::Tensor> Tensor_vec_anchor_scaling_;
  std::vector<torch::Tensor> Tensor_vec_anchor_rotation_;

  std::shared_ptr<torch::optim::Adam> optimizer_;

  // ===========================================================================
  // Anchor Tracking Tensors
  // ===========================================================================

  torch::Tensor anchor_chunk_ids_;        // [N] encoded spatial chunk per anchor
  torch::Tensor anchor_ids_;              // [N] unique persistent ID
  torch::Tensor exist_since_iter_;        // [N] creation iteration
  torch::Tensor offset_gradient_accum_;   // [N*K, 1] gradient normalization
  torch::Tensor offset_denom_;            // [N*K, 1]
  torch::Tensor opacity_accum_;           // [N, 1] opacity running sum
  torch::Tensor anchor_denom_;            // [N, 1] visibility counter

  // ===========================================================================
  // Chunk Tracking State
  // ===========================================================================

  torch::Tensor chunks_loaded_from_disk_;  // [C] chunk IDs currently in GPU
  torch::Tensor chunks_on_disk_;           // [D] all chunk IDs saved to disk
  torch::Tensor chunk_anchor_counts_;      // [D] anchor count per disk chunk
  std::unordered_map<int64_t, float> chunk_access_times_;  // LRU timestamps

  int64_t next_anchor_id_ = 0;
  int64_t max_anchors_in_memory_ = 300000;

  // ===========================================================================
  // Configuration
  // ===========================================================================

  float chunk_size_ = 20.0f;
  int n_offsets_ = 5;
  int feat_dim_ = 32;
  float voxel_size_ = 0.01f;
  std::string storage_base_path_;
  torch::DeviceType device_type_ = torch::kCUDA;
  float spatial_lr_scale_ = 1.0f;
  bool is_initialized_ = false;

  // Densification parameters
  int update_depth_ = 3;
  int update_init_factor_ = 100;
  int update_hierarchy_factor_ = 4;
};

}  // namespace scaffold_chungs
