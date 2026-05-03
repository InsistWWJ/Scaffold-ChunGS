/**
 * Scaffold-ChunGS: Configuration structs for anchor-based 3DGS SLAM.
 */

#pragma once

#include <string>
#include <vector>

// =============================================================================
// Anchor Model Configuration
// =============================================================================

struct AnchorModelConfig {
  // Anchor structure
  int n_offsets = 5;          // Child Gaussians per anchor (K)
  int feat_dim = 32;          // Anchor feature dimension (D)
  float voxel_size = 0.01f;   // Initial voxel size for anchor placement
  int sh_degree = 1;          // Spherical harmonics degree (unused: MLP predicts colors)

  // MLP architecture flags
  bool use_feat_bank = false;     // View-adaptive feature blending
  bool add_opacity_dist = true;   // Feed view distance into opacity MLP
  bool add_cov_dist = true;       // Feed view distance into covariance MLP
  bool add_color_dist = true;     // Feed view distance into color MLP
  int appearance_dim = 0;         // Per-camera appearance embedding (0 = disabled)

  // Densification parameters
  int update_depth = 3;           // Hierarchical levels for anchor growing
  int update_init_factor = 100;   // Initial voxel multiplication factor
  int update_hierarchy_factor = 4;  // Voxel size shrink per level
  float densify_grad_threshold = 0.0002f;
  float densify_opacity_threshold = 0.005f;
  int densify_check_interval = 100;

  // Pruning parameters
  float prune_opacity_threshold = 0.005f;
  int prune_check_interval = 100;
  int prune_success_threshold = 80;
};

// =============================================================================
// Chunk Memory Configuration
// =============================================================================

struct ChunkConfig {
  float chunk_size = 20.0f;            // World-unit size of each chunk
  int64_t max_anchors_in_memory = 300000;  // Eviction threshold
  int new_anchor_chunk_density = 10;   // Min anchors per chunk for new points
  std::string storage_base_path = "./chunks";  // Disk storage directory
};

// =============================================================================
// Optimization Configuration
// =============================================================================

struct OptimizationConfig {
  // Learning rates (anchor parameters)
  float anchor_position_lr_init = 0.00005f;
  float anchor_position_lr_decay = 0.99998f;
  float anchor_feature_lr = 0.005f;
  float anchor_opacity_lr = 0.1f;
  float anchor_scaling_lr = 0.01f;
  float anchor_rotation_lr = 0.002f;
  float offset_lr = 0.0001f;

  // MLP learning rates
  float mlp_opacity_lr = 0.005f;
  float mlp_cov_lr = 0.005f;
  float mlp_color_lr = 0.0025f;

  // Loss weights
  float lambda_dssim = 0.2f;
  float lambda_depth = 0.1f;
  float lambda_isotropic = 0.01f;

  // Training
  int max_num_iterations = -1;         // -1 = unlimited
  int new_keyframe_times_of_use = 8;
  int stable_num_iter_existence = 1;
  bool smooth_l1 = false;
  float spatial_lr_scale = 1.0f;
};

// =============================================================================
// Camera Configuration
// =============================================================================

struct CameraConfig {
  float z_near = 0.01f;
  float z_far = 100.0f;
};

// =============================================================================
// Keyframe Configuration
// =============================================================================

struct KeyframeConfig {
  float large_rotation_threshold = 1.0f;     // radians
  float large_translation_threshold = 0.1f;   // world units
  int min_num_initial_map_kfs = 10;
  int gaus_pyramid_sub_levels = 1;
  std::vector<float> gaus_pyramid_factors = {1.0f, 0.5f, 0.25f};
};

// =============================================================================
// Master Configuration
// =============================================================================

struct ScaffoldChunGSConfig {
  AnchorModelConfig anchor;
  ChunkConfig chunk;
  OptimizationConfig optimization;
  CameraConfig camera;
  KeyframeConfig keyframe;

  // Load from OpenCV FileStorage YAML
  static ScaffoldChunGSConfig fromYAML(const std::string& config_path);

  // Data device
  std::string data_device = "cuda";
  bool white_background = false;
};
