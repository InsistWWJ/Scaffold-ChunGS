/**
 * Scaffold-ChunGS: Configuration structs for anchor-based 3DGS SLAM.
 */

#pragma once

#include "depth_estimator.h"
#include "loop_closing.h"
#include "tracking.h"

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
// Mapper Configuration (DiskChunGS-aligned)
// =============================================================================

struct MapperConfig {
  float min_depth = 0.01f;
  float max_depth = 100.0f;
  int min_num_initial_map_kfs = 10;
  int new_keyframe_times_of_use = 8;
  int local_BA_increased_times_of_use = 4;
  int loop_closure_increased_times_of_use = 8;
  int stable_num_iter_existence = 1;
  int loop_closure_optimization_iterations = 1000;
  float loop_closure_memory_multiplier = 8.0f;
  bool auto_distribute_learning_rates = true;
};

// =============================================================================
// Gaussian Pyramid Configuration (DiskChunGS-aligned)
// =============================================================================

struct GausPyramidConfig {
  int num_sub_levels = 1;
  std::vector<float> factors = {1.0f, 0.5f, 0.25f};
  std::vector<int> times_of_use = {8, 4, 2};
};

// =============================================================================
// Pipeline Flags (DiskChunGS-aligned)
// =============================================================================

struct PipelineConfig {
  bool convert_SHs = false;        // SH conversion (unused: MLP predicts colors)
  bool compute_cov3D = true;       // Compute 3D covariance from scaling+rotation
  bool use_pose_optimization = false;  // Joint pose + Gaussian optimization
  bool use_exposure_optimization = false;
};

// =============================================================================
// Viewer Configuration
// =============================================================================

struct ViewerConfig {
  int window_width = 1920;
  int window_height = 1080;
  std::string window_title = "Scaffold-ChunGS Viewer";
  bool fullscreen = false;
  bool vsync = true;

  // Rendering
  float point_size = 3.0f;
  float frustum_scale = 0.5f;
  int max_display_gaussians = 500000;
  bool show_cameras = true;
  bool show_points = true;
  bool show_grid = true;

  // Camera
  float orbit_distance = 5.0f;
  float orbit_azimuth = 0.0f;
  float orbit_elevation = 0.5f;
  float orbit_sensitivity = 0.005f;
  float zoom_sensitivity = 0.1f;

  // Colors
  float bg_color_r = 0.15f;
  float bg_color_g = 0.15f;
  float bg_color_b = 0.15f;
  float camera_color_r = 0.0f;
  float camera_color_g = 1.0f;
  float camera_color_b = 0.0f;
  float trajectory_color_r = 1.0f;
  float trajectory_color_g = 0.5f;
  float trajectory_color_b = 0.0f;
};

// =============================================================================
// Master Configuration
// =============================================================================

struct ScaffoldChunGSConfig {
  // Model and scene
  AnchorModelConfig anchor;
  ChunkConfig chunk;

  // Pipeline (DiskChunGS-aligned sections)
  OptimizationConfig optimization;
  CameraConfig camera;
  MapperConfig mapper;
  KeyframeConfig keyframe;
  GausPyramidConfig gaus_pyramid;
  PipelineConfig pipeline;

  // Subsystem configs
  TrackingConfig tracking;
  LoopClosingConfig loop_closing;
  StereoSGBMConfig stereo_sgbm;
  ViewerConfig viewer;
  bool enable_viewer = false;

  // Load from OpenCV FileStorage YAML
  static ScaffoldChunGSConfig fromYAML(const std::string& config_path);

  // Data device
  std::string data_device = "cuda";
  bool white_background = false;
};
