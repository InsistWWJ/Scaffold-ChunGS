/**
 * Scaffold-ChunGS: Main training loop.
 *
 * Orchestrates:
 *   1. Keyframe selection
 *   2. Anchor visibility culling + chunk memory management
 *   3. Anchor-to-Gaussian expansion via MLP
 *   4. Rendering
 *   5. Loss computation + backward pass
 *   6. Periodic anchor growing/pruning
 */

#include "scaffold_chunks/gaussian_model.h"
#include "scaffold_chunks/gaussian_renderer.h"
#include "scaffold_chunks/gaussian_keyframe.h"
#include "scaffold_chunks/gaussian_scene.h"
#include "scaffold_chunks/keyframe_selection.h"
#include "scaffold_chunks/frustum_culler.h"
#include "scaffold_chunks/config.h"
#include "scaffold_chunks/training_loss.h"

#include <iostream>
#include <chrono>
#include <memory>
#include <cmath>

namespace scaffold_chungs {

// =============================================================================
// ScaffoldTrainer
// =============================================================================

class ScaffoldTrainer {
 public:
  ScaffoldTrainer(std::shared_ptr<GaussianModel> model,
                  std::shared_ptr<GaussianScene> scene,
                  const OptimizationConfig& opt_cfg,
                  const KeyframeConfig& kf_cfg)
      : model_(model), scene_(scene), opt_cfg_(opt_cfg), kf_cfg_(kf_cfg),
        current_iteration_(0) {
    keyframe_selector_ = std::make_shared<KeyframeSelection>(scene_);
    preallocateBuffers();
  }

  float trainOneIteration();
  void run();

  void signalStop() { stopped_ = true; }
  bool stopped() const { return stopped_; }
  int currentIteration() const { return current_iteration_; }

  std::shared_ptr<GaussianModel> model() const { return model_; }
  std::shared_ptr<GaussianScene> scene() const { return scene_; }

 private:
  void preallocateBuffers();

  std::shared_ptr<GaussianModel> model_;
  std::shared_ptr<GaussianScene> scene_;
  std::shared_ptr<KeyframeSelection> keyframe_selector_;
  OptimizationConfig opt_cfg_;
  KeyframeConfig kf_cfg_;

  int current_iteration_ = 0;
  bool stopped_ = false;

  // Pre-allocated render buffers to avoid per-frame allocation
  torch::Tensor bg_color_;
  bool buffers_allocated_ = false;
};

void ScaffoldTrainer::preallocateBuffers() {
  auto device = model_->deviceType() == torch::kCUDA ?
      torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  bg_color_ = torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  buffers_allocated_ = true;
}

// =============================================================================
// Single Training Iteration
// =============================================================================

float ScaffoldTrainer::trainOneIteration() {
  // Step 1: Select keyframe
  auto kf = keyframe_selector_->getNextKeyframe();
  if (!kf) {
    return -1.0f;
  }

  // Ensure keyframe tensors are on GPU
  kf->transferToGPU(model_->deviceType());

  // Step 2: Compute transforms directly on model's device
  auto dev = model_->deviceType();
  kf->computeTransformTensors(dev);
  auto camera_center = kf->getCameraCenter(dev);
  auto w2v = kf->worldViewTransform();
  auto proj = kf->projectionMatrix();

  // Step 3: Frustum cull + load visible chunks
  Eigen::Vector3f cc_eigen(camera_center[0].item<float>(),
                           camera_center[1].item<float>(),
                           camera_center[2].item<float>());

  Eigen::Matrix4f wvp_eigen = Eigen::Matrix4f::Identity();
  if (model_->deviceType() == torch::kCPU) {
    // Already on CPU — direct accessor
    auto fp_acc = torch::matmul(proj, w2v).accessor<float, 2>();
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        wvp_eigen(r, c) = fp_acc[r][c];
  } else {
    // GPU tensor — copy to CPU first
    torch::Tensor full_proj = torch::matmul(proj, w2v).cpu();
    auto fp_acc = full_proj.accessor<float, 2>();
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        wvp_eigen(r, c) = fp_acc[r][c];
  }

  torch::Tensor visible_mask = model_->cullVisibleAnchors(
      cc_eigen, wvp_eigen, /*manage_memory=*/true);

  // Step 4: Render
  auto output = ScaffoldRenderer::render(
      *model_, visible_mask, camera_center, w2v, proj,
      kf->FoVx(), kf->FoVy(),
      kf->imageHeight(), kf->imageWidth(),
      bg_color_, 1.0f);

  // Step 5: Compute loss (returns tensors in autograd graph)
  torch::Tensor gt_image = kf->getGTImage();
  torch::Tensor gt_depth = kf->getGTDepth();
  torch::Tensor mask = kf->getUndistortMask();

  // Note: placeholder rasterizer uses .item() calls which break autograd;
  // gradients will flow once CUDA rasterizer (diff_gaussian_rasterization)
  // is integrated. The MLP outputs remain in the autograd graph.
  auto loss = computeLosses(
      output.color, output.depth, gt_image, gt_depth,
      output.radii, mask,
      opt_cfg_.lambda_dssim, opt_cfg_.lambda_depth,
      opt_cfg_.lambda_isotropic);

  // Step 6: Backward — propagates gradients through MLP (when rasterizer supports it)
  if (loss.total.requires_grad() || loss.total.item<float>() > 0) {
    loss.total.backward();
  }

  // Step 7: Optimizer step
  model_->optimizerStep(visible_mask, visible_mask.sum().item<int>());
  model_->optimizerZeroGrad();

  // Step 8: Record loss for keyframe selection
  keyframe_selector_->recordLoss(kf->fid(), loss.total.item<float>());

  // Step 9: Periodic densification (adjustAnchors handles optimizer internally)
  if (current_iteration_ % opt_cfg_.densify_check_interval == 0 &&
      current_iteration_ > 0) {
    model_->adjustAnchors(
        opt_cfg_.densify_check_interval,
        opt_cfg_.prune_success_threshold,
        opt_cfg_.densify_grad_threshold,
        opt_cfg_.densify_opacity_threshold);
  }

  current_iteration_++;

  if (current_iteration_ % 50 == 0) {
    std::cout << "[Iter " << current_iteration_ << "] "
              << "L1=" << loss.l1_color
              << " SSIM=" << loss.ssim_value
              << " Iso=" << loss.isotropic
              << " Total=" << loss.total.item<float>()
              << " Anchors=" << model_->getNumAnchors()
              << " KF=" << kf->fid() << "\n";
  }

  return loss.total.item<float>();
}

// =============================================================================
// Full Training Loop
// =============================================================================

void ScaffoldTrainer::run() {
  std::cout << "[Trainer] Starting training loop"
            << " (max_iters=" << opt_cfg_.max_num_iterations << ")\n";

  auto start_time = std::chrono::steady_clock::now();
  int training_kf_count = 0;

  while (!stopped_) {
    if (opt_cfg_.max_num_iterations > 0 &&
        current_iteration_ >= opt_cfg_.max_num_iterations) {
      break;
    }

    float loss = trainOneIteration();
    if (loss < 0) {
      auto kfs = scene_->getAllKeyframes();
      for (auto& [fid, kf] : kfs) {
        kf->setTimesOfUse(opt_cfg_.new_keyframe_times_of_use);
      }
      continue;
    }
    training_kf_count++;
  }

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
      end_time - start_time).count();

  std::cout << "[Trainer] Finished " << training_kf_count
            << " training iterations in " << elapsed << "s"
            << " (" << (elapsed > 0 ? training_kf_count / elapsed : 0)
            << " it/s)\n";

  model_->saveAllChunks();
  std::cout << "[Trainer] Total anchors: " << model_->countAllAnchors()
            << " (" << model_->countAllGaussians() << " gaussians)\n";
}

// =============================================================================
// Config Loading (YAML via OpenCV FileStorage)
// =============================================================================

ScaffoldChunGSConfig ScaffoldChunGSConfig::fromYAML(const std::string& config_path) {
  cv::FileStorage fs(config_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Cannot open config file: " + config_path);
  }

  ScaffoldChunGSConfig cfg;

  // Model section
  cfg.anchor.n_offsets = static_cast<int>(fs["Anchor.n_offsets"]);
  cfg.anchor.feat_dim = static_cast<int>(fs["Anchor.feat_dim"]);
  cfg.anchor.voxel_size = static_cast<float>(fs["Anchor.voxel_size"]);
  cfg.anchor.sh_degree = static_cast<int>(fs["Anchor.sh_degree"]);
  cfg.anchor.use_feat_bank = static_cast<int>(fs["Anchor.use_feat_bank"]) != 0;
  cfg.anchor.add_opacity_dist = static_cast<int>(fs["Anchor.add_opacity_dist"]) != 0;
  cfg.anchor.add_cov_dist = static_cast<int>(fs["Anchor.add_cov_dist"]) != 0;
  cfg.anchor.add_color_dist = static_cast<int>(fs["Anchor.add_color_dist"]) != 0;
  cfg.anchor.appearance_dim = static_cast<int>(fs["Anchor.appearance_dim"]);
  cfg.anchor.update_depth = static_cast<int>(fs["Anchor.update_depth"]);
  cfg.anchor.update_init_factor = static_cast<int>(fs["Anchor.update_init_factor"]);
  cfg.anchor.update_hierarchy_factor = static_cast<int>(fs["Anchor.update_hierarchy_factor"]);
  cfg.anchor.densify_grad_threshold = static_cast<float>(fs["Anchor.densify_grad_threshold"]);
  cfg.anchor.densify_opacity_threshold =
      static_cast<float>(fs["Anchor.densify_opacity_threshold"]);
  cfg.anchor.densify_check_interval = static_cast<int>(fs["Anchor.densify_check_interval"]);
  cfg.anchor.prune_opacity_threshold =
      static_cast<float>(fs["Anchor.prune_opacity_threshold"]);
  cfg.anchor.prune_check_interval = static_cast<int>(fs["Anchor.prune_check_interval"]);
  cfg.anchor.prune_success_threshold = static_cast<int>(fs["Anchor.prune_success_threshold"]);

  // Chunk section
  cfg.chunk.chunk_size = static_cast<float>(fs["Chunk.chunk_size"]);
  cfg.chunk.max_anchors_in_memory = static_cast<int>(fs["Chunk.max_anchors_in_memory"]);
  cfg.chunk.new_anchor_chunk_density = static_cast<int>(fs["Chunk.new_anchor_chunk_density"]);
  cfg.chunk.storage_base_path = static_cast<std::string>(fs["Chunk.storage_base_path"]);

  // Optimization section
  cfg.optimization.anchor_position_lr_init =
      static_cast<float>(fs["Optimization.anchor_position_lr_init"]);
  cfg.optimization.anchor_position_lr_decay =
      static_cast<float>(fs["Optimization.anchor_position_lr_decay"]);
  cfg.optimization.anchor_feature_lr =
      static_cast<float>(fs["Optimization.anchor_feature_lr"]);
  cfg.optimization.anchor_opacity_lr =
      static_cast<float>(fs["Optimization.anchor_opacity_lr"]);
  cfg.optimization.anchor_scaling_lr =
      static_cast<float>(fs["Optimization.anchor_scaling_lr"]);
  cfg.optimization.anchor_rotation_lr =
      static_cast<float>(fs["Optimization.anchor_rotation_lr"]);
  cfg.optimization.offset_lr = static_cast<float>(fs["Optimization.offset_lr"]);
  cfg.optimization.mlp_opacity_lr = static_cast<float>(fs["Optimization.mlp_opacity_lr"]);
  cfg.optimization.mlp_cov_lr = static_cast<float>(fs["Optimization.mlp_cov_lr"]);
  cfg.optimization.mlp_color_lr = static_cast<float>(fs["Optimization.mlp_color_lr"]);
  cfg.optimization.lambda_dssim = static_cast<float>(fs["Optimization.lambda_dssim"]);
  cfg.optimization.lambda_depth = static_cast<float>(fs["Optimization.lambda_depth"]);
  cfg.optimization.lambda_isotropic = static_cast<float>(fs["Optimization.lambda_isotropic"]);
  cfg.optimization.max_num_iterations = static_cast<int>(fs["Optimization.max_num_iterations"]);
  cfg.optimization.new_keyframe_times_of_use =
      static_cast<int>(fs["Optimization.new_keyframe_times_of_use"]);
  cfg.optimization.stable_num_iter_existence =
      static_cast<int>(fs["Optimization.stable_num_iter_existence"]);
  cfg.optimization.smooth_l1 = static_cast<int>(fs["Optimization.smooth_l1"]) != 0;
  cfg.optimization.spatial_lr_scale = static_cast<float>(fs["Optimization.spatial_lr_scale"]);

  // Camera section
  cfg.camera.z_near = static_cast<float>(fs["Camera.z_near"]);
  cfg.camera.z_far = static_cast<float>(fs["Camera.z_far"]);

  // Keyframe section
  cfg.keyframe.large_rotation_threshold =
      static_cast<float>(fs["Keyframe.large_rotation_threshold"]);
  cfg.keyframe.large_translation_threshold =
      static_cast<float>(fs["Keyframe.large_translation_threshold"]);
  cfg.keyframe.min_num_initial_map_kfs =
      static_cast<int>(fs["Keyframe.min_num_initial_map_kfs"]);
  cfg.keyframe.gaus_pyramid_sub_levels =
      static_cast<int>(fs["Keyframe.gaus_pyramid_sub_levels"]);

  // Parse gaus_pyramid_factors if present
  cv::FileNode factors_node = fs["Keyframe.gaus_pyramid_factors"];
  if (!factors_node.empty()) {
    cfg.keyframe.gaus_pyramid_factors.clear();
    for (const auto& v : factors_node) {
      cfg.keyframe.gaus_pyramid_factors.push_back(static_cast<float>(v));
    }
  }

  // Tracking section
  {
    cv::FileNode node = fs["Tracking"];
    if (!node.empty()) {
      cfg.tracking.max_orb_features = static_cast<int>(node["max_orb_features"]);
      cfg.tracking.scale_factor = static_cast<float>(node["scale_factor"]);
      cfg.tracking.n_levels = static_cast<int>(node["n_levels"]);
      cfg.tracking.fast_threshold = static_cast<int>(node["fast_threshold"]);
      cfg.tracking.match_ratio_threshold = static_cast<float>(node["match_ratio_threshold"]);
      cfg.tracking.min_pnp_inliers = static_cast<int>(node["min_pnp_inliers"]);
      cfg.tracking.pnp_ransac_reproj_threshold =
          static_cast<float>(node["pnp_ransac_reproj_threshold"]);
      cfg.tracking.use_depth_for_pnp = static_cast<int>(node["use_depth_for_pnp"]) != 0;
      cfg.tracking.min_depth = static_cast<float>(node["min_depth"]);
      cfg.tracking.max_depth = static_cast<float>(node["max_depth"]);
    }
  }

  // LoopClosing section
  {
    cv::FileNode node = fs["LoopClosing"];
    if (!node.empty()) {
      cfg.loop_closing.vocab_size = static_cast<int>(node["vocab_size"]);
      cfg.loop_closing.vocab_tree_levels = static_cast<int>(node["vocab_tree_levels"]);
      cfg.loop_closing.min_consecutive_loop = static_cast<int>(node["min_consecutive_loop"]);
      cfg.loop_closing.loop_score_threshold = static_cast<float>(node["loop_score_threshold"]);
      cfg.loop_closing.min_3d_matches = static_cast<int>(node["min_3d_matches"]);
      cfg.loop_closing.max_loop_distance = static_cast<float>(node["max_loop_distance"]);
      cfg.loop_closing.use_icp_refinement = static_cast<int>(node["use_icp_refinement"]) != 0;
      cfg.loop_closing.max_optimization_iterations =
          static_cast<int>(node["max_optimization_iterations"]);
      cfg.loop_closing.optimization_convergence =
          static_cast<float>(node["optimization_convergence"]);
    }
  }

  // StereoSGBM section
  {
    cv::FileNode node = fs["StereoSGBM"];
    if (!node.empty()) {
      cfg.stereo_sgbm.min_disparity = static_cast<int>(node["min_disparity"]);
      cfg.stereo_sgbm.num_disparities = static_cast<int>(node["num_disparities"]);
      cfg.stereo_sgbm.block_size = static_cast<int>(node["block_size"]);
      cfg.stereo_sgbm.p1 = static_cast<int>(node["p1"]);
      cfg.stereo_sgbm.p2 = static_cast<int>(node["p2"]);
      cfg.stereo_sgbm.disp12_max_diff = static_cast<int>(node["disp12_max_diff"]);
      cfg.stereo_sgbm.pre_filter_cap = static_cast<int>(node["pre_filter_cap"]);
      cfg.stereo_sgbm.uniqueness_ratio = static_cast<int>(node["uniqueness_ratio"]);
      cfg.stereo_sgbm.speckle_window_size = static_cast<int>(node["speckle_window_size"]);
      cfg.stereo_sgbm.speckle_range = static_cast<int>(node["speckle_range"]);
      cfg.stereo_sgbm.use_wls_filter = static_cast<int>(node["use_wls_filter"]) != 0;
    }
  }

  // Viewer section
  {
    cv::FileNode node = fs["Viewer"];
    if (!node.empty()) {
      cfg.viewer.window_width = static_cast<int>(node["window_width"]);
      cfg.viewer.window_height = static_cast<int>(node["window_height"]);
      cfg.viewer.window_title = static_cast<std::string>(node["window_title"]);
      cfg.viewer.fullscreen = static_cast<int>(node["fullscreen"]) != 0;
      cfg.viewer.vsync = static_cast<int>(node["vsync"]) != 0;
      cfg.viewer.point_size = static_cast<float>(node["point_size"]);
      cfg.viewer.frustum_scale = static_cast<float>(node["frustum_scale"]);
      cfg.viewer.max_display_gaussians = static_cast<int>(node["max_display_gaussians"]);
      cfg.viewer.show_cameras = static_cast<int>(node["show_cameras"]) != 0;
      cfg.viewer.show_points = static_cast<int>(node["show_points"]) != 0;
      cfg.viewer.show_grid = static_cast<int>(node["show_grid"]) != 0;
      cfg.viewer.orbit_distance = static_cast<float>(node["orbit_distance"]);
      cfg.viewer.orbit_azimuth = static_cast<float>(node["orbit_azimuth"]);
      cfg.viewer.orbit_elevation = static_cast<float>(node["orbit_elevation"]);
      cfg.viewer.orbit_sensitivity = static_cast<float>(node["orbit_sensitivity"]);
      cfg.viewer.zoom_sensitivity = static_cast<float>(node["zoom_sensitivity"]);
      cfg.viewer.bg_color_r = static_cast<float>(node["bg_color_r"]);
      cfg.viewer.bg_color_g = static_cast<float>(node["bg_color_g"]);
      cfg.viewer.bg_color_b = static_cast<float>(node["bg_color_b"]);
      cfg.viewer.camera_color_r = static_cast<float>(node["camera_color_r"]);
      cfg.viewer.camera_color_g = static_cast<float>(node["camera_color_g"]);
      cfg.viewer.camera_color_b = static_cast<float>(node["camera_color_b"]);
      cfg.viewer.trajectory_color_r = static_cast<float>(node["trajectory_color_r"]);
      cfg.viewer.trajectory_color_g = static_cast<float>(node["trajectory_color_g"]);
      cfg.viewer.trajectory_color_b = static_cast<float>(node["trajectory_color_b"]);
    }
  }

  cfg.enable_viewer = static_cast<int>(fs["enable_viewer"]) != 0;

  // Mapper section (DiskChunGS-aligned)
  {
    cv::FileNode node = fs["Mapper"];
    if (!node.empty()) {
      cfg.mapper.min_depth = static_cast<float>(node["min_depth"]);
      cfg.mapper.max_depth = static_cast<float>(node["max_depth"]);
      cfg.mapper.min_num_initial_map_kfs = static_cast<int>(node["min_num_initial_map_kfs"]);
      cfg.mapper.new_keyframe_times_of_use =
          static_cast<int>(node["new_keyframe_times_of_use"]);
      cfg.mapper.local_BA_increased_times_of_use =
          static_cast<int>(node["local_BA_increased_times_of_use"]);
      cfg.mapper.loop_closure_increased_times_of_use =
          static_cast<int>(node["loop_closure_increased_times_of_use"]);
      cfg.mapper.stable_num_iter_existence =
          static_cast<int>(node["stable_num_iter_existence"]);
      cfg.mapper.loop_closure_optimization_iterations =
          static_cast<int>(node["loop_closure_optimization_iterations"]);
      cfg.mapper.loop_closure_memory_multiplier =
          static_cast<float>(node["loop_closure_memory_multiplier"]);
      cfg.mapper.auto_distribute_learning_rates =
          static_cast<int>(node["auto_distribute_learning_rates"]) != 0;
    }
  }

  // GausPyramid section
  {
    cv::FileNode node = fs["GausPyramid"];
    if (!node.empty()) {
      cfg.gaus_pyramid.num_sub_levels = static_cast<int>(node["num_sub_levels"]);
      cv::FileNode factors_node = node["factors"];
      if (!factors_node.empty()) {
        cfg.gaus_pyramid.factors.clear();
        for (const auto& v : factors_node) {
          cfg.gaus_pyramid.factors.push_back(static_cast<float>(v));
        }
      }
      cv::FileNode times_node = node["times_of_use"];
      if (!times_node.empty()) {
        cfg.gaus_pyramid.times_of_use.clear();
        for (const auto& v : times_node) {
          cfg.gaus_pyramid.times_of_use.push_back(static_cast<int>(v));
        }
      }
    }
  }

  // Pipeline section
  {
    cv::FileNode node = fs["Pipeline"];
    if (!node.empty()) {
      cfg.pipeline.convert_SHs = static_cast<int>(node["convert_SHs"]) != 0;
      cfg.pipeline.compute_cov3D = static_cast<int>(node["compute_cov3D"]) != 0;
      cfg.pipeline.use_pose_optimization =
          static_cast<int>(node["use_pose_optimization"]) != 0;
      cfg.pipeline.use_exposure_optimization =
          static_cast<int>(node["use_exposure_optimization"]) != 0;
    }
  }

  // Device
  cfg.data_device = static_cast<std::string>(fs["Device.data_device"]);
  cfg.white_background = static_cast<int>(fs["Device.white_background"]) != 0;

  fs.release();
  return cfg;
}

}  // namespace scaffold_chungs
