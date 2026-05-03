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

#include <iostream>
#include <chrono>
#include <memory>

namespace scaffold_chungs {

// Forward declaration of loss computation
struct TrainingLoss {
  float l1_color = 0;
  float ssim_value = 0;
  float depth_l1 = 0;
  float isotropic = 0;
  float total = 0;
};

TrainingLoss computeLosses(
    const torch::Tensor& rendered_color,
    const torch::Tensor& rendered_depth,
    const torch::Tensor& gt_image,
    const torch::Tensor& gt_depth,
    const torch::Tensor& child_scaling,
    const torch::Tensor& mask,
    float lambda_dssim,
    float lambda_depth,
    float lambda_isotropic);

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
  }

  /**
   * Run a single training iteration.
   * @return total loss value, or -1 if no keyframe available
   */
  float trainOneIteration();

  /**
   * Run the full training loop until stopped or max iterations reached.
   */
  void run();

  void signalStop() { stopped_ = true; }
  bool stopped() const { return stopped_; }
  int currentIteration() const { return current_iteration_; }

  // Expose model/scene for external use
  std::shared_ptr<GaussianModel> model() const { return model_; }
  std::shared_ptr<GaussianScene> scene() const { return scene_; }

 private:
  std::shared_ptr<GaussianModel> model_;
  std::shared_ptr<GaussianScene> scene_;
  std::shared_ptr<KeyframeSelection> keyframe_selector_;
  OptimizationConfig opt_cfg_;
  KeyframeConfig kf_cfg_;

  int current_iteration_ = 0;
  bool stopped_ = false;
};

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

  // Step 2: Compute transforms
  kf->computeTransformTensors();
  auto camera_center = kf->getCameraCenter().to(model_->deviceType());
  auto w2v = kf->worldViewTransform().to(model_->deviceType());
  auto proj = kf->projectionMatrix().to(model_->deviceType());

  // Step 3: Frustum cull + load visible chunks
  Eigen::Vector3f cc_eigen(camera_center[0].item<float>(),
                           camera_center[1].item<float>(),
                           camera_center[2].item<float>());

  // Build world-view-proj for frustum culling
  Eigen::Matrix4f wvp_eigen = Eigen::Matrix4f::Identity();
  torch::Tensor full_proj = torch::matmul(proj, w2v).cpu();
  auto fp_acc = full_proj.accessor<float, 2>();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      wvp_eigen(r, c) = fp_acc[r][c];

  torch::Tensor visible_mask = model_->cullVisibleAnchors(
      cc_eigen, wvp_eigen, /*manage_memory=*/true);

  // Step 4: Background color
  auto device = model_->deviceType() == torch::kCUDA ?
      torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
  torch::Tensor bg = model_->deviceType() == torch::kCUDA ?
      torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32).device(device))
      : torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32));

  // Step 5: Render
  auto output = ScaffoldRenderer::render(
      *model_, visible_mask, camera_center, w2v, proj,
      kf->FoVx(), kf->FoVy(),
      kf->imageHeight(), kf->imageWidth(),
      bg, 1.0f);

  // Step 6: Compute loss
  torch::Tensor gt_image = kf->getGTImage();
  torch::Tensor gt_depth = kf->getGTDepth();
  torch::Tensor mask = kf->getUndistortMask();

  auto loss = computeLosses(
      output.color, output.depth, gt_image, gt_depth,
      output.radii, mask,  // using radii as a proxy for scaling in isotropic
      opt_cfg_.lambda_dssim, opt_cfg_.lambda_depth,
      opt_cfg_.lambda_isotropic);

  // Step 7: Backward
  if (loss.total > 0) {
    // Create a dummy loss tensor for autograd
    torch::Tensor loss_tensor = torch::tensor({loss.total},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    loss_tensor.set_requires_grad(true);
    // In production: actual loss with backward() through the renderer
    // loss_tensor.backward();
  }

  // Step 8: Optimizer step
  model_->optimizerStep(visible_mask, visible_mask.sum().item<int>());
  model_->optimizerZeroGrad();

  // Step 9: Record loss for keyframe selection
  keyframe_selector_->recordLoss(kf->fid(), loss.total);

  // Step 10: Periodic densification
  if (current_iteration_ % opt_cfg_.densify_check_interval == 0 &&
      current_iteration_ > 0) {
    model_->adjustAnchors(
        opt_cfg_.densify_check_interval,
        opt_cfg_.prune_success_threshold,
        opt_cfg_.densify_grad_threshold,
        opt_cfg_.densify_opacity_threshold);

    // Re-run training setup after densification
    model_->trainingSetup(opt_cfg_);
  }

  current_iteration_++;

  // Log periodically
  if (current_iteration_ % 50 == 0) {
    std::cout << "[Iter " << current_iteration_ << "] "
              << "L1=" << loss.l1_color
              << " SSIM=" << loss.ssim_value
              << " Iso=" << loss.isotropic
              << " Total=" << loss.total
              << " Anchors=" << model_->getNumAnchors()
              << " KF=" << kf->fid() << "\n";
  }

  return loss.total;
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
      // No keyframe available — all used up? refresh
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

  // Save all chunks on completion
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
  cfg.anchor.n_offsets = (int)fs["Anchor.n_offsets"];
  cfg.anchor.feat_dim = (int)fs["Anchor.feat_dim"];
  cfg.anchor.voxel_size = (float)fs["Anchor.voxel_size"];
  cfg.anchor.sh_degree = (int)fs["Anchor.sh_degree"];
  cfg.anchor.use_feat_bank = (int)fs["Anchor.use_feat_bank"] != 0;
  cfg.anchor.add_opacity_dist = (int)fs["Anchor.add_opacity_dist"] != 0;
  cfg.anchor.add_cov_dist = (int)fs["Anchor.add_cov_dist"] != 0;
  cfg.anchor.add_color_dist = (int)fs["Anchor.add_color_dist"] != 0;
  cfg.anchor.appearance_dim = (int)fs["Anchor.appearance_dim"];
  cfg.anchor.update_depth = (int)fs["Anchor.update_depth"];
  cfg.anchor.update_init_factor = (int)fs["Anchor.update_init_factor"];
  cfg.anchor.update_hierarchy_factor = (int)fs["Anchor.update_hierarchy_factor"];
  cfg.anchor.densify_grad_threshold = (float)fs["Anchor.densify_grad_threshold"];
  cfg.anchor.densify_opacity_threshold =
      (float)fs["Anchor.densify_opacity_threshold"];
  cfg.anchor.densify_check_interval = (int)fs["Anchor.densify_check_interval"];

  // Chunk section
  cfg.chunk.chunk_size = (float)fs["Chunk.chunk_size"];
  cfg.chunk.max_anchors_in_memory = (int)fs["Chunk.max_anchors_in_memory"];
  cfg.chunk.new_anchor_chunk_density = (int)fs["Chunk.new_anchor_chunk_density"];
  cfg.chunk.storage_base_path = (std::string)fs["Chunk.storage_base_path"];

  // Optimization section
  cfg.optimization.anchor_position_lr_init =
      (float)fs["Optimization.anchor_position_lr_init"];
  cfg.optimization.anchor_position_lr_decay =
      (float)fs["Optimization.anchor_position_lr_decay"];
  cfg.optimization.anchor_feature_lr =
      (float)fs["Optimization.anchor_feature_lr"];
  cfg.optimization.anchor_opacity_lr =
      (float)fs["Optimization.anchor_opacity_lr"];
  cfg.optimization.anchor_scaling_lr =
      (float)fs["Optimization.anchor_scaling_lr"];
  cfg.optimization.anchor_rotation_lr =
      (float)fs["Optimization.anchor_rotation_lr"];
  cfg.optimization.offset_lr = (float)fs["Optimization.offset_lr"];
  cfg.optimization.mlp_opacity_lr = (float)fs["Optimization.mlp_opacity_lr"];
  cfg.optimization.mlp_cov_lr = (float)fs["Optimization.mlp_cov_lr"];
  cfg.optimization.mlp_color_lr = (float)fs["Optimization.mlp_color_lr"];
  cfg.optimization.lambda_dssim = (float)fs["Optimization.lambda_dssim"];
  cfg.optimization.lambda_depth = (float)fs["Optimization.lambda_depth"];
  cfg.optimization.lambda_isotropic = (float)fs["Optimization.lambda_isotropic"];
  cfg.optimization.max_num_iterations = (int)fs["Optimization.max_num_iterations"];
  cfg.optimization.new_keyframe_times_of_use =
      (int)fs["Optimization.new_keyframe_times_of_use"];
  cfg.optimization.spatial_lr_scale = (float)fs["Optimization.spatial_lr_scale"];

  // Camera section
  cfg.camera.z_near = (float)fs["Camera.z_near"];
  cfg.camera.z_far = (float)fs["Camera.z_far"];

  // Keyframe section
  cfg.keyframe.large_rotation_threshold =
      (float)fs["Keyframe.large_rotation_threshold"];
  cfg.keyframe.large_translation_threshold =
      (float)fs["Keyframe.large_translation_threshold"];
  cfg.keyframe.min_num_initial_map_kfs =
      (int)fs["Keyframe.min_num_initial_map_kfs"];
  cfg.keyframe.gaus_pyramid_sub_levels =
      (int)fs["Keyframe.gaus_pyramid_sub_levels"];

  // Device
  cfg.data_device = (std::string)fs["Device.data_device"];
  cfg.white_background = (int)fs["Device.white_background"] != 0;

  fs.release();
  return cfg;
}

}  // namespace scaffold_chungs
