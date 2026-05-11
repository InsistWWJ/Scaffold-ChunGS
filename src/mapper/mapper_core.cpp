/**
 * Scaffold-ChunGS: SLAM mapper pipeline (DiskChunGS-aligned).
 *
 * Two-phase design:
 *   Phase 1 — Accumulate min_num_initial_map_kfs, batch-seed, one training step
 *   Phase 2 — Poll MappingOperations from SLAM system, dispatch, train iteratively
 */

#include "scaffold_chunks/gaussian_mapper.h"
#include "scaffold_chunks/gaussian_renderer.h"
#include "scaffold_chunks/frustum_culler.h"
#include "scaffold_chunks/training_loss.h"
#include "scaffold_chunks/chunk_types.h"
#include "scaffold_chunks/slam_system.h"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace scaffold_chungs {

// =============================================================================
// Construction
// =============================================================================

ScaffoldMapper::ScaffoldMapper(std::shared_ptr<SLAMSystemInterface> slam_system,
                               const ScaffoldChunGSConfig& cfg,
                               std::shared_ptr<DepthEstimator> depth_estimator)
    : slam_system_(std::move(slam_system)),
      cfg_(cfg),
      depth_estimator_(std::move(depth_estimator)),
      external_mode_(false),
      initial_map_kfs_needed_(cfg.mapper.min_num_initial_map_kfs),
      loop_closure_optimization_iterations_(
          cfg.mapper.loop_closure_optimization_iterations),
      loop_closure_memory_multiplier_(cfg.mapper.loop_closure_memory_multiplier) {

  device_ = (cfg.data_device == "cuda" && torch::cuda::is_available())
                ? torch::kCUDA
                : torch::kCPU;

  model_ = std::make_shared<GaussianModel>(cfg_.anchor, cfg_.chunk, device_);
  scene_ = std::make_shared<GaussianScene>();
  kf_selector_ = std::make_shared<KeyframeSelection>(scene_);

  opt_params_ = cfg_.optimization;

  if (cfg_.anchor.appearance_dim > 0) {
    model_->mlp().setAppearanceEmbedding(256);
  }

  if (cfg_.enable_viewer) {
    viewer_ = std::make_unique<GaussianViewer>(cfg_.viewer);
  }

  // Pre-allocate background tensor
  auto bg_device = device_ == torch::kCUDA ? torch::Device(torch::kCUDA)
                                           : torch::Device(torch::kCPU);
  background_ = torch::zeros({3},
      torch::TensorOptions().dtype(torch::kFloat32).device(bg_device));

  std::cout << "[Mapper] Initialized on "
            << (device_ == torch::kCUDA ? "CUDA" : "CPU")
            << " | " << (external_mode_ ? "External" : "SLAM") << " mode"
            << (viewer_ ? " | Viewer enabled" : "")
            << "\n";
}

ScaffoldMapper::ScaffoldMapper(const ScaffoldChunGSConfig& cfg,
                               std::shared_ptr<DepthEstimator> depth_estimator)
    : cfg_(cfg),
      depth_estimator_(std::move(depth_estimator)),
      external_mode_(true),
      initial_map_kfs_needed_(cfg.mapper.min_num_initial_map_kfs),
      loop_closure_optimization_iterations_(
          cfg.mapper.loop_closure_optimization_iterations),
      loop_closure_memory_multiplier_(cfg.mapper.loop_closure_memory_multiplier) {

  device_ = (cfg.data_device == "cuda" && torch::cuda::is_available())
                ? torch::kCUDA
                : torch::kCPU;

  model_ = std::make_shared<GaussianModel>(cfg_.anchor, cfg_.chunk, device_);
  scene_ = std::make_shared<GaussianScene>();
  kf_selector_ = std::make_shared<KeyframeSelection>(scene_);

  opt_params_ = cfg_.optimization;

  if (cfg_.anchor.appearance_dim > 0) {
    model_->mlp().setAppearanceEmbedding(256);
  }

  if (cfg_.enable_viewer) {
    viewer_ = std::make_unique<GaussianViewer>(cfg_.viewer);
  }

  auto bg_device = device_ == torch::kCUDA ? torch::Device(torch::kCUDA)
                                           : torch::Device(torch::kCPU);
  background_ = torch::zeros({3},
      torch::TensorOptions().dtype(torch::kFloat32).device(bg_device));

  std::cout << "[Mapper] Initialized on "
            << (device_ == torch::kCUDA ? "CUDA" : "CPU")
            << " | External mode"
            << (viewer_ ? " | Viewer enabled" : "")
            << "\n";
}

ScaffoldMapper::~ScaffoldMapper() {
  stop();
}

// =============================================================================
// Lifecycle
// =============================================================================

void ScaffoldMapper::start() {
  if (running_.load()) return;
  running_.store(true);
  {
    std::lock_guard<std::mutex> lock(mutex_state_);
    state_ = MapperState::kInitializing;
  }
  mapper_thread_ = std::make_unique<std::thread>(&ScaffoldMapper::run, this);

  if (viewer_) {
    viewer_->start();
  }

  std::cout << "[Mapper] Thread started\n";
}

void ScaffoldMapper::stop() {
  if (!running_.load()) return;
  running_.store(false);
  signalStop(true);

  // Shut down SLAM system
  if (slam_system_) {
    slam_system_->shutdown();
  }

  if (mapper_thread_ && mapper_thread_->joinable()) {
    mapper_thread_->join();
  }

  if (viewer_) {
    viewer_->stop();
  }

  if (model_ && model_->isInitialized()) {
    model_->saveAllChunks();
  }

  std::cout << "[Mapper] Stopped. Total anchors: "
            << (model_ ? model_->countAllAnchors() : 0)
            << " (" << (model_ ? model_->countAllGaussians() : 0)
            << " gaussians)\n";
}

// =============================================================================
// Frame Processing
// =============================================================================

void ScaffoldMapper::processFrame(const cv::Mat& image,
                                   const cv::Mat& depth,
                                   const Eigen::Matrix4f& external_pose,
                                   double timestamp) {
  if (!running_.load()) return;

  // Check if frame ingestion is paused (during loop closure)
  if (pause_frame_ingestion_.load()) return;

  std::lock_guard<std::mutex> lock(mutex_frame_);
  // Auto-generate timestamp if not provided
  double ts = (timestamp < 0) ? frame_counter_ / 30.0 : timestamp;
  frame_queue_.emplace_back(image.clone(),
                            depth.empty() ? cv::Mat() : depth.clone(),
                            external_pose, ts);
  frame_counter_++;
}

// =============================================================================
// Two-Phase Run (matching DiskChunGS)
// =============================================================================

void ScaffoldMapper::run() {
  std::cout << "[Mapper] Starting two-phase mapping loop\n";

  // =========================================================================
  // Phase 1 — Initial Mapping
  // =========================================================================
  std::cout << "[Mapper] Phase 1: Waiting for initial map conditions (need "
            << initial_map_kfs_needed_ << " KFs)...\n";

  while (!isStopped()) {
    if (hasMetInitialMappingConditions()) {
      std::cout << "[Mapper] Building initial map\n";

      // Consume all pending operations
      if (slam_system_ && slam_system_->hasMappingOperation()) {
        combineMappingOperations();
      }

      // Batch-seed Gaussians from all keyframes
      auto all_kfs = scene_->getAllKeyframes();
      if (!initial_mapped_) {
        for (auto& [fid, kf] : all_kfs) {
          sampleGaussians(kf);
          if (!initial_mapped_) {
            model_->trainingSetup(opt_params_);
            initial_mapped_ = true;
          }
        }
      } else {
        for (auto& [fid, kf] : all_kfs) {
          sampleGaussians(kf);
        }
      }

      if (!initial_mapped_) {
        model_->trainingSetup(opt_params_);
        initial_mapped_ = true;
      }

      {
        std::lock_guard<std::mutex> lock(mutex_state_);
        state_ = MapperState::kTracking;
      }

      trainForOneIteration();
      std::cout << "[Mapper] Initial map complete. Entering Phase 2.\n";
      break;
    }

    // Check for incoming frames while waiting
    std::tuple<cv::Mat, cv::Mat, Eigen::Matrix4f, double> frame_data;
    {
      std::lock_guard<std::mutex> lock(mutex_frame_);
      if (!frame_queue_.empty()) {
        frame_data = std::move(frame_queue_.front());
        frame_queue_.pop_front();
      }
    }

    auto& image = std::get<0>(frame_data);
    if (!image.empty()) {
      auto& depth = std::get<1>(frame_data);
      auto& pose = std::get<2>(frame_data);
      double ts = std::get<3>(frame_data);

      if (external_mode_) {
        // External mode: force keyframe creation
        handleNewKeyframeExternal(image, pose, depth);
      } else if (slam_system_) {
        // Feed to SLAM system
        Eigen::Matrix4f Tcw;
        switch (slam_system_->sensorType()) {
          case SLAMSystemInterface::MONOCULAR:
            Tcw = slam_system_->trackMonocular(image, ts);
            break;
          case SLAMSystemInterface::STEREO:
            Tcw = slam_system_->trackStereo(image, depth, ts);
            break;
          case SLAMSystemInterface::RGBD:
            Tcw = slam_system_->trackRGBD(image, depth, ts);
            break;
        }
        (void)Tcw;  // Pose consumed by SLAM system internally
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (slam_system_ && slam_system_->isShutDown()) break;
  }

  // =========================================================================
  // Phase 2 — Incremental Mapping
  // =========================================================================
  std::cout << "[Mapper] Phase 2: Incremental mapping\n";

  while (!isStopped()) {
    // Process SLAM operations
    if (hasMetIncrementalMappingConditions()) {
      combineMappingOperations();
    }

    // Process any pending frames
    if (!external_mode_) {
      std::tuple<cv::Mat, cv::Mat, Eigen::Matrix4f, double> frame_data;
      {
        std::lock_guard<std::mutex> lock(mutex_frame_);
        if (!frame_queue_.empty()) {
          frame_data = std::move(frame_queue_.front());
          frame_queue_.pop_front();
        }
      }
      auto& image = std::get<0>(frame_data);
      if (!image.empty() && slam_system_) {
        auto& depth = std::get<1>(frame_data);
        double ts = std::get<3>(frame_data);
        switch (slam_system_->sensorType()) {
          case SLAMSystemInterface::MONOCULAR:
            slam_system_->trackMonocular(image, ts);
            break;
          case SLAMSystemInterface::STEREO:
            slam_system_->trackStereo(image, depth, ts);
            break;
          case SLAMSystemInterface::RGBD:
            slam_system_->trackRGBD(image, depth, ts);
            break;
        }
      }
    }

    // One training iteration
    trainForOneIteration();

    // Check SLAM shutdown
    if (slam_system_ && slam_system_->isShutDown()) {
      std::cout << "[Mapper] SLAM system shut down. Finishing...\n";
      break;
    }
  }

  std::cout << "[Mapper] Mapping loop ended at iteration " << current_iteration_
            << "\n";
}

// =============================================================================
// Initial / Incremental Conditions (matching DiskChunGS)
// =============================================================================

bool ScaffoldMapper::hasMetInitialMappingConditions() {
  if (external_mode_) {
    // External mode: met when enough frames queued
    return initial_map_kfs_.size() >= static_cast<size_t>(initial_map_kfs_needed_);
  }
  if (slam_system_ && !slam_system_->isShutDown() &&
      slam_system_->numKeyframes() >= static_cast<unsigned long>(initial_map_kfs_needed_)) {
    return true;
  }
  return false;
}

bool ScaffoldMapper::hasMetIncrementalMappingConditions() {
  if (slam_system_ && !slam_system_->isShutDown() &&
      slam_system_->hasMappingOperation()) {
    return true;
  }
  return false;
}

// =============================================================================
// Mapping Operation Dispatch (matching DiskChunGS mapper_operations.cpp)
// =============================================================================

void ScaffoldMapper::combineMappingOperations() {
  std::vector<MappingOperation> localBAOps;
  std::vector<MappingOperation> loopClosureOps;
  std::vector<MappingOperation> scaleRefinementOps;

  while (slam_system_->hasMappingOperation()) {
    MappingOperation opr = slam_system_->getAndPopMappingOperation();
    switch (opr.operation_type) {
      case MappingOperation::LocalMappingBA:
        localBAOps.push_back(std::move(opr));
        break;
      case MappingOperation::LoopClosingBA:
        loopClosureOps.push_back(std::move(opr));
        break;
      case MappingOperation::ScaleRefinement:
        scaleRefinementOps.push_back(std::move(opr));
        break;
    }
  }

  // Batch process local BA
  if (!localBAOps.empty()) {
    processLocalMappingBABatch(localBAOps);
  }

  // Process each loop closure individually (extended training)
  for (auto& opr : loopClosureOps) {
    pause_frame_ingestion_.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    processLoopClosureBA(opr);
    // Extended optimization after loop closure
    loop_closure_iteration_ = true;
    for (int i = 0; i < loop_closure_optimization_iterations_; i++) {
      if (isStopped()) break;
      trainForOneIteration();
    }
    loop_closure_iteration_ = false;
    pause_frame_ingestion_.store(false, std::memory_order_release);

    std::cout << "[Mapper] Loop closure optimization complete ("
              << loop_closure_optimization_iterations_ << " iters)\n";
  }

  // Scale refinements
  for (auto& opr : scaleRefinementOps) {
    processScaleRefinement(opr);
  }
}

// =============================================================================
// Local BA Processing (matching DiskChunGS)
// =============================================================================

void ScaffoldMapper::processLocalMappingBABatch(
    std::vector<MappingOperation>& operations) {

  for (auto& opr : operations) {
    auto& associated_kfs = opr.associatedKeyFrames();
    for (auto& kf_tuple : associated_kfs) {
      auto kf_id = std::get<0>(kf_tuple);
      auto kf = scene_->getKeyframe(static_cast<int64_t>(kf_id));

      if (kf) {
        // Existing keyframe: update pose
        auto& new_pose = std::get<2>(kf_tuple);
        kf->setPose(new_pose);
        kf->setTimesOfUse(
            kf->remainingTimesOfUse() + opt_params_.new_keyframe_times_of_use);
      } else {
        // New keyframe: create from ORB-SLAM data
        handleNewKeyframeFromSLAM(kf_tuple);
      }
    }
  }
}

// =============================================================================
// Loop Closure Processing (matching DiskChunGS)
// =============================================================================

void ScaffoldMapper::processLoopClosureBA(MappingOperation& opr) {
  std::cout << "[Mapper] Processing loop closure BA...\n";

  {
    std::lock_guard<std::mutex> lock(mutex_state_);
    state_ = MapperState::kLoopClosing;
  }

  auto& associated_kfs = opr.associatedKeyFrames();

  // Process each affected keyframe
  for (auto& kf_tuple : associated_kfs) {
    auto kf_id = std::get<0>(kf_tuple);
    auto& corrected_pose = std::get<2>(kf_tuple);

    auto kf = scene_->getKeyframe(static_cast<int64_t>(kf_id));
    if (kf) {
      Eigen::Matrix4f old_pose = kf->getPose();
      kf->setPose(corrected_pose);

      // Transform Gaussian positions for this KF's anchors
      Eigen::Matrix4f delta = corrected_pose * old_pose.inverse();
      // Full implementation: apply delta to chunk-associated Gaussians
      (void)delta;
    }
  }

  // Submit loop event to viewer
  if (viewer_) {
    ViewerFrameData vfd;
    vfd.valid = true;
    vfd.current_iteration = current_iteration_;
    vfd.recent_loops.push_back({
        static_cast<int64_t>(std::get<0>(associated_kfs.front())),
        static_cast<int64_t>(std::get<0>(associated_kfs.back())),
        opr.scale
    });
    viewer_->submitFrameData(vfd);
  }

  {
    std::lock_guard<std::mutex> lock(mutex_state_);
    state_ = MapperState::kTracking;
  }
}

// =============================================================================
// Scale Refinement (stub — matching DiskChunGS "not implemented")
// =============================================================================

void ScaffoldMapper::processScaleRefinement(MappingOperation& /*opr*/) {
  std::cout << "[Mapper] Scale refinement (not yet implemented)\n";
}

// =============================================================================
// Handle New Keyframe from SLAM System
// =============================================================================

void ScaffoldMapper::handleNewKeyframeFromSLAM(
    const MappingOperation::KFTuple& kf_tuple, bool skip_sampling) {

  auto kf_id = static_cast<int64_t>(std::get<0>(kf_tuple));
  auto& pose = std::get<2>(kf_tuple);
  auto& image = std::get<3>(kf_tuple);
  auto& auxiliary = std::get<5>(kf_tuple);

  if (image.empty()) return;

  auto kf = std::make_shared<GaussianKeyframe>(kf_id, current_iteration_);
  kf->setPose(pose);

  float fx = fx_, fy = fy_, cx = cx_, cy = cy_;
  kf->setIntrinsics(fx, fy, cx, cy, image.cols, image.rows,
                    cfg_.camera.z_near, cfg_.camera.z_far);

  // Transfer to device
  kf->computeTransformTensors(device_);
  kf->setTrainingImage(image);

  // Set depth if available (auxiliary image = depth for RGB-D, right for stereo)
  if (!auxiliary.empty()) {
    kf->setDepthMap(auxiliary);
  }

  kf->setTimesOfUse(opt_params_.new_keyframe_times_of_use);
  kf->transferToGPU(device_);

  scene_->addKeyframe(kf);

  // Update last keyframe pose
  last_keyframe_pose_ = pose;

  // Seed Gaussians if not in skip mode
  if (!skip_sampling && depth_estimator_) {
    std::lock_guard<std::mutex> lock(mutex_render_);
    sampleGaussians(kf);
  }

  // Submit to viewer
  submitToViewer(0.0f);
}

// =============================================================================
// Handle New Keyframe (External Mode)
// =============================================================================

void ScaffoldMapper::handleNewKeyframeExternal(const cv::Mat& image,
                                                const Eigen::Matrix4f& pose,
                                                const cv::Mat& depth) {
  if (image.empty()) return;

  auto kf = std::make_shared<GaussianKeyframe>(next_keyframe_id_++,
                                               current_iteration_);
  kf->setPose(pose);

  float fx = fx_, fy = fy_;
  float cx = image.cols / 2.0f, cy = image.rows / 2.0f;
  kf->setIntrinsics(fx, fy, cx, cy, image.cols, image.rows,
                    cfg_.camera.z_near, cfg_.camera.z_far);

  kf->computeTransformTensors(device_);
  kf->setTrainingImage(image);

  if (!depth.empty()) {
    kf->setDepthMap(depth);
  }

  kf->setTimesOfUse(opt_params_.new_keyframe_times_of_use);
  kf->transferToGPU(device_);

  scene_->addKeyframe(kf);
  last_keyframe_pose_ = pose;

  // Check for initial map accumulation
  MapperState current_state;
  {
    std::lock_guard<std::mutex> lock(mutex_state_);
    current_state = state_;
  }

  if (current_state == MapperState::kInitializing) {
    initial_map_kfs_.push_back(kf);
  }

  if (current_state != MapperState::kInitializing && depth_estimator_) {
    std::lock_guard<std::mutex> lock(mutex_render_);
    sampleGaussians(kf);
  }

  submitToViewer(0.0f);
}

// =============================================================================
// Single Training Iteration (matching DiskChunGS)
// =============================================================================

void ScaffoldMapper::trainForOneIteration() {
  // Select keyframe
  auto kf = kf_selector_->getNextKeyframe();
  if (!kf) return;

  kf->transferToGPU(device_);
  kf->computeTransformTensors(device_);

  auto dev = device_;
  auto camera_center = kf->getCameraCenter(dev);
  auto w2v = kf->worldViewTransform();
  auto proj = kf->projectionMatrix();

  // Frustum cull
  Eigen::Vector3f cc_eigen(camera_center[0].item<float>(),
                           camera_center[1].item<float>(),
                           camera_center[2].item<float>());

  Eigen::Matrix4f wvp_eigen = Eigen::Matrix4f::Identity();
  torch::Tensor full_proj = torch::matmul(proj, w2v);
  if (dev == torch::kCUDA) full_proj = full_proj.cpu();
  auto fp_acc = full_proj.accessor<float, 2>();
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      wvp_eigen(r, c) = fp_acc[r][c];

  torch::Tensor vis_mask = model_->cullVisibleAnchors(cc_eigen, wvp_eigen, true);

  // Render
  auto output = ScaffoldRenderer::render(
      *model_, vis_mask, camera_center, w2v, proj,
      kf->FoVx(), kf->FoVy(),
      kf->imageHeight(), kf->imageWidth(),
      background_, 1.0f);

  // Compute loss
  auto loss = computeLosses(
      output.color, output.depth,
      kf->getGTImage(), kf->getGTDepth(),
      output.radii, kf->getUndistortMask(),
      opt_params_.lambda_dssim,
      opt_params_.lambda_depth,
      opt_params_.lambda_isotropic);

  // Backward
  float loss_val = 0.0f;
  if (loss.total.requires_grad() || loss.total.item<float>() > 0) {
    loss.total.backward();
    loss_val = loss.total.item<float>();
  }

  // Optimizer step
  model_->optimizerStep(vis_mask, vis_mask.sum().item<int>());
  model_->optimizerZeroGrad();

  // Record loss
  kf_selector_->recordLoss(kf->fid(), loss_val);
  kf->decrementTimesOfUse();

  // Periodic densification
  if (current_iteration_ % cfg_.anchor.densify_check_interval == 0 &&
      current_iteration_ > 0) {
    model_->adjustAnchors(
        cfg_.anchor.densify_check_interval,
        cfg_.anchor.prune_success_threshold,
        cfg_.anchor.densify_grad_threshold,
        cfg_.anchor.densify_opacity_threshold);
  }

  current_iteration_++;

  // Logging
  if (current_iteration_ % 50 == 0) {
    std::cout << "[Iter " << current_iteration_ << "] "
              << "L1=" << loss.l1_color
              << " SSIM=" << loss.ssim_value
              << " Total=" << loss_val
              << " Anchors=" << model_->getNumAnchors()
              << " KF=" << kf->fid() << "\n";
  }

  // Viewer snapshot every 10 iterations
  if (current_iteration_ % 10 == 0) {
    submitToViewer(loss_val);
  }
}

// =============================================================================
// Gaussian Seeding via Depth Back-Projection
// =============================================================================

void ScaffoldMapper::seedGaussians(std::shared_ptr<GaussianKeyframe> kf,
                                    const DepthResult& depth) {
  if (!depth.valid || !model_) return;

  int H = kf->imageHeight();
  int W = kf->imageWidth();

  float fx = kf->FoVx() > 0 ?
      0.5f * W / std::tan(kf->FoVx() * 0.5f) : fx_;
  float fy = kf->FoVy() > 0 ?
      0.5f * H / std::tan(kf->FoVy() * 0.5f) : fy_;
  float cx = W / 2.0f;
  float cy = H / 2.0f;

  auto device = model_->deviceType() == torch::kCUDA ?
      torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

  torch::Tensor depth_map = depth.depth_map.to(device);
  if (depth_map.dim() == 3) depth_map = depth_map.squeeze(0);

  auto y_idx = torch::arange(H, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto x_idx = torch::arange(W, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto [grid_y, grid_x] = torch::meshgrid({y_idx, x_idx}, "ij");

  torch::Tensor X = (grid_x - cx) * depth_map / fx;
  torch::Tensor Y = (grid_y - cy) * depth_map / fy;
  torch::Tensor Z = depth_map;

  torch::Tensor valid_depth = (depth_map > 0.01f) & (depth_map < 100.0f);
  torch::Tensor points = torch::stack({
      X.index({valid_depth}),
      Y.index({valid_depth}),
      Z.index({valid_depth})}, 1);

  if (points.size(0) == 0) return;

  // Transform to world
  Eigen::Matrix4f Tcw = kf->getPose();
  Eigen::Matrix3f R = Tcw.block<3, 3>(0, 0);
  Eigen::Vector3f t = Tcw.block<3, 1>(0, 3);

  auto R_tensor = torch::from_blob(R.data(), {3, 3},
      torch::TensorOptions().dtype(torch::kFloat32)).to(device).t();
  auto t_tensor = torch::from_blob(t.data(), {3},
      torch::TensorOptions().dtype(torch::kFloat32)).to(device);

  torch::Tensor world_pts = torch::matmul(points, R_tensor) + t_tensor;

  // Colors from GT image
  torch::Tensor img_tensor = kf->getGTImage().to(device);
  if (img_tensor.dim() == 3 && img_tensor.size(0) == 3) {
    torch::Tensor r = img_tensor[0].index({valid_depth});
    torch::Tensor g = img_tensor[1].index({valid_depth});
    torch::Tensor b = img_tensor[2].index({valid_depth});
    torch::Tensor colors = torch::stack({r, g, b}, 1);

    if (model_->isInitialized()) {
      model_->addAnchors(world_pts, colors);
    } else {
      model_->initializeFromPoints(world_pts, colors);
    }
  }
}

void ScaffoldMapper::sampleGaussians(std::shared_ptr<GaussianKeyframe> kf) {
  if (!depth_estimator_) return;

  DepthResult dr;
  // Use keyframe's depth if available (e.g., RGB-D or stereo), else estimate
  auto gt_depth = kf->getGTDepth();
  if (gt_depth.numel() > 0) {
    dr.depth_map = gt_depth;
    dr.confidence = torch::ones({1, kf->imageHeight(), kf->imageWidth()},
                                 torch::kFloat32);
    dr.valid = true;
  } else {
    // Estimate from the keyframe's training image
    // (would need cv::Mat — use depth_estimator directly)
    dr.valid = false;
  }

  if (dr.valid) {
    seedGaussians(kf, dr);
  }
}

// =============================================================================
// Local Window Optimization (used for loop closure re-optimization)
// =============================================================================

void ScaffoldMapper::optimizeLocalWindow(int64_t center_kf_id) {
  if (!model_ || !model_->isInitialized()) return;

  auto all_kfs = scene_->getAllKeyframes();
  if (all_kfs.empty()) return;

  std::vector<std::shared_ptr<GaussianKeyframe>> window;
  for (auto it = all_kfs.lower_bound(center_kf_id - kLocalWindowSize);
       it != all_kfs.end() && static_cast<int>(window.size()) < kLocalWindowSize * 2;
       ++it) {
    window.push_back(it->second);
  }

  auto dev = device_;
  for (int iter = 0; iter < 5; ++iter) {
    for (auto& kf : window) {
      if (!kf || kf->remainingTimesOfUse() <= 0) continue;

      kf->transferToGPU(dev);
      kf->computeTransformTensors(dev);

      auto cc = kf->getCameraCenter(dev);
      auto w2v = kf->worldViewTransform();
      auto proj = kf->projectionMatrix();

      Eigen::Vector3f cc_eigen(cc[0].item<float>(),
                               cc[1].item<float>(),
                               cc[2].item<float>());

      torch::Tensor full_proj = torch::matmul(proj, w2v);
      if (dev == torch::kCUDA) full_proj = full_proj.cpu();
      Eigen::Matrix4f wvp;
      auto fp_acc = full_proj.accessor<float, 2>();
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          wvp(r, c) = fp_acc[r][c];

      torch::Tensor vis_mask = model_->cullVisibleAnchors(cc_eigen, wvp, true);

      auto output = ScaffoldRenderer::render(
          *model_, vis_mask, cc, w2v, proj,
          kf->FoVx(), kf->FoVy(),
          kf->imageHeight(), kf->imageWidth(),
          background_, 1.0f);

      auto loss = computeLosses(
          output.color, output.depth,
          kf->getGTImage(), kf->getGTDepth(),
          output.radii, kf->getUndistortMask(),
          opt_params_.lambda_dssim,
          opt_params_.lambda_depth,
          opt_params_.lambda_isotropic);

      if (loss.total.requires_grad() || loss.total.item<float>() > 0) {
        loss.total.backward();
      }

      model_->optimizerStep(vis_mask, vis_mask.sum().item<int>());
      model_->optimizerZeroGrad();

      kf_selector_->recordLoss(kf->fid(), loss.total.item<float>());
      kf->decrementTimesOfUse();
    }
  }
}

// =============================================================================
// Viewer Submission
// =============================================================================

void ScaffoldMapper::submitToViewer(float last_loss) {
  if (!viewer_ || !viewer_->isOpen()) return;

  ViewerFrameData data;
  data.total_anchors = model_ ? model_->countAllAnchors() : 0;
  data.total_gaussians = model_ ? model_->countAllGaussians() : 0;
  data.current_iteration = current_iteration_;
  data.last_loss = last_loss;
  data.valid = true;

  auto all_kfs = scene_->getAllKeyframes();
  data.keyframe_poses.reserve(all_kfs.size());
  data.keyframe_ids.reserve(all_kfs.size());
  for (auto& [fid, kf_ptr] : all_kfs) {
    data.keyframe_poses.push_back(kf_ptr->getPose());
    data.keyframe_ids.push_back(fid);
  }

  viewer_->submitFrameData(data);
}

// =============================================================================
// Keyframe Decision
// =============================================================================

bool ScaffoldMapper::isKeyframe(const Eigen::Matrix4f& current_pose,
                                 const Eigen::Matrix4f& last_kf_pose) const {
  if (last_kf_pose.isIdentity()) return true;

  Eigen::Vector3f t_curr = current_pose.block<3, 1>(0, 3);
  Eigen::Vector3f t_last = last_kf_pose.block<3, 1>(0, 3);
  float translation = (t_curr - t_last).norm();

  Eigen::Matrix3f R_curr = current_pose.block<3, 3>(0, 0);
  Eigen::Matrix3f R_last = last_kf_pose.block<3, 3>(0, 0);
  float rotation = Eigen::AngleAxisf(R_curr.transpose() * R_last).angle();

  return (translation > cfg_.keyframe.large_translation_threshold) ||
         (rotation > cfg_.keyframe.large_rotation_threshold);
}

// =============================================================================
// Accessors
// =============================================================================

size_t ScaffoldMapper::totalKeyframes() const {
  return scene_ ? scene_->size() : 0;
}

}  // namespace scaffold_chungs
