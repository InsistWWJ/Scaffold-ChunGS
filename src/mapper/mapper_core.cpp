/**
 * Scaffold-ChunGS: SLAM mapper pipeline implementation.
 */

#include "scaffold_chunks/gaussian_mapper.h"
#include "scaffold_chunks/gaussian_renderer.h"
#include "scaffold_chunks/frustum_culler.h"
#include "scaffold_chunks/training_loss.h"
#include "scaffold_chunks/chunk_types.h"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace scaffold_chungs {

// =============================================================================
// Construction
// =============================================================================

ScaffoldMapper::ScaffoldMapper(const ScaffoldChunGSConfig& cfg,
                               std::shared_ptr<DepthEstimator> depth_estimator)
    : cfg_(cfg),
      depth_estimator_(std::move(depth_estimator)),
      initial_map_kfs_needed_(cfg.keyframe.min_num_initial_map_kfs) {

  device_ = (cfg.data_device == "cuda" && torch::cuda::is_available())
                ? torch::kCUDA
                : torch::kCPU;

  model_ = std::make_shared<GaussianModel>(cfg_.anchor, cfg_.chunk, device_);
  scene_ = std::make_shared<GaussianScene>();
  kf_selector_ = std::make_shared<KeyframeSelection>(scene_);

  if (cfg_.anchor.appearance_dim > 0) {
    model_->mlp().setAppearanceEmbedding(256);  // max expected cameras
  }

  std::cout << "[Mapper] Initialized on "
            << (device_ == torch::kCUDA ? "CUDA" : "CPU") << "\n";
}

ScaffoldMapper::~ScaffoldMapper() {
  stop();
}

// =============================================================================
// Lifecycle
// =============================================================================

void ScaffoldMapper::start() {
  if (running_) return;
  running_ = true;
  state_ = MapperState::kInitializing;
  mapper_thread_ = std::make_unique<std::thread>(&ScaffoldMapper::mappingLoop, this);
  std::cout << "[Mapper] Thread started\n";
}

void ScaffoldMapper::stop() {
  if (!running_) return;
  running_ = false;
  state_ = MapperState::kShuttingDown;

  if (mapper_thread_ && mapper_thread_->joinable()) {
    mapper_thread_->join();
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
                                   const Eigen::Matrix4f& initial_pose) {
  if (!running_) return;
  std::lock_guard<std::mutex> lock(mutex_frame_);
  frame_queue_.emplace_back(image.clone(), depth.empty() ? cv::Mat() : depth.clone(),
                            initial_pose);
}

// =============================================================================
// Mapping Loop
// =============================================================================

void ScaffoldMapper::mappingLoop() {
  while (running_) {
    // Dequeue next frame
    std::tuple<cv::Mat, cv::Mat, Eigen::Matrix4f> frame_data;
    {
      std::lock_guard<std::mutex> lock(mutex_frame_);
      if (frame_queue_.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
      }
      frame_data = std::move(frame_queue_.front());
      frame_queue_.pop_front();
    }

    if (pause_mapping_) continue;

    auto& image = std::get<0>(frame_data);
    auto& depth = std::get<1>(frame_data);
    auto& initial_pose = std::get<2>(frame_data);

    // Step 1: Track current frame
    Eigen::Matrix4f Tcw = trackFrame(image, initial_pose);

    // Step 2: Estimate depth (if not provided)
    DepthResult depth_result;
    if (!depth.empty()) {
      // Use provided depth (e.g., from RGB-D sensor)
      depth_result.depth_map = torch::from_blob(
          const_cast<uchar*>(depth.data),
          {1, depth.rows, depth.cols}, torch::kFloat32).clone();
      depth_result.confidence = torch::ones(
          {1, depth.rows, depth.cols}, torch::kFloat32);
      depth_result.valid = true;
    } else if (depth_estimator_) {
      depth_result = depth_estimator_->estimateDepth(image);
    }

    // Step 3: Check if this should be a keyframe
    bool is_kf = isKeyframe(Tcw, last_keyframe_pose_);
    if (!is_kf) continue;

    // Update last keyframe pose
    last_keyframe_pose_ = Tcw;

    // Step 4: Create keyframe
    auto kf = std::make_shared<GaussianKeyframe>(next_keyframe_id_++,
                                                   current_iteration_);
    kf->setPose(Tcw);

    // Intrinsics (default to reasonable values; override from config)
    float fx = 500.0f, fy = 500.0f;
    float cx = image.cols / 2.0f, cy = image.rows / 2.0f;
    kf->setIntrinsics(fx, fy, cx, cy, image.cols, image.rows,
                      cfg_.camera.z_near, cfg_.camera.z_far);
    kf->computeTransformTensors(device_);

    kf->setTrainingImage(image);
    if (depth_result.valid) {
      cv::Mat depth_mat(depth.rows, depth.cols, CV_32FC1,
                        const_cast<float*>(depth_result.depth_map.data_ptr<float>()));
      kf->setDepthMap(depth_mat);
    }
    kf->setTimesOfUse(cfg_.optimization.new_keyframe_times_of_use);
    kf->transferToGPU(device_);

    // Step 5: Handle keyframe
    handleNewKeyframe(kf);
  }
}

// =============================================================================
// New Keyframe Handling
// =============================================================================

void ScaffoldMapper::handleNewKeyframe(std::shared_ptr<GaussianKeyframe> kf) {
  scene_->addKeyframe(kf);

  // Initial map building
  if (state_ == MapperState::kInitializing) {
    initial_map_kfs_.push_back(kf);
    if (static_cast<int>(initial_map_kfs_.size()) >= initial_map_kfs_needed_) {
      // Initialize model from accumulated keyframe depths
      std::cout << "[Mapper] Building initial map from "
                << initial_map_kfs_.size() << " keyframes\n";

      if (depth_estimator_) {
        for (auto& init_kf : initial_map_kfs_) {
          // Seed Gaussians from keyframe depth
          DepthResult dr;
          dr.valid = true;
          dr.depth_map = init_kf->getGTDepth();
          dr.confidence = torch::ones({1, init_kf->imageHeight(),
                                        init_kf->imageWidth()}, torch::kFloat32);
          seedNewGaussians(init_kf, dr);
        }
      }

      // Set up training
      if (model_->isInitialized()) {
        model_->trainingSetup(cfg_.optimization);
      }
      state_ = MapperState::kTracking;
    }
    return;
  }

  // Normal tracking: seed new Gaussians, then optimize
  if (depth_estimator_) {
    DepthResult dr = depth_estimator_->estimateDepth(cv::Mat());
    if (dr.valid) {
      seedNewGaussians(kf, dr);
    }
  }

  // Local window optimization
  optimizeLocalWindow(kf->fid());
}

// =============================================================================
// Tracking (pose estimation stub — replace with ORB-SLAM3 or visual odometry)
// =============================================================================

Eigen::Matrix4f ScaffoldMapper::trackFrame(const cv::Mat& image,
                                            const Eigen::Matrix4f& initial_pose) {
  // Placeholder: returns identity or initial pose.
  // Replace with actual tracking (ORB-SLAM3, DROID-SLAM, etc.)
  return initial_pose.isIdentity() ? Eigen::Matrix4f::Identity() : initial_pose;
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
// Gaussian Seeding
// =============================================================================

void ScaffoldMapper::seedNewGaussians(std::shared_ptr<GaussianKeyframe> kf,
                                       const DepthResult& depth) {
  if (!depth.valid || !model_) return;

  auto device = model_->deviceType() == torch::kCUDA ?
      torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

  // Back-project depth to 3D points
  float fx = kf->FoVx() > 0 ?
      0.5f * kf->imageWidth() / std::tan(kf->FoVx() * 0.5f) : 500.0f;
  float fy = kf->FoVy() > 0 ?
      0.5f * kf->imageHeight() / std::tan(kf->FoVy() * 0.5f) : 500.0f;
  float cx = kf->imageWidth() / 2.0f;
  float cy = kf->imageHeight() / 2.0f;

  int H = kf->imageHeight();
  int W = kf->imageWidth();

  torch::Tensor depth_map = depth.depth_map.to(device);
  if (depth_map.dim() == 3) depth_map = depth_map.squeeze(0);  // [H, W]

  // Create pixel grid
  auto y_idx = torch::arange(H, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto x_idx = torch::arange(W, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto [grid_y, grid_x] = torch::meshgrid({y_idx, x_idx}, "ij");

  // Back-project
  torch::Tensor X = (grid_x - cx) * depth_map / fx;
  torch::Tensor Y = (grid_y - cy) * depth_map / fy;
  torch::Tensor Z = depth_map;

  torch::Tensor valid_depth = (depth_map > 0.01f) & (depth_map < 100.0f);
  torch::Tensor points = torch::stack({
      X.index({valid_depth}),
      Y.index({valid_depth}),
      Z.index({valid_depth})}, 1);  // [P, 3]

  if (points.size(0) == 0) return;

  // Transform to world coordinates
  Eigen::Matrix4f Tcw = kf->getPose();
  Eigen::Matrix3f R = Tcw.block<3, 3>(0, 0);
  Eigen::Vector3f t = Tcw.block<3, 1>(0, 3);

  auto R_tensor = torch::from_blob(R.data(),
      {3, 3}, torch::TensorOptions().dtype(torch::kFloat32)).to(device).t();
  auto t_tensor = torch::from_blob(t.data(), {3},
      torch::TensorOptions().dtype(torch::kFloat32)).to(device);

  torch::Tensor world_pts = torch::matmul(points, R_tensor) + t_tensor;

  // Colors from image
  torch::Tensor img_tensor = kf->getGTImage().to(device);
  if (img_tensor.dim() == 3 && img_tensor.size(0) == 3) {
    // [3, H, W] -> select by valid_depth mask
    torch::Tensor r = img_tensor[0].index({valid_depth});
    torch::Tensor g = img_tensor[1].index({valid_depth});
    torch::Tensor b = img_tensor[2].index({valid_depth});
    torch::Tensor colors = torch::stack({r, g, b}, 1);  // [P, 3]

    if (model_->isInitialized()) {
      model_->addAnchors(world_pts, colors);
    } else {
      model_->initializeFromPoints(world_pts, colors);
    }
  }
}

// =============================================================================
// Local Window Optimization
// =============================================================================

void ScaffoldMapper::optimizeLocalWindow(int64_t center_kf_id) {
  if (!model_ || !model_->isInitialized()) return;

  auto all_kfs = scene_->getAllKeyframes();
  if (all_kfs.empty()) return;

  // Find kLocalWindowSize nearest keyframes by ID proximity
  std::vector<std::shared_ptr<GaussianKeyframe>> window;
  for (auto it = all_kfs.lower_bound(center_kf_id - kLocalWindowSize);
       it != all_kfs.end() && static_cast<int>(window.size()) < kLocalWindowSize * 2;
       ++it) {
    window.push_back(it->second);
  }

  // Run a few optimization iterations on the local window
  auto device = model_->deviceType();
  torch::Tensor bg = torch::zeros({3},
      torch::TensorOptions().dtype(torch::kFloat32).device(
          device == torch::kCUDA ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU)));

  for (int iter = 0; iter < 5; ++iter) {
    for (auto& kf : window) {
      if (!kf || kf->remainingTimesOfUse() <= 0) continue;

      kf->transferToGPU(device);
      kf->computeTransformTensors(device);

      auto cc = kf->getCameraCenter(device);
      auto w2v = kf->worldViewTransform();
      auto proj = kf->projectionMatrix();

      Eigen::Vector3f cc_eigen(cc[0].item<float>(),
                               cc[1].item<float>(),
                               cc[2].item<float>());

      auto fp = torch::matmul(proj, w2v).cpu();
      Eigen::Matrix4f wvp;
      auto fp_acc = fp.accessor<float, 2>();
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          wvp(r, c) = fp_acc[r][c];

      torch::Tensor vis_mask = model_->cullVisibleAnchors(cc_eigen, wvp, true);

      auto output = ScaffoldRenderer::render(
          *model_, vis_mask, cc, w2v, proj,
          kf->FoVx(), kf->FoVy(),
          kf->imageHeight(), kf->imageWidth(),
          bg, 1.0f);

      auto loss = computeLosses(
          output.color, output.depth,
          kf->getGTImage(), kf->getGTDepth(),
          output.radii, kf->getUndistortMask(),
          cfg_.optimization.lambda_dssim,
          cfg_.optimization.lambda_depth,
          cfg_.optimization.lambda_isotropic);

      if (loss.total.requires_grad() || loss.total.item<float>() > 0) {
        loss.total.backward();
      }

      model_->optimizerStep(vis_mask, vis_mask.sum().item<int>());
      model_->optimizerZeroGrad();

      kf_selector_->recordLoss(kf->fid(), loss.total.item<float>());
      kf->decrementTimesOfUse();
    }

    current_iteration_++;

    // Periodic densification
    if (current_iteration_ % cfg_.anchor.densify_check_interval == 0) {
      model_->adjustAnchors(
          cfg_.anchor.densify_check_interval,
          cfg_.anchor.prune_success_threshold,
          cfg_.anchor.densify_grad_threshold,
          cfg_.anchor.densify_opacity_threshold);
    }
  }
}

// =============================================================================
// Loop Closure
// =============================================================================

void ScaffoldMapper::performLoopClosure(int64_t from_kf, int64_t to_kf,
                                         const Eigen::Matrix4f& relative_pose) {
  std::cout << "[Mapper] Loop closure: " << from_kf << " -> " << to_kf << "\n";
  state_ = MapperState::kLoopClosing;

  // 1. Apply pose correction to keyframes in the loop
  auto kf_from = scene_->getKeyframe(from_kf);
  auto kf_to = scene_->getKeyframe(to_kf);
  if (!kf_from || !kf_to) {
    state_ = MapperState::kTracking;
    return;
  }

  // 2. Transform affected Gaussians
  // 3. Re-optimize affected chunks
  // 4. Handle chunk redistribution

  state_ = MapperState::kTracking;
  std::cout << "[Mapper] Loop closure complete\n";
}

// =============================================================================
// Accessors
// =============================================================================

size_t ScaffoldMapper::totalKeyframes() const {
  return scene_ ? scene_->size() : 0;
}

}  // namespace scaffold_chungs
