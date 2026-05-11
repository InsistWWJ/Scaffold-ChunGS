/**
 * Scaffold-ChunGS: SLAM Mapper Pipeline (DiskChunGS-aligned).
 *
 * Architecture:
 *   SLAM System (ORB-SLAM3 or builtin) → MappingOperations → GaussianMapper
 *
 * Two-phase design:
 *   Phase 1 — Initial mapping: accumulate N keyframes, batch-seed Gaussians
 *   Phase 2 — Incremental mapping: consume MappingOperations, train iteratively
 *
 * Coordinates:
 *   1. Tracking (pose estimation from SLAM system)
 *   2. Keyframe selection (motion-based + covisibility)
 *   3. Depth estimation (mono/stereo/guided MVS)
 *   4. Gaussian seeding (new anchors from depth point cloud)
 *   5. Training (anchor MLP optimization via rendering loss)
 *   6. Loop closure (pose graph optimization + Gaussian transform)
 *   7. Chunk memory management (LRU eviction)
 */

#pragma once

#include <torch/torch.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <atomic>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "depth_estimator.h"
#include "gaussian_keyframe.h"
#include "gaussian_model.h"
#include "gaussian_scene.h"
#include "gaussian_viewer.h"
#include "keyframe_selection.h"
#include "slam_system.h"

namespace scaffold_chungs {

// =============================================================================
// Mapper State
// =============================================================================

enum class MapperState {
  kIdle,
  kInitializing,     // Building initial map from first N keyframes
  kTracking,         // Normal SLAM operation
  kLoopClosing,      // Processing a loop closure
  kShuttingDown,
};

// =============================================================================
// Sensor type (matches ORB-SLAM3)
// =============================================================================

enum class SensorType { MONOCULAR = 0, STEREO = 1, RGBD = 2 };

// =============================================================================
// SLAM Mapper (DiskChunGS-aligned)
// =============================================================================

class ScaffoldMapper {
 public:
  // ---- Constructors (matching DiskChunGS) ----

  /** Full SLAM mode: takes a SLAM system for tracking + loop closing. */
  ScaffoldMapper(std::shared_ptr<SLAMSystemInterface> slam_system,
                 const ScaffoldChunGSConfig& cfg,
                 std::shared_ptr<DepthEstimator> depth_estimator = nullptr);

  /** External pose mode: poses provided via processFrame(). */
  ScaffoldMapper(const ScaffoldChunGSConfig& cfg,
                 std::shared_ptr<DepthEstimator> depth_estimator = nullptr);

  ~ScaffoldMapper();

  // ---- Lifecycle ----

  /** Start the mapper thread (two-phase: initial → incremental). */
  void start();

  /** Signal stop and join thread. */
  void stop();

  /** Process a new frame through the SLAM system and enqueue for mapping.
   *  In external mode, the pose is used directly. */
  void processFrame(const cv::Mat& image,
                    const cv::Mat& depth = cv::Mat(),
                    const Eigen::Matrix4f& external_pose = Eigen::Matrix4f::Identity(),
                    double timestamp = -1.0);

  // ---- Core Training ----

  /** Single training iteration (keyframe selection → cull → render → loss → step). */
  void trainForOneIteration();

  /** Run the two-phase mapping loop (blocking; called from mapper thread). */
  void run();

  // ---- Accessors ----

  void setPauseMapping(bool pause) { pause_mapping_.store(pause); }
  bool isMappingPaused() const { return pause_mapping_.load(); }

  MapperState state() const {
    std::lock_guard<std::mutex> lock(mutex_state_);
    return state_;
  }

  std::shared_ptr<GaussianModel> model() const { return model_; }
  std::shared_ptr<GaussianScene> scene() const { return scene_; }
  size_t totalKeyframes() const;

  int iteration() const { return current_iteration_; }
  bool isStopped() const { return stopped_.load(); }
  void signalStop(bool going_to_stop = true) { stopped_.store(going_to_stop); }

  GaussianViewer* viewer() const { return viewer_.get(); }
  SLAMSystemInterface* slamSystem() const { return slam_system_.get(); }
  bool isExternalMode() const { return external_mode_; }

 private:
  // ---- Two-Phase Conditions (matching DiskChunGS) ----

  bool hasMetInitialMappingConditions();
  bool hasMetIncrementalMappingConditions();

  // ---- Mapping Operation Dispatch ----

  void combineMappingOperations();
  void processLocalMappingBABatch(std::vector<MappingOperation>& operations);
  void processLoopClosureBA(MappingOperation& opr);
  void processScaleRefinement(MappingOperation& opr);

  // ---- Keyframe Handling ----

  /** Handle a new keyframe tuple from the SLAM system. */
  void handleNewKeyframeFromSLAM(const MappingOperation::KFTuple& kf_tuple,
                                 bool skip_sampling = false);

  /** Handle a new keyframe (external mode). */
  void handleNewKeyframeExternal(const cv::Mat& image,
                                  const Eigen::Matrix4f& pose,
                                  const cv::Mat& depth = cv::Mat());

  // ---- Gaussian Operations ----

  void seedGaussians(std::shared_ptr<GaussianKeyframe> kf,
                     const DepthResult& depth);
  void sampleGaussians(std::shared_ptr<GaussianKeyframe> kf);
  void optimizeLocalWindow(int64_t center_kf_id);

  // ---- Viewer ----

  void submitToViewer(float last_loss = 0.0f);

  // ---- Keyframe Decision ----

  bool isKeyframe(const Eigen::Matrix4f& current_pose,
                  const Eigen::Matrix4f& last_kf_pose) const;

  // ===========================================================================
  // State
  // ===========================================================================

  MapperState state_ = MapperState::kIdle;
  mutable std::mutex mutex_state_;

  ScaffoldChunGSConfig cfg_;
  torch::DeviceType device_ = torch::kCPU;
  bool external_mode_ = false;

  // Core components
  std::shared_ptr<SLAMSystemInterface> slam_system_;
  std::shared_ptr<GaussianModel> model_;
  std::shared_ptr<GaussianScene> scene_;
  std::shared_ptr<DepthEstimator> depth_estimator_;
  std::shared_ptr<KeyframeSelection> kf_selector_;
  std::unique_ptr<GaussianViewer> viewer_;

  // Intrinsics
  float fx_ = 500.0f, fy_ = 500.0f;
  float cx_ = 320.0f, cy_ = 240.0f;
  int img_width_ = 640, img_height_ = 480;

  // Training
  int current_iteration_ = 0;
  torch::Tensor background_;
  bool initial_mapped_ = false;

  // Optimization parameters
  OptimizationConfig opt_params_;

  // ---- Threading ----

  std::unique_ptr<std::thread> mapper_thread_;
  std::mutex mutex_frame_;
  std::deque<std::tuple<cv::Mat, cv::Mat, Eigen::Matrix4f, double>> frame_queue_;
  std::atomic<bool> running_{false};
  std::atomic<bool> pause_mapping_{false};
  std::atomic<bool> stopped_{false};

  // ---- Loop Closure Control ----

  std::atomic<bool> pause_frame_ingestion_{false};
  bool loop_closure_iteration_ = false;
  int loop_closure_optimization_iterations_ = 1000;
  float loop_closure_memory_multiplier_ = 8.0f;

  // ---- Tracking State ----

  Eigen::Matrix4f last_keyframe_pose_ = Eigen::Matrix4f::Identity();
  int64_t next_keyframe_id_ = 0;

  // ---- Local Optimization ----

  static constexpr int kLocalWindowSize = 8;

  // ---- Initial Map ----

  int initial_map_kfs_needed_ = 10;
  std::vector<std::shared_ptr<GaussianKeyframe>> initial_map_kfs_;

  // ---- Frame Counter ----

  int frame_counter_ = 0;

  // ---- Mutexes ----

  mutable std::mutex mutex_render_;
  mutable std::mutex mutex_settings_;
};

}  // namespace scaffold_chungs
