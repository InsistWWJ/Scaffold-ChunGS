/**
 * Scaffold-ChunGS: SLAM Mapper Pipeline.
 *
 * Coordinates the full mapping pipeline:
 *   1. Tracking (pose estimation from ORB-SLAM3 or visual odometry)
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

#include <deque>
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
#include "keyframe_selection.h"

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
// Tracking Result
// =============================================================================

struct TrackingResult {
  Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();
  bool success = false;
  float num_inliers = 0;
  float num_matches = 0;
};

// =============================================================================
// SLAM Mapper
// =============================================================================

class ScaffoldMapper {
 public:
  ScaffoldMapper(const ScaffoldChunGSConfig& cfg,
                 std::shared_ptr<DepthEstimator> depth_estimator = nullptr);

  ~ScaffoldMapper();

  // ---- Lifecycle ----

  /** Start the mapper thread. */
  void start();

  /** Signal stop and join thread. */
  void stop();

  /** Process a new RGB-D frame (or RGB + estimated depth). */
  void processFrame(const cv::Mat& image,
                    const cv::Mat& depth = cv::Mat(),
                    const Eigen::Matrix4f& initial_pose = Eigen::Matrix4f::Identity());

  // ---- Accessors ----

  MapperState state() const { return state_; }
  std::shared_ptr<GaussianModel> model() const { return model_; }
  std::shared_ptr<GaussianScene> scene() const { return scene_; }
  size_t totalKeyframes() const;

  // ---- Configuration ----

  void setPauseMapping(bool pause) { pause_mapping_ = pause; }
  bool isMappingPaused() const { return pause_mapping_; }

 private:
  void mappingLoop();
  void handleNewKeyframe(std::shared_ptr<GaussianKeyframe> kf);
  void performLoopClosure(int64_t from_kf, int64_t to_kf,
                          const Eigen::Matrix4f& relative_pose);
  Eigen::Matrix4f trackFrame(const cv::Mat& image,
                             const Eigen::Matrix4f& initial_pose);
  bool isKeyframe(const Eigen::Matrix4f& current_pose,
                  const Eigen::Matrix4f& last_kf_pose) const;
  void seedNewGaussians(std::shared_ptr<GaussianKeyframe> kf,
                        const DepthResult& depth);
  void optimizeLocalWindow(int64_t center_kf_id);

  // ---- State ----

  MapperState state_ = MapperState::kIdle;
  ScaffoldChunGSConfig cfg_;
  torch::DeviceType device_ = torch::kCPU;

  std::shared_ptr<GaussianModel> model_;
  std::shared_ptr<GaussianScene> scene_;
  std::shared_ptr<DepthEstimator> depth_estimator_;
  std::shared_ptr<KeyframeSelection> kf_selector_;

  // ---- Threading ----

  std::unique_ptr<std::thread> mapper_thread_;
  std::mutex mutex_state_;
  std::mutex mutex_frame_;
  std::deque<std::tuple<cv::Mat, cv::Mat, Eigen::Matrix4f>> frame_queue_;
  bool running_ = false;
  bool pause_mapping_ = false;

  // ---- Tracking State ----

  Eigen::Matrix4f last_keyframe_pose_ = Eigen::Matrix4f::Identity();
  int64_t next_keyframe_id_ = 0;
  int64_t current_iteration_ = 0;

  // ---- Local Optimization ----

  static constexpr int kLocalWindowSize = 8;

  // ---- Initial Map ----

  int initial_map_kfs_needed_ = 10;
  std::vector<std::shared_ptr<GaussianKeyframe>> initial_map_kfs_;
};

}  // namespace scaffold_chungs
