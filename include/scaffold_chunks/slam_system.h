/**
 * Scaffold-ChunGS: SLAM System Interface (DiskChunGS-aligned).
 *
 * Abstracts the tracking/loop-closing backend behind a unified interface.
 * Two implementations:
 *   - ORBSLAM3Adapter:  Wraps ORB-SLAM3 (tracking + DBoW2 loop closing)
 *   - BuiltinAdapter:   Self-contained ORB+PnP + k-means BoVW fallback
 *
 * Both produce MappingOperations consumed by ScaffoldMapper via the
 * same dispatch pattern as DiskChunGS.
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/core.hpp>

namespace scaffold_chungs {

// =============================================================================
// MappingOperation — mirrors ORB-SLAM3's Atlas::MappingOperation
// =============================================================================

class MappingOperation {
 public:
  enum OprType { LocalMappingBA = 1, LoopClosingBA = 2, ScaleRefinement = 3 };

  MappingOperation(OprType type,
                   float scale = 1.0f,
                   Eigen::Matrix4f T = Eigen::Matrix4f::Identity(),
                   size_t nKFs = 0)
      : operation_type(type), scale(scale), transform(T) {
    associated_kfs.reserve(nKFs);
  }

  MappingOperation(const MappingOperation&) = default;
  MappingOperation(MappingOperation&&) = default;
  MappingOperation& operator=(const MappingOperation&) = default;
  MappingOperation& operator=(MappingOperation&&) = default;

  /// Keyframe tuple: (id, camera_id, SE3 pose, image, is_loop_kf, auxiliary)
  using KFTuple = std::tuple<unsigned long,      // kf_id
                             unsigned long,      // camera_id
                             Eigen::Matrix4f,    // Tcw pose
                             cv::Mat,            // main image
                             bool,               // is loop closure KF
                             cv::Mat>;           // auxiliary image (depth/right)

  void addKeyframe(unsigned long id, unsigned long cam_id,
                   const Eigen::Matrix4f& pose, const cv::Mat& image,
                   bool is_loop_kf = false, const cv::Mat& auxiliary = cv::Mat()) {
    std::lock_guard<std::mutex> lock(mutex_kfs);
    associated_kfs.emplace_back(id, cam_id, pose, image.clone(),
                                is_loop_kf, auxiliary.empty() ? cv::Mat()
                                                              : auxiliary.clone());
  }

  std::vector<KFTuple>& associatedKeyFrames() { return associated_kfs; }
  const std::vector<KFTuple>& associatedKeyFrames() const { return associated_kfs; }

  // Public fields (matching ORB-SLAM3 access pattern)
  OprType operation_type;
  float scale;
  Eigen::Matrix4f transform;  // T (for loop closure correction)

 private:
  std::vector<KFTuple> associated_kfs;
  mutable std::mutex mutex_kfs;
};

// =============================================================================
// SLAM System Interface
// =============================================================================

class SLAMSystemInterface {
 public:
  virtual ~SLAMSystemInterface() = default;

  /// Track a monocular frame. Returns Tcw or Identity on failure.
  virtual Eigen::Matrix4f trackMonocular(const cv::Mat& image,
                                         double timestamp) = 0;

  /// Track a stereo pair.
  virtual Eigen::Matrix4f trackStereo(const cv::Mat& left,
                                      const cv::Mat& right,
                                      double timestamp) = 0;

  /// Track an RGB-D frame.
  virtual Eigen::Matrix4f trackRGBD(const cv::Mat& image,
                                    const cv::Mat& depth,
                                    double timestamp) = 0;

  /// Number of keyframes in the current map.
  virtual unsigned long numKeyframes() const = 0;

  /// True if the SLAM system has shut down.
  virtual bool isShutDown() const = 0;

  /// Shut down the SLAM system.
  virtual void shutdown() = 0;

  // ---- Mapping operation queue (mirrors ORB-SLAM3 Atlas) ----

  virtual bool hasMappingOperation() const = 0;
  virtual MappingOperation getAndPopMappingOperation() = 0;

  /// Sensor type
  enum SensorType { MONOCULAR = 0, STEREO = 1, RGBD = 2 };
  virtual SensorType sensorType() const = 0;
};

// =============================================================================
// Builtin SLAM Adapter (ORB+PnP tracking + k-means BoVW loop closing)
// =============================================================================

class TrackingModule;
class LoopClosingModule;

class BuiltinSLAMAdapter : public SLAMSystemInterface {
 public:
  BuiltinSLAMAdapter(SensorType sensor, float fx, float fy, float cx, float cy,
                     int width, int height);

  ~BuiltinSLAMAdapter() override;

  Eigen::Matrix4f trackMonocular(const cv::Mat& image, double timestamp) override;
  Eigen::Matrix4f trackStereo(const cv::Mat& left, const cv::Mat& right,
                              double timestamp) override;
  Eigen::Matrix4f trackRGBD(const cv::Mat& image, const cv::Mat& depth,
                            double timestamp) override;

  unsigned long numKeyframes() const override;
  bool isShutDown() const override;
  void shutdown() override;

  bool hasMappingOperation() const override;
  MappingOperation getAndPopMappingOperation() override;
  SensorType sensorType() const override { return sensor_; }

  /// Access the underlying modules for direct use by the mapper.
  TrackingModule* tracking() const { return tracking_.get(); }
  LoopClosingModule* loopClosing() const { return loop_closing_.get(); }

 private:
  void pushMappingOperation(MappingOperation opr);
  void detectAndEnqueueLoop(int64_t kf_id, const cv::Mat& image,
                            const Eigen::Matrix4f& pose);

  SensorType sensor_;
  std::unique_ptr<TrackingModule> tracking_;
  std::unique_ptr<LoopClosingModule> loop_closing_;

  std::queue<MappingOperation> op_queue_;
  mutable std::mutex mutex_queue_;

  std::atomic<bool> shut_down_{false};
  int64_t next_kf_id_ = 0;
  unsigned long num_keyframes_ = 0;
  int frame_count_since_last_kf_ = 0;
};

}  // namespace scaffold_chungs
