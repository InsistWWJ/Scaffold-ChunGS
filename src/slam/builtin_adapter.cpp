/**
 * Scaffold-ChunGS: Builtin SLAM Adapter implementation.
 *
 * Wraps TrackingModule + LoopClosingModule behind the SLAMSystemInterface,
 * producing MappingOperations in the same format as ORB-SLAM3's Atlas.
 */

#include "scaffold_chunks/slam_system.h"
#include "scaffold_chunks/tracking.h"
#include "scaffold_chunks/loop_closing.h"
#include "scaffold_chunks/config.h"

#include <iostream>

namespace scaffold_chungs {

// =============================================================================
// Construction / Destruction
// =============================================================================

BuiltinSLAMAdapter::BuiltinSLAMAdapter(SensorType sensor,
                                       float fx, float fy, float cx, float cy,
                                       int width, int height)
    : sensor_(sensor) {
  // Use default configs; can be overridden externally
  TrackingConfig track_cfg;
  track_cfg.use_depth_for_pnp = (sensor == RGBD);

  LoopClosingConfig loop_cfg;

  tracking_ = std::make_unique<TrackingModule>(track_cfg, fx, fy, cx, cy,
                                                width, height);
  loop_closing_ = std::make_unique<LoopClosingModule>(loop_cfg, fx, fy, cx, cy);

  std::cout << "[BuiltinSLAM] Initialized (sensor="
            << (sensor == MONOCULAR ? "mono" : sensor == STEREO ? "stereo" : "rgbd")
            << ")\n";
}

BuiltinSLAMAdapter::~BuiltinSLAMAdapter() {
  shutdown();
}

// =============================================================================
// Tracking Methods
// =============================================================================

Eigen::Matrix4f BuiltinSLAMAdapter::trackMonocular(const cv::Mat& image,
                                                    double /*timestamp*/) {
  TrackingResult result = tracking_->track(image, cv::Mat(),
                                            tracking_->predictMotionModel());
  if (!result.success) return Eigen::Matrix4f::Identity();
  return result.Tcw;
}

Eigen::Matrix4f BuiltinSLAMAdapter::trackStereo(const cv::Mat& left,
                                                 const cv::Mat& right,
                                                 double /*timestamp*/) {
  // Stereo: compute disparity via block matching for depth, then PnP
  // For now, pass right as "depth" — tracking module back-projects matched keypoints
  TrackingResult result = tracking_->track(left, right,  // right used as depth hint
                                            tracking_->predictMotionModel());
  if (!result.success) return Eigen::Matrix4f::Identity();
  return result.Tcw;
}

Eigen::Matrix4f BuiltinSLAMAdapter::trackRGBD(const cv::Mat& image,
                                               const cv::Mat& depth,
                                               double /*timestamp*/) {
  TrackingResult result = tracking_->track(image, depth,
                                            tracking_->predictMotionModel());
  if (!result.success) return Eigen::Matrix4f::Identity();
  return result.Tcw;
}

// =============================================================================
// Keyframe / Operation Interface
// =============================================================================

unsigned long BuiltinSLAMAdapter::numKeyframes() const {
  return num_keyframes_;
}

bool BuiltinSLAMAdapter::isShutDown() const {
  return shut_down_.load();
}

void BuiltinSLAMAdapter::shutdown() {
  shut_down_.store(true);
}

bool BuiltinSLAMAdapter::hasMappingOperation() const {
  std::lock_guard<std::mutex> lock(mutex_queue_);
  return !op_queue_.empty();
}

MappingOperation BuiltinSLAMAdapter::getAndPopMappingOperation() {
  std::lock_guard<std::mutex> lock(mutex_queue_);
  if (op_queue_.empty()) {
    return MappingOperation(MappingOperation::LocalMappingBA);
  }
  MappingOperation opr = std::move(op_queue_.front());
  op_queue_.pop();
  return opr;
}

void BuiltinSLAMAdapter::pushMappingOperation(MappingOperation opr) {
  std::lock_guard<std::mutex> lock(mutex_queue_);
  op_queue_.push(std::move(opr));
}

// =============================================================================
// Loop Detection → Enqueue LoopClosingBA Operation
// =============================================================================

void BuiltinSLAMAdapter::detectAndEnqueueLoop(
    int64_t kf_id, const cv::Mat& image, const Eigen::Matrix4f& pose) {

  // Extract ORB descriptors for loop detection
  cv::Mat gray;
  if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image;
  }

  cv::Ptr<cv::ORB> orb = cv::ORB::create(500);
  std::vector<cv::KeyPoint> kpts;
  cv::Mat desc;
  orb->detectAndCompute(gray, cv::noArray(), kpts, desc);

  if (desc.empty()) return;

  // Register with loop closing database
  loop_closing_->addKeyframe(kf_id, desc, pose);

  // Detect loop (every 10 keyframes)
  if (kf_id % 10 == 0 && kf_id > 50) {
    auto candidate = loop_closing_->detectLoop(kf_id, desc, pose);
    if (candidate) {
      std::cout << "[BuiltinSLAM] Loop detected: " << candidate->query_kf_id
                << " <-> " << candidate->match_kf_id
                << " (score=" << candidate->score << ")\n";

      // Enqueue a LoopClosingBA operation
      MappingOperation opr(MappingOperation::LoopClosingBA);
      opr.transform = candidate->relative_pose;
      opr.addKeyframe(candidate->query_kf_id, 0, pose, image, true);
      opr.addKeyframe(candidate->match_kf_id, 0, pose, image, false);

      pushMappingOperation(std::move(opr));
    }
  }
}

}  // namespace scaffold_chungs
