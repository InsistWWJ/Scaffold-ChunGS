/**
 * Scaffold-ChunGS: Depth estimation module interface.
 *
 * Supports three depth sources:
 *   1. Monocular depth (DepthAnything via TensorRT)
 *   2. Stereo depth (ORB-SLAM3 stereo matching)
 *   3. Guided MVS (multi-view stereo with optical flow)
 *
 * The depth module produces per-keyframe depth maps used for:
 *   - Initial Gaussian seeding
 *   - Training supervision (depth loss)
 */

#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <optional>
#include <string>

namespace scaffold_chungs {

// =============================================================================
// Depth Source Type
// =============================================================================

enum class DepthSource {
  kNone = 0,
  kMonoDepth,     // Single-image depth estimation
  kStereoDepth,   // Stereo pair disparity -> depth
  kGuidedMVS,     // Multi-view stereo guided by optical flow
};

// =============================================================================
// Depth Estimation Result
// =============================================================================

struct DepthResult {
  torch::Tensor depth_map;    // [1, H, W] float32, metric depth in meters
  torch::Tensor confidence;   // [1, H, W] float32, [0, 1] per-pixel confidence
  bool valid = false;
};

// =============================================================================
// Depth Estimator Base
// =============================================================================

class DepthEstimator {
 public:
  virtual ~DepthEstimator() = default;

  /** Estimate depth for a single image (monocular). */
  virtual DepthResult estimateDepth(const cv::Mat& image) = 0;

  /** Estimate depth from a stereo pair. */
  virtual DepthResult estimateDepthStereo(const cv::Mat& left,
                                          const cv::Mat& right) {
    (void)left; (void)right;
    return DepthResult{};
  }

  /** Multi-view depth refinement with neighboring keyframes. */
  virtual DepthResult refineDepthMVS(
      const cv::Mat& image,
      const cv::Mat& initial_depth,
      const std::vector<cv::Mat>& neighbor_images,
      const std::vector<Eigen::Matrix4f>& neighbor_poses) {
    (void)image; (void)initial_depth;
    (void)neighbor_images; (void)neighbor_poses;
    return DepthResult{};
  }

  DepthSource sourceType() const { return source_type_; }
  bool isInitialized() const { return initialized_; }

 protected:
  DepthSource source_type_ = DepthSource::kNone;
  bool initialized_ = false;
};

// =============================================================================
// DepthAnything TensorRT Estimator (monocular)
// =============================================================================

class MonoDepthEstimator : public DepthEstimator {
 public:
  explicit MonoDepthEstimator(const std::string& engine_path = "");

  bool initialize(const std::string& engine_path);
  DepthResult estimateDepth(const cv::Mat& image) override;

 private:
  std::string engine_path_;
  // Placeholder for TensorRT engine handle
};

// =============================================================================
// Stereo Depth Estimator (ORB-SLAM3 compatible)
// =============================================================================

class StereoDepthEstimator : public DepthEstimator {
 public:
  StereoDepthEstimator(float baseline, float fx);

  DepthResult estimateDepth(const cv::Mat& image) override {
    (void)image;
    return DepthResult{};  // Stereo requires a pair
  }
  DepthResult estimateDepthStereo(const cv::Mat& left,
                                  const cv::Mat& right) override;

 private:
  float baseline_;
  float fx_;
};

// =============================================================================
// Depth Module Factory
// =============================================================================

std::shared_ptr<DepthEstimator> createDepthEstimator(DepthSource source,
    const std::string& model_path = "");

}  // namespace scaffold_chungs
