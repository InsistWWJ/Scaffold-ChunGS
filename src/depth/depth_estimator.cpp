/**
 * Scaffold-ChunGS: Depth estimation module implementations.
 */

#include "scaffold_chunks/depth_estimator.h"

#include <iostream>

namespace scaffold_chungs {

// =============================================================================
// MonoDepthEstimator
// =============================================================================

MonoDepthEstimator::MonoDepthEstimator(const std::string& engine_path)
    : engine_path_(engine_path) {
  source_type_ = DepthSource::kMonoDepth;
}

bool MonoDepthEstimator::initialize(const std::string& engine_path) {
  if (!engine_path.empty()) engine_path_ = engine_path;

  // Placeholder: load TensorRT engine
  // In production, this loads a DepthAnything TensorRT engine:
  //   engine_ = loadTensorRTEngine(engine_path_);
  //   context_ = engine_->createExecutionContext();

  initialized_ = !engine_path_.empty();
  if (initialized_) {
    std::cout << "[Depth] MonoDepth estimator initialized: " << engine_path_ << "\n";
  }
  return initialized_;
}

DepthResult MonoDepthEstimator::estimateDepth(const cv::Mat& image) {
  DepthResult result;

  if (!initialized_ || image.empty()) {
    result.valid = false;
    return result;
  }

  // Placeholder: run TensorRT inference
  // In production:
  //   1. Preprocess image (resize, normalize)
  //   2. Run TensorRT inference
  //   3. Postprocess output -> metric depth
  //
  // For now, return a dummy depth map matching image size
  auto opt = torch::TensorOptions().dtype(torch::kFloat32);
  result.depth_map = torch::ones({1, image.rows, image.cols}, opt) * 5.0f;
  result.confidence = torch::ones({1, image.rows, image.cols}, opt);
  result.valid = true;

  return result;
}

// =============================================================================
// StereoDepthEstimator
// =============================================================================

StereoDepthEstimator::StereoDepthEstimator(float baseline, float fx)
    : baseline_(baseline), fx_(fx) {
  source_type_ = DepthSource::kStereoDepth;
  initialized_ = true;
}

DepthResult StereoDepthEstimator::estimateDepthStereo(const cv::Mat& left,
                                                       const cv::Mat& right) {
  DepthResult result;

  if (left.empty() || right.empty()) {
    result.valid = false;
    return result;
  }

  // Placeholder: stereo matching -> disparity -> depth
  // In production, this uses ORB-SLAM3's stereo matcher or RAFT-Stereo.
  //
  // depth = baseline * fx / disparity
  //
  auto opt = torch::TensorOptions().dtype(torch::kFloat32);
  result.depth_map = torch::full({1, left.rows, left.cols}, 5.0f, opt);
  result.confidence = torch::ones({1, left.rows, left.cols}, opt);
  result.valid = true;

  return result;
}

// =============================================================================
// Factory
// =============================================================================

std::shared_ptr<DepthEstimator> createDepthEstimator(DepthSource source,
    const std::string& model_path) {
  switch (source) {
    case DepthSource::kMonoDepth: {
      auto est = std::make_shared<MonoDepthEstimator>(model_path);
      est->initialize(model_path);
      return est;
    }
    case DepthSource::kStereoDepth: {
      // Default: 0.12m baseline, 500px focal length
      return std::make_shared<StereoDepthEstimator>(0.12f, 500.0f);
    }
    default:
      return nullptr;
  }
}

}  // namespace scaffold_chungs
