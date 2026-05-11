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

StereoDepthEstimator::StereoDepthEstimator(float baseline, float fx,
                                           const StereoSGBMConfig& sgbm_cfg)
    : baseline_(baseline), fx_(fx), sgbm_cfg_(sgbm_cfg) {
  source_type_ = DepthSource::kStereoDepth;

  // Create SGBM matcher with config
  sgbm_ = cv::StereoSGBM::create(
      sgbm_cfg_.min_disparity,
      sgbm_cfg_.num_disparities,
      sgbm_cfg_.block_size,
      sgbm_cfg_.p1,
      sgbm_cfg_.p2,
      sgbm_cfg_.disp12_max_diff,
      sgbm_cfg_.pre_filter_cap,
      sgbm_cfg_.uniqueness_ratio,
      sgbm_cfg_.speckle_window_size,
      sgbm_cfg_.speckle_range);

  if (sgbm_) {
    initialized_ = true;
  }
}

DepthResult StereoDepthEstimator::estimateDepthStereo(const cv::Mat& left,
                                                       const cv::Mat& right) {
  DepthResult result;

  if (left.empty() || right.empty() || !initialized_) {
    result.valid = false;
    return result;
  }

  // Ensure grayscale
  cv::Mat left_gray, right_gray;
  if (left.channels() == 3) {
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
  } else {
    left_gray = left;
  }
  if (right.channels() == 3) {
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);
  } else {
    right_gray = right;
  }

  // Compute disparity via SGBM
  cv::Mat disparity_s16;
  sgbm_->compute(left_gray, right_gray, disparity_s16);

  // Convert to float disparity (divided by 16 for SGBM fixed-point)
  cv::Mat disparity_f32;
  disparity_s16.convertTo(disparity_f32, CV_32FC1, 1.0 / 16.0);

  // Compute depth: Z = baseline * fx / disparity
  cv::Mat depth =
      (baseline_ * fx_) / (disparity_f32 + 1e-6f);  // small epsilon to avoid /0

  // Filter invalid depths
  cv::Mat valid_mask = (disparity_f32 > 0.5f) & (depth > 0.01f) & (depth < 200.0f);

  // Confidence from disparity gradient (high gradient = low confidence near edges)
  cv::Mat disp_grad_x, disp_grad_y;
  cv::Sobel(disparity_f32, disp_grad_x, CV_32FC1, 1, 0, 3);
  cv::Sobel(disparity_f32, disp_grad_y, CV_32FC1, 0, 1, 3);
  cv::Mat grad_mag = cv::abs(disp_grad_x) + cv::abs(disp_grad_y);
  // Confidence = exp(-grad_mag / threshold), high where disparity is smooth
  cv::Mat confidence;
  cv::exp(-grad_mag / 3.0f, confidence);
  confidence.setTo(0.0f, ~valid_mask);

  // Apply speckle filter on depth (median filter to remove isolated outliers)
  cv::medianBlur(depth, depth, 5);
  depth.setTo(0.0f, ~valid_mask);
  depth.setTo(200.0f, depth > 200.0f);

  // Convert to torch tensors
  auto opt = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor depth_tensor = torch::from_blob(
      depth.data, {left.rows, left.cols}, opt).clone().unsqueeze(0);  // [1, H, W]
  torch::Tensor conf_tensor = torch::from_blob(
      confidence.data, {left.rows, left.cols}, opt).clone().unsqueeze(0);

  result.depth_map = depth_tensor;
  result.confidence = conf_tensor;
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
      // Default: 0.12m baseline, 500px focal length, standard SGBM params
      StereoSGBMConfig sgbm_cfg;
      return std::make_shared<StereoDepthEstimator>(0.12f, 500.0f, sgbm_cfg);
    }
    default:
      return nullptr;
  }
}

}  // namespace scaffold_chungs
