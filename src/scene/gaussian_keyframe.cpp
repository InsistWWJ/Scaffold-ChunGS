/**
 * Scaffold-ChunGS: Keyframe implementation.
 */

#include "scaffold_chunks/gaussian_keyframe.h"

namespace scaffold_chungs {

void GaussianKeyframe::setPose(const Eigen::Matrix4f& Tcw) {
  Tcw_ = Tcw;
}

void GaussianKeyframe::setPose(const Eigen::Quaternionf& q,
                                const Eigen::Vector3f& t) {
  Tcw_.setIdentity();
  Tcw_.block<3,3>(0,0) = q.toRotationMatrix();
  Tcw_.block<3,1>(0,3) = t;
}

Eigen::Matrix4f GaussianKeyframe::getPose() const {
  return Tcw_;
}

Eigen::Matrix3f GaussianKeyframe::getRotationMatrix() const {
  return Tcw_.block<3,3>(0,0);
}

Eigen::Vector3f GaussianKeyframe::getTranslation() const {
  return Tcw_.block<3,1>(0,3);
}

torch::Tensor GaussianKeyframe::getCameraCenter() const {
  // Camera center in world coordinates = -R^T * t
  Eigen::Matrix3f R = Tcw_.block<3,3>(0,0);
  Eigen::Vector3f t = Tcw_.block<3,1>(0,3);
  Eigen::Vector3f center = -R.transpose() * t;

  return torch::tensor({center.x(), center.y(), center.z()},
      torch::TensorOptions().dtype(torch::kFloat32));
}

// =============================================================================
// Camera Intrinsics
// =============================================================================

void GaussianKeyframe::setIntrinsics(float fx, float fy, float cx, float cy,
                                     int width, int height,
                                     float znear, float zfar) {
  fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;
  image_width_ = width; image_height_ = height;
  znear_ = znear; zfar_ = zfar;

  FoVx_ = 2.0f * std::atan(static_cast<float>(width) / (2.0f * fx));
  FoVy_ = 2.0f * std::atan(static_cast<float>(height) / (2.0f * fy));
}

// =============================================================================
// Transform Tensors
// =============================================================================

void GaussianKeyframe::computeTransformTensors() {
  auto opt = torch::TensorOptions().dtype(torch::kFloat32);

  // World-to-view: inverse of Tcw
  Eigen::Matrix4f w2v = Tcw_.inverse();

  std::vector<float> w2v_flat;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      w2v_flat.push_back(w2v(r, c));
  world_view_transform_ = torch::tensor(w2v_flat, opt).reshape({4, 4}).t();

  // Projection matrix (OpenGL style)
  float tan_half_fovx = std::tan(FoVx_ * 0.5f);
  float tan_half_fovy = std::tan(FoVy_ * 0.5f);

  std::vector<float> proj_flat = {
    1.0f / tan_half_fovx, 0, 0, 0,
    0, 1.0f / tan_half_fovy, 0, 0,
    0, 0, (znear_ + zfar_) / (znear_ - zfar_),
        2.0f * znear_ * zfar_ / (znear_ - zfar_),
    0, 0, -1, 0
  };
  projection_matrix_ = torch::tensor(proj_flat, opt).reshape({4, 4}).t();

  // Full proj = projection * world_view
  full_proj_transform_ = torch::matmul(projection_matrix_, world_view_transform_);

  // Camera center
  camera_center_ = getCameraCenter();
}

// =============================================================================
// Training Data
// =============================================================================

void GaussianKeyframe::setTrainingImage(const cv::Mat& image) {
  // Convert BGR → RGB and to float32 [0,1] tensor [3, H, W]
  cv::Mat rgb;
  cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

  std::vector<float> data((float*)rgb.data, (float*)rgb.data + rgb.total() * 3);
  auto opt = torch::TensorOptions().dtype(torch::kFloat32);
  gt_image_ = torch::tensor(data, opt)
      .reshape({image.rows, image.cols, 3})
      .permute({2, 0, 1});  // [3, H, W]
}

void GaussianKeyframe::setDepthMap(const cv::Mat& depth) {
  cv::Mat depth_f32;
  depth.convertTo(depth_f32, CV_32FC1);
  std::vector<float> data((float*)depth_f32.data,
                          (float*)depth_f32.data + depth_f32.total());
  auto opt = torch::TensorOptions().dtype(torch::kFloat32);
  gt_depth_ = torch::tensor(data, opt)
      .reshape({1, depth.rows, depth.cols});  // [1, H, W]
}

torch::Tensor GaussianKeyframe::getUndistortMask() const {
  if (undistort_mask_.defined()) return undistort_mask_;
  return torch::ones({1, image_height_, image_width_},
      torch::TensorOptions().dtype(torch::kFloat32));
}

// =============================================================================
// Memory Management
// =============================================================================

void GaussianKeyframe::transferToCPU() {
  if (gt_image_.defined() && gt_image_.is_cuda()) {
    gt_image_ = gt_image_.cpu();
  }
  if (gt_depth_.defined() && gt_depth_.is_cuda()) {
    gt_depth_ = gt_depth_.cpu();
  }
}

void GaussianKeyframe::transferToGPU(torch::DeviceType device) {
  if (gt_image_.defined() && !gt_image_.is_cuda()) {
    gt_image_ = gt_image_.to(device);
  }
  if (gt_depth_.defined() && !gt_depth_.is_cuda()) {
    gt_depth_ = gt_depth_.to(device);
  }
}

void GaussianKeyframe::clearGPUData() {
  gt_image_ = torch::Tensor();
  gt_depth_ = torch::Tensor();
}

}  // namespace scaffold_chungs
