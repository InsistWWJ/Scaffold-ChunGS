/**
 * Scaffold-ChunGS: Keyframe representing a camera observation used for
 * training and rendering anchor-based Gaussians.
 *
 * Adapted from DiskChunGS / Photo-SLAM for the scaffold anchor pipeline.
 */

#pragma once

#include <torch/torch.h>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <memory>
#include <vector>

namespace scaffold_chungs {

class GaussianKeyframe {
 public:
  GaussianKeyframe() = default;

  explicit GaussianKeyframe(int64_t fid, int creation_iter = 0)
      : fid_(fid), creation_iter_(creation_iter) {}

  // ===========================================================================
  // Pose Management
  // ===========================================================================

  void setPose(const Eigen::Matrix4f& Tcw);  // camera-to-world
  void setPose(const Eigen::Quaternionf& q, const Eigen::Vector3f& t);

  Eigen::Matrix4f getPose() const;           // returns Tcw
  Eigen::Matrix3f getRotationMatrix() const;
  Eigen::Vector3f getTranslation() const;
  torch::Tensor getCameraCenter(torch::DeviceType device = torch::kCPU) const;

  // ===========================================================================
  // Transform Tensors (computed from pose + intrinsics)
  // ===========================================================================

  void computeTransformTensors(torch::DeviceType device = torch::kCPU);

  torch::Tensor worldViewTransform() const { return world_view_transform_; }
  torch::Tensor projectionMatrix() const { return projection_matrix_; }
  torch::Tensor fullProjTransform() const { return full_proj_transform_; }

  // ===========================================================================
  // Camera Intrinsics
  // ===========================================================================

  void setIntrinsics(float fx, float fy, float cx, float cy,
                     int width, int height,
                     float znear = 0.01f, float zfar = 100.0f);

  float FoVx() const { return FoVx_; }
  float FoVy() const { return FoVy_; }
  int imageWidth() const { return image_width_; }
  int imageHeight() const { return image_height_; }
  float znear() const { return znear_; }
  float zfar() const { return zfar_; }

  // ===========================================================================
  // Training Data
  // ===========================================================================

  void setTrainingImage(const cv::Mat& image);
  void setDepthMap(const cv::Mat& depth);

  torch::Tensor getGTImage() const { return gt_image_; }
  torch::Tensor getGTDepth() const { return gt_depth_; }
  torch::Tensor getUndistortMask() const { return undistort_mask_; }

  // ===========================================================================
  // Memory Management
  // ===========================================================================

  void transferToCPU();
  void transferToGPU(torch::DeviceType device = torch::kCUDA);
  void clearGPUData();

  // ===========================================================================
  // Accessors
  // ===========================================================================

  int64_t fid() const { return fid_; }
  int creationIter() const { return creation_iter_; }
  int remainingTimesOfUse() const { return remaining_times_of_use_; }
  void decrementTimesOfUse() { if (remaining_times_of_use_ > 0) --remaining_times_of_use_; }
  void setTimesOfUse(int n) { remaining_times_of_use_ = n; }

 public:
  int64_t fid_ = 0;
  int creation_iter_ = 0;
  int remaining_times_of_use_ = 0;

 private:
  // Camera
  float fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
  int image_width_ = 0, image_height_ = 0;
  float FoVx_ = 0, FoVy_ = 0;
  float znear_ = 0.01f, zfar_ = 100.0f;

  // Pose
  Eigen::Matrix4f Tcw_ = Eigen::Matrix4f::Identity();
  torch::Tensor world_view_transform_;  // [4, 4]
  torch::Tensor projection_matrix_;     // [4, 4]
  torch::Tensor full_proj_transform_;   // [4, 4]
  torch::Tensor camera_center_;         // [3]

  // Image data
  torch::Tensor gt_image_;
  torch::Tensor gt_depth_;
  torch::Tensor undistort_mask_;
};

}  // namespace scaffold_chungs
