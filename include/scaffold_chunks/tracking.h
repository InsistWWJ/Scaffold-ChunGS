/**
 * Scaffold-ChunGS: Visual Odometry Tracking Module.
 *
 * Tracks camera pose frame-to-frame using:
 *   1. ORB feature extraction + FLANN matching
 *   2. 3D-2D PnP + RANSAC (with depth back-projection)
 *   3. Constant velocity motion model
 *   4. Fallback: Essential matrix decomposition (5-point) when depth unavailable
 */

#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <memory>
#include <vector>

namespace scaffold_chungs {

// =============================================================================
// Tracking Configuration
// =============================================================================

struct TrackingConfig {
  int max_orb_features = 1000;
  float scale_factor = 1.2f;
  int n_levels = 8;
  int fast_threshold = 20;
  float match_ratio_threshold = 0.75f;     // Lowe's ratio test
  int min_pnp_inliers = 15;
  float pnp_ransac_reproj_threshold = 3.0f;
  bool use_depth_for_pnp = true;            // Back-project to 3D for PnP
  float min_depth = 0.1f;
  float max_depth = 80.0f;
};

// =============================================================================
// Tracking Result
// =============================================================================

struct TrackingResult {
  Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();
  bool success = false;
  int num_inliers = 0;
  int num_matches = 0;
};

// =============================================================================
// Tracking Module
// =============================================================================

class TrackingModule {
 public:
  TrackingModule(const TrackingConfig& cfg,
                 float fx, float fy, float cx, float cy,
                 int width, int height);

  /** Estimate camera pose Tcw for the current frame.
   *  @param image       Current RGB image (CV_8UC3 or CV_8UC1)
   *  @param depth       Optional depth map (CV_32FC1 in meters), may be empty
   *  @param initial_pose Prior pose guess (from motion model or external)
   *  @return TrackingResult with estimated Tcw and quality metrics
   */
  TrackingResult track(const cv::Mat& image,
                       const cv::Mat& depth,
                       const Eigen::Matrix4f& initial_pose);

  /** Reset internal state (after loop closure or tracking loss). */
  void reset();

  /** Update camera intrinsics. */
  void setIntrinsics(float fx, float fy, float cx, float cy);

  /** Check if tracking has been initialized. */
  bool isInitialized() const { return is_initialized_; }

 private:
  // Extract ORB features and descriptors
  void extractORB(const cv::Mat& image,
                  std::vector<cv::KeyPoint>& keypoints,
                  cv::Mat& descriptors) const;

  // Match descriptors using FLANN with Lowe's ratio test
  std::vector<cv::DMatch> matchFeatures(const cv::Mat& desc_prev,
                                        const cv::Mat& desc_curr) const;

  // Build FLANN index from previous descriptors
  void buildFlannIndex(const cv::Mat& descriptors);

  // Back-project keypoints to 3D using depth
  std::vector<cv::Point3f> toPoints3D(const std::vector<cv::KeyPoint>& kpts,
                                       const cv::Mat& depth_map) const;

  // PnP + RANSAC pose estimation
  Eigen::Matrix4f solvePnPRansac(const std::vector<cv::Point3f>& pts3d,
                                  const std::vector<cv::Point2f>& pts2d,
                                  int& num_inliers) const;

  // Essential matrix decomposition for monocular
  Eigen::Matrix4f solveEssential(const std::vector<cv::Point2f>& pts_prev,
                                  const std::vector<cv::Point2f>& pts_curr,
                                  int& num_inliers) const;

  // Constant velocity motion model prediction
  Eigen::Matrix4f predictMotionModel() const;

  TrackingConfig cfg_;
  float fx_, fy_, cx_, cy_;
  int width_, height_;

  cv::Ptr<cv::ORB> orb_;

  // Frame-to-frame state
  cv::Mat prev_image_;
  std::vector<cv::KeyPoint> prev_keypoints_;
  cv::Mat prev_descriptors_;
  cv::Ptr<cv::flann::Index> flann_index_;
  bool is_initialized_ = false;
  int consecutive_failures_ = 0;

  // Motion model
  Eigen::Matrix4f prev_pose_ = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f velocity_ = Eigen::Matrix4f::Identity();  // T_{k-1} → T_k
};

}  // namespace scaffold_chungs
