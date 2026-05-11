/**
 * Scaffold-ChunGS: Visual Odometry Tracking Module implementation.
 */

#include "scaffold_chunks/tracking.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

namespace scaffold_chungs {

// =============================================================================
// Construction
// =============================================================================

TrackingModule::TrackingModule(const TrackingConfig& cfg,
                               float fx, float fy, float cx, float cy,
                               int width, int height)
    : cfg_(cfg), fx_(fx), fy_(fy), cx_(cx), cy_(cy),
      width_(width), height_(height) {
  orb_ = cv::ORB::create(cfg_.max_orb_features, cfg_.scale_factor,
                         cfg_.n_levels, cfg_.fast_threshold);
}

// =============================================================================
// Reset
// =============================================================================

void TrackingModule::reset() {
  prev_image_.release();
  prev_keypoints_.clear();
  prev_descriptors_.release();
  if (flann_index_) flann_index_.release();
  is_initialized_ = false;
  consecutive_failures_ = 0;
  prev_pose_ = Eigen::Matrix4f::Identity();
  velocity_ = Eigen::Matrix4f::Identity();
}

// =============================================================================
// Intrinsics Update
// =============================================================================

void TrackingModule::setIntrinsics(float fx, float fy, float cx, float cy) {
  fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;
}

// =============================================================================
// Main Tracking Entry Point
// =============================================================================

TrackingResult TrackingModule::track(const cv::Mat& image,
                                      const cv::Mat& depth,
                                      const Eigen::Matrix4f& initial_pose) {
  TrackingResult result;

  if (image.empty()) {
    result.success = false;
    return result;
  }

  // Convert to grayscale if needed
  cv::Mat gray;
  if (image.channels() == 3) {
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image;
  }

  // Extract ORB features on current frame
  std::vector<cv::KeyPoint> curr_keypoints;
  cv::Mat curr_descriptors;
  extractORB(gray, curr_keypoints, curr_descriptors);

  if (curr_keypoints.empty()) {
    result.success = false;
    return result;
  }

  // First frame or after reset: store and return identity
  if (!is_initialized_ || prev_descriptors_.empty()) {
    prev_image_ = gray.clone();
    prev_keypoints_ = curr_keypoints;
    prev_descriptors_ = curr_descriptors.clone();
    buildFlannIndex(prev_descriptors_);
    is_initialized_ = true;
    prev_pose_ = initial_pose;
    result.Tcw = initial_pose;
    result.success = true;
    result.num_matches = 0;
    result.num_inliers = 0;
    return result;
  }

  // Match features between previous and current frame
  std::vector<cv::DMatch> matches = matchFeatures(prev_descriptors_, curr_descriptors);
  result.num_matches = static_cast<int>(matches.size());

  int min_matches = std::max(15, cfg_.min_pnp_inliers);
  if (static_cast<int>(matches.size()) < min_matches) {
    consecutive_failures_++;
    result.success = false;
    return result;
  }

  // Extract matched 2D points
  std::vector<cv::Point2f> pts_prev, pts_curr;
  pts_prev.reserve(matches.size());
  pts_curr.reserve(matches.size());
  for (const auto& m : matches) {
    pts_prev.push_back(prev_keypoints_[m.queryIdx].pt);
    pts_curr.push_back(curr_keypoints_[m.trainIdx].pt);
  }

  int num_inliers = 0;

  // Strategy 1: If depth is available, back-project to 3D and use PnP
  if (cfg_.use_depth_for_pnp && !depth.empty()) {
    std::vector<cv::Point3f> pts3d = toPoints3D(prev_keypoints_, depth);
    if (!pts3d.empty()) {
      // Only keep matches where both 2D and 3D points are valid
      std::vector<cv::Point3f> pts3d_matched;
      std::vector<cv::Point2f> pts2d_matched;
      pts3d_matched.reserve(matches.size());
      pts2d_matched.reserve(matches.size());
      for (const auto& m : matches) {
        const auto& p3 = pts3d[m.queryIdx];
        if (std::isfinite(p3.x) && std::isfinite(p3.y) && std::isfinite(p3.z)) {
          pts3d_matched.push_back(p3);
          pts2d_matched.push_back(pts_curr[m.trainIdx]);
        }
      }
      if (static_cast<int>(pts3d_matched.size()) >= cfg_.min_pnp_inliers) {
        result.Tcw = solvePnPRansac(pts3d_matched, pts2d_matched, num_inliers);
        result.success = (num_inliers >= cfg_.min_pnp_inliers);
        result.num_inliers = num_inliers;
      }
    }
  }

  // Strategy 2: Fallback to Essential matrix (monocular, 5-point algorithm)
  if (!result.success) {
    result.Tcw = solveEssential(pts_prev, pts_curr, num_inliers);
    result.success = (num_inliers >= cfg_.min_pnp_inliers);
    result.num_inliers = num_inliers;
  }

  // If tracking succeeded, update state
  if (result.success) {
    // Update velocity (motion model)
    velocity_ = prev_pose_.inverse() * result.Tcw;
    prev_pose_ = result.Tcw;
    consecutive_failures_ = 0;

    // Store current frame as previous for next iteration
    prev_image_ = gray.clone();
    prev_keypoints_ = curr_keypoints;
    prev_descriptors_ = curr_descriptors.clone();
    buildFlannIndex(prev_descriptors_);
  } else {
    consecutive_failures_++;
  }

  return result;
}

// =============================================================================
// ORB Feature Extraction
// =============================================================================

void TrackingModule::extractORB(const cv::Mat& image,
                                 std::vector<cv::KeyPoint>& keypoints,
                                 cv::Mat& descriptors) const {
  keypoints.clear();
  descriptors.release();

  if (orb_) {
    orb_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
  }
}

// =============================================================================
// FLANN Matching with Lowe's Ratio Test
// =============================================================================

std::vector<cv::DMatch> TrackingModule::matchFeatures(
    const cv::Mat& desc_prev, const cv::Mat& desc_curr) const {

  std::vector<cv::DMatch> matches;

  if (desc_prev.empty() || desc_curr.empty() ||
      desc_prev.type() != CV_32F || desc_curr.type() != CV_32F) {
    return matches;
  }

  // For binary descriptors (ORB uses 8-bit), use Hamming distance via BFMatcher
  // For float descriptors (FLANN-compatible), use FLANN
  cv::Ptr<cv::DescriptorMatcher> matcher;
  if (desc_prev.type() == CV_8U) {
    matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
  } else {
    matcher = cv::FlannBasedMatcher::create();
  }

  // Get the 2 nearest matches for Lowe's ratio test
  std::vector<std::vector<cv::DMatch>> knn_matches;
  matcher->knnMatch(desc_curr, desc_prev, knn_matches, 2);

  matches.reserve(knn_matches.size());
  for (const auto& knn : knn_matches) {
    if (knn.size() < 2) continue;
    if (knn[0].distance < cfg_.match_ratio_threshold * knn[1].distance) {
      // Swap so queryIdx = prev, trainIdx = curr (consistent with convention)
      matches.emplace_back(knn[0].trainIdx, knn[0].queryIdx, knn[0].distance);
    }
  }

  return matches;
}

// =============================================================================
// FLANN Index Construction
// =============================================================================

void TrackingModule::buildFlannIndex(const cv::Mat& descriptors) {
  if (descriptors.empty() || descriptors.type() != CV_32F) {
    flann_index_.release();
    return;
  }
  // Build a L2-indexed FLANN for float descriptors
  flann_index_ = cv::makePtr<cv::flann::Index>(
      descriptors, cv::flann::KDTreeIndexParams(4));
}

// =============================================================================
// Depth Back-Projection to 3D
// =============================================================================

std::vector<cv::Point3f> TrackingModule::toPoints3D(
    const std::vector<cv::KeyPoint>& kpts,
    const cv::Mat& depth_map) const {

  std::vector<cv::Point3f> pts3d;
  pts3d.reserve(kpts.size());

  if (depth_map.empty()) {
    pts3d.resize(kpts.size(), cv::Point3f(NAN, NAN, NAN));
    return pts3d;
  }

  for (const auto& kp : kpts) {
    int u = static_cast<int>(std::round(kp.pt.x));
    int v = static_cast<int>(std::round(kp.pt.y));

    if (u < 0 || u >= depth_map.cols || v < 0 || v >= depth_map.rows) {
      pts3d.emplace_back(NAN, NAN, NAN);
      continue;
    }

    float d;
    if (depth_map.type() == CV_32FC1) {
      d = depth_map.at<float>(v, u);
    } else if (depth_map.type() == CV_16UC1) {
      d = depth_map.at<uint16_t>(v, u) * 0.001f;  // mm -> meters
    } else {
      pts3d.emplace_back(NAN, NAN, NAN);
      continue;
    }

    if (d < cfg_.min_depth || d > cfg_.max_depth) {
      pts3d.emplace_back(NAN, NAN, NAN);
      continue;
    }

    // Back-project: X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy,  Z = d
    float x = (kp.pt.x - cx_) * d / fx_;
    float y = (kp.pt.y - cy_) * d / fy_;
    pts3d.emplace_back(x, y, d);
  }

  return pts3d;
}

// =============================================================================
// PnP + RANSAC Pose Estimation
// =============================================================================

Eigen::Matrix4f TrackingModule::solvePnPRansac(
    const std::vector<cv::Point3f>& pts3d,
    const std::vector<cv::Point2f>& pts2d,
    int& num_inliers) const {

  Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();
  num_inliers = 0;

  if (static_cast<int>(pts3d.size()) < cfg_.min_pnp_inliers ||
      pts3d.size() != pts2d.size()) {
    return Tcw;
  }

  cv::Mat K = (cv::Mat_<float>(3, 3) << fx_, 0.0f, cx_,
                                         0.0f, fy_, cy_,
                                         0.0f, 0.0f, 1.0f);

  cv::Mat rvec, tvec;
  std::vector<int> inliers;

  bool ok = cv::solvePnPRansac(
      pts3d, pts2d, K, cv::noArray(),
      rvec, tvec, false,               // useExtrinsicGuess = false
      100,                              // iterations
      cfg_.pnp_ransac_reproj_threshold,
      0.99,                             // confidence
      inliers,
      cv::SOLVEPNP_ITERATIVE);

  if (!ok || static_cast<int>(inliers.size()) < cfg_.min_pnp_inliers) {
    num_inliers = static_cast<int>(inliers.size());
    return Tcw;
  }

  num_inliers = static_cast<int>(inliers.size());

  // Refine with only inliers
  if (static_cast<int>(inliers.size()) < static_cast<int>(pts3d.size())) {
    std::vector<cv::Point3f> pts3d_in;
    std::vector<cv::Point2f> pts2d_in;
    pts3d_in.reserve(inliers.size());
    pts2d_in.reserve(inliers.size());
    for (int idx : inliers) {
      pts3d_in.push_back(pts3d[idx]);
      pts2d_in.push_back(pts2d[idx]);
    }
    cv::solvePnP(pts3d_in, pts2d_in, K, cv::noArray(), rvec, tvec, true,
                 cv::SOLVEPNP_ITERATIVE);
  }

  // Convert rotation vector to matrix
  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);

  Tcw.setIdentity();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      Tcw(r, c) = R_cv.at<float>(r, c);
    }
  }
  Tcw(0, 3) = tvec.at<float>(0);
  Tcw(1, 3) = tvec.at<float>(1);
  Tcw(2, 3) = tvec.at<float>(2);

  return Tcw;
}

// =============================================================================
// Essential Matrix Decomposition (Monocular Fallback)
// =============================================================================

Eigen::Matrix4f TrackingModule::solveEssential(
    const std::vector<cv::Point2f>& pts_prev,
    const std::vector<cv::Point2f>& pts_curr,
    int& num_inliers) const {

  Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();
  num_inliers = 0;

  int min_pts = std::max(5, cfg_.min_pnp_inliers);
  if (static_cast<int>(pts_prev.size()) < min_pts ||
      pts_prev.size() != pts_curr.size()) {
    return Tcw;
  }

  cv::Mat K = (cv::Mat_<float>(3, 3) << fx_, 0.0f, cx_,
                                         0.0f, fy_, cy_,
                                         0.0f, 0.0f, 1.0f);

  // Find essential matrix with RANSAC
  cv::Mat inlier_mask;
  cv::Mat E = cv::findEssentialMat(
      pts_prev, pts_curr, K,
      cv::RANSAC, 0.999, 1.0,  // 1px RANSAC threshold
      inlier_mask);

  if (E.empty()) {
    return Tcw;
  }

  num_inliers = cv::countNonZero(inlier_mask);
  if (num_inliers < cfg_.min_pnp_inliers) {
    return Tcw;
  }

  // Recover pose from essential matrix
  cv::Mat R, t;
  int recovered = cv::recoverPose(E, pts_prev, pts_curr, K, R, t, inlier_mask);

  if (recovered <= 0) {
    return Tcw;
  }

  // Build Tcw = [R | t]
  Tcw.setIdentity();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      Tcw(r, c) = R.at<double>(r, c);
    }
  }
  Tcw(0, 3) = static_cast<float>(t.at<double>(0));
  Tcw(1, 3) = static_cast<float>(t.at<double>(1));
  Tcw(2, 3) = static_cast<float>(t.at<double>(2));

  return Tcw;
}

// =============================================================================
// Constant Velocity Motion Model
// =============================================================================

Eigen::Matrix4f TrackingModule::predictMotionModel() const {
  return prev_pose_ * velocity_;
}

}  // namespace scaffold_chungs
