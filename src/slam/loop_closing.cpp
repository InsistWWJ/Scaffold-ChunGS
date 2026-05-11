/**
 * Scaffold-ChunGS: Loop Closing Module implementation.
 */

#include "scaffold_chunks/loop_closing.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>

namespace scaffold_chungs {

// =============================================================================
// SE(3) Utility: Log map
// =============================================================================

namespace {

/// Logarithm map: SE(3) -> se(3) twist coordinates (6x1).
/// Returns [v; w] where w is rotation axis * angle, v is translation.
Eigen::Matrix<float, 6, 1> se3Log(const Eigen::Matrix4f& T) {
  Eigen::Matrix<float, 6, 1> xi;

  Eigen::Matrix3f R = T.block<3, 3>(0, 0);
  Eigen::Vector3f t = T.block<3, 1>(0, 3);

  // Clamp trace to valid acos range
  float trace_R = R.trace();
  float cos_theta = (trace_R - 1.0f) / 2.0f;
  cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));
  float theta = std::acos(cos_theta);

  Eigen::Vector3f omega;
  if (theta < 1e-8f) {
    omega = Eigen::Vector3f::Zero();
  } else {
    omega = Eigen::Vector3f(
        R(2, 1) - R(1, 2),
        R(0, 2) - R(2, 0),
        R(1, 0) - R(0, 1)) * (0.5f * theta / std::sin(theta));
  }
  xi.block<3, 1>(3, 0) = omega;

  // Compute V^{-1} for translation
  if (theta < 1e-8f) {
    xi.block<3, 1>(0, 0) = t;
  } else {
    float sin_theta = std::sin(theta);
    float A = sin_theta / theta;
    float B = (1.0f - cos_theta) / (theta * theta);
    float C = (1.0f - A) / (theta * theta);

    Eigen::Matrix3f V_inv = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f wx = Eigen::Matrix3f::Zero();
    wx(0, 1) = -omega(2); wx(0, 2) = omega(1);
    wx(1, 0) = omega(2); wx(1, 2) = -omega(0);
    wx(2, 0) = -omega(1); wx(2, 1) = omega(0);

    V_inv += 0.5f * wx + C * wx * wx;
    xi.block<3, 1>(0, 0) = V_inv * t;
  }

  return xi;
}

/// Exponential map: se(3) twist -> SE(3) matrix
Eigen::Matrix4f se3Exp(const Eigen::Matrix<float, 6, 1>& xi) {
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();

  Eigen::Vector3f omega = xi.block<3, 1>(3, 0);
  Eigen::Vector3f v = xi.block<3, 1>(0, 0);

  float theta = omega.norm();

  Eigen::Matrix3f R;
  if (theta < 1e-8f) {
    R = Eigen::Matrix3f::Identity();
    T.block<3, 1>(0, 3) = v;
  } else {
    Eigen::Vector3f axis = omega / theta;
    Eigen::AngleAxisf aa(theta, axis);
    R = aa.toRotationMatrix();

    // Rodriguez formula for V
    float sin_t = std::sin(theta);
    float cos_t = std::cos(theta);
    float V_coeff = (1.0f - cos_t) / (theta * theta);
    float W_coeff = (theta - sin_t) / (theta * theta * theta);

    Eigen::Matrix3f wx = Eigen::Matrix3f::Zero();
    wx(0, 1) = -axis(2); wx(0, 2) = axis(1);
    wx(1, 0) = axis(2); wx(1, 2) = -axis(0);
    wx(2, 0) = -axis(1); wx(2, 1) = axis(0);

    Eigen::Matrix3f V = Eigen::Matrix3f::Identity() +
                         V_coeff * wx + W_coeff * wx * wx;
    T.block<3, 1>(0, 3) = V * v * theta;
  }

  T.block<3, 3>(0, 0) = R;
  return T;
}

/// Hat operator: R^3 -> so(3) skew-symmetric matrix
Eigen::Matrix3f skew(const Eigen::Vector3f& w) {
  Eigen::Matrix3f S;
  S << 0, -w(2), w(1),
       w(2), 0, -w(0),
       -w(1), w(0), 0;
  return S;
}

}  // anonymous namespace

// =============================================================================
// Construction
// =============================================================================

LoopClosingModule::LoopClosingModule(const LoopClosingConfig& cfg,
                                     float fx, float fy, float cx, float cy)
    : cfg_(cfg), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

// =============================================================================
// Keyframe Database Management
// =============================================================================

void LoopClosingModule::addKeyframe(int64_t kf_id,
                                     const cv::Mat& descriptors,
                                     const Eigen::Matrix4f& pose) {
  std::lock_guard<std::mutex> lock(mutex_database_);

  KeyframeEntry entry;
  entry.id = kf_id;
  entry.pose = pose;
  entry.descriptors = descriptors.clone();
  entry.bow = computeBoVW(descriptors);

  kf_database_[kf_id] = entry;

  // Accumulate for vocabulary building
  if (!vocab_built_ && descriptors.rows > 0) {
    desc_buffer_.push_back(descriptors.clone());
    if (static_cast<int>(desc_buffer_.size()) >= kVocabBuildThreshold) {
      buildVocabulary();
    }
  }
}

void LoopClosingModule::removeKeyframe(int64_t kf_id) {
  std::lock_guard<std::mutex> lock(mutex_database_);
  kf_database_.erase(kf_id);
}

// =============================================================================
// Loop Detection
// =============================================================================

std::optional<LoopCandidate> LoopClosingModule::detectLoop(
    int64_t kf_id, const cv::Mat& descriptors,
    const Eigen::Matrix4f& pose) {
  std::lock_guard<std::mutex> lock(mutex_database_);

  if (kf_database_.size() < 10) return std::nullopt;

  BoVWSignature query_bow = computeBoVW(descriptors);

  // Find best match excluding recent keyframes
  int64_t best_id = -1;
  float best_score = 0.0f;

  for (const auto& [other_id, entry] : kf_database_) {
    // Skip self and temporally close keyframes
    if (std::abs(other_id - kf_id) < 50) continue;

    float s = similarity(query_bow, entry.bow);
    if (s > best_score) {
      best_score = s;
      best_id = other_id;
    }
  }

  if (best_score < cfg_.loop_score_threshold) {
    consecutive_detections_ = 0;
    last_loop_candidate_ = -1;
    return std::nullopt;
  }

  LoopCandidate candidate;
  candidate.query_kf_id = kf_id;
  candidate.match_kf_id = best_id;
  candidate.score = best_score;

  // Check for temporal consistency (consecutive detections)
  if (best_id == last_loop_candidate_) {
    consecutive_detections_++;
  } else {
    consecutive_detections_ = 1;
    last_loop_candidate_ = best_id;
  }

  if (consecutive_detections_ < cfg_.min_consecutive_loop) {
    return std::nullopt;
  }

  return candidate;
}

// =============================================================================
// Loop Verification
// =============================================================================

std::optional<Eigen::Matrix4f> LoopClosingModule::verifyLoop(
    int64_t kf_from, int64_t kf_to,
    const std::vector<cv::Point3f>& pts3d,
    const std::vector<cv::Point2f>& pts2d,
    const cv::Mat& desc_from, const cv::Mat& desc_to) {

  // Step 1: 3D-2D PnP registration
  int num_pnp_inliers = 0;
  Eigen::Matrix4f T_rel = registerPnP(pts3d, pts2d, num_pnp_inliers);

  if (num_pnp_inliers < cfg_.min_3d_matches) {
    return std::nullopt;
  }

  // Step 2: Check translation magnitude
  float trans = T_rel.block<3, 1>(0, 3).norm();
  if (trans > cfg_.max_loop_distance) {
    std::cout << "[LoopClosing] Loop rejected: translation " << trans
              << " > " << cfg_.max_loop_distance << "\n";
    return std::nullopt;
  }

  return T_rel;
}

// =============================================================================
// Pose Graph Operations
// =============================================================================

void LoopClosingModule::addPoseGraphEdge(const PoseGraphEdge& edge) {
  std::lock_guard<std::mutex> lock(mutex_graph_);
  pose_graph_edges_.push_back(edge);
}

// =============================================================================
// Pose Graph Optimization (Gauss-Newton on SE(3) manifold)
// =============================================================================

std::map<int64_t, Eigen::Matrix4f> LoopClosingModule::optimizePoseGraph() {
  std::lock_guard<std::mutex> lock_graph(mutex_graph_);
  std::lock_guard<std::mutex> lock_db(mutex_database_);

  // Collect all nodes
  std::unordered_set<int64_t> node_set;
  for (const auto& edge : pose_graph_edges_) {
    node_set.insert(edge.from_id);
    node_set.insert(edge.to_id);
  }

  std::vector<int64_t> node_ids(node_set.begin(), node_set.end());
  std::sort(node_ids.begin(), node_ids.end());

  int N = static_cast<int>(node_ids.size());

  // Index map
  std::map<int64_t, int> index;
  for (int i = 0; i < N; ++i) {
    index[node_ids[i]] = i;
  }

  // Initial poses
  std::vector<Eigen::Matrix4f> poses(N);
  for (int i = 0; i < N; ++i) {
    auto it = kf_database_.find(node_ids[i]);
    if (it != kf_database_.end()) {
      poses[i] = it->second.pose;
    } else {
      poses[i] = Eigen::Matrix4f::Identity();
    }
  }

  // Gauss-Newton iterations
  for (int iter = 0; iter < cfg_.max_optimization_iterations; ++iter) {
    // Build linear system: J^T J Δx = -J^T e
    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(6 * N, 6 * N);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(6 * N);
    float total_error = 0.0f;

    for (const auto& edge : pose_graph_edges_) {
      int i = index[edge.from_id];
      int j = index[edge.to_id];

      // Current relative: T_i_to_j_est = Ti^{-1} * Tj
      Eigen::Matrix4f T_rel_est = poses[i].inverse() * poses[j];
      Eigen::Matrix4f T_rel_meas = edge.relative_pose;

      // Error: log(T_rel_meas^{-1} * T_rel_est) in se(3)
      Eigen::Matrix4f T_err = T_rel_meas.inverse() * T_rel_est;
      Eigen::Matrix<float, 6, 1> e = se3Log(T_err);

      float w = edge.weight;
      total_error += w * e.squaredNorm();

      // Jacobians w.r.t. delta_i and delta_j
      Eigen::Matrix3f R_ij = T_rel_meas.block<3, 3>(0, 0);
      Eigen::Vector3f t_ij = T_rel_meas.block<3, 1>(0, 3);

      // Adjoint of T_rel_meas^{-1}
      Eigen::Matrix<float, 6, 6> Adj_inv;
      Adj_inv.setZero();
      Eigen::Matrix3f R_inv = R_ij.transpose();
      Adj_inv.block<3, 3>(0, 0) = R_inv;
      Adj_inv.block<3, 3>(0, 3) = skew(R_inv * t_ij) * R_inv;
      Adj_inv.block<3, 3>(3, 0) = Eigen::Matrix3f::Zero();
      Adj_inv.block<3, 3>(3, 3) = R_inv;

      // J_i = -Adj(T_rel_err^{-1}) * Adj(T_j^{-1})
      // For simplicity, use the analytical Jacobians:
      // J_i = -Adj(T_err^{-1}),  J_j = I (approximation for small error)
      Eigen::Matrix<float, 6, 6> J_i = -Adj_inv;
      Eigen::Matrix<float, 6, 6> J_j = Eigen::Matrix<float, 6, 6>::Identity();

      // Accumulate into H and b
      for (int r = 0; r < 6; ++r) {
        int row_i = 6 * i + r;
        int row_j = 6 * j + r;
        for (int c = 0; c < 6; ++c) {
          int col_i = 6 * i + c;
          int col_j = 6 * j + c;

          float Ji_rc = J_i(r, c);
          float Jj_rc = J_j(r, c);

          H(row_i, col_i) += w * Ji_rc * Ji_rc;
          H(row_i, col_j) += w * Ji_rc * Jj_rc;
          H(row_j, col_i) += w * Jj_rc * Ji_rc;
          H(row_j, col_j) += w * Jj_rc * Jj_rc;
        }
        b(row_i) -= w * J_i(r, r) * e(r);
        b(row_j) -= w * J_j(r, r) * e(r);
      }
    }

    // Fix first pose (gauge freedom)
    for (int k = 0; k < 6; ++k) {
      H(0, k) = 0.0f;
      H(k, 0) = 0.0f;
    }
    H(0, 0) = 1.0f;
    b(0) = 0.0f;

    // Solve normal equations
    Eigen::VectorXf dx = H.ldlt().solve(b);

    // Update poses on manifold
    for (int i = 0; i < N; ++i) {
      Eigen::Matrix<float, 6, 1> dxi = dx.segment<6>(6 * i);
      poses[i] = poses[i] * se3Exp(dxi);
    }

    // Convergence check
    if (dx.norm() < cfg_.optimization_convergence * N) {
      std::cout << "[LoopClosing] Pose graph converged in " << (iter + 1)
                << " iterations, error = " << total_error << "\n";
      break;
    }
  }

  // Build result map
  std::map<int64_t, Eigen::Matrix4f> result;
  for (int i = 0; i < N; ++i) {
    result[node_ids[i]] = poses[i];
  }

  // Update database with optimized poses
  for (auto& [id, entry] : kf_database_) {
    auto it = result.find(id);
    if (it != result.end()) {
      entry.pose = it->second;
    }
  }

  return result;
}

// =============================================================================
// Correction Distribution
// =============================================================================

std::map<int64_t, Eigen::Matrix4f> LoopClosingModule::computeCorrections(
    int64_t loop_from, int64_t loop_to,
    const Eigen::Matrix4f& relative_pose) {
  std::lock_guard<std::mutex> lock(mutex_database_);

  std::map<int64_t, Eigen::Matrix4f> corrections;

  auto it_from = kf_database_.find(loop_from);
  auto it_to = kf_database_.find(loop_to);
  if (it_from == kf_database_.end() || it_to == kf_database_.end()) {
    return corrections;
  }

  // Compute the accumulated drift
  Eigen::Matrix4f T_from = it_from->second.pose;
  Eigen::Matrix4f T_to = it_to->second.pose;
  Eigen::Matrix4f T_accumulated = T_from.inverse() * T_to;

  // Drift error: how much the observed relative differs from the loop constraint
  Eigen::Matrix4f T_drift = relative_pose.inverse() * T_accumulated;

  // Distribute the correction across keyframes between loop_from and loop_to
  int64_t span = std::abs(loop_to - loop_from);
  if (span == 0) span = 1;

  for (auto& [kf_id, entry] : kf_database_) {
    if (kf_id < loop_from || kf_id > loop_to) continue;

    // Linear interpolation weight: 0 at loop_from, 1 at loop_to
    float w = static_cast<float>(kf_id - loop_from) / static_cast<float>(span);

    // Interpolate the correction on SE(3)
    Eigen::Matrix<float, 6, 1> drift_xi = se3Log(T_drift);
    Eigen::Matrix4f correction = se3Exp(w * drift_xi);

    corrections[kf_id] = correction * entry.pose;
  }

  return corrections;
}

// =============================================================================
// Reset
// =============================================================================

void LoopClosingModule::reset() {
  std::lock_guard<std::mutex> lock_db(mutex_database_);
  std::lock_guard<std::mutex> lock_graph(mutex_graph_);

  kf_database_.clear();
  pose_graph_edges_.clear();
  desc_buffer_.clear();
  vocabulary_.release();
  vocab_built_ = false;
  last_loop_candidate_ = -1;
  consecutive_detections_ = 0;
}

// =============================================================================
// BoVW Computation
// =============================================================================

LoopClosingModule::BoVWSignature LoopClosingModule::computeBoVW(
    const cv::Mat& descriptors) const {

  BoVWSignature sig;
  sig.bow_vector.resize(cfg_.vocab_size, 0.0f);

  if (descriptors.empty() || !vocab_built_ || vocabulary_.empty()) {
    return sig;
  }

  // For each descriptor, find nearest vocabulary word
  // Use FLANN for efficient matching
  cv::Ptr<cv::DescriptorMatcher> matcher;
  if (descriptors.type() == CV_8U) {
    matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
  } else {
    matcher = cv::FlannBasedMatcher::create();
  }

  std::vector<cv::DMatch> matches;
  matcher->match(descriptors, vocabulary_, matches);

  for (const auto& m : matches) {
    if (m.trainIdx >= 0 && m.trainIdx < cfg_.vocab_size) {
      sig.bow_vector[m.trainIdx] += 1.0f;
    }
  }

  // L2-normalize
  float norm = 0.0f;
  for (float v : sig.bow_vector) norm += v * v;
  norm = std::sqrt(norm);
  if (norm > 1e-8f) {
    for (float& v : sig.bow_vector) v /= norm;
  }

  return sig;
}

float LoopClosingModule::similarity(const BoVWSignature& a,
                                     const BoVWSignature& b) const {
  if (a.bow_vector.empty() || b.bow_vector.empty()) return 0.0f;

  // Cosine similarity (dot product of L2-normalized vectors)
  float dot = 0.0f;
  for (size_t i = 0; i < a.bow_vector.size() && i < b.bow_vector.size(); ++i) {
    dot += a.bow_vector[i] * b.bow_vector[i];
  }
  return dot;
}

// =============================================================================
// Vocabulary Building
// =============================================================================

void LoopClosingModule::buildVocabulary() {
  if (desc_buffer_.empty() || vocab_built_) return;

  // Concatenate all descriptors
  int total_rows = 0;
  for (const auto& d : desc_buffer_) total_rows += d.rows;
  if (total_rows < cfg_.vocab_size) return;

  cv::Mat all_descriptors(total_rows, desc_buffer_[0].cols, desc_buffer_[0].type());
  int offset = 0;
  for (const auto& d : desc_buffer_) {
    d.copyTo(all_descriptors.rowRange(offset, offset + d.rows));
    offset += d.rows;
  }

  // Convert to float if binary
  cv::Mat all_float;
  if (all_descriptors.type() != CV_32F) {
    all_descriptors.convertTo(all_float, CV_32F);
  } else {
    all_float = all_descriptors;
  }

  // k-means clustering
  cv::Mat labels;
  int attempts = 3;
  cv::kmeans(all_float, cfg_.vocab_size, labels,
             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                              100, 0.01),
             attempts, cv::KMEANS_PP_CENTERS, vocabulary_);

  vocab_built_ = (vocabulary_.rows == cfg_.vocab_size);
  if (vocab_built_) {
    std::cout << "[LoopClosing] Vocabulary built: " << vocabulary_.rows
              << " words from " << total_rows << " descriptors\n";
  }
}

// =============================================================================
// Pairwise PnP Registration
// =============================================================================

Eigen::Matrix4f LoopClosingModule::registerPnP(
    const std::vector<cv::Point3f>& pts3d,
    const std::vector<cv::Point2f>& pts2d,
    int& num_inliers) const {

  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  num_inliers = 0;

  if (static_cast<int>(pts3d.size()) < cfg_.min_3d_matches) return T;

  cv::Mat K = (cv::Mat_<float>(3, 3) << fx_, 0.0f, cx_,
                                         0.0f, fy_, cy_,
                                         0.0f, 0.0f, 1.0f);

  cv::Mat rvec, tvec;
  std::vector<int> inliers;

  bool ok = cv::solvePnPRansac(pts3d, pts2d, K, cv::noArray(),
                                rvec, tvec, false, 200, 4.0f, 0.99, inliers,
                                cv::SOLVEPNP_ITERATIVE);

  if (!ok) return T;

  num_inliers = static_cast<int>(inliers.size());

  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      T(r, c) = R_cv.at<float>(r, c);
    }
  }
  T(0, 3) = tvec.at<float>(0);
  T(1, 3) = tvec.at<float>(1);
  T(2, 3) = tvec.at<float>(2);

  return T;
}

// =============================================================================
// ICP Refinement
// =============================================================================

Eigen::Matrix4f LoopClosingModule::refineICP(
    const std::vector<cv::Point3f>& src,
    const std::vector<cv::Point3f>& dst,
    const Eigen::Matrix4f& initial_guess) const {

  if (src.size() < 10 || dst.size() < 10) return initial_guess;

  // Build Eigen point matrices
  Eigen::MatrixXf S(3, src.size());
  Eigen::MatrixXf D(3, dst.size());
  for (size_t i = 0; i < src.size(); ++i) {
    S(0, i) = src[i].x; S(1, i) = src[i].y; S(2, i) = src[i].z;
  }
  for (size_t i = 0; i < dst.size(); ++i) {
    D(0, i) = dst[i].x; D(1, i) = dst[i].y; D(2, i) = dst[i].z;
  }

  Eigen::Matrix4f T = initial_guess;
  Eigen::Matrix3f R = T.block<3, 3>(0, 0);
  Eigen::Vector3f t = T.block<3, 1>(0, 3);

  for (int iter = 0; iter < 20; ++iter) {
    // Nearest neighbor correspondences (brute-force for small sets)
    std::vector<int> correspondences(src.size(), -1);
    for (size_t i = 0; i < src.size(); ++i) {
      Eigen::Vector3f s_transformed = R * S.col(i) + t;
      float best_dist = std::numeric_limits<float>::max();
      int best_j = -1;
      for (size_t j = 0; j < dst.size(); ++j) {
        float dist = (s_transformed - D.col(j)).squaredNorm();
        if (dist < best_dist) {
          best_dist = dist;
          best_j = static_cast<int>(j);
        }
      }
      correspondences[i] = best_j;
    }

    // Compute centroids
    Eigen::Vector3f centroid_s = Eigen::Vector3f::Zero();
    Eigen::Vector3f centroid_d = Eigen::Vector3f::Zero();
    int N = 0;
    for (size_t i = 0; i < src.size(); ++i) {
      if (correspondences[i] < 0) continue;
      centroid_s += S.col(i);
      centroid_d += D.col(correspondences[i]);
      N++;
    }
    if (N < 5) break;
    centroid_s /= static_cast<float>(N);
    centroid_d /= static_cast<float>(N);

    // Cross-covariance matrix
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    for (size_t i = 0; i < src.size(); ++i) {
      if (correspondences[i] < 0) continue;
      Eigen::Vector3f s_centered = S.col(i) - centroid_s;
      Eigen::Vector3f d_centered = D.col(correspondences[i]) - centroid_d;
      H += s_centered * d_centered.transpose();
    }

    // SVD for rotation
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f R_new = svd.matrixV() * svd.matrixU().transpose();
    if (R_new.determinant() < 0) {
      Eigen::Matrix3f V = svd.matrixV();
      V.col(2) *= -1.0f;
      R_new = V * svd.matrixU().transpose();
    }

    Eigen::Vector3f t_new = centroid_d - R_new * centroid_s;

    // Convergence check
    float rot_change = (R_new.transpose() * R - Eigen::Matrix3f::Identity()).norm();
    float trans_change = (t_new - t).norm();
    R = R_new;
    t = t_new;

    if (rot_change < 1e-6f && trans_change < 1e-6f) break;
  }

  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t;
  return T;
}

}  // namespace scaffold_chungs
