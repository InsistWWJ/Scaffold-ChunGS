/**
 * Scaffold-ChunGS: Loop Closing Module.
 *
 * Detects loop closures and corrects accumulated drift via:
 *   1. BoVW place recognition (ORB descriptors + k-means vocabulary)
 *   2. Pairwise registration (PnP + ICP on Gaussian means)
 *   3. Pose graph optimization on SE(3) manifold
 *   4. Global correction of keyframe poses + Gaussian positions
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace scaffold_chungs {

// =============================================================================
// Loop Closing Configuration
// =============================================================================

struct LoopClosingConfig {
  // Vocabulary
  int vocab_size = 1000;           // k-means cluster count for BoVW
  int vocab_tree_levels = 4;      // Hierarchical vocabulary depth

  // Detection thresholds
  int min_consecutive_loop = 3;    // Consecutive detections before acceptance
  float loop_score_threshold = 0.05f;   // Minimum similarity score

  // Registration
  int min_3d_matches = 30;         // Minimum 3D-2D correspondences for PnP
  float max_loop_distance = 20.0f;  // Maximum translation for valid loop (meters)
  bool use_icp_refinement = true;   // Refine with ICP on Gaussian means

  // Pose graph
  int max_optimization_iterations = 10;
  float optimization_convergence = 1e-6f;
};

// =============================================================================
// Loop Detection Result
// =============================================================================

struct LoopCandidate {
  int64_t query_kf_id = -1;
  int64_t match_kf_id = -1;
  float score = 0.0f;
  Eigen::Matrix4f relative_pose = Eigen::Matrix4f::Identity();
  int num_inliers = 0;
};

// =============================================================================
// Pose Graph Edge
// =============================================================================

struct PoseGraphEdge {
  int64_t from_id;
  int64_t to_id;
  Eigen::Matrix4f relative_pose;  // T_from_to
  float weight = 1.0f;
};

// =============================================================================
// Loop Closing Module
// =============================================================================

class LoopClosingModule {
 public:
  LoopClosingModule(const LoopClosingConfig& cfg, float fx, float fy,
                    float cx, float cy);

  /** Add a keyframe descriptor to the recognition database. */
  void addKeyframe(int64_t kf_id, const cv::Mat& descriptors,
                   const Eigen::Matrix4f& pose);

  /** Remove a keyframe from the database. */
  void removeKeyframe(int64_t kf_id);

  /** Detect a loop closure for the given keyframe.
   *  Returns a candidate if a valid loop is found, or empty. */
  std::optional<LoopCandidate> detectLoop(int64_t kf_id,
                                          const cv::Mat& descriptors,
                                          const Eigen::Matrix4f& pose);

  /** Verify a loop candidate with geometric registration.
   *  @param kf_from      Query keyframe ID
   *  @param kf_to        Matched keyframe ID
   *  @param pts3d       3D points in the matched keyframe (Gaussian means)
   *  @param pts2d       2D observations in the query frame
   *  @param desc_from    ORB descriptors of query frame
   *  @param desc_to      ORB descriptors of matched frame
   *  @return Verified relative pose, or empty on failure
   */
  std::optional<Eigen::Matrix4f> verifyLoop(
      int64_t kf_from, int64_t kf_to,
      const std::vector<cv::Point3f>& pts3d,
      const std::vector<cv::Point2f>& pts2d,
      const cv::Mat& desc_from, const cv::Mat& desc_to);

  /** Add an edge to the pose graph (e.g., from sequential odometry). */
  void addPoseGraphEdge(const PoseGraphEdge& edge);

  /** Run pose graph optimization over all accumulated edges.
   *  @return Map of keyframe ID -> corrected pose
   */
  std::map<int64_t, Eigen::Matrix4f> optimizePoseGraph();

  /** Compute the correction for a keyframe chain affected by a loop.
   *  Distributes the loop error across the loop's keyframes. */
  std::map<int64_t, Eigen::Matrix4f> computeCorrections(
      int64_t loop_from, int64_t loop_to,
      const Eigen::Matrix4f& relative_pose);

  /** Clear all internal state. */
  void reset();

 private:
  // BoVW representation for a keyframe
  struct BoVWSignature {
    std::vector<float> bow_vector;  // Normalized histogram over vocabulary
  };

  // Build a BoVW signature from ORB descriptors
  BoVWSignature computeBoVW(const cv::Mat& descriptors) const;

  // Cosine similarity between two BoVW vectors
  float similarity(const BoVWSignature& a, const BoVWSignature& b) const;

  // Pairwise 3D-2D PnP registration
  Eigen::Matrix4f registerPnP(const std::vector<cv::Point3f>& pts3d,
                               const std::vector<cv::Point2f>& pts2d,
                               int& num_inliers) const;

  // ICP refinement on 3D point sets
  Eigen::Matrix4f refineICP(const std::vector<cv::Point3f>& src,
                             const std::vector<cv::Point3f>& dst,
                             const Eigen::Matrix4f& initial_guess) const;

  // Build k-means vocabulary from accumulated descriptors
  void buildVocabulary();

  LoopClosingConfig cfg_;
  float fx_, fy_, cx_, cy_;

  // Vocabulary (built when enough descriptors accumulated)
  cv::Mat vocabulary_;  // [vocab_size x 32] cluster centers
  bool vocab_built_ = false;

  // Descriptor buffer for vocabulary building
  std::vector<cv::Mat> desc_buffer_;
  static constexpr int kVocabBuildThreshold = 100;

  // Keyframe database
  struct KeyframeEntry {
    int64_t id;
    Eigen::Matrix4f pose;
    BoVWSignature bow;
    cv::Mat descriptors;
  };
  std::map<int64_t, KeyframeEntry> kf_database_;
  mutable std::mutex mutex_database_;

  // Pose graph
  std::vector<PoseGraphEdge> pose_graph_edges_;
  std::mutex mutex_graph_;

  // Consecutive loop detection tracking
  int64_t last_loop_candidate_ = -1;
  int consecutive_detections_ = 0;
};

}  // namespace scaffold_chungs
