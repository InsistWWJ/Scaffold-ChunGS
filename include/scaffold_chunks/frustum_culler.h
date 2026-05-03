/**
 * Scaffold-ChunGS: Frustum culling for anchor-based scene management.
 *
 * Two-level culling:
 *   1. Chunk-level: AABB-frustum intersection test (fast, conservative).
 *   2. Anchor-level: per-anchor frustum test on visible chunks (precise).
 */

#pragma once

#include <torch/torch.h>
#include <Eigen/Core>
#include <vector>

#include "chunk_types.h"
#include "gaussian_model.h"

namespace scaffold_chungs {

class FrustumCuller {
 public:
  FrustumCuller() = default;

  /**
   * Compute which chunks intersect the camera frustum.
   *
   * Uses the standard frustum plane test on chunk AABBs.
   * Returns a sorted list of visible chunk IDs.
   */
  static std::vector<int64_t> cullChunks(
      const Eigen::Matrix4f& world_view_proj,
      const std::vector<ChunkCoord>& all_chunks,
      float chunk_size);

  /**
   * Per-anchor frustum test using the approximated bounding sphere of each anchor.
   * An anchor's sphere center = anchor position, radius = max offset * max scale.
   *
   * Returns a [N] bool tensor; True = anchor is visible.
   */
  static torch::Tensor cullAnchors(
      const GaussianModel& model,
      const Eigen::Vector3f& camera_center,
      const Eigen::Matrix4f& world_view_proj,
      const std::vector<int64_t>& visible_chunk_ids);

  /**
   * Build frustum planes from a projection-view matrix.
   * Returns 6 planes (near, far, left, right, top, bottom) in [A,B,C,D] format.
   */
  static torch::Tensor buildFrustumPlanes(const Eigen::Matrix4f& wvp);

  /**
   * Build a mask for anchors belonging to the given chunk IDs.
   */
  static torch::Tensor anchorMaskFromChunks(
      const GaussianModel& model,
      const std::vector<int64_t>& chunk_ids);

  /**
   * Test if an AABB intersects a set of frustum planes (conservative).
   */
  static bool aabbIntersectsFrustum(
      const AABB& aabb, const torch::Tensor& frustum_planes);
};

}  // namespace scaffold_chungs
