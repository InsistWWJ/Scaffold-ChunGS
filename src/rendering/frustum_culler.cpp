/**
 * Scaffold-ChunGS: Frustum culling for anchor visibility.
 */

#include "scaffold_chunks/frustum_culler.h"

#include <algorithm>

namespace scaffold_chungs {

// =============================================================================
// Frustum Plane Construction
// =============================================================================

torch::Tensor FrustumCuller::buildFrustumPlanes(const Eigen::Matrix4f& wvp) {
  // Extract plane equations from the combined world-view-projection matrix
  // Each plane: Ax + By + Cz + D <= 0 means "inside"
  //
  // wvp is column-major: wvp.col(i) is column i
  // The 6 frustum planes (left, right, bottom, top, near, far) in NDC

  Eigen::RowVector4f r0 = wvp.row(0);
  Eigen::RowVector4f r1 = wvp.row(1);
  Eigen::RowVector4f r2 = wvp.row(2);
  Eigen::RowVector4f r3 = wvp.row(3);

  // Convert to torch
  auto opt = torch::TensorOptions().dtype(torch::kFloat32);

  auto to_tensor = [&](const Eigen::RowVector4f& v) {
    return torch::tensor({v.x(), v.y(), v.z(), v.w()}, opt);
  };

  // Left:   r3 + r0
  // Right:  r3 - r0
  // Bottom: r3 + r1
  // Top:    r3 - r1
  // Near:   r3 + r2
  // Far:    r3 - r2
  torch::Tensor planes = torch::stack({
      to_tensor(r3 + r0),
      to_tensor(r3 - r0),
      to_tensor(r3 + r1),
      to_tensor(r3 - r1),
      to_tensor(r3 + r2),
      to_tensor(r3 - r2),
  }, 0);  // [6, 4]

  // Normalize each plane
  auto norms = torch::norm(planes.index({torch::indexing::Slice(),
                                         torch::indexing::Slice(0, 3)}),
                           2, 1, true);  // [6, 1]
  planes = planes / (norms + 1e-8f);

  return planes;
}

// =============================================================================
// AABB-Frustum Intersection
// =============================================================================

bool FrustumCuller::aabbIntersectsFrustum(
    const AABB& aabb, const torch::Tensor& frustum_planes) {
  // Conservative test: AABB is outside if it's completely on the negative
  // side of any plane. Requires frustum_planes to be on CPU.
  TORCH_CHECK(!frustum_planes.is_cuda(),
              "aabbIntersectsFrustum expects CPU-plane tensor (accessor-based)");
  auto pla = frustum_planes.accessor<float, 2>();

  for (int p = 0; p < 6; ++p) {
    float a = pla[p][0], b = pla[p][1], c = pla[p][2], d = pla[p][3];

    // Find the AABB vertex that would be furthest in the negative direction
    float px = (a > 0) ? aabb.max.x() : aabb.min.x();
    float py = (b > 0) ? aabb.max.y() : aabb.min.y();
    float pz = (c > 0) ? aabb.max.z() : aabb.min.z();

    // If this vertex is still outside (negative), reject the whole AABB
    if (a * px + b * py + c * pz + d < 0) {
      return false;
    }
  }
  return true;
}

// =============================================================================
// Chunk-Level Culling
// =============================================================================

std::vector<int64_t> FrustumCuller::cullChunks(
    const Eigen::Matrix4f& world_view_proj,
    const std::vector<ChunkCoord>& all_chunks,
    float chunk_size) {

  torch::Tensor planes = buildFrustumPlanes(world_view_proj);

  std::vector<int64_t> visible;
  for (const auto& coord : all_chunks) {
    AABB aabb = getChunkAABB(coord, chunk_size);
    if (aabbIntersectsFrustum(aabb, planes)) {
      visible.push_back(encodeChunkCoord(coord));
    }
  }

  // Sort for cache coherence
  std::sort(visible.begin(), visible.end());

  return visible;
}

// =============================================================================
// Anchor-Level Culling
// =============================================================================

torch::Tensor FrustumCuller::cullAnchors(
    const GaussianModel& model,
    const Eigen::Vector3f& camera_center,
    const Eigen::Matrix4f& world_view_proj,
    const std::vector<int64_t>& visible_chunk_ids) {

  torch::Tensor chunk_mask = anchorMaskFromChunks(model, visible_chunk_ids);
  const auto& anchor = model.anchor();
  auto device = anchor.device();
  int64_t N = anchor.size(0);

  torch::Tensor visible_mask = torch::zeros({N},
      torch::TensorOptions().dtype(torch::kBool).device(device));

  if (!chunk_mask.any().item<bool>()) {
    return visible_mask;
  }

  // Indices of anchors belonging to visible chunks
  torch::Tensor chunk_indices = torch::where(chunk_mask)[0];  // [K]

  // Per-anchor sphere radius (approximated from max scale axis)
  const auto& scaling = model.anchorScaling();
  torch::Tensor radius = scaling.index({torch::indexing::Slice(),
                                         torch::indexing::Slice(0, 3)})
      .abs().max(1).values * model.getNumOffsets();  // [N]

  // Gather visible subset
  torch::Tensor selected_xyz = anchor.index({chunk_indices});   // [K, 3]
  torch::Tensor selected_r = radius.index({chunk_indices});      // [K]

  // Build frustum planes, transfer coefficients to GPU
  torch::Tensor planes = buildFrustumPlanes(world_view_proj).to(device);  // [6, 4]

  // Batch sphere-vs-frustum test on GPU:
  //   signed distance to plane = dot(xyz, normal) + offset
  //   anchor is inside if distance >= -radius for all 6 planes
  torch::Tensor norms = planes.index({torch::indexing::Slice(),
                                      torch::indexing::Slice(0, 3)});    // [6, 3]
  torch::Tensor offsets = planes.index({torch::indexing::Slice(), 3});   // [6]
  torch::Tensor dists = torch::matmul(selected_xyz, norms.t()) + offsets.unsqueeze(0);  // [K, 6]

  torch::Tensor inside = (dists >= -selected_r.unsqueeze(1)).all(1);  // [K] bool

  // Scatter results back to the full-size mask
  visible_mask.index_put_({chunk_indices}, inside);

  return visible_mask;
}

// =============================================================================
// Anchor Mask from Chunks
// =============================================================================

torch::Tensor FrustumCuller::anchorMaskFromChunks(
    const GaussianModel& model,
    const std::vector<int64_t>& chunk_ids) {

  if (chunk_ids.empty()) {
    return torch::zeros({model.getNumAnchors()},
        torch::TensorOptions().dtype(torch::kBool).device(model.deviceType()));
  }

  auto opt = torch::TensorOptions().dtype(torch::kInt64).device(model.deviceType());
  torch::Tensor chunk_tensor = torch::tensor(chunk_ids, opt);
  return model.createAnchorMaskFromChunks(chunk_tensor);
}

}  // namespace scaffold_chungs
