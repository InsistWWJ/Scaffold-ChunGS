/**
 * Scaffold-ChunGS: Fusion of Scaffold-GS anchor representation
 * with DiskChunGS chunk-based out-of-core memory management.
 *
 * Copyright (C) 2025 Scaffold-ChunGS
 * Licensed under GPLv3.
 */

#pragma once

#include <torch/torch.h>

#include <Eigen/Core>
#include <cstdint>
#include <memory>
#include <unordered_map>

// =============================================================================
// Spatial Data Structures
// =============================================================================

struct AABB {
  Eigen::Vector3f min;
  Eigen::Vector3f max;

  AABB() : min(Eigen::Vector3f::Zero()), max(Eigen::Vector3f::Zero()) {}
  AABB(const Eigen::Vector3f& min_val, const Eigen::Vector3f& max_val)
      : min(min_val), max(max_val) {}
};

struct ChunkCoord {
  int64_t x, y, z;

  bool operator==(const ChunkCoord& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  bool operator<(const ChunkCoord& other) const {
    if (x != other.x) return x < other.x;
    if (y != other.y) return y < other.y;
    return z < other.z;
  }
};

struct ChunkCoordHash {
  std::size_t operator()(const ChunkCoord& coord) const {
    std::size_t h1 = std::hash<int64_t>{}(coord.x);
    std::size_t h2 = std::hash<int64_t>{}(coord.y);
    std::size_t h3 = std::hash<int64_t>{}(coord.z);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

// =============================================================================
// Chunk Coordinate Utilities
// =============================================================================

inline ChunkCoord getChunkCoord(const Eigen::Vector3f& position,
                                float chunk_size) {
  float half_chunk = chunk_size * 0.5f;
  return ChunkCoord{
      static_cast<int64_t>(std::floor((position.x() + half_chunk) / chunk_size)),
      static_cast<int64_t>(std::floor((position.y() + half_chunk) / chunk_size)),
      static_cast<int64_t>(std::floor((position.z() + half_chunk) / chunk_size))};
}

inline Eigen::Vector3f getChunkCenter(const ChunkCoord& coord,
                                      float chunk_size) {
  return Eigen::Vector3f(coord.x * chunk_size, coord.y * chunk_size,
                         coord.z * chunk_size);
}

inline AABB getChunkAABB(const ChunkCoord& coord, float chunk_size) {
  float half = chunk_size * 0.5f;
  Eigen::Vector3f c(coord.x * chunk_size, coord.y * chunk_size,
                    coord.z * chunk_size);
  return AABB(c - Eigen::Vector3f::Constant(half),
              c + Eigen::Vector3f::Constant(half));
}

// =============================================================================
// 64-bit Chunk Coordinate Encoding
// =============================================================================

inline int64_t encodeChunkCoord(const ChunkCoord& coord) {
  const int64_t OFFSET = 1048576;  // 2^20
  int64_t x = coord.x + OFFSET;
  int64_t y = coord.y + OFFSET;
  int64_t z = coord.z + OFFSET;
  return x * (1LL << 42) + y * (1LL << 21) + z;
}

inline ChunkCoord decodeChunkCoord(int64_t chunk_id) {
  const int64_t OFFSET = 1048576;
  const int64_t FIELD_SIZE = 1LL << 21;
  int64_t z = (chunk_id % FIELD_SIZE) - OFFSET;
  int64_t y = ((chunk_id / FIELD_SIZE) % FIELD_SIZE) - OFFSET;
  int64_t x = (chunk_id / (FIELD_SIZE * FIELD_SIZE)) - OFFSET;
  return ChunkCoord{x, y, z};
}

// =============================================================================
// Torch Tensor Chunk Utilities
// =============================================================================

inline torch::Tensor encodeChunkCoordsTensor(const torch::Tensor& chunk_coords) {
  const int64_t OFFSET = 1048576;
  auto x = chunk_coords.index({torch::indexing::Slice(), 0}) + OFFSET;
  auto y = chunk_coords.index({torch::indexing::Slice(), 1}) + OFFSET;
  auto z = chunk_coords.index({torch::indexing::Slice(), 2}) + OFFSET;
  return x * (1LL << 42) + y * (1LL << 21) + z;
}

inline torch::Tensor decodeChunkCoordsTensor(const torch::Tensor& encoded_ids) {
  const int64_t OFFSET = 1048576;
  const int64_t FIELD_SIZE = 1LL << 21;
  torch::Tensor z = (encoded_ids % FIELD_SIZE) - OFFSET;
  torch::Tensor y = ((encoded_ids / FIELD_SIZE) % FIELD_SIZE) - OFFSET;
  torch::Tensor x = (encoded_ids / (FIELD_SIZE * FIELD_SIZE)) - OFFSET;
  return torch::stack({x, y, z}, 1);
}

inline torch::Tensor computeChunkIds(const torch::Tensor& positions,
                                     float chunk_size) {
  torch::NoGradGuard no_grad;
  float half_chunk = chunk_size * 0.5f;
  torch::Tensor shifted = positions + half_chunk;
  torch::Tensor coords = torch::floor(shifted / chunk_size).to(torch::kInt64);
  return encodeChunkCoordsTensor(coords);
}

// =============================================================================
// Binary Tensor Serialization Header
// =============================================================================

struct TensorHeader {
  uint32_t dims = 0;
  uint32_t sizes[8] = {};
  uint32_t dtype = 0;
  uint64_t data_size = 0;
};
