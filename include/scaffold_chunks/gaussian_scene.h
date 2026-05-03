/**
 * Scaffold-ChunGS: Scene container managing keyframes for training.
 */

#pragma once

#include <map>
#include <memory>
#include <mutex>

#include "gaussian_keyframe.h"

namespace scaffold_chungs {

class GaussianScene {
 public:
  GaussianScene() = default;

  void addKeyframe(std::shared_ptr<GaussianKeyframe> kf);
  std::shared_ptr<GaussianKeyframe> getKeyframe(int64_t fid);
  std::map<int64_t, std::shared_ptr<GaussianKeyframe>> getAllKeyframes() const;

  size_t size() const;
  float computeSceneRadius() const;

 private:
  std::map<int64_t, std::shared_ptr<GaussianKeyframe>> keyframes_;
  mutable std::mutex mutex_kfs_;
};

}  // namespace scaffold_chungs
