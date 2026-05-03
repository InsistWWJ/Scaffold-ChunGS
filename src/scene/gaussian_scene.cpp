/**
 * Scaffold-ChunGS: Scene container implementation.
 */

#include "scaffold_chunks/gaussian_scene.h"

namespace scaffold_chungs {

void GaussianScene::addKeyframe(std::shared_ptr<GaussianKeyframe> kf) {
  std::lock_guard<std::mutex> lock(mutex_kfs_);
  keyframes_[kf->fid()] = kf;
}

std::shared_ptr<GaussianKeyframe> GaussianScene::getKeyframe(int64_t fid) {
  std::lock_guard<std::mutex> lock(mutex_kfs_);
  auto it = keyframes_.find(fid);
  if (it != keyframes_.end()) return it->second;
  return nullptr;
}

std::map<int64_t, std::shared_ptr<GaussianKeyframe>>
GaussianScene::getAllKeyframes() const {
  std::lock_guard<std::mutex> lock(mutex_kfs_);
  return keyframes_;
}

size_t GaussianScene::size() const {
  std::lock_guard<std::mutex> lock(mutex_kfs_);
  return keyframes_.size();
}

float GaussianScene::computeSceneRadius() const {
  std::lock_guard<std::mutex> lock(mutex_kfs_);
  if (keyframes_.empty()) return 1.0f;

  Eigen::Vector3f centroid(0, 0, 0);
  for (const auto& [id, kf] : keyframes_) {
    centroid += kf->getTranslation();
  }
  centroid /= keyframes_.size();

  float max_dist = 0;
  for (const auto& [id, kf] : keyframes_) {
    float d = (kf->getTranslation() - centroid).norm();
    if (d > max_dist) max_dist = d;
  }
  return max_dist;
}

}  // namespace scaffold_chungs
