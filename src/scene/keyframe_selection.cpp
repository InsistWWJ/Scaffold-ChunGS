/**
 * Scaffold-ChunGS: Keyframe selection strategy implementation.
 *
 * Uses a loss-weighted probability distribution that favors recent
 * and high-loss keyframes during training.
 */

#include "scaffold_chunks/keyframe_selection.h"

#include <algorithm>
#include <iostream>

namespace scaffold_chungs {

KeyframeSelection::KeyframeSelection(std::shared_ptr<GaussianScene> scene)
    : scene_(scene) {
  std::random_device rd;
  rng_ = std::mt19937(rd());
}

std::shared_ptr<GaussianKeyframe> KeyframeSelection::getNextKeyframe() {
  auto all_kfs = scene_->getAllKeyframes();
  if (all_kfs.empty()) return nullptr;

  // Build selection weights from loss + usage
  std::vector<int64_t> fids;
  std::vector<float> weights;

  float total_weight = 0;
  for (const auto& [fid, kf] : all_kfs) {
    if (kf->remaining_times_of_use_ <= 0) continue;

    float loss = 1.0f;
    auto it = kf_losses_.find(fid);
    if (it != kf_losses_.end()) loss = it->second + 0.1f;

    int used = 0;
    auto uit = kf_used_times_.find(fid);
    if (uit != kf_used_times_.end()) used = uit->second;

    // Higher weight for higher loss + lower usage
    float weight = loss / (used + 1.0f);

    fids.push_back(fid);
    weights.push_back(weight);
    total_weight += weight;
  }

  if (fids.empty()) return nullptr;

  // Normalize
  for (auto& w : weights) w /= total_weight;

  // Sample
  std::discrete_distribution<int64_t> dist(weights.begin(), weights.end());
  int64_t idx = dist(rng_);
  int64_t selected_fid = fids[idx];

  auto selected = scene_->getKeyframe(selected_fid);
  if (selected) {
    selected->decrementTimesOfUse();
    kf_used_times_[selected_fid]++;
  }
  return selected;
}

void KeyframeSelection::recordLoss(int64_t fid, float loss) {
  // Exponential moving average of loss
  auto it = kf_losses_.find(fid);
  if (it != kf_losses_.end()) {
    kf_losses_[fid] = 0.9f * it->second + 0.1f * loss;
  } else {
    kf_losses_[fid] = loss;
  }
}

}  // namespace scaffold_chungs
