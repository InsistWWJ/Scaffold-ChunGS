/**
 * Scaffold-ChunGS: Training keyframe selection strategy.
 */

#pragma once

#include <map>
#include <memory>
#include <random>

#include "gaussian_keyframe.h"
#include "gaussian_scene.h"

namespace scaffold_chungs {

class KeyframeSelection {
 public:
  KeyframeSelection(std::shared_ptr<GaussianScene> scene);

  /**
   * Select the next keyframe for training using a loss-weighted
   * distribution that favors recent and high-loss keyframes.
   */
  std::shared_ptr<GaussianKeyframe> getNextKeyframe();

  /**
   * Record the loss for a keyframe (used to update selection weights).
   */
  void recordLoss(int64_t fid, float loss);

 private:
  std::shared_ptr<GaussianScene> scene_;
  std::map<int64_t, float> kf_losses_;
  std::map<int64_t, int> kf_used_times_;
  std::mt19937 rng_;
};

}  // namespace scaffold_chungs
