/**
 * Scaffold-ChunGS: Training loss types shared across trainer and loss modules.
 */

#pragma once

#include <torch/torch.h>

namespace scaffold_chungs {

struct TrainingLoss {
  torch::Tensor total;
  float l1_color = 0;
  float ssim_value = 0;
  float depth_l1 = 0;
  float isotropic = 0;
};

TrainingLoss computeLosses(
    const torch::Tensor& rendered_color,
    const torch::Tensor& rendered_depth,
    const torch::Tensor& gt_image,
    const torch::Tensor& gt_depth,
    const torch::Tensor& child_scaling,
    const torch::Tensor& mask,
    float lambda_dssim = 0.2f,
    float lambda_depth = 0.1f,
    float lambda_isotropic = 0.01f);

}  // namespace scaffold_chungs
