/**
 * Scaffold-ChunGS: Training loss functions.
 *
 * Combines:
 *   - L1 color loss (primary)
 *   - SSIM loss (structural similarity)
 *   - Isotropic regularization (encourages spherical Gaussians)
 *   - Optional depth loss
 */

#include "scaffold_chunks/gaussian_renderer.h"

#include <torch/torch.h>
#include <iostream>
#include <mutex>

namespace scaffold_chungs {

// =============================================================================
// SSIM Loss (simplified)
// =============================================================================

// Cached Gaussian kernel — computed once, reused across calls
static torch::Tensor gaussianWindowCached(int size, float sigma) {
  static torch::Tensor cached;
  static std::mutex mtx;
  static int cached_size = 0;
  static float cached_sigma = 0.f;

  std::lock_guard<std::mutex> lock(mtx);
  if (cached.defined() && cached_size == size && cached_sigma == sigma) {
    return cached;
  }

  auto x = torch::arange(size, torch::TensorOptions().dtype(torch::kFloat32));
  x = x - (size - 1) / 2.0f;
  x = torch::exp(-0.5f * (x / sigma).pow(2));
  x = x / x.sum();
  cached = x.outer(x);
  cached_size = size;
  cached_sigma = sigma;
  return cached;
}

static torch::Tensor ssim(const torch::Tensor& img1,
                          const torch::Tensor& img2,
                          float C1 = 0.01f * 0.01f,
                          float C2 = 0.03f * 0.03f) {
  int window_size = 11;
  float sigma = 1.5f;
  auto window = gaussianWindowCached(window_size, sigma)
      .unsqueeze(0).unsqueeze(0)
      .to(img1.device());
  window = window / window.sum();

  torch::Tensor mu1 = torch::conv2d(
      img1.unsqueeze(1), window,
      torch::Conv2dOptions().padding(window_size / 2).groups(1));
  torch::Tensor mu2 = torch::conv2d(
      img2.unsqueeze(1), window,
      torch::Conv2dOptions().padding(window_size / 2).groups(1));

  torch::Tensor mu1_sq = mu1.pow(2);
  torch::Tensor mu2_sq = mu2.pow(2);
  torch::Tensor mu1_mu2 = mu1 * mu2;

  torch::Tensor sigma1_sq = torch::conv2d(
      (img1.unsqueeze(1)).pow(2), window,
      torch::Conv2dOptions().padding(window_size / 2).groups(1)) - mu1_sq;
  torch::Tensor sigma2_sq = torch::conv2d(
      (img2.unsqueeze(1)).pow(2), window,
      torch::Conv2dOptions().padding(window_size / 2).groups(1)) - mu2_sq;
  torch::Tensor sigma12 = torch::conv2d(
      (img1 * img2).unsqueeze(1), window,
      torch::Conv2dOptions().padding(window_size / 2).groups(1)) - mu1_mu2;

  torch::Tensor ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) /
      ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

  return ssim_map.mean();
}

// =============================================================================
// Isotropic Loss
// =============================================================================

static torch::Tensor isotropicLoss(const torch::Tensor& scaling) {
  // scaling: [M, 3] — per-axis log-scales
  // Penalize non-uniformity: encourage all 3 axes to have the same scale
  torch::Tensor mean_scale = scaling.mean(1, true);  // [M, 1]
  torch::Tensor diff = scaling - mean_scale;          // [M, 3]
  return diff.pow(2).mean();
}

// =============================================================================
// Compute Training Losses (returns tensors for autograd)
// =============================================================================

struct TrainingLoss {
  torch::Tensor total;       // combined loss tensor (for backward())
  float l1_color = 0;       // scalar values for logging
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
    float lambda_isotropic = 0.01f) {

  TrainingLoss loss;
  auto device = rendered_color.device();
  auto gt = gt_image.to(device);
  auto valid_mask = mask.to(device);

  if (gt.size(0) == 0 || rendered_color.size(0) == 0) {
    loss.total = torch::tensor(0.f, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    return loss;
  }

  // Resize if needed
  if (gt.size(1) != rendered_color.size(1) || gt.size(2) != rendered_color.size(2)) {
    gt = torch::nn::functional::interpolate(
        gt.unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{rendered_color.size(1),
                                       rendered_color.size(2)})
            .mode(torch::kBilinear)
            .align_corners(false)).squeeze(0);
    valid_mask = torch::nn::functional::interpolate(
        valid_mask.unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{rendered_color.size(1),
                                       rendered_color.size(2)})
            .mode(torch::kNearest)).squeeze(0).to(torch::kBool);
  }

  torch::Tensor mask_float = valid_mask.to(torch::kFloat32);

  // ---- L1 Color Loss ----
  torch::Tensor diff = (rendered_color - gt).abs();
  diff = diff * mask_float.unsqueeze(0);
  torch::Tensor l1_tensor = diff.sum() / (mask_float.sum() * 3.0f + 1e-8f);

  // ---- SSIM Loss ----
  torch::Tensor masked_render = rendered_color * mask_float.unsqueeze(0);
  torch::Tensor masked_gt = gt * mask_float.unsqueeze(0);
  torch::Tensor ssim_val = ssim(masked_render, masked_gt);
  torch::Tensor ssim_tensor = 1.0f - ssim_val;

  // ---- Depth L1 Loss ----
  torch::Tensor d_tensor = torch::tensor(0.f,
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  if (gt_depth.defined() && gt_depth.numel() > 0) {
    auto gt_d = gt_depth.to(device);
    if (gt_d.size(1) != rendered_depth.size(1) || gt_d.size(2) != rendered_depth.size(2)) {
      gt_d = torch::nn::functional::interpolate(
          gt_d.unsqueeze(0),
          torch::nn::functional::InterpolateFuncOptions()
              .size(std::vector<int64_t>{rendered_depth.size(1),
                                         rendered_depth.size(2)})
              .mode(torch::kBilinear)
              .align_corners(false)).squeeze(0);
    }
    torch::Tensor depth_mask = (gt_d > 0.01f).to(torch::kFloat32);
    torch::Tensor d_diff = (rendered_depth - gt_d).abs();
    d_diff = d_diff * depth_mask;
    d_tensor = d_diff.sum() / (depth_mask.sum() + 1e-8f);
  }

  // ---- Isotropic Loss ----
  torch::Tensor iso_tensor = torch::tensor(0.f,
      torch::TensorOptions().dtype(torch::kFloat32).device(device));
  if (child_scaling.defined() && child_scaling.numel() > 0) {
    iso_tensor = isotropicLoss(child_scaling);
  }

  // ---- Combined total (as tensor, for autograd) ----
  float l1_weight = 1.0f - lambda_dssim;
  loss.total = l1_weight * l1_tensor +
               lambda_dssim * ssim_tensor +
               lambda_depth * d_tensor +
               lambda_isotropic * iso_tensor;

  // Detached scalars for logging (does not break autograd chain)
  loss.l1_color = l1_tensor.item<float>();
  loss.ssim_value = ssim_tensor.item<float>();
  loss.depth_l1 = d_tensor.item<float>();
  loss.isotropic = iso_tensor.item<float>();

  return loss;
}

// =============================================================================
// Exposure Correction (optional)
// =============================================================================

torch::Tensor applyExposureTransform(
    const torch::Tensor& colors,
    float exposure_a,
    float exposure_b) {
  return torch::exp(exposure_a) * colors + exposure_b;
}

}  // namespace scaffold_chungs
