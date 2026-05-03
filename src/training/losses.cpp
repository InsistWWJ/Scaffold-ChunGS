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

namespace scaffold_chungs {

// =============================================================================
// SSIM Loss (simplified)
// =============================================================================

static torch::Tensor gaussianWindow(int size, float sigma) {
  auto x = torch::arange(size, torch::TensorOptions().dtype(torch::kFloat32));
  x = x - (size - 1) / 2.0f;
  x = torch::exp(-0.5f * (x / sigma).pow(2));
  x = x / x.sum();
  return x.outer(x);
}

static torch::Tensor ssim(const torch::Tensor& img1,
                          const torch::Tensor& img2,
                          float C1 = 0.01f * 0.01f,
                          float C2 = 0.03f * 0.03f) {
  // img1, img2: [C, H, W]
  int window_size = 11;
  float sigma = 1.5f;
  auto window = gaussianWindow(window_size, sigma)
      .unsqueeze(0).unsqueeze(0)
      .to(img1.device());  // [1, 1, 11, 11]
  window = window / window.sum();

  // Per-channel SSIM with 2D convolution
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
// Compute Training Losses
// =============================================================================

struct TrainingLoss {
  float l1_color = 0;
  float ssim_value = 0;
  float depth_l1 = 0;
  float isotropic = 0;
  float total = 0;
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

  // Ensure same device
  auto device = rendered_color.device();
  auto gt = gt_image.to(device);
  auto valid_mask = mask.to(device);

  if (gt.size(0) == 0 || rendered_color.size(0) == 0) {
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

  // Apply mask
  torch::Tensor mask_float = valid_mask.to(torch::kFloat32);

  // ---- L1 Color Loss ----
  torch::Tensor diff = (rendered_color - gt).abs();
  diff = diff * mask_float.unsqueeze(0);
  loss.l1_color = diff.sum().item<float>() /
      (mask_float.sum().item<float>() * 3.0f + 1e-8f);

  // ---- SSIM Loss ----
  torch::Tensor masked_render = rendered_color * mask_float.unsqueeze(0);
  torch::Tensor masked_gt = gt * mask_float.unsqueeze(0);
  torch::Tensor ssim_val = ssim(masked_render, masked_gt);
  loss.ssim_value = 1.0f - ssim_val.item<float>();

  // ---- Depth L1 Loss ----
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
    loss.depth_l1 = d_diff.sum().item<float>() /
        (depth_mask.sum().item<float>() + 1e-8f);
  }

  // ---- Isotropic Loss ----
  if (child_scaling.defined() && child_scaling.numel() > 0) {
    loss.isotropic = isotropicLoss(child_scaling).item<float>();
  }

  // ---- Total ----
  float l1_weight = 1.0f - lambda_dssim;
  loss.total = l1_weight * loss.l1_color +
               lambda_dssim * loss.ssim_value +
               lambda_depth * loss.depth_l1 +
               lambda_isotropic * loss.isotropic;

  return loss;
}

// =============================================================================
// Exposure Correction (optional)
// =============================================================================

torch::Tensor applyExposureTransform(
    const torch::Tensor& colors,
    float exposure_a,
    float exposure_b) {
  // colors: [3, H, W] or [M, 3]
  return torch::exp(exposure_a) * colors + exposure_b;
}

}  // namespace scaffold_chungs
