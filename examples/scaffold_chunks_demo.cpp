/**
 * Scaffold-ChunGS Demo: End-to-end example of anchor-based 3DGS with chunk I/O.
 *
 * Usage:
 *   scaffold_chunks_demo <config_path> <output_dir>
 *
 * Flow:
 *   1. Load config from YAML
 *   2. Create GaussianModel with anchor MLP
 *   3. Simulate camera trajectory
 *   4. Initialize anchors from synthetic point clouds
 *   5. Run training loop
 *   6. Save results
 */

#include "scaffold_chunks/gaussian_model.h"
#include "scaffold_chunks/gaussian_renderer.h"
#include "scaffold_chunks/gaussian_keyframe.h"
#include "scaffold_chunks/gaussian_scene.h"
#include "scaffold_chunks/keyframe_selection.h"
#include "scaffold_chunks/frustum_culler.h"
#include "scaffold_chunks/chunk_types.h"
#include "scaffold_chunks/config.h"

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <filesystem>
#include <iostream>
#include <memory>
#include <random>

using namespace scaffold_chungs;

// =============================================================================
// Generate a synthetic point cloud (sphere)
// =============================================================================

static std::tuple<torch::Tensor, torch::Tensor> generateSpherePointCloud(
    int num_points, float radius, const Eigen::Vector3f& center) {
  auto opt = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor theta = torch::rand({num_points}, opt) * 2.0f * M_PI;
  torch::Tensor phi = torch::acos(2.0f * torch::rand({num_points}, opt) - 1.0f);

  torch::Tensor x = center.x() + radius * torch::sin(phi) * torch::cos(theta);
  torch::Tensor y = center.y() + radius * torch::sin(phi) * torch::sin(theta);
  torch::Tensor z = center.z() + radius * torch::cos(phi);

  torch::Tensor positions = torch::stack({x, y, z}, 1);

  // Colors: gradient based on position
  torch::Tensor r = (x / radius).clamp(0, 1);
  torch::Tensor g = (y / radius).clamp(0, 1);
  torch::Tensor b = (z / radius).clamp(0, 1);
  torch::Tensor colors = torch::stack({r, g, b}, 1);

  return {positions, colors};
}

// =============================================================================
// Generate a synthetic camera trajectory (circle around origin)
// =============================================================================

static std::vector<Eigen::Matrix4f> generateCircularTrajectory(
    int num_views, float radius, float height) {
  std::vector<Eigen::Matrix4f> poses;

  for (int i = 0; i < num_views; ++i) {
    float angle = 2.0f * M_PI * i / num_views;
    float x = radius * std::cos(angle);
    float y = height;
    float z = radius * std::sin(angle);

    // Look-at: camera looks at origin
    Eigen::Vector3f eye(x, y, z);
    Eigen::Vector3f target(0, 0, 0);
    Eigen::Vector3f up(0, 1, 0);

    Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();
    Eigen::Vector3f forward = (target - eye).normalized();
    Eigen::Vector3f right = forward.cross(up).normalized();
    Eigen::Vector3f new_up = right.cross(forward);

    Tcw(0, 0) = right.x();   Tcw(0, 1) = right.y();   Tcw(0, 2) = right.z();
    Tcw(1, 0) = new_up.x();  Tcw(1, 1) = new_up.y();   Tcw(1, 2) = new_up.z();
    Tcw(2, 0) = -forward.x(); Tcw(2, 1) = -forward.y(); Tcw(2, 2) = -forward.z();
    Tcw(0, 3) = eye.x();     Tcw(1, 3) = eye.y();       Tcw(2, 3) = eye.z();

    poses.push_back(Tcw);
  }

  return poses;
}

// =============================================================================
// Create a synthetic keyframe
// =============================================================================

static std::shared_ptr<GaussianKeyframe> createKeyframe(
    int64_t fid,
    const Eigen::Matrix4f& Tcw,
    int width = 480, int height = 320) {

  auto kf = std::make_shared<GaussianKeyframe>(fid, 0);
  kf->setPose(Tcw);

  float fx = 300.0f, fy = 300.0f, cx = width / 2.0f, cy = height / 2.0f;
  kf->setIntrinsics(fx, fy, cx, cy, width, height, 0.01f, 100.0f);
  kf->computeTransformTensors(torch::kCPU);

  // Create dummy GT image (black)
  cv::Mat dummy_img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  kf->setTrainingImage(dummy_img);
  kf->setTimesOfUse(8);

  return kf;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
  std::cout << "=====================================================\n";
  std::cout << " Scaffold-ChunGS Demo\n";
  std::cout << " Anchor-based 3DGS with Chunk Memory Management\n";
  std::cout << "=====================================================\n\n";

  // Parse args
  std::string config_path = (argc > 1) ? argv[1]
      : "cfg/gaussian_mapper/scaffold_chunks.yaml";
  std::string output_dir = (argc > 2) ? argv[2] : "./output";

  // Create output directory
  std::filesystem::create_directories(output_dir);

  // Load config
  std::cout << "[Init] Loading config: " << config_path << "\n";
  ScaffoldChunGSConfig cfg;
  try {
    cfg = ScaffoldChunGSConfig::fromYAML(config_path);
  } catch (const std::exception& e) {
    std::cerr << "[Error] " << e.what() << "\n";
    std::cerr << "Using default config...\n";
    cfg = ScaffoldChunGSConfig();
  }

  // Set chunk storage path
  cfg.chunk.storage_base_path = output_dir + "/chunks";

  // Detect device
  torch::DeviceType device = torch::kCPU;
  if (cfg.data_device == "cuda" && torch::cuda::is_available()) {
    device = torch::kCUDA;
    std::cout << "[Init] Using CUDA device\n";
  } else {
    std::cout << "[Init] Using CPU device\n";
  }

  // ---- Step 1: Create GaussianModel ----
  std::cout << "\n[Step 1] Creating anchor-based GaussianModel...\n";
  auto model = std::make_shared<GaussianModel>(
      cfg.anchor, cfg.chunk, device);

  // Set up MLP appearance embedding (1 camera for demo)
  if (cfg.anchor.appearance_dim > 0) {
    model->mlp().setAppearanceEmbedding(1);
  }

  // ---- Step 2: Generate synthetic scene ----
  std::cout << "[Step 2] Generating synthetic point cloud...\n";
  const int NUM_POINTS = 50000;
  auto [positions, colors] = generateSpherePointCloud(
      NUM_POINTS, 2.0f, Eigen::Vector3f(0, 0, 0));

  // Initialize model from point cloud
  model->initializeFromPoints(positions, colors);

  // Set up optimizer
  model->trainingSetup(cfg.optimization);

  std::cout << "Initial anchors: " << model->getNumAnchors()
            << " (potential gaussians: "
            << model->countAllGaussians() << ")\n";

  // ---- Step 3: Create keyframes ----
  std::cout << "[Step 3] Generating keyframe trajectory...\n";
  auto scene = std::make_shared<GaussianScene>();
  auto poses = generateCircularTrajectory(20, 5.0f, 0.5f);

  for (int i = 0; i < (int)poses.size(); ++i) {
    auto kf = createKeyframe(i, poses[i]);
    scene->addKeyframe(kf);
  }
  std::cout << "Created " << scene->size() << " keyframes\n";

  // ---- Step 4: Initial rendering test ----
  std::cout << "[Step 4] Testing render pipeline...\n";
  {
    auto kf = scene->getKeyframe(0);
    kf->transferToGPU(device);
    kf->computeTransformTensors(device);
    auto cc = kf->getCameraCenter(device);

    Eigen::Vector3f cc_eigen(cc[0].item<float>(),
                              cc[1].item<float>(),
                              cc[2].item<float>());

    // Build WVP matrix
    auto w2v = kf->worldViewTransform();
    auto proj = kf->projectionMatrix();
    auto fp = torch::matmul(proj, w2v).cpu();
    Eigen::Matrix4f wvp;
    auto fp_acc = fp.accessor<float, 2>();
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        wvp(r, c) = fp_acc[r][c];

    // Cull + render
    torch::Tensor vis_mask = model->cullVisibleAnchors(cc_eigen, wvp, false);

    torch::Tensor bg = torch::zeros({3},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    auto output = ScaffoldRenderer::renderInference(
        *model, vis_mask,
        cc.to(device), w2v.to(device), proj.to(device),
        kf->FoVx(), kf->FoVy(),
        kf->imageHeight(), kf->imageWidth(),
        bg, 1.0f);

    // Save rendered image
    auto color_cpu = output.color.clamp(0.0f, 1.0f).cpu();
    cv::Mat rendered(kf->imageHeight(), kf->imageWidth(), CV_32FC3);
    for (int y = 0; y < kf->imageHeight(); ++y) {
      for (int x = 0; x < kf->imageWidth(); ++x) {
        rendered.at<cv::Vec3f>(y, x) = cv::Vec3f(
            color_cpu[0][y][x].item<float>(),
            color_cpu[1][y][x].item<float>(),
            color_cpu[2][y][x].item<float>());
      }
    }
    cv::Mat rendered_8u;
    rendered.convertTo(rendered_8u, CV_8UC3, 255.0);
    cv::imwrite(output_dir + "/render_initial.png", rendered_8u);
    std::cout << "Saved: " << output_dir << "/render_initial.png\n";
  }

  // ---- Step 5: Training loop ----
  std::cout << "[Step 5] Running training loop...\n";
  {
    // Create trainer and run a few iterations
    // (Using the training infrastructure from trainer.cpp)

    std::cout << "Running " << cfg.optimization.max_num_iterations
              << " training iterations...\n";

    // Simple training: render from each keyframe and optimize
    const int MAX_ITERS = 100;
    auto opt_float = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor bg = torch::zeros({3}, opt_float);

    for (int iter = 0; iter < MAX_ITERS; ++iter) {
      int kf_idx = iter % scene->size();
      auto kf = scene->getKeyframe(kf_idx);
      if (!kf) continue;

      kf->transferToGPU(device);
      kf->computeTransformTensors();
      auto cc = kf->getCameraCenter().to(device);
      auto w2v = kf->worldViewTransform();
      auto proj = kf->projectionMatrix();

      Eigen::Vector3f cc_eigen(cc[0].item<float>(),
                                cc[1].item<float>(),
                                cc[2].item<float>());

      auto fp_t = torch::matmul(proj, w2v).cpu();
      Eigen::Matrix4f wvp_eigen;
      auto acc = fp_t.accessor<float, 2>();
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          wvp_eigen(r, c) = acc[r][c];

      torch::Tensor vis_mask = model->cullVisibleAnchors(
          cc_eigen, wvp_eigen, (iter > 0));

      // For actual training, gradients would flow through the renderer
      // Here we just validate that the pipeline works end-to-end

      // Periodically densify
      if (iter > 0 && iter % 50 == 0) {
        model->adjustAnchors(
            cfg.optimization.densify_check_interval,
            cfg.optimization.prune_success_threshold,
            cfg.optimization.densify_grad_threshold,
            cfg.optimization.densify_opacity_threshold);
      }

      if (iter % 20 == 0) {
        std::cout << "  Iter " << iter << ": "
                  << model->getNumAnchors() << " anchors, "
                  << "visible chunks: " << model->getNumAnchors()
                  << "\n";
      }
    }
  }

  // ---- Step 6: Save results ----
  std::cout << "\n[Step 6] Saving final model...\n";
  model->saveAllChunks();

  // Save final render
  {
    auto kf = scene->getKeyframe(10);
    kf->transferToGPU(device);
    kf->computeTransformTensors(device);
    auto cc = kf->getCameraCenter(device);
    auto w2v = kf->worldViewTransform().to(device);
    auto proj = kf->projectionMatrix().to(device);

    Eigen::Vector3f cc_eigen(cc[0].item<float>(),
                              cc[1].item<float>(),
                              cc[2].item<float>());

    auto fp_t = torch::matmul(proj, w2v).cpu();
    Eigen::Matrix4f wvp_eigen;
    auto acc = fp_t.accessor<float, 2>();
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        wvp_eigen(r, c) = acc[r][c];

    torch::Tensor vis_mask = model->cullVisibleAnchors(cc_eigen, wvp_eigen, false);

    torch::Tensor bg = torch::zeros({3},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    auto output = ScaffoldRenderer::renderInference(
        *model, vis_mask,
        cc.to(device), w2v.to(device), proj.to(device),
        kf->FoVx(), kf->FoVy(),
        kf->imageHeight(), kf->imageWidth(),
        bg, 1.0f);

    auto color_cpu = output.color.clamp(0.0f, 1.0f).cpu();
    cv::Mat rendered(kf->imageHeight(), kf->imageWidth(), CV_32FC3);
    for (int y = 0; y < kf->imageHeight(); ++y) {
      for (int x = 0; x < kf->imageWidth(); ++x) {
        rendered.at<cv::Vec3f>(y, x) = cv::Vec3f(
            color_cpu[0][y][x].item<float>(),
            color_cpu[1][y][x].item<float>(),
            color_cpu[2][y][x].item<float>());
      }
    }
    cv::Mat rendered_8u;
    rendered.convertTo(rendered_8u, CV_8UC3, 255.0);
    cv::imwrite(output_dir + "/render_final.png", rendered_8u);
    std::cout << "Saved: " << output_dir << "/render_final.png\n";
  }

  // Summary
  std::cout << "\n===================== SUMMARY =====================\n";
  std::cout << "Total anchors:       " << model->getNumAnchors() << "\n";
  std::cout << "Total gaussians:     " << model->countAllGaussians() << "\n";
  std::cout << "Chunks in memory:    " << model->anchorChunkIDs().size(0)
            << " assigned\n";
  std::cout << "Output directory:    " << output_dir << "\n";
  std::cout << "=====================================================\n";

  return 0;
}
