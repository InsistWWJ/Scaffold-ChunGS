# Scaffold-ChunGS: Compressed 3D Gaussian SLAM with Chunk-Based Out-of-Core Memory Management

**Scaffold-ChunGS** fuses two state-of-the-art 3D Gaussian SLAM techniques:

| Component | Source | Key Innovation |
|-----------|--------|---------------|
| **Anchor-Based Scene Representation** | [Compact_GSSLAM](https://github.com/dtc111111/Compact_GSSLAM) (IJCV 2025) | Scaffold-GS: voxelized anchors with MLP-decoded child Gaussians — 5–10× memory compression |
| **Chunk-Based Memory Management** | [DiskChunGS](https://github.com/leggedrobotics/DiskChunGS) (IEEE RA-L 2026) | Spatial chunk partitioning with LRU eviction to disk — supports arbitrarily large scenes |

By combining these techniques in a unified C++/LibTorch framework, Scaffold-ChunGS achieves **higher reconstruction density per GPU byte** than standard 3DGS while preserving **unbounded scene scalability** through out-of-core chunk I/O.

<p align="center">
  <b>C++17 &nbsp;|&nbsp; CUDA &nbsp;|&nbsp; LibTorch &nbsp;|&nbsp; OpenCV &nbsp;|&nbsp; Eigen3</b>
</p>

---

## Architecture Overview

```
                   ┌──────────────────────────────┐
 RGB-D / Stereo ──▶│   GaussianKeyframe (pose +   │
                   │   intrinsics + image pyramid) │
                   └──────────────┬───────────────┘
                                  │
                   ┌──────────────▼───────────────┐
                   │     FrustumCuller             │
                   │  Chunk-level AABB test ──▶    │
                   │  Anchor-level sphere test     │
                   └──────────────┬───────────────┘
                                  │ visible anchor mask [N] bool
                   ┌──────────────▼───────────────┐
                   │       AnchorMLP               │
                   │  anchor_feat [V,D] + ob_view  │
                   │       │                       │
                   │  ┌────┼────┬──────────┐       │
                   │  ▼    ▼    ▼          ▼       │
                   │ mlp_  mlp_ mlp_    (app_emb)  │
                   │ opacity cov  color            │
                   │  │     │    │                 │
                   │  ▼     ▼    ▼                 │
                   │ [V*K] child Gaussians         │
                   │  ▶ opacity mask filter        │
                   │  ▶ xyz = anchor + offset*scale│
                   │  ▶ scale = sigmoid(residual)  │
                   │  ▶ rot = normalize(quat)      │
                   └──────────────┬───────────────┘
                                  │ [M, 3/3/1/3/4] per-Gaussian params
                   ┌──────────────▼───────────────┐
                   │   CUDA Gaussian Rasterizer    │
                   │   Tile-based α-compositing    │
                   └──────────────┬───────────────┘
                                  │ rendered RGB + depth
                   ┌──────────────▼───────────────┐
                   │  L1 + SSIM + Isotropic Loss   │
                   │  backward() → Adam step       │
                   └──────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  Chunk Memory Manager (LRU)                         │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
  │  │ Chunk A  │  │ Chunk B  │  │ Chunk C  │  GPU RAM │
  │  │ (active) │  │ (active) │  │ (active) │          │
  │  └──────────┘  └──────────┘  └──────────┘          │
  │       ▲              │              │               │
  │       │ load         │ evict        │ evict         │
  │       │              ▼              ▼               │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
  │  │ Chunk D  │  │ Chunk E  │  │ Chunk F  │  DISK    │
  │  │ (on disk)│  │ (on disk)│  │ (on disk)│          │
  │  └──────────┘  └──────────┘  └──────────┘          │
  └─────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Anchor = Voxel Center + Feature Vector**. Unlike standard 3DGS where every Gaussian has 59 independent parameters, Scaffold-ChunGS stores only sparse _anchors_ (typical `voxel_size = 0.01`). Each anchor's `feat_dim=32` feature and `n_offsets=5` child offsets are decoded by MLPs at render time into the actual child Gaussian parameters. This compresses storage by roughly **1.7–10×** depending on `n_offsets`.

2. **MLPs are view-dependent**. The `mlp_opacity`, `mlp_cov`, and `mlp_color` networks take `[anchor_feat | view_direction | distance]` as input, enabling the model to adapt child Gaussians to different viewpoints — critical for SLAM where training views are sparse.

3. **Chunks operate at anchor granularity**. Each chunk file (`.schun` binary format) stores all anchor tensors + Adam optimizer states for its spatial region. The LRU eviction policy tracks access timestamps and swaps the least-recently-visible chunks to disk when `max_anchors_in_memory` is exceeded.

4. **Anchor growing/densification** uses hierarchical gradient-based seeding (same as Compact_GSSLAM: `update_depth=3` levels, `update_hierarchy_factor=4`), ensuring new anchors are placed where photometric error is high.

---

## Project Structure

```
Scaffold-ChunGS/
├── CMakeLists.txt                  # Build system (CUDA + LibTorch + OpenCV)
├── README.md
├── cfg/
│   └── gaussian_mapper/
│       └── scaffold_chunks.yaml    # Master configuration
├── include/scaffold_chunks/
│   ├── chunk_types.h               # ChunkCoord, 64-bit encoding, AABB
│   ├── config.h                    # Config structs (YAML → C++)
│   ├── anchor_mlp.h                # AnchorMLP decoders
│   ├── gaussian_model.h            # Core anchor-based GaussianModel
│   ├── gaussian_renderer.h         # ScaffoldRenderer pipeline
│   ├── frustum_culler.h            # Two-level frustum culling
│   ├── gaussian_keyframe.h         # Keyframe (pose + intrinsics)
│   ├── gaussian_scene.h            # Keyframe container
│   └── keyframe_selection.h        # Loss-weighted KF selection
├── src/
│   ├── model/
│   │   ├── model_core.cpp          # Anchor CRUD, voxelization
│   │   ├── model_mlp.cpp           # MLP forward/backward (core expansion)
│   │   ├── model_storage.cpp       # Chunk binary I/O (magic: "SCHN")
│   │   ├── model_memory.cpp        # LRU eviction, load/save chunks
│   │   └── model_optimization.cpp  # Adam setup, anchor growing/pruning
│   ├── rendering/
│   │   ├── gaussian_renderer.cpp   # Anchor → GPU rasterizer
│   │   └── frustum_culler.cpp      # Plane extraction + AABB/sphere tests
│   ├── scene/
│   │   ├── gaussian_keyframe.cpp   # Pose, intrinsics, transform tensors
│   │   ├── gaussian_scene.cpp      # Thread-safe KF map
│   │   └── keyframe_selection.cpp  # Probabilistic KF sampling
│   └── training/
│       ├── losses.cpp              # L1 + SSIM + isotropic + depth loss
│       └── trainer.cpp             # Training loop + YAML config loader
├── scripts/                        # Utility scripts (data prep, eval)
├── examples/
│   └── scaffold_chunks_demo.cpp    # End-to-end demo with synthetic scene
├── third_party/                    # External dependencies (build-time)
└── thesis/                         # Related academic thesis (LaTeX)
```

---

## Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| LibTorch (PyTorch C++) | ≥ 2.0 | Tensor ops, Adam optimizer, `torch::nn` MLPs |
| CUDA Toolkit | ≥ 11.8 | GPU rasterizer, custom kernels |
| Eigen3 | ≥ 3.4 | Linear algebra (SE3, AABB) |
| OpenCV | ≥ 4.5 | Image I/O, YAML config parsing |
| OpenMP | — | CPU parallelization |
| GLFW + OpenGL | *(optional)* | Real-time viewer |

---

## Installation

### Prerequisites

- **OS**: Ubuntu 22.04 / 20.04 (primary), Windows (build-only)
- **GPU**: NVIDIA GPU with CUDA compute capability ≥ 8.6 (RTX 30xx+), or 8.7 (Jetson AGX Orin)
- **Compiler**: GCC ≥ 9, NVCC ≥ 11.8

### 1. Install LibTorch

Download from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip -d /workspace/third_party/libtorch
```

On **Jetson**, PyTorch is typically installed via pip; update `CMakeLists.txt` to resolve Torch via Python:

```bash
export TORCH_CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
```

### 2. Install OpenCV & Eigen3

```bash
# Ubuntu
sudo apt install libopencv-dev libeigen3-dev

# Or build OpenCV from source for CUDA support
cd /workspace/third_party
git clone https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/workspace/third_party/install/opencv \
      -DWITH_CUDA=ON ..
make -j$(nproc) && make install
```

### 3. Clone & Build

```bash
git clone --recursive git@github.com:InsistWWJ/Scaffold-ChunGS.git
cd Scaffold-ChunGS

# Auto-detect architecture
mkdir build && cd build
cmake ..
make -j$(nproc)

# Binaries are at build/bin/
# Libraries are at build/lib/
```

---

## Quick Start

### 1. Configure

Edit `cfg/gaussian_mapper/scaffold_chunks.yaml`:

```yaml
# Scene representation
Anchor.n_offsets: 5        # Children per anchor
Anchor.feat_dim: 32        # Anchor feature dimension
Anchor.voxel_size: 0.01    # Anchor placement resolution

# Memory management
Chunk.chunk_size: 20.0             # Chunk size (world units)
Chunk.max_anchors_in_memory: 300000  # Eviction threshold
Chunk.storage_base_path: "./chunks"  # Disk storage location

# Training
Optimization.lambda_dssim: 0.2
Optimization.lambda_depth: 0.1
Optimization.lambda_isotropic: 0.01
```

### 2. Run the Demo (Synthetic Scene)

```bash
./build/bin/scaffold_chunks_demo \
    cfg/gaussian_mapper/scaffold_chunks.yaml \
    ./output_demo
```

This runs an end-to-end test: generates a synthetic sphere point cloud, creates a 20-view circular camera trajectory, initializes anchors, renders the scene, runs 100 training iterations with periodic anchor growing/pruning, and saves rendered images to `./output_demo/`.

**Expected output:**
- `output_demo/render_initial.png` — rendering before training
- `output_demo/render_final.png` — rendering after training
- `output_demo/chunks/` — chunk binary files (`.schun`)

### 3. Integration with Real Data

To use with your own dataset, implement a data loader that:

```cpp
// 1. Create model
auto model = std::make_shared<GaussianModel>(anchor_cfg, chunk_cfg, device);
model->mlp().setAppearanceEmbedding(num_cameras);

// 2. Add initial point cloud
model->initializeFromPoints(point_xyz, point_colors);
model->trainingSetup(opt_cfg);

// 3. Create keyframes
auto scene = std::make_shared<GaussianScene>();
for (each camera view) {
    auto kf = std::make_shared<GaussianKeyframe>(frame_id);
    kf->setPose(Tcw);
    kf->setIntrinsics(fx, fy, cx, cy, W, H, znear, zfar);
    kf->setTrainingImage(image);
    kf->setDepthMap(depth);
    kf->setTimesOfUse(8);
    kf->computeTransformTensors();
    scene->addKeyframe(kf);
}

// 4. Train
for (int iter = 0; iter < max_iters; ++iter) {
    auto kf = keyframe_selector->getNextKeyframe();
    torch::Tensor vis_mask = model->cullVisibleAnchors(
        camera_center, world_view_proj, true);

    auto output = ScaffoldRenderer::render(
        *model, vis_mask, camera_center,
        world_view_transform, projection_matrix,
        FoVx, FoVy, H, W, bg_color);

    auto loss = computeLosses(output.color, output.depth,
                              kf->getGTImage(), kf->getGTDepth(),
                              child_scaling, mask,
                              lambda_dssim, lambda_depth, lambda_isotropic);

    loss.backward();
    model->optimizerStep(visibility, N);
    model->optimizerZeroGrad();

    if (iter % 100 == 0) {
        model->adjustAnchors(check_interval, success_threshold,
                            grad_threshold, min_opacity);
        model->trainingSetup(opt_cfg);  // rebuild optimizer
    }
}

// 5. Save
model->saveAllChunks();
```

---

## Configuration Reference

### Anchor Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Anchor.n_offsets` | int | 5 | Children per anchor (K). Higher = more Gaussians/anchor, better quality, more VRAM. |
| `Anchor.feat_dim` | int | 32 | Anchor feature dimension (D). Higher = more expressive, larger model. |
| `Anchor.voxel_size` | float | 0.01 | Initial voxel size for anchor placement. Smaller = denser anchors. |
| `Anchor.sh_degree` | int | 1 | Spherical harmonics degree (unused: MLPs predict colors directly). |
| `Anchor.use_feat_bank` | bool | 0 | Enable view-adaptive feature blending (3-resolution attention). |
| `Anchor.add_opacity_dist` | bool | 1 | Feed view distance into opacity MLP. |
| `Anchor.add_cov_dist` | bool | 1 | Feed view distance into covariance MLP. |
| `Anchor.add_color_dist` | bool | 1 | Feed view distance into color MLP. |
| `Anchor.appearance_dim` | int | 0 | Per-camera appearance embedding dim (0 = disabled). |
| `Anchor.update_depth` | int | 3 | Hierarchical levels for anchor growing. |
| `Anchor.update_init_factor` | int | 100 | Initial voxel multiplier for new anchor placement. |
| `Anchor.update_hierarchy_factor` | int | 4 | Voxel shrink factor per densification depth. |
| `Anchor.densify_grad_threshold` | float | 0.0002 | Min gradient norm to trigger anchor growing. |
| `Anchor.densify_check_interval` | int | 100 | Iterations between densification cycles. |

### Chunk Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Chunk.chunk_size` | float | 20.0 | Size of each spatial chunk (world units). |
| `Chunk.max_anchors_in_memory` | int | 300000 | Eviction threshold (number of anchors). ~300K anchors × K offsets ≈ 1.5M Gaussians. |
| `Chunk.new_anchor_chunk_density` | int | 10 | Minimum anchors per chunk for adding new points. |
| `Chunk.storage_base_path` | string | "./chunks" | Directory for `.schun` binary files. |

### Optimization Parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `Optimization.anchor_position_lr_init` | float | 5e-5 | Initial position LR. |
| `Optimization.anchor_position_lr_decay` | float | 0.99998 | Per-iteration position LR decay. |
| `Optimization.anchor_feature_lr` | float | 0.005 | Feature vector LR. |
| `Optimization.anchor_opacity_lr` | float | 0.1 | Base opacity LR. |
| `Optimization.anchor_scaling_lr` | float | 0.01 | Base scale LR. |
| `Optimization.anchor_rotation_lr` | float | 0.002 | Base rotation LR. |
| `Optimization.offset_lr` | float | 1e-4 | Child offset LR. |
| `Optimization.mlp_opacity_lr` | float | 0.005 | Opacity MLP weights LR. |
| `Optimization.mlp_cov_lr` | float | 0.005 | Covariance MLP weights LR. |
| `Optimization.mlp_color_lr` | float | 0.0025 | Color MLP weights LR. |
| `Optimization.lambda_dssim` | float | 0.2 | SSIM loss weight. |
| `Optimization.lambda_depth` | float | 0.1 | Depth L1 loss weight. |
| `Optimization.lambda_isotropic` | float | 0.01 | Isotropic regularization weight. |
| `Optimization.max_num_iterations` | int | -1 | Max training iters (-1 = unlimited). |
| `Optimization.new_keyframe_times_of_use` | int | 8 | How many times each new KF is used for training. |

---

## Chunk Binary Format (`.schun`)

```
Offset  Size    Field
─────────────────────────────────────────
0       4       Magic: 0x5343484E ("SCHN")
4       4       Version: 1
8       8       Chunk ID (encoded int64)
16      4       Number of anchors (uint32)
20      N       Tensor data (anchor parameters)
        ┊       [TensorHeader + raw bytes] × 13 tensors:
        ┊         anchor, offset, anchor_feat, anchor_scaling,
        ┊         anchor_rotation, anchor_opacity,
        ┊         exist_since, anchor_chunk_ids, anchor_ids,
        ┊         offset_gradient_accum, offset_denom,
        ┊         opacity_accum, anchor_denom
        ┊       [step_count + exp_avg + exp_avg_sq] × 8 param groups
─────────────────────────────────────────
```

Files are named: `p<n>_p<n>_p<n>.schun` (e.g., `p2_n5_p0.schun` for chunk at x=2, y=-5, z=0).

---

## Performance Benchmarks

Estimated figures (pending real-world validation):

| Metric | Standard 3DGS | Scaffold-ChunGS |
|--------|:---:|:---:|
| Bytes per Gaussian (storage) | ~240 | ~28 (for K=5) |
| Scene size @ 8GB VRAM | ~30M Gaussians | ~300K anchors → ~1.5M active Gaussians |
| Max scene extent | GPU-limited | Unlimited (disk-backed) |
| Render quality (PSNR) | baseline | comparable (+ isotropic regularization) |
| Training speed | baseline | +10–20% (MLP overhead per iteration) |

The 300K anchor threshold produces up to 1.5M active child Gaussians in memory, while arbitrary numbers of inactive chunks reside on disk, enabling effectively unbounded scene reconstruction.

---

## Limitations & Future Work

1. **No loop closure yet** — this is a core representation library; the pose-graph + loop detection pipeline from Compact_GSSLAM (NetVLAD + GICP + 3DGS registration) and DiskChunGS (ORB-SLAM3 BA) can be integrated on top.
2. **No real-time viewer** — the OpenGL ImGui viewer from DiskChunGS can be ported.
3. **CUDA rasterizer placeholder** — the `renderGaussiansCUDA()` function in `gaussian_renderer.cpp` uses simplified point splatting. Replace with INRIA's `diff_gaussian_rasterization` for production quality.
4. **Optimizer rebuild after anchor growing** — `catAnchorsToOptimizer()` currently recommends re-running `trainingSetup()`. A proper optimizer-state extension (matching Compact_GSSLAM's behavior) is needed for efficiency.
5. **No DepthLab integration** — the diffusion-based depth completion model is not yet wired in.

---

## License

Scaffold-ChunGS is licensed under the **GNU General Public License v3.0** (GPL v3).

This project incorporates code and concepts from:
- [DiskChunGS](https://github.com/leggedrobotics/DiskChunGS) (GPL v3, ETH Zurich)
- [Compact_GSSLAM / VCGS-SLAM](https://github.com/dtc111111/Compact_GSSLAM) (IJCV 2025)
- [Scaffold-GS](https://github.com/city-super/Scaffold-GS) (Inria non-commercial research license)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Inria non-commercial research license)

Commercial use of the underlying 3DGS rasterizer and Scaffold-GS components requires a separate license from Inria.

---

## Citation

If you use Scaffold-ChunGS in your research, please cite both foundational works:

```bibtex
@ARTICLE{feldmann2026diskchungs,
  author={Feldmann, Casimir and Wilder-Smith, Maximum and Patil, Vaishakh
          and Oechsle, Michael and Niemeyer, Michael and Tateno, Keisuke
          and Hutter, Marco},
  journal={IEEE Robotics and Automation Letters},
  title={DiskChunGS: Large-Scale 3D Gaussian SLAM Through Chunk-Based
         Memory Management},
  year={2026}, volume={11}, number={4}, pages={5009-5016},
  doi={10.1109/LRA.2026.3668704}}

@ARTICLE{chen2025compactgsslam,
  author={Chen, Tiancheng and Ding, Peng and Duan, Peihu and Zhang, Zhaoying
          and Shi, Guodong and Cui, Chen and Ji, Xiaoxue and Zhu, Feng},
  journal={International Journal of Computer Vision},
  title={VCGS-R-SLAM: Voxelized 3D Gaussian Representation for Dense Visual
         SLAM on Embedded Vision System},
  year={2025}}
```

```bibtex
@inproceedings{scaffoldgs,
  author={Lu, Tao and Yu, Mulin and Xu, Linning and Xiangli, Yuanbo
          and Wang, Limin and Lin, Dahua and Dai, Bo},
  title={Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering},
  booktitle={CVPR}, year={2024}}
```

---

## Acknowledgements

This project builds upon outstanding open-source contributions:

- [DiskChunGS](https://github.com/leggedrobotics/DiskChunGS) — chunk-based out-of-core 3DGS SLAM (ETH Zurich / Google)
- [Compact_GSSLAM](https://github.com/dtc111111/Compact_GSSLAM) — voxelized/scaffold Gaussian SLAM (IJCV 2025)
- [Scaffold-GS](https://github.com/city-super/Scaffold-GS) — anchor-based Gaussian representation (CVPR 2024)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — differential rasterizer (Inria)
- [CaRtGS](https://github.com/DapengFeng/cartgs) / [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM) — photo-realistic GS-SLAM baselines
