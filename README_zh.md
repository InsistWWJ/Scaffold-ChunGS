# Scaffold-ChunGS：基于分块外存管理的压缩式 3D Gaussian SLAM

**Scaffold-ChunGS** 融合了两项最先进的 3D Gaussian SLAM 技术：

| 模块 | 来源 | 核心创新 |
|-----------|--------|---------------|
| **Anchor 场景表征** | [Compact_GSSLAM](https://github.com/dtc111111/Compact_GSSLAM) (IJCV 2025) | Scaffold-GS：体素化 anchor + MLP 解码子 Gaussian，内存压缩 5–10× |
| **Chunk 分块内存管理** | [DiskChunGS](https://github.com/leggedrobotics/DiskChunGS) (IEEE RA-L 2026) | 空间分块 + LRU 换出到磁盘，支持任意大规模场景 |

通过在统一的 C++/LibTorch 框架中结合上述技术，Scaffold-ChunGS 在**单位 GPU 显存内实现了比标准 3DGS 更高的重建密度**，同时通过外存分块 I/O 保持了**无限的场景可扩展性**。

<p align="center">
  <b>C++17 &nbsp;|&nbsp; CUDA &nbsp;|&nbsp; LibTorch &nbsp;|&nbsp; OpenCV &nbsp;|&nbsp; Eigen3</b>
</p>

---

## 架构概览

```
                   ┌──────────────────────────────┐
 RGB-D / Stereo ──▶│   GaussianKeyframe (位姿 +    │
                   │   内参 + 图像金字塔)           │
                   └──────────────┬───────────────┘
                                  │
                   ┌──────────────▼───────────────┐
                   │     FrustumCuller             │
                   │  Chunk 级 AABB 测试 ──▶       │
                   │  Anchor 级球体测试             │
                   └──────────────┬───────────────┘
                                  │ 可见 anchor 掩码 [N] bool
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
                   │ [V*K] 子 Gaussian              │
                   │  ▶ 透明度掩码过滤              │
                   │  ▶ xyz = anchor + offset*scale │
                   │  ▶ scale = sigmoid(residual)   │
                   │  ▶ rot = normalize(quat)       │
                   └──────────────┬───────────────┘
                                  │ [M, 3/3/1/3/4] 逐 Gaussian 参数
                   ┌──────────────▼───────────────┐
                   │   CUDA Gaussian 光栅化器      │
                   │   基于 Tile 的 α 合成          │
                   └──────────────┬───────────────┘
                                  │ 渲染 RGB + 深度
                   ┌──────────────▼───────────────┐
                   │  L1 + SSIM + 各向同性损失      │
                   │  backward() → Adam 步进       │
                   └──────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │  Chunk 内存管理器 (LRU)                              │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
  │  │ Chunk A  │  │ Chunk B  │  │ Chunk C  │  GPU 显存│
  │  │ (活跃)   │  │ (活跃)   │  │ (活跃)   │          │
  │  └──────────┘  └──────────┘  └──────────┘          │
  │       ▲              │              │               │
  │       │ 加载         │ 换出         │ 换出          │
  │       │              ▼              ▼               │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
  │  │ Chunk D  │  │ Chunk E  │  │ Chunk F  │  磁盘    │
  │  │ (磁盘中) │  │ (磁盘中) │  │ (磁盘中) │          │
  │  └──────────┘  └──────────┘  └──────────┘          │
  └─────────────────────────────────────────────────────┘
```

### 核心设计决策

1. **Anchor = 体素中心 + 特征向量**。与标准 3DGS（每个 Gaussian 约 59 个独立参数）不同，Scaffold-ChunGS 仅存储稀疏的 _anchor_（典型 `voxel_size = 0.01`）。每个 anchor 的 32 维特征和 5 个子偏移量在渲染时由 MLP 解码为实际的子 Gaussian 参数。根据 `n_offsets` 的取值，这可将存储压缩约 **1.7–10×**。

2. **MLP 依赖视角**。`mlp_opacity`、`mlp_cov` 和 `mlp_color` 网络以 `[anchor_feat | view_direction | distance]` 作为输入，使模型能够针对不同视点自适应调整子 Gaussian——这对于训练视角稀疏的 SLAM 至关重要。

3. **分块在 Anchor 粒度上操作**。每个 chunk 文件（`.schun` 二进制格式）存储其空间区域内所有 anchor 张量及其 Adam 优化器状态。LRU 淘汰策略追踪访问时间戳，当 `max_anchors_in_memory` 超限时，将最近最少可见的 chunk 交换到磁盘。

4. **Anchor 生长/加密**采用基于层次梯度的种子策略（与 Compact_GSSLAM 一致：`update_depth=3` 层，`update_hierarchy_factor=4`），确保新 anchor 被放置在光度误差较大的区域。

---

## 项目结构

```
Scaffold-ChunGS/
├── CMakeLists.txt                  # 构建系统 (CUDA + LibTorch + OpenCV)
├── README.md
├── README_zh.md                    # 中文说明文档
├── cfg/
│   └── gaussian_mapper/
│       └── scaffold_chunks.yaml    # 主配置文件
├── include/scaffold_chunks/
│   ├── chunk_types.h               # ChunkCoord、64 位编码、AABB
│   ├── config.h                    # 配置结构体 (YAML → C++)
│   ├── anchor_mlp.h                # AnchorMLP 解码器
│   ├── gaussian_model.h            # 核心 anchor-based GaussianModel
│   ├── gaussian_renderer.h         # ScaffoldRenderer 渲染管线
│   ├── frustum_culler.h            # 两级视锥裁剪
│   ├── gaussian_keyframe.h         # 关键帧 (位姿 + 内参)
│   ├── gaussian_scene.h            # 关键帧容器
│   └── keyframe_selection.h        # 基于损失的 KF 选取
├── src/
│   ├── model/
│   │   ├── model_core.cpp          # Anchor 增删查改、体素化
│   │   ├── model_mlp.cpp           # MLP 前向/反向传播
│   │   ├── model_storage.cpp       # Chunk 二进制 I/O (魔数: "SCHN")
│   │   ├── model_memory.cpp        # LRU 淘汰、chunk 加载/保存
│   │   └── model_optimization.cpp  # Adam 设置、anchor 生长/剪枝
│   ├── rendering/
│   │   ├── gaussian_renderer.cpp   # Anchor → GPU 光栅化器
│   │   └── frustum_culler.cpp      # 平面提取 + AABB/球体测试
│   ├── scene/
│   │   ├── gaussian_keyframe.cpp   # 位姿、内参、变换张量
│   │   ├── gaussian_scene.cpp      # 线程安全 KF 映射
│   │   └── keyframe_selection.cpp  # 概率化 KF 采样
│   └── training/
│       ├── losses.cpp              # L1 + SSIM + 各向同性 + 深度损失
│       └── trainer.cpp             # 训练循环 + YAML 配置加载
├── scripts/                        # 辅助脚本 (数据准备、评估)
├── examples/
│   └── scaffold_chunks_demo.cpp    # 端到端合成场景演示
├── third_party/                    # 第三方依赖 (构建时)
└── thesis/                         # 相关学位论文 (LaTeX)
```

---

## 依赖项

| 依赖库 | 版本要求 | 用途 |
|-----------|---------|---------|
| LibTorch (PyTorch C++) | ≥ 2.0 | 张量运算、Adam 优化器、`torch::nn` MLP |
| CUDA Toolkit | ≥ 11.4 | GPU 光栅化器、自定义内核 |
| Eigen3 | ≥ 3.4 | 线性代数 (SE3、AABB) |
| OpenCV | ≥ 4.5 | 图像 I/O、YAML 配置解析 |
| OpenMP | — | CPU 并行化 |
| GLFW + OpenGL | *(可选)* | 实时可视化 |

---

## 安装

### 前置条件

- **操作系统**：Ubuntu 22.04 / 20.04（主力平台），Windows（仅构建），JetPack 5.1.3+（Jetson Orin）
- **GPU**：NVIDIA GPU，CUDA 计算能力 ≥ 8.6（RTX 30xx+），或 8.7（Jetson Orin Nano / AGX Orin）
- **编译器**：GCC ≥ 9，NVCC ≥ 11.4

### 1. 安装 LibTorch

**桌面端 (x86_64)：**

```bash
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip -d /workspace/third_party/libtorch
```

**Jetson Orin (aarch64, JetPack 5.1.3+)：**

Jetson 上的 PyTorch/LibTorch 必须通过 JetPack 提供的 pip wheel 安装（JP5 编译自 CUDA 11.4，JP6 编译自 CUDA 12.x）：

```bash
# 通过 pip 安装 PyTorch（包含 LibTorch CMake 配置文件）
pip3 install torch torchvision

# 导出 CMake 前缀路径，使构建系统能找到 Torch
export TORCH_CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
```

CMakeLists.txt 会自动检测 aarch64 并搜索常见的 JetPack Python 路径（3.8/3.10/3.11）。设置 `TORCH_CMAKE_PREFIX_PATH` 可覆盖所有默认路径。

### 2. 安装 OpenCV & Eigen3

```bash
# Ubuntu / Jetson（JetPack 预装 OpenCV 4.5+）
sudo apt install libopencv-dev libeigen3-dev
```

仅在**桌面端 (x86_64)** 需要从源码编译带 CUDA 的 OpenCV：

```bash
cd /workspace/third_party
git clone https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/workspace/third_party/install/opencv \
      -DWITH_CUDA=ON ..
make -j$(nproc) && make install
```

### 3. 克隆并构建

```bash
git clone --recursive git@github.com:InsistWWJ/Scaffold-ChunGS.git
cd Scaffold-ChunGS

# 自动检测架构（x86_64 → sm_86，aarch64 → sm_87）
mkdir build && cd build
cmake ..
make -j$(nproc)

# 可执行文件：build/bin/
# 库文件：build/lib/
```

**Jetson 注意事项：**
- aarch64 上 CUDA 架构自动设为 `sm_87`
- OpenCV 从系统路径解析（JetPack 预装路径）
- Torch 通过 pip 安装的 PyTorch CMake 配置查找
- 若找不到 OpenCV，请手动安装：`sudo apt install libopencv-dev`
- Orin Nano 为 6 核 CPU，推荐 `make -j4` 以保留系统资源

---

## 快速开始

### 1. 配置

编辑 `cfg/gaussian_mapper/scaffold_chunks.yaml`：

```yaml
# 场景表征
Anchor.n_offsets: 5        # 每个 anchor 的子 Gaussian 数
Anchor.feat_dim: 32        # Anchor 特征维度
Anchor.voxel_size: 0.01    # Anchor 放置分辨率

# 内存管理
Chunk.chunk_size: 20.0             # Chunk 大小（世界单位）
Chunk.max_anchors_in_memory: 120000  # 淘汰阈值（8GB Jetson；12GB+ dGPU 可用 300000）
Chunk.storage_base_path: "./chunks"  # 磁盘存储路径

# 训练
Optimization.lambda_dssim: 0.2
Optimization.lambda_depth: 0.1
Optimization.lambda_isotropic: 0.01
```

### 2. 运行演示（合成场景）

```bash
./build/bin/scaffold_chunks_demo \
    cfg/gaussian_mapper/scaffold_chunks.yaml \
    ./output_demo
```

执行一个端到端测试：生成合成球面点云、创建包含 20 个视角的圆形相机轨迹、初始化 anchor、渲染场景、运行 100 次训练迭代并定期进行 anchor 生长/剪枝，将渲染图像保存到 `./output_demo/`。

**预期输出：**
- `output_demo/render_initial.png` — 训练前渲染
- `output_demo/render_final.png` — 训练后渲染
- `output_demo/chunks/` — chunk 二进制文件 (`.schun`)

### 3. 集成真实数据

如需使用自有数据集，请实现数据加载器：

```cpp
// 1. 创建模型
auto model = std::make_shared<GaussianModel>(anchor_cfg, chunk_cfg, device);
model->mlp().setAppearanceEmbedding(num_cameras);

// 2. 添加初始点云
model->initializeFromPoints(point_xyz, point_colors);
model->trainingSetup(opt_cfg);

// 3. 创建关键帧
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

// 4. 训练
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
        model->trainingSetup(opt_cfg);  // 重建优化器
    }
}

// 5. 保存
model->saveAllChunks();
```

---

## 配置参考

### Anchor 参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|---------|-------------|
| `Anchor.n_offsets` | int | 5 | 每个 anchor 的子 Gaussian 数 (K)。值越大，Gaussian/anchor 越多，质量越好，显存占用越高。 |
| `Anchor.feat_dim` | int | 32 | Anchor 特征维度 (D)。值越大，表达能力越强，模型越大。 |
| `Anchor.voxel_size` | float | 0.01 | Anchor 放置的初始体素大小。值越小，anchor 越密。 |
| `Anchor.sh_degree` | int | 1 | 球谐函数阶数（未使用：MLP 直接预测颜色）。 |
| `Anchor.use_feat_bank` | bool | 0 | 启用视角自适应特征融合（3 分辨率注意力）。 |
| `Anchor.add_opacity_dist` | bool | 1 | 向透明度 MLP 传入观察距离。 |
| `Anchor.add_cov_dist` | bool | 1 | 向协方差 MLP 传入观察距离。 |
| `Anchor.add_color_dist` | bool | 1 | 向颜色 MLP 传入观察距离。 |
| `Anchor.appearance_dim` | int | 0 | 逐相机外观嵌入维度（0 = 禁用）。 |
| `Anchor.update_depth` | int | 3 | Anchor 生长的层级深度。 |
| `Anchor.update_init_factor` | int | 100 | 新 anchor 放置的初始体素乘数。 |
| `Anchor.update_hierarchy_factor` | int | 4 | 每层加密的体素缩小因子。 |
| `Anchor.densify_grad_threshold` | float | 0.0002 | 触发 anchor 生长的最小梯度范数。 |
| `Anchor.densify_check_interval` | int | 100 | 加密周期（迭代次数间隔）。 |

### Chunk 参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|---------|-------------|
| `Chunk.chunk_size` | float | 20.0 | 每个空间 chunk 的大小（世界单位）。 |
| `Chunk.max_anchors_in_memory` | int | 120000 | 淘汰阈值（anchor 数量）。默认 120K 适配 8GB Jetson；12GB+ dGPU 可增至 300K。 |
| `Chunk.new_anchor_chunk_density` | int | 10 | 向 chunk 添加新点所需的最小 anchor 数。 |
| `Chunk.storage_base_path` | string | "./chunks" | `.schun` 二进制文件的存储目录。 |

### 优化参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|---------|-------------|
| `Optimization.anchor_position_lr_init` | float | 5e-5 | 位置初始学习率。 |
| `Optimization.anchor_position_lr_decay` | float | 0.99998 | 每次迭代的位置学习率衰减。 |
| `Optimization.anchor_feature_lr` | float | 0.005 | 特征向量学习率。 |
| `Optimization.anchor_opacity_lr` | float | 0.1 | 基础透明度学习率。 |
| `Optimization.anchor_scaling_lr` | float | 0.01 | 基础缩放学习率。 |
| `Optimization.anchor_rotation_lr` | float | 0.002 | 基础旋转学习率。 |
| `Optimization.offset_lr` | float | 1e-4 | 子偏移学习率。 |
| `Optimization.mlp_opacity_lr` | float | 0.005 | 透明度 MLP 权重学习率。 |
| `Optimization.mlp_cov_lr` | float | 0.005 | 协方差 MLP 权重学习率。 |
| `Optimization.mlp_color_lr` | float | 0.0025 | 颜色 MLP 权重学习率。 |
| `Optimization.lambda_dssim` | float | 0.2 | SSIM 损失权重。 |
| `Optimization.lambda_depth` | float | 0.1 | 深度 L1 损失权重。 |
| `Optimization.lambda_isotropic` | float | 0.01 | 各向同性正则化权重。 |
| `Optimization.max_num_iterations` | int | -1 | 最大训练迭代次数（-1 = 无限制）。 |
| `Optimization.new_keyframe_times_of_use` | int | 8 | 每个新关键帧用于训练的次数。 |

---

## Chunk 二进制格式 (`.schun`)

```
Offset  大小   字段
─────────────────────────────────────────
0       4      魔数: 0x5343484E ("SCHN")
4       4      版本: 1
8       8      Chunk ID (编码 int64)
16      4      Anchor 数量 (uint32)
20      N      张量数据 (anchor 参数)
        ┊       [TensorHeader + 原始字节] × 13 个张量:
        ┊         anchor, offset, anchor_feat, anchor_scaling,
        ┊         anchor_rotation, anchor_opacity,
        ┊         exist_since, anchor_chunk_ids, anchor_ids,
        ┊         offset_gradient_accum, offset_denom,
        ┊         opacity_accum, anchor_denom
        ┊       [step_count + exp_avg + exp_avg_sq] × 8 参数组
─────────────────────────────────────────
```

文件命名规则：`p<n>_p<n>_p<n>.schun`（例如，chunk 位于 x=2, y=-5, z=0 时文件名为 `p2_n5_p0.schun`）。

---

## 性能基准

以下为估算数据（尚待实测验证）：

| 指标 | 标准 3DGS | Scaffold-ChunGS |
|--------|:---:|:---:|
| 每 Gaussian 存储字节数 | ~240 | ~28（K=5 时） |
| 8GB 显存场景规模 (Jetson) | ~30M Gaussian | ~120K anchor → ~600K 活跃 Gaussian |
| 12GB 显存场景规模 (dGPU) | ~30M Gaussian | ~300K anchor → ~1.5M 活跃 Gaussian |
| 最大场景范围 | 受限于 GPU 显存 | 无限制（磁盘可扩展） |
| 渲染质量 (PSNR) | 基准线 | 相当（+ 各向同性正则化） |
| 训练速度 | 基准线 | +10–20%（每次迭代的 MLP 开销） |

Anchor 阈值（Jetson 8GB 为 120K，桌面 12GB+ 为 300K）可在内存中容纳最多 5 倍于该数量的活跃子 Gaussian，而任意数量的非活跃 chunk 驻留在磁盘上，从而实现几乎无界的场景重建。

---

## 局限性与未来工作

1. **尚无回环检测** — 目前是一个核心表征库；来自 Compact_GSSLAM（NetVLAD + GICP + 3DGS 配准）和 DiskChunGS（ORB-SLAM3 BA）的位姿图 + 回环检测管线可在上层集成。
2. **暂无实时可视化** — DiskChunGS 中的 OpenGL ImGui 可视化模块可移植。
3. **CUDA 光栅化器为占位实现** — `gaussian_renderer.cpp` 中的 `renderGaussiansCUDA()` 函数使用简化的点喷射渲染。生产环境下请替换为 INRIA 的 `diff_gaussian_rasterization` 以获得更高质量。
4. **Anchor 生长后需重建优化器** — `catAnchorsToOptimizer()` 目前建议重新运行 `trainingSetup()`。需要实现适当的优化器状态扩展（匹配 Compact_GSSLAM 的行为）以提高效率。
5. **尚未集成 DepthLab** — 基于扩散的深度补全模型尚未接入。

---

## 许可证

Scaffold-ChunGS 基于 **GNU General Public License v3.0**（GPL v3）许可。

本项目包含来自以下项目的代码和概念：
- [DiskChunGS](https://github.com/leggedrobotics/DiskChunGS) (GPL v3, ETH Zurich)
- [Compact_GSSLAM / VCGS-SLAM](https://github.com/dtc111111/Compact_GSSLAM) (IJCV 2025)
- [Scaffold-GS](https://github.com/city-super/Scaffold-GS) (Inria 非商业研究许可)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Inria 非商业研究许可)

底层 3DGS 光栅化器和 Scaffold-GS 组件的商业使用需获得 Inria 的单独许可。

---

## 引用

如果您在研究中使用了 Scaffold-ChunGS，请同时引用以下两项基础工作：

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
```

```bibtex
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

## 致谢

本项目基于以下杰出的开源成果构建：

- [DiskChunGS](https://github.com/leggedrobotics/DiskChunGS) — 基于分块外存的 3DGS SLAM（ETH Zurich / Google）
- [Compact_GSSLAM](https://github.com/dtc111111/Compact_GSSLAM) — 体素化/Scaffold Gaussian SLAM（IJCV 2025）
- [Scaffold-GS](https://github.com/city-super/Scaffold-GS) — 基于 Anchor 的 Gaussian 表征（CVPR 2024）
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — 微分光栅化器（Inria）
- [CaRtGS](https://github.com/DapengFeng/cartgs) / [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM) — 照片级真实感 GS-SLAM 基线
