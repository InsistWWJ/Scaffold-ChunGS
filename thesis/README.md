# Scaffold-ChunGS 学士学位论文

中国地质大学（武汉）本科毕业设计

**题目：** 基于锚点压缩与分块外存管理的大场景3D高斯SLAM系统设计与实现

**作者：** 万文杰 | **学号：** 20221003247 | **专业：** 遥感科学与技术

---

## 论文概述

本论文针对标准 3DGS 的两大瓶颈——单高斯 59 参数/240 bytes 的存储开销，以及 GPU 显存对场景规模的硬约束——设计并实现 Scaffold-ChunGS 统一框架。核心思路是在 C++17/CUDA/LibTorch 技术栈上融合 **Scaffold-GS 锚点压缩**（体素化 anchor + 视角条件 MLP 解码器，~8.6× 压缩比）与 **DiskChunGS 分块外存管理**（空间分块 + LRU 淘汰 + `.schun` 自包含二进制格式），并围绕"块级调度、锚点级存储、子高斯级渲染"三层粒度构建 10 大模块的完整 SLAM 管线。

### 章节内容

| 章 | 内容 |
|---|------|
| 第一章 | 研究背景与意义、国内外研究现状（国外 3DGS-SLAM / 压缩与扩展 / 硬件加速，国内轻量化与工程化研究）、现有工作不足与本文定位 |
| 第二章 | 3DGS 原理、Scaffold-GS 锚点表示、DiskChunGS 分块管理、视觉 SLAM 基础、锚点压缩与分块管理耦合可行性的五维理论推理（存储复杂度 / GPU 上界 / I/O 摊销 / 优化一致性 / 综合推论） |
| 第三章（核心） | 系统设计与实现——10 大模块：锚点场景表示（初始化 / 增量 / MLP / 生长剪枝）、分块外存管理（编码 / LRU / 异步 I/O / .schun 格式）、双后端渲染与训练（视锥剔除 / INRIA CUDA + LibTorch 回退 / L1+SSIM+Depth+Isotropic / 调度器）、深度估计（SGBM 双目 + TensorRT 单目）、视觉跟踪（ORB+FLANN+PnP+RANSAC）、回环闭合（k-means BoVW+SE(3)）、SLAM 系统接口（SLAMSystemInterface + MappingOperation 队列）、两阶段 Mapper 管线（Phase 1 初始建图 + Phase 2 combineMappingOperations 分发）、实时可视化（GLFW+OpenGL+ImGui） |
| 第四章 | 实验与分析：合成场景端到端验证、压缩效果定量分析（不同 K/D 的压缩-质量权衡）、分块 LRU 行为与 I/O 分析、消融实验设计（视角条件 / 特征维度 / 块大小）、KITTI 数据集评测方案（ATE/RPE/渲染/磁盘占用） |
| 第五章 | 工作总结（4 项）与未来展望（7 项：光栅化器升级 / 回环升级 / 可视化 splat 渲染 / 优化器增量 / DepthLab / 真数评测 / 嵌入式适配） |

### 参考文献

`thesis.bib` 包含约 64 条参考文献，覆盖：3DGS 经典（Kerbl 2023）、锚点压缩（Scaffold-GS / Compact 3DGS-SLAM）、分块外存（DiskChunGS）、GS-SLAM 系统（SplaTAM / GS-SLAM / Photo-SLAM 等）、大场景与层次化（GigaSLAM / VPGS-SLAM / LODGE）、动态与多传感器（GS-LIVO / VIGS-SLAM）、硬件加速（REACT3D / SPLATONIC）、回环与优化（LoopSplat 等）、NeRF 基础、数据集（TUM / Replica / ScanNet / KITTI）、中文文献 12 篇。

---

## 编译

### 环境要求

- TeX Live 2023+（含 xelatex + biber）
- ctex 宏包（中文支持）
- biblatex-gb7714-2005（GB/T 7714-2005 参考文献格式）
- 中文字体：SimSun（宋体）、SimHei（黑体）、KaiTi（楷体）、FangSong（仿宋）

### 编译命令

```bash
# 完整编译
make

# 或手动执行
xelatex -synctex=1 -interaction=nonstopmode main.tex
biber main
xelatex -synctex=1 -interaction=nonstopmode main.tex
xelatex -synctex=1 -interaction=nonstopmode main.tex

# 清理辅助文件
make clean

# 持续编译（监控文件变化自动重编）
make watch
```

### 编译产物

- `main.pdf` — 完整论文（含封面、原创性声明、中英文摘要、目录、正文、参考文献、致谢）
- `aigc-check.pdf` — AIGC 检测专用版（仅含摘要与正文，已移除个人信息、目录、致谢、参考文献等）
- `abstract-only.pdf` — 摘要独立 PDF

---

## 文件结构

```
thesis/
├── main.tex                     # 主文件（封面/声明/摘要/目录/正文/参考文献/致谢）
├── aigc-check.tex               # AIGC 检测专用文档（独立 preamble）
├── thesis.bib                   # 参考文献（GB/T 7714-2005 顺序编码制，~64 条）
├── Makefile                     # 编译脚本（xelatex → biber → xelatex × 2）
├── latexmkrc                    # latexmk 配置（持续编译）
├── chapters/
│   ├── ch1-intro.tex            # 第一章 绪论（背景 / 国内外现状 / 研究内容 / 结构）
│   ├── ch2-background.tex       # 第二章 相关技术基础（3DGS / Scaffold-GS / DiskChunGS / SLAM / 理论推理）
│   ├── ch3-design.tex           # 第三章 系统设计与实现（10 模块详细设计 + 文件组织表）
│   ├── ch4-experiments.tex      # 第四章 实验与分析（合成场景 / 压缩 / 内存 / 消融 / KITTI 方案）
│   └── ch5-summary.tex          # 第五章 总结与展望
├── figures/                     # 图片目录（系统架构图等）
└── README.md
```

---

## 格式规范

| 项目 | 设置 |
|------|------|
| 页面 | A4，上/下 3cm，左/右 3cm，双面印制（twoside） |
| 正文字体 | 宋体 (SimSun) + Times New Roman，12 磅（小四），固定行距 20 磅 |
| 标题字体 | 黑体 (SimHei) |
| 章标题 | 黑体 16 磅（三号），加粗居中，"第X章" |
| 节标题 | 黑体 14 磅（四号），加粗居中 |
| 小节标题 | 黑体 12 磅（小四），加粗居左 |
| 页眉 | 奇数页：中国地质大学学士学位论文；偶数页：作者姓名：论文题目 |
| 页码 | 宋体 10.5 磅（五号），外侧 |
| 图/表编号 | 章内编号（如"图 3.2""表 3.1"） |
| 图/表标题 | 宋体 10.5 磅（五号），图题居下居中，表题居上居中 |
| 参考文献 | GB/T 7714-2005 顺序编码制，biber 编译 |
| 代码 | listings 宏包，浅灰底纹，行号左侧，C++ 语法高亮 |

---

## 待办清单

- [x] 在 `main.tex` 封面中填入个人信息（学院、专业、姓名、学号、导师、职称）
- [x] 将页眉中的姓名替换为 万文杰
- [x] 在原创性声明中填入论文完整标题
- [ ] 填写封面中培养单位、导师职称信息
- [ ] 运行 demo 获取实验数据，填入第四章各表格（PSNR / SSIM / LPIPS / 显存 / 磁盘占用）
- [ ] 绘制系统架构图 → `figures/architecture.pdf`
- [ ] 插入渲染对比图（初始 vs 训练后）
- [ ] 完成 KITTI 数据集评测（序列 00-08 训练 / 09-10 测试）并填入实验表格
- [ ] 撰写致谢
- [ ] 对照学校 Word 模板微调格式
- [ ] 查重

---

## AIGC 检测文档

`aigc-check.tex` 是专为 AIGC 检测准备的独立 LaTeX 文档，与 `main.tex` 共享相同的正文内容（5 章），但做了以下处理：

- 移除：封面、原创性声明、目录、图清单、表清单、致谢、附录
- 移除：页眉中的个人信息（仅保留页码）
- 移除：biblatex 引用宏包（`\cite` 命令静默丢弃，正文中引用标记消失）
- 保留：完整的摘要（中英文）与全部 5 章正文

该文档的存在是为了在知网/维普等 AIGC 检测系统中获得合理的检测结果——移除因格式元素和参考文献引用导致的误判。编译方式与 `main.tex` 相同。

---

## 学校参考文档

位于项目根目录 `论文要求/` 下：

- `学士学位论文写作规范（2018版）.pdf` — 写作规范正文（封面/声明/摘要/正文/图表/参考文献/致谢格式）
- `学士学位论文写作规范附件.docx` — 附件格式（封面模板 / 原创性声明 / 目录 / 参考文献示例）
- `本科毕业论文（设计）相关附件.docx` — 过程文档（任务书 / 指导登记表 / 开题审核表 / 评阅表 / 答辩记录）
- `关于印发《本科毕业论文（设计）工作管理办法》的通知.doc` — 学校管理办法
