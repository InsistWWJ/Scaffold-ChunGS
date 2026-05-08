# Scaffold-ChunGS 毕业论文

## 编译

### 环境要求

- TeX Live 2023+（含 xelatex + biber）
- ctex 宏包
- biblatex-gb7714-2015（GB/T 7714-2015 参考文献格式）

### 编译命令

```bash
# 首次编译（或清理后）
make

# 增量编译
latexmk -xelatex main.tex

# 清理辅助文件
make clean

# 持续编译（监控文件变化）
make watch
```

## 文件结构

```
thesis/
├── main.tex                    # 主文件（文档类、宏包、封面、摘要、目录）
├── thesis.bib                  # 参考文献（GB/T 7714-2015）
├── Makefile                    # 构建脚本
├── latexmkrc                   # latexmk 配置
├── chapters/
│   ├── ch1-intro.tex           # 第一章 绪论
│   ├── ch2-background.tex      # 第二章 相关技术基础
│   ├── ch3-design.tex          # 第三章 系统设计与实现（核心）
│   ├── ch4-experiments.tex     # 第四章 实验与分析
│   └── ch5-summary.tex         # 第五章 总结与展望
├── figures/                    # 图片目录
└── README.md
```

## 学校模板匹配

学校有 Word 模板和写作要求。收到模板文件后，需要对齐以下设置：

- `main.tex` 中的页边距（geometry）
- 字体设置（中文字体、英文字体、字号）
- 封面信息表结构
- 章节标题格式
- 参考文献格式确认

## 待办事项

- [ ] 填入学校信息（学院、专业、姓名、学号、导师）
- [ ] 获取并对照 Word 模板调整格式
- [ ] 运行合成场景实验，填入第四章定量数据
- [ ] 绘制系统架构图（figure/architecture.pdf）
- [ ] 插入渲染对比图（初始 vs 训练后）
- [ ] 撰写致谢
- [ ] 查重与格式检查
