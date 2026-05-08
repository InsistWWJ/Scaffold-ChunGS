# Scaffold-ChunGS 学士学位论文

中国地质大学（武汉）本科毕业设计

## 编译

### 环境要求

- TeX Live 2023+（含 xelatex + biber）
- ctex 宏包（中文支持）
- biblatex-gb7714-2005（GB/T 7714-2005 参考文献格式）

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

# 持续编译（监控文件变化）
make watch
```

## 文件结构

```
thesis/
├── main.tex                    # 主文件（封面/原创性声明/摘要/目录/正文/参考文献/致谢）
├── thesis.bib                  # 参考文献（GB/T 7714-2005 顺序编码制）
├── Makefile                    # 编译脚本
├── latexmkrc                   # latexmk 配置
├── chapters/
│   ├── ch1-intro.tex           # 第一章 绪论
│   ├── ch2-background.tex      # 第二章 相关技术基础
│   ├── ch3-design.tex          # 第三章 系统设计与实现（核心章节）
│   ├── ch4-experiments.tex     # 第四章 实验与分析
│   └── ch5-summary.tex         # 第五章 总结与展望
├── figures/                    # 图片目录
└── README.md
```

## 格式规范

| 项目 | 设置 |
|------|------|
| 页面 | A4, 上2.54cm, 下2.54cm, 左3.17cm, 右3.17cm |
| 正文字体 | 宋体 (SimSun), 小四 |
| 标题字体 | 黑体 (SimHei) |
| 参考文献 | GB/T 7714-2005 顺序编码制 |
| 页眉 | 奇数页: 中国地质大学 / 偶数页: 姓名：题目 |
| 装订 | 双面印制 (twoside) |

## 待办清单

- [ ] 在 `main.tex` 封面中填入个人信息（学院、专业、姓名、学号、导师、职称）
- [ ] 将页眉中的 `XXX` 替换为你的姓名
- [ ] 在原创性声明中填入论文完整标题
- [ ] 运行 demo 获取实验数据，填入第四章表格
- [ ] 绘制系统架构图 → `figures/architecture.pdf`
- [ ] 插入渲染对比图（初始 vs 训练后）
- [ ] 撰写致谢
- [ ] 对照学校 Word 模板微调格式
- [ ] 查重
- [ ] 双面打印装订

## 学校参考文档

- `论文要求/学士学位论文写作规范（2018版）.pdf` — 写作规范正文
- `论文要求/学士学位论文写作规范附件.docx` — 封面/原创性声明/目录/参考文献格式
- `论文要求/本科毕业论文（设计）相关附件.docx` — 任务书/指导登记表/开题审核表/评阅/答辩记录
- `论文要求/关于印发《本科毕业论文（设计）工作管理办法》的通知.doc` — 管理办法
