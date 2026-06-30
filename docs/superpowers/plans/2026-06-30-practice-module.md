# 实践环节模块 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** 在 `docs/xidian-micro-major/` 下创建第三个模块「实践环节：Python基础与MNIST实战」

**Architecture:** 从两本 PDF 提取内容，按 CLAUDE_WORKFLOW.md 规范整理为结构化笔记（中英双语），并提取可运行 MNIST 代码

**Tech Stack:** Markdown + Python (PyTorch, torchvision)

---

### Task 1: 创建目录结构和 Python 快速入门笔记

**Files:**
- Create: `docs/xidian-micro-major/实践环节：Python基础与MNIST实战/01-Python快速入门.md`
- Create: `docs/xidian-micro-major/实践环节：Python基础与MNIST实战/01-Python快速入门.en.md`

- [ ] **Step 1: 创建目录**

```bash
mkdir -p "docs/xidian-micro-major/实践环节：Python基础与MNIST实战/code"
```

- [ ] **Step 2: 整理中文笔记**

从《从0开始Python快速入门-面向深度学习实践.pdf》提取内容，按 CLAUDE_WORKFLOW.md 规范整理：
- 固定开头格式
- 中文数字章节
- 表格对比（如 NumPy vs 列表、PyTorch vs NumPy）
- LaTeX 公式
- 代码块标注 python
- Blockquote 提取关键观点
- 末尾交叉引用

- [ ] **Step 3: 创建英文镜像 .en.md**

- [ ] **Step 4: 提交**

### Task 2: MNIST 实验手册笔记

**Files:**
- Create: `docs/xidian-micro-major/实践环节：Python基础与MNIST实战/02-MNIST实验手册.md`
- Create: `docs/xidian-micro-major/实践环节：Python基础与MNIST实战/02-MNIST实验手册.en.md`

- [ ] **Step 1: 整理中文笔记**

从《第一次实践环节-MNIST实验手册.pdf》提取内容，同上规范。
重点：数据加载、网络构建、训练循环、评估指标、可视化

- [ ] **Step 2: 创建英文镜像 .en.md**

- [ ] **Step 3: 提交**

### Task 3: 可运行 MNIST 代码

**Files:**
- Create: `docs/xidian-micro-major/实践环节：Python基础与MNIST实战/code/requirements.txt`
- Create: `docs/xidian-micro-major/实践环节：Python基础与MNIST实战/code/mnist_train.py`
- Create: `docs/xidian-micro-major/实践环节：Python基础与MNIST实战/code/mnist_infer.py`

- [ ] **Step 1: 创建 requirements.txt**

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
```

- [ ] **Step 2: 创建 mnist_train.py**

完整训练脚本：数据加载 → 网络定义(CNN) → 训练循环 → 模型保存 → 可视化

- [ ] **Step 3: 创建 mnist_infer.py**

推理脚本：加载模型 → 单张/批量推理 → 可视化预测结果

- [ ] **Step 4: 提交**

### Task 4: 更新 README 和交叉引用

**Files:**
- Modify: `docs/xidian-micro-major/README.md`
- Modify: `docs/xidian-micro-major/README.en.md`
- Modify: `docs/xidian-micro-major/实践环节：Python基础与MNIST实战/01-Python快速入门.md`
- Modify: `docs/xidian-micro-major/实践环节：Python基础与MNIST实战/02-MNIST实验手册.md`

- [ ] **Step 1: 更新 README.md 目录树，加入新模块**

- [ ] **Step 2: 更新 README.en.md 目录树**

- [ ] **Step 3: 更新笔记间交叉引用**

- [ ] **Step 4: 提交**
