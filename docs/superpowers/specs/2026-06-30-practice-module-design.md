# 实践环节模块设计

## 背景

项目 `docs/xidian-micro-major/` 下现有两个理论模块：
- 具身智能感知基础：机器学习与深度学习
- 具身智能认知进阶：多模态大模型与语义理解

现需新增实践板块，容纳两本 PDF 的内容：
- 《从0开始Python快速入门-面向深度学习实践》(31页)
- 《第一次实践环节-MNIST实验手册》(22页)

## 决策

在 `docs/xidian-micro-major/` 下新建第三个模块「实践环节：Python基础与MNIST实战」，与两个理论模块平级。

## 目录结构

```
docs/xidian-micro-major/
├── 实践环节：Python基础与MNIST实战/
│   ├── 01-Python快速入门.md
│   ├── 01-Python快速入门.en.md
│   ├── 02-MNIST实验手册.md
│   ├── 02-MNIST实验手册.en.md
│   └── code/
│       ├── requirements.txt
│       ├── mnist_train.py
│       └── mnist_infer.py
```

## 笔记整理规范

遵循 `CLAUDE_WORKFLOW.md` 的 6 步工作流：
1. 读取原始 PDF
2. 整理为结构化笔记（中文数字章节、表格优先、LaTeX、代码块）
3. 创建英文镜像 `.en.md`
4. 更新交叉引用
5. 更新 README 目录树
6. 汇报结果

## 可运行代码

`code/` 目录存放从 PDF 中提取的完整可运行代码，依赖通过 `requirements.txt` 管理。

## 约束

- 开头固定格式：`# 标题` → `> 西安电子科技大学 · 具身智能微专业` → `> 课程笔记`
- 不空壳、不泛化、双语对齐
- 与其他模块保持一致的格式和语气
