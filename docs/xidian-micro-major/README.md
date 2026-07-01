# 西安电子科技大学 · 具身智能微专业笔记

English: [README.en.md](./README.en.md)

## 背景

西安电子科技大学开设了面向本科生的具身智能微专业。本目录存放该微专业的课程笔记、实验记录和学习总结。

这是国内最早一批面向本科生的具身智能系统性课程之一。记录这些内容的目的：

- 保留第一手学习路径和踩坑经验
- 为后续学习者提供可参考的笔记模板
- 与 Embodied AI Lab 的实验互相印证

## 使用方式

1. 在本目录下按课程或主题创建 `.md` 文件
2. 写完后由 Claude 协助完善格式、补充交叉引用
3. 完成的笔记可选择性地同步到公开仓库

## 目录结构

```text
docs/xidian-micro-major/
├── README.md
├── README.en.md
├── CLAUDE_WORKFLOW.md
├── 具身智能感知基础：机器学习与深度学习/    ← 第一模块（理论）
│   ├── 具身智能感知基础：机器视觉与深度学习.md
│   ├── What_is_neural_network.md
│   ├── From_traditional_image_processing_to_deep_learning.md
│   ├── What_is_Convolution_network.md
│   ├── Commonly_used_convolution_operators.md
│   └── *.en.md                              ← 英文镜像
├── 具身智能认知进阶：多模态大模型与语义理解/  ← 第二模块（理论）
│   ├── How_to_Build_and_Train_a_Neural_Network_p1.md
│   ├── How_to_Build_and_Train_a_Neural_Network_p2.md
│   └── *.en.md                              ← 英文镜像
└── 实践环节：Python基础与MNIST实战/          ← 第三模块（实践）
    ├── 01-Python快速入门.md
    ├── 01-Python快速入门.en.md
    ├── 02-MNIST实验手册.md
    ├── 02-MNIST实验手册.en.md
    └── code/
        ├── requirements.txt
        ├── mnist_train.py
        └── mnist_infer.py
└── 实践环节：图像超分辨率实战/               ← 第四模块（实践）
    ├── 01-图像超分辨率实验手册.md
    ├── 01-图像超分辨率实验手册.en.md
    └── code/
        ├── requirements.txt
        ├── train_sr.py
        └── infer_sr.py
```

## 相关文档

- [Embodied AI Lab 课程全景](../../README.md)
- [Python-first, ROS2-ready 路线说明](../curriculum/python-first-ros2-ready.md)
