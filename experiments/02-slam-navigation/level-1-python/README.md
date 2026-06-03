# SLAM 与导航 (SLAM & Navigation) — Level 1: Python 入门

English: [README.en.md](./README.en.md)

> 本目录为 Level 1 Python 入门实验的规划目录，当前候选实验正在调研中。

## 候选实验

| 实验 | 核心问题 | 前置知识 | 状态 |
|------|---------|---------|------|
| `s01-grid-search` | 已知地图中最短路径怎么找？ | 无 | 候选 |
| `s02-mcl-localization` | 已知地图中机器人在哪？ | 粒子滤波 | 候选 |
| `s03-ekf-slam` | 未知环境中如何同时定位与建图？ | 扩展卡尔曼滤波 | 候选 |

## 目录结构（规划）

```text
level-1-python/
├── scripts/        # 可运行的 Python 脚本
├── tutorials/      # 入门教程
├── utils/          # 工具函数
├── output/         # 运行输出（不跟踪）
└── requirements.txt
```
