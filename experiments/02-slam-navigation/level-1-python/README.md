# SLAM 与导航 (SLAM & Navigation) — Level 1: Core Python Lab

English: [README.en.md](./README.en.md)

本目录是 SLAM 与导航方向的 Level 1 Python 实验入口。目标是在不依赖 ROS2 的前提下，先把定位、建图和路径规划的核心直觉跑通、看见、解释清楚。

## 当前入口

| 实验 | 核心问题 | 前置知识 | 状态 |
|------|---------|---------|------|
| `s01-grid-search` | 已知地图中最短路径怎么找？ | 无 | 已完成 |
| `s02-mcl-localization` | 已知地图中机器人在哪？ | 粒子滤波 | 候选 |
| `s03-ekf-slam` | 未知环境中如何同时定位与建图？ | 扩展卡尔曼滤波 | 候选 |

## 快速运行

```bash
pip install -r requirements.txt
python scripts/grid_search.py --map maze --algorithm astar
python scripts/grid_search.py --map maze --algorithm dijkstra
python scripts/grid_search.py --map maze --compare
```

运行后输出保存在 `output/grid_search/`，主要观察 `grid_search.png`、`grid_search_astar.png`、`grid_search_dijkstra.png` 和 JSON 指标。

## 学习路线

1. 先运行 `s01-grid-search`，观察 A* 和 Dijkstra 在已知栅格地图中的搜索差异。
2. 再进入 `s02-mcl-localization`，把“怎么走”之前的“我在哪”问题补齐。
3. 最后进入 `s03-ekf-slam`，理解未知地图中定位和建图为什么会耦合。

## 实验说明

- [Grid Search 路径规划教程](tutorials/grid_search.md)

## 目录结构

```text
level-1-python/
├── planners/       # 路径规划算法实现
├── scripts/        # 可运行的 Python 脚本
├── tutorials/      # 入门教程
├── output/         # 运行输出（不跟踪）
└── requirements.txt
```
