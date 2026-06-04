# RL 与模仿学习 (RL and Imitation) — Level 1: Core Python Lab

English: [README.en.md](./README.en.md)

## 定位

本目录是强化学习与模仿学习方向的 **Level 1 入口**。Level 1 是当前主产品：纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Python 3.10+ 运行环境中直接运行。

## 规划中的实验

| 实验 | 核心问题 | 状态 |
|------|---------|------|
| Q-learning | 如何用表格方法学习最优策略？ | 规划中 |
| 深度强化学习 | 如何用神经网络逼近策略函数？ | 规划中 |
| 模仿学习 | 如何从示教数据中学习行为？ | 规划中 |

## 与 Level 2 的关系

Level 1 用 Python 建立策略学习直觉。Level 2（`../level-2-ros2-bridge/`）将这些算法接入 ROS2 / C++ / 真实机器人软件栈。

## 相关文档

- [Python-first, ROS2-ready 路线说明](../../../docs/curriculum/python-first-ros2-ready.md)
- [RL 与模仿学习方向页](../README.md)
