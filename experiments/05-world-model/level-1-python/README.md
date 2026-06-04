# 世界模型 (World Models) — Level 1: Core Python Lab

English: [README.en.md](./README.en.md)

## 定位

本目录是世界模型方向的 **Level 1 入口**。Level 1 是当前主产品：纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Python 3.10+ 运行环境中直接运行。

## 规划中的实验

| 实验 | 核心问题 | 状态 |
|------|---------|------|
| 动力学预测 | 如何学习物理世界的前向模型？ | 规划中 |
| 短期 rollout | 如何用预测模型推演未来状态？ | 规划中 |

## 与 Level 2 的关系

Level 1 用 Python 建立预测建模直觉。Level 2（`../level-2-ros2-bridge/`）将这些算法接入 ROS2 / C++ / 真实机器人软件栈。

## 相关文档

- [Python-first, ROS2-ready 路线说明](../../../docs/curriculum/python-first-ros2-ready.md)
- [世界模型方向页](../README.md)
