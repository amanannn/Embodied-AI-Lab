# 仿真与 Sim-to-Real (Simulation and Sim-to-Real) — Level 1: Core Python Lab

English: [README.en.md](./README.en.md)

## 定位

本目录是仿真与 Sim-to-Real 方向的 **Level 1 入口**。Level 1 是当前主产品：纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Manjaro 或任意 Python 环境中直接运行。

## 规划中的实验

| 实验 | 核心问题 | 状态 |
|------|---------|------|
| 2D 仿真基础 | 如何用轻量仿真器理解世界步进和观测循环？ | 规划中 |
| 域随机化 | 如何通过随机化仿真参数提高迁移性？ | 规划中 |

## 与 Level 2 的关系

Level 1 用 Python 建立仿真基础直觉。Level 2（`../level-2-ros2-bridge/`）将这些概念接入 Gazebo / Isaac Sim 等更真实的仿真环境。

## 相关文档

- [Python-first, ROS2-ready 路线说明](../../../docs/curriculum/python-first-ros2-ready.md)
- [仿真与 Sim-to-Real 方向页](../README.md)
