# 运动控制 (Motion Control) — Level 1: Core Python Lab

English: [README.en.md](./README.en.md)

## 定位

本目录是运动控制方向的 **Level 1 入口**。Level 1 是当前主产品：纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Manjaro 或任意 Python 环境中直接运行。

## 规划中的实验

| 实验 | 核心问题 | 状态 |
|------|---------|------|
| PID 控制 | 如何用反馈控制让系统稳定跟踪目标？ | 规划中 |
| 轨迹优化 | 如何生成平滑、可行的运动轨迹？ | 规划中 |

## 与 Level 2 的关系

Level 1 用 Python 建立控制算法直觉。Level 2（`../level-2-ros2-bridge/`）将这些算法接入 ROS2 / C++ / 真实机器人软件栈。

## 相关文档

- [Python-first, ROS2-ready 路线说明](../../../docs/curriculum/python-first-ros2-ready.md)
- [运动控制方向页](../README.md)
