# 机器人操作 (Manipulation) — Level 1: Core Python Lab

English: [README.en.md](./README.en.md)

## 定位

本目录是机器人操作方向的 **Level 1 入口**。Level 1 是当前主产品：纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Manjaro 或任意 Python 环境中直接运行。

## 规划中的实验

| 实验 | 核心问题 | 状态 |
|------|---------|------|
| 机器人运动学 | 如何计算机械臂的正/逆运动学？ | 规划中 |
| 机械臂动力学 | 如何控制机械臂的力和运动？ | 规划中 |
| 抓取与放置 | 如何规划和执行抓取-放置任务？ | 规划中 |

## 与 Level 2 的关系

Level 1 用 Python 建立操作算法直觉。Level 2（`../level-2-ros2-bridge/`）将这些算法接入 ROS2 / C++ / 真实机器人软件栈。

## 相关文档

- [Python-first, ROS2-ready 路线说明](../../../docs/curriculum/python-first-ros2-ready.md)
- [机器人操作方向页](../README.md)
