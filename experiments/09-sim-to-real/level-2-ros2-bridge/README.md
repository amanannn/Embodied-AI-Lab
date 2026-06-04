# 仿真与 Sim-to-Real (Simulation and Sim-to-Real) — Level 2: ROS2 / C++ / Mixed Bridge

English: [README.en.md](./README.en.md)

## 定位

本目录是仿真与 Sim-to-Real 方向的 **工程桥接层**。它的目标不是用 C++ 重写 Level 1 的 Python 代码，而是把 Level 1 建立的算法直觉接入 ROS2 / C++ / 真实机器人软件栈。

## 与 Level 1 的关系

| | Level 1 (Python) | Level 2 (ROS2/C++) |
|---|---|---|
| 目标 | 建立算法直觉 | 桥接真实系统 |
| 环境 | Manjaro / 任意 Python | Ubuntu + ROS2 |
| 依赖 | Python 3.10+, pip | ROS2, CMake, C++ 编译工具链 |
| 输出 | CSV / numpy 数组 | ROS2 消息 / 话题 |

## 桥接方向

仿真与 Sim-to-Real 方向的 Level 2 候选桥接主题包括：

- **仿真器桥接**：将 2D 仿真器接入 Gazebo 或 Isaac Sim
- **域随机化桥接**：将域随机化概念应用到更真实的仿真环境
- **迁移诊断**：建立仿真到真实的性能对比和诊断工具
- **数字孪生**：接入真实机器人的数字孪生框架

## 当前状态

本目录处于接口规划阶段。具体实现将根据 Level 1 实验的成熟度逐步推进。

## 设备要求

- Ubuntu LTS，版本需与目标 ROS2 发行版匹配
- 示例组合：ROS2 Humble + Ubuntu 22.04，ROS2 Jazzy + Ubuntu 24.04
- C++ 编译工具链（CMake, colcon）
- 可选：Gazebo, Isaac Sim

## 相关文档

- [Python-first, ROS2-ready 路线说明](../../../docs/curriculum/python-first-ros2-ready.md)
- [ROS2 Bridge 接口契约](../../../shared/ros2_interfaces/README.md)
- [仿真与 Sim-to-Real Level 1](../level-1-python/README.md)
