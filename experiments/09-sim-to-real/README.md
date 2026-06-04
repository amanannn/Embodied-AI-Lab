# Simulation and Sim-to-Real

English: [README.en.md](./README.en.md)

## Core Question

如何在仿真中构建训练环境并减少与真实具身系统的差距？

## Why This Direction Matters

仿真实验是可扩展的，但迁移质量决定了这些工作是否在仿真器之外有意义。

Phase 1 给此方向带来直接的仓库关联：`archive/legacy-experiments/04-robot-sim` 映射到此，未来可复用的仿真器组件预计将迁移到 `shared/sim2d` 而非埋在单个实验中。

## Level Structure

- `level-1-python`：**当前主产品**。纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Python 3.10+ 运行环境中直接运行。包含仿真器基础与域随机化概念。
- `level-2-ros2-bridge`：**工程桥接层**。将 Level 1 的仿真基础接入 ROS2 / C++ / 更丰富的仿真器，面向 Ubuntu 开发环境。
- `level-3-research`：**研究扩展层**。迁移、辨识与数字孪生扩展。

## Representative Experiments

- `r01-sim2d-foundation`
- 后续从 `archive/legacy-experiments/04-robot-sim` 迁移到方向内实验和 `shared/sim2d` 的桥接

## Suggested Entry Point

使用简化仿真器基础理解世界步进和观测循环，再进入迁移导向的工作。在 Phase 1，此页面明确仓库尚未包含此方向的迁移实验，即使方向已拥有 sim-to-real 路径。

## Research Extensions

域随机化、系统辨识、仿真器保真度权衡、迁移诊断。
