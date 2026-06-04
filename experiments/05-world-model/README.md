# World Models

English: [README.en.md](./README.en.md)

## Core Question

机器人能否学习预测性动力学或潜在结构，以在行动前进行规划？

## Why This Direction Matters

预测建模帮助机器人推理后果，而不是仅在当前时间步做出反应。

在本仓库中，Phase 1 先建立方向着陆页，尚未包含迁移实验，仅保留定位所需的占位工作。

## Level Structure

- `level-1-python`：**当前主产品**。纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Python 3.10+ 运行环境中直接运行。包含动力学预测与短期展开。
- `level-2-ros2-bridge`：**工程桥接层**。将 Level 1 的预测模型接入 ROS2 / C++ / 真实机器人软件栈，面向 Ubuntu 开发环境。
- `level-3-research`：**研究扩展层**。潜在世界模型与基于模型的强化学习。

## Representative Experiments

- `w01-dynamics-prediction-world-model`
- 后续从控制、RL 和 sim-to-real 迁移的桥接实验

## Suggested Entry Point

在接触潜在或多模态世界模型之前，先在玩具环境中建立简单预测模型。在 Phase 1，此页面作为课程契约，即使仓库尚未包含该方向的迁移实验。

## Research Extensions

潜在展开、在学习模型上规划、多模态预测、基于模型的具身智能体。
