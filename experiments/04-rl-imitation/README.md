# RL and Imitation

English: [README.en.md](./README.en.md)

## Core Question

机器人如何从奖励信号、示教或策略迭代中学习行为？

## Why This Direction Matters

当手写控制逻辑过于脆弱或成本太高时，基于学习的行为变得重要。

## Level Structure

- `level-1-python`：**当前主产品**。纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Python 3.10+ 运行环境中直接运行。包含 Q-learning、深度强化学习与模仿学习。
- `level-2-ros2-bridge`：**工程桥接层**。将 Level 1 的策略学习算法接入 ROS2 / C++ / 真实机器人软件栈，面向 Ubuntu 开发环境。
- `level-3-research`：**研究扩展层**。sim-to-real RL、离线 RL 与迁移扩展。

## Representative Experiments

- `l01-q-learning`
- `l02-deep-rl`
- `l03-imitation-learning`

## Suggested Entry Point

先用表格 Q-learning 建立直觉，再进入深度策略或基于示教的行为学习。

## Research Extensions

离线 RL、策略迁移、安全探索、示教高效学习。
