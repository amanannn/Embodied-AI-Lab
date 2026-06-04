# LLM and Robotics

English: [README.en.md](./README.en.md)

## Core Question

语言模型如何分解任务、调用工具并支持机器人决策？

## Why This Direction Matters

大模型越来越多地被用作规划和交互层，但它们必须扎根于可执行的面向机器人的行为。

本仓库的 Phase 1 状态在此很重要：先建立着陆页，以便后续工具使用和 ROS2 连接的智能体工作有明确归属。

## Level Structure

- `level-1-python`：**当前主产品**。纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Python 3.10+ 运行环境中直接运行。包含任务分解与玩具世界工具使用。
- `level-2-ros2-bridge`：**工程桥接层**。将 Level 1 的 LLM 规划算法接入 ROS2 / C++ / 真实机器人软件栈，面向 Ubuntu 开发环境。
- `level-3-research`：**研究扩展层**。长期、多模态与智能体机器人系统。

## Representative Experiments

- `a01-llm-task-planning-agent`
- 后续连接语言规划到 ROS2 执行的工具使用中间件桥接

## Suggested Entry Point

从受限环境中的工具使用规划开始，再尝试真实中间件集成。在本仓库中，Phase 1 意味着方向在迁移实验落地前就已定义，此页面为后续扎根智能体工作设定范围。

## Research Extensions

扎根规划、多模态智能体、执行监控、长期机器人任务编排。
