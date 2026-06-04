# Motion Control

English: [README.en.md](./README.en.md)

## Core Question

机器人如何准确、鲁棒地跟踪期望运动，并提供可解释的反馈行为？

## Why This Direction Matters

控制是让模型、传感器和执行器从"期望轨迹"变成"可靠物理运动"的关键环节。

## Level Structure

- `level-1-python`：**当前主产品**。纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Python 3.10+ 运行环境中直接运行。包含 PID 与轨迹优化。
- `level-2-ros2-bridge`：**工程桥接层**。将 Level 1 的控制算法接入 ROS2 / C++ / 真实机器人软件栈，面向 Ubuntu 开发环境。
- `level-3-research`：**研究扩展层**。鲁棒控制、MPC、力控与安全约束。

## Representative Experiments

- `c01-pid-control-playground`（已完成）：Python + 浏览器互动 PID 控制实验
- `c02-trajectory-optimization`
- 后续实时桥接工作

## Suggested Entry Point

从 `level-1-python` 的 `c01-pid-control-playground` 入手，先用滑块观察 `Kp / Ki / Kd` 如何改变超调、振荡、稳定时间和抗扰恢复，再用轨迹优化把反馈控制与更平滑的运动生成连接起来。

## Research Extensions

力控、阻抗行为、MPC、扰动鲁棒控制等更真实系统中的研究方向。
