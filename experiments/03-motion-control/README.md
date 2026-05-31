# Motion Control

English: [README.en.md](./README.en.md)

## Core Question

机器人如何准确、鲁棒地跟踪期望运动，并提供可解释的反馈行为？

## Why This Direction Matters

控制是让模型、传感器和执行器从"期望轨迹"变成"可靠物理运动"的关键环节。

## Level Structure

- `level-1-python`：PID 与轨迹优化
- `level-2-cpp-or-mixed`：实时控制强化
- `level-3-research`：鲁棒控制、MPC、力控与安全约束

## Representative Experiments

- `c01-pid-control-lab`
- `c02-trajectory-optimization`
- 后续实时桥接工作

## Suggested Entry Point

从 PID 入手，再用轨迹优化把反馈控制与更平滑的运动生成连接起来。

## Research Extensions

力控、阻抗行为、MPC、扰动鲁棒控制等更真实系统中的研究方向。
