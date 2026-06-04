# Vertical Applications

English: [README.en.md](./README.en.md)

## Core Question

具身 AI 系统如何在巡检或工业流程等真实场景环境中变得有用？

## Why This Direction Matters

应用方向迫使课程在具体操作约束下整合感知、规划、控制和系统思维。

## Level Structure

- `level-1-python`：**当前主产品**。纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Python 3.10+ 运行环境中直接运行。包含场景范围的应用练习。
- `level-2-ros2-bridge`：**工程桥接层**。将 Level 1 的应用练习接入 ROS2 / C++ / 真实机器人软件栈，面向 Ubuntu 开发环境。
- `level-3-research`：**研究扩展层**。领域特定的系统设计扩展。

## Representative Experiments

- `x01-autonomous-inspection-capstone`

## Suggested Entry Point

完成至少一条感知路径和一条控制或导航路径后，再进入此方向。

## Research Extensions

巡检、制造、仓储、操作约束下的领域特定具身系统设计。
