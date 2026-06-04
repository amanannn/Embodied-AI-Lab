# Manipulation

English: [README.en.md](./README.en.md)

## Core Question

机器人如何移动身体和末端执行器以有目的地与物体交互？

## Why This Direction Matters

操作是运动学、动力学、感知和控制与物体交互的交汇点。

## Level Structure

- `level-1-python`：**当前主产品**。纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Manjaro 或任意 Python 环境中直接运行。包含运动学、机械臂动力学与抓取。
- `level-2-ros2-bridge`：**工程桥接层**。将 Level 1 的操作算法接入 ROS2 / C++ / 真实机器人软件栈，面向 Ubuntu 开发环境。
- `level-3-research`：**研究扩展层**。灵巧操作、力引导操作与双臂协调。

## Representative Experiments

- `m01-robot-kinematics`
- `m02-manipulator-dynamics-control`
- `m03-grasping-and-placing`

## Suggested Entry Point

从运动学入手，再进入机械臂动力学，最后攻克抓取管线。

## Research Extensions

力引导操作、灵巧手、双臂协调、任务条件化操作。
