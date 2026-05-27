# SLAM 与导航

English: [README.en.md](./README.en.md)

## Core Question

机器人如何知道自己在哪里、如何构建地图、以及如何安全地到达目标？

## Why This Direction Matters

SLAM 与导航把状态估计扩展成真正的移动机器人闭环：定位、地图、规划和执行在这里开始组合成系统。

## Level Structure

- `level-1-python`：MCL、EKF-SLAM 与路径规划
- `level-2-cpp-or-mixed`：SLAM 前端与 ROS2 导航集成
- `level-3-research`：语义 SLAM、动态场景导航与更大规模地图

## Representative Experiments

- `s01-mcl-localization`
- `s02-ekf-slam`
- `s03-path-planning`

## Suggested Entry Point

建议在完成卡尔曼滤波基线后，先进入 MCL，再扩展到 EKF-SLAM 与路径规划。

## Research Extensions

语义地图、动态避障、跨场景导航、更大尺度地图构建，以及工业级导航栈集成。
