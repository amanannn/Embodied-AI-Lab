# SLAM 与导航

English: [README.en.md](./README.en.md)

## Core Question

机器人如何知道自己在哪里、如何构建地图、以及如何安全地到达目标？

## Why This Direction Matters

SLAM 与导航把状态估计扩展成真正的移动机器人闭环：定位、地图、规划和执行在这里开始组合成系统。

## Level Structure

- `level-1-python`：Grid Search 路径规划、MCL 与 EKF-SLAM
- `level-2-cpp-or-mixed`：SLAM 前端与 ROS2 导航集成
- `level-3-research`：语义 SLAM、动态场景导航与更大规模地图

## Representative Experiments（候选）

以下为候选实验，尚未实现：

- `s01-grid-search` — A* 与 Dijkstra 路径规划（候选）
- `s02-mcl-localization` — 蒙特卡洛定位（候选）
- `s03-ekf-slam` — 扩展卡尔曼 SLAM（候选）

## Suggested Entry Point

建议在完成卡尔曼滤波基线后，先进入路径规划，再扩展到 MCL 定位与 EKF-SLAM。

## Research Extensions

语义地图、动态避障、跨场景导航、更大尺度地图构建，以及工业级导航栈集成。
