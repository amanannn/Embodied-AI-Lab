# SLAM and Navigation

中文： [README.md](./README.md)

## Core Question

How does a robot know where it is, build maps, and move toward goals safely?

## Why This Direction Matters

SLAM and navigation turn state estimation into a real mobile-robot loop: localization, mapping, planning, and execution begin to operate as one system here.

## Level Structure

- `level-1-python`: **Current main product**. Pure Python implementations with no ROS2 / Gazebo / Isaac / GPU dependencies, runnable in a Python 3.10+ runtime environment. Covers Grid Search path planning, MCL, and EKF-SLAM.
- `level-2-ros2-bridge`: **Engineering bridge layer**. Connects Level 1 localization and planning algorithms into ROS2 / C++ / real robot software stacks, targeting Ubuntu environments.
- `level-3-research`: **Research extension layer**. Semantic SLAM, dynamic-scene navigation, and larger-scale mapping.

## Representative Experiments

Level 1 now includes three implemented core labs covering path planning, known-map localization, and basic SLAM:

- `s01-grid-search` — A* and Dijkstra path planning (complete)
- `s02-mcl-localization` — Monte Carlo Localization (complete)
- `s03-ekf-slam` — Extended Kalman Filter SLAM (complete)

## Suggested Entry Point

After completing the Kalman baseline, enter [`level-1-python`](level-1-python/README.en.md): run Grid Search first, then MCL localization, then EKF-SLAM.

## Research Extensions

Semantic maps, dynamic obstacle handling, cross-scene navigation, larger mapping pipelines, and production-grade navigation stack integration.
