# SLAM and Navigation

中文： [README.md](./README.md)

## Core Question

How does a robot know where it is, build maps, and move toward goals safely?

## Why This Direction Matters

SLAM and navigation turn state estimation into a real mobile-robot loop: localization, mapping, planning, and execution begin to operate as one system here.

## Level Structure

- `level-1-python`: Grid Search path planning, MCL, and EKF-SLAM
- `level-2-cpp-or-mixed`: SLAM frontend and ROS2 navigation integration
- `level-3-research`: semantic SLAM, dynamic-scene navigation, and larger-scale mapping

## Representative Experiments (Candidates)

The following are candidate experiments, not yet implemented:

- `s01-grid-search` — A* and Dijkstra path planning (candidate)
- `s02-mcl-localization` — Monte Carlo Localization (candidate)
- `s03-ekf-slam` — Extended Kalman Filter SLAM (candidate)

## Suggested Entry Point

After completing the Kalman baseline, start with path planning, then expand into MCL localization and EKF-SLAM.

## Research Extensions

Semantic maps, dynamic obstacle handling, cross-scene navigation, larger mapping pipelines, and production-grade navigation stack integration.
