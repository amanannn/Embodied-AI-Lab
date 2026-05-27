# SLAM and Navigation

中文： [README.md](./README.md)

## Core Question

How does a robot know where it is, build maps, and move toward goals safely?

## Why This Direction Matters

SLAM and navigation turn state estimation into a real mobile-robot loop: localization, mapping, planning, and execution begin to operate as one system here.

## Level Structure

- `level-1-python`: MCL, EKF-SLAM, and path planning
- `level-2-cpp-or-mixed`: SLAM frontend and ROS2 navigation integration
- `level-3-research`: semantic SLAM, dynamic-scene navigation, and larger-scale mapping

## Representative Experiments

- `s01-mcl-localization`
- `s02-ekf-slam`
- `s03-path-planning`

## Suggested Entry Point

After completing the Kalman baseline, start with MCL and then expand into EKF-SLAM and path planning.

## Research Extensions

Semantic maps, dynamic obstacle handling, cross-scene navigation, larger mapping pipelines, and production-grade navigation stack integration.
