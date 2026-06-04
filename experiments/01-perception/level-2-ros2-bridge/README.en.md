# Perception — Level 2: ROS2 / C++ / Mixed Bridge

中文： [README.md](./README.md)

## Positioning

This directory is the **engineering bridge layer** for the perception direction. Its goal is not to rewrite Level 1 Python code in C++, but to connect the algorithmic intuition built in Level 1 into ROS2 / C++ / real robot software stacks.

## Relationship to Level 1

| | Level 1 (Python) | Level 2 (ROS2/C++) |
|---|---|---|
| Goal | Build algorithmic intuition | Bridge into real systems |
| Environment | Manjaro / any Python | Ubuntu + ROS2 |
| Dependencies | Python 3.10+, pip | ROS2, CMake, C++ toolchain |
| Output | CSV / numpy arrays | ROS2 messages / topics |

## Bridge Directions

Candidate bridge themes for the perception direction include:

- **State estimation bridge**: Connect KF / EKF / UKF / PF Python implementations into ROS2 topics via pybind11 or ROS2 Python nodes
- **Sensor fusion bridge**: Connect multi-sensor fusion algorithms into standard `sensor_msgs` messages
- **Vision pipeline bridge**: Wrap camera calibration, ArUco detection, and optical flow OpenCV pipelines as ROS2 nodes
- **Point cloud processing**: Extend from 2D vision to 3D point clouds, connecting into `sensor_msgs/PointCloud2`

## Current State

This directory is in the interface planning stage. Specific implementations will progress based on the maturity of Level 1 experiments.

## Device Requirements

- Ubuntu LTS matched to the target ROS2 distribution
- Example pairs: ROS2 Humble + Ubuntu 22.04, ROS2 Jazzy + Ubuntu 24.04
- C++ build toolchain (CMake, colcon)
- Optional: pybind11 (Python ↔ C++ bindings)

## Related Documentation

- [Python-first, ROS2-ready Path](../../../docs/curriculum/python-first-ros2-ready.en.md)
- [ROS2 Bridge Interface Contract](../../../shared/ros2_interfaces/README.en.md)
- [Perception Level 1](../level-1-python/README.en.md)
