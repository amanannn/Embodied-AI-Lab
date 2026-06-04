# SLAM and Navigation — Level 2: ROS2 / C++ / Mixed Bridge

中文： [README.md](./README.md)

## Positioning

This directory is the **engineering bridge layer** for the SLAM and navigation direction. Its goal is not to rewrite Level 1 Python code in C++, but to connect the algorithmic intuition built in Level 1 into ROS2 / C++ / real robot software stacks.

## Relationship to Level 1

| | Level 1 (Python) | Level 2 (ROS2/C++) |
|---|---|---|
| Goal | Build algorithmic intuition | Bridge into real systems |
| Environment | Python 3.10+ runtime environment | Ubuntu + ROS2 |
| Dependencies | Python 3.10+, pip | ROS2, CMake, C++ toolchain |
| Output | CSV / numpy arrays | ROS2 messages / topics |

## Bridge Directions

Candidate bridge themes for the SLAM and navigation direction include:

- **Path planning bridge**: Connect A* / Dijkstra Python implementations into ROS2 navigation stack (`nav_msgs`)
- **Localization bridge**: Connect MCL localization into ROS2's AMCL or custom localization nodes
- **SLAM frontend**: Extend EKF-SLAM Python prototype into a ROS2 SLAM frontend
- **Navigation integration**: Explore ROS2 Navigation2 integration to understand the perception-planning-execution loop

## Current State

This directory is in the interface planning stage. Specific implementations will progress based on the maturity of Level 1 experiments.

## Device Requirements

- Ubuntu LTS matched to the target ROS2 distribution
- Example pairs: ROS2 Humble + Ubuntu 22.04, ROS2 Jazzy + Ubuntu 24.04
- C++ build toolchain (CMake, colcon)
- Optional: Navigation2 stack

## Related Documentation

- [Python-first, ROS2-ready Path](../../../docs/curriculum/python-first-ros2-ready.en.md)
- [ROS2 Bridge Interface Contract](../../../shared/ros2_interfaces/README.en.md)
- [SLAM and Navigation Level 1](../level-1-python/README.en.md)
