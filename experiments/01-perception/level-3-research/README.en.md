# Perception — Level 3: Research Extension

中文： [README.md](./README.md)

## Positioning

This directory is the **research extension layer** for the perception direction. Level 3 connects course versions to research questions, papers, and deeper implementation paths — it does not promise vague "complete implementations" but clarifies research boundaries.

## Research Directions

| Direction | Research Question | Prerequisites |
|-----------|------------------|---------------|
| Multi-sensor Fusion | How to fuse information from heterogeneous sensors (camera, LiDAR, IMU)? | Level 1 fusion foundations + Level 2 ROS2 interfaces |
| 3D Perception | How to reconstruct 3D scene structure from 2D images? | Level 1 camera calibration + point cloud foundations |
| Semantic Reconstruction | How to add semantic information to 3D reconstruction? | Level 1 + deep learning foundations |
| Open-vocabulary Perception | How to recognize object categories not seen during training? | Level 1 + vision-language model foundations |

## Relationship to Level 1/2

- Level 1 builds sensor, filtering, and vision algorithm intuition
- Level 2 connects perception algorithms into ROS2 / C++ real systems
- Level 3 explores more open research questions and extension boundaries

## Related Documentation

- [Python-first, ROS2-ready Path](../../../docs/curriculum/python-first-ros2-ready.en.md)
- [Perception Direction Page](../README.en.md)
