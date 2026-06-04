# 感知 (Perception) — Level 3: Research Extension

English: [README.en.md](./README.en.md)

## 定位

本目录是感知方向的 **研究扩展层**。Level 3 连接课程版本与研究问题、论文和进一步实现路径，不承诺空泛的"完整实现"，而是明确研究边界。

## 研究方向

| 方向 | 研究问题 | 前置要求 |
|------|---------|---------|
| 多传感器融合 | 如何融合异构传感器（相机、LiDAR、IMU）的信息？ | Level 1 融合基础 + Level 2 ROS2 接口 |
| 3D 感知 | 如何从 2D 图像重建 3D 场景结构？ | Level 1 相机标定 + 点云基础 |
| 语义重建 | 如何在 3D 重建中加入语义信息？ | Level 1 + 深度学习基础 |
| 开放词汇感知 | 如何识别训练时未见过的物体类别？ | Level 1 + 视觉语言模型基础 |

## 与 Level 1/2 的关系

- Level 1 建立传感器、滤波和视觉的算法直觉
- Level 2 将感知算法接入 ROS2 / C++ 真实系统
- Level 3 探索更开放的研究问题和扩展边界

## 相关文档

- [Python-first, ROS2-ready 路线说明](../../../docs/curriculum/python-first-ros2-ready.md)
- [感知方向页](../README.md)
