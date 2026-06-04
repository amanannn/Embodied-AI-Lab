# 运动控制 (Motion Control) — Level 3: Research Extension

English: [README.en.md](./README.en.md)

## 定位

本目录是运动控制方向的 **研究扩展层**。Level 3 连接课程版本与研究问题、论文和进一步实现路径，不承诺空泛的"完整实现"，而是明确研究边界。

## 研究方向

| 方向 | 研究问题 | 前置要求 |
|------|---------|---------|
| 鲁棒控制 | 如何在模型不确定性和外部扰动下保持稳定控制？ | Level 1 PID + 控制理论基础 |
| 模型预测控制 (MPC) | 如何在约束条件下优化未来多步控制序列？ | Level 1 轨迹优化 + 优化基础 |
| 力控与阻抗控制 | 如何实现与环境的安全交互？ | Level 2 实时控制接口 |

## 与 Level 1/2 的关系

- Level 1 建立 PID 和轨迹优化的算法直觉
- Level 2 将控制算法接入 ROS2 / C++ 实时系统
- Level 3 探索更开放的研究问题和扩展边界

## 相关文档

- [Python-first, ROS2-ready 路线说明](../../../docs/curriculum/python-first-ros2-ready.md)
- [运动控制方向页](../README.md)
