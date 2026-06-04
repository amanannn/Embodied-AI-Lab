# 世界模型 (World Models) — Level 2: ROS2 / C++ / Mixed Bridge

English: [README.en.md](./README.en.md)

## 定位

本目录是世界模型方向的 **工程桥接层**。它的目标不是用 C++ 重写 Level 1 的 Python 代码，而是把 Level 1 建立的算法直觉接入 ROS2 / C++ / 真实机器人软件栈。

## 与 Level 1 的关系

| | Level 1 (Python) | Level 2 (ROS2/C++) |
|---|---|---|
| 目标 | 建立算法直觉 | 桥接真实系统 |
| 环境 | Python 3.10+ 运行环境 | Ubuntu + ROS2 |
| 依赖 | Python 3.10+, pip | ROS2, CMake, C++ 编译工具链 |
| 输出 | CSV / numpy 数组 | ROS2 消息 / 话题 |

## 桥接方向

世界模型方向的 Level 2 候选桥接主题包括：

- **预测模型桥接**：将动力学预测模型接入 ROS2 规划节点
- **rollout 桥接**：将 Python rollout 管线接入 C++ 实时推理
- **规划耦合**：将预测模型与 ROS2 规划器耦合
- **模型服务**：将训练好的模型封装为 ROS2 服务

## 当前状态

本目录处于接口规划阶段。具体实现将根据 Level 1 实验的成熟度逐步推进。

## 设备要求

- Ubuntu LTS，版本需与目标 ROS2 发行版匹配
- 示例组合：ROS2 Humble + Ubuntu 22.04，ROS2 Jazzy + Ubuntu 24.04
- C++ 编译工具链（CMake, colcon）
- 可选：CUDA（如需 GPU 推理）

## 相关文档

- [Python-first, ROS2-ready 路线说明](../../../docs/curriculum/python-first-ros2-ready.md)
- [ROS2 Bridge 接口契约](../../../shared/ros2_interfaces/README.md)
- [世界模型 Level 1](../level-1-python/README.md)
