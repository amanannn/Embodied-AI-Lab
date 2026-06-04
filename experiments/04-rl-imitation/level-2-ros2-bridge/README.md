# RL 与模仿学习 (RL and Imitation) — Level 2: ROS2 / C++ / Mixed Bridge

English: [README.en.md](./README.en.md)

## 定位

本目录是强化学习与模仿学习方向的 **工程桥接层**。它的目标不是用 C++ 重写 Level 1 的 Python 代码，而是把 Level 1 建立的算法直觉接入 ROS2 / C++ / 真实机器人软件栈。

## 与 Level 1 的关系

| | Level 1 (Python) | Level 2 (ROS2/C++) |
|---|---|---|
| 目标 | 建立算法直觉 | 桥接真实系统 |
| 环境 | Python 3.10+ 运行环境 | Ubuntu + ROS2 |
| 依赖 | Python 3.10+, pip | ROS2, CMake, C++ 编译工具链 |
| 输出 | CSV / numpy 数组 | ROS2 消息 / 话题 |

## 桥接方向

RL 与模仿学习方向的 Level 2 候选桥接主题包括：

- **策略部署桥接**：将训练好的策略模型接入 ROS2 执行节点
- **环境接口桥接**：将 Gym-like 环境接口映射到 ROS2 服务/动作
- **示教数据桥接**：将示教数据采集管线接入 ROS2 消息系统
- **在线学习桥接**：将在线 RL 算法接入实时机器人控制循环

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
- [RL 与模仿学习 Level 1](../level-1-python/README.md)
