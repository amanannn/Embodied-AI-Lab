# Python-first, ROS2-ready 路线说明

English: [python-first-ros2-ready.en.md](./python-first-ros2-ready.en.md)

## 定位

本项目采用 **Python-first, ROS2-ready** 的路线：

- **Level 1** 用纯 Python 建立核心算法直觉
- **Level 2** 用 ROS2 / C++ / Mixed Bridge 连接真实机器人软件栈
- **Level 3** 连接研究问题与扩展边界

这不是"先写 Python 再写 C++"的简化说法，而是一条有意设计的学习路径：先用最低门槛跑通算法，再用工程手段接入真实系统，最后用研究视角探索开放问题。

## 为什么这样设计

### Level 1: Core Python Lab

Level 1 是当前主产品。

- 纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖
- 可在 Python 3.10+ 运行环境中直接运行
- 目标是让本科生和入门学习者在 30 分钟内跑通第一个实验
- 强调从零实现、可视化、快速反馈和直觉建立

Level 1 不是"低配版本"，而是算法直觉的建立层。很多核心概念（卡尔曼滤波、路径规划、强化学习）用 Python 实现反而更清晰。

### Level 2: ROS2 / C++ / Mixed Bridge

Level 2 是工程桥接层。

- 将 Level 1 的算法直觉接入 ROS2 / C++ / 真实机器人软件栈
- 面向 Ubuntu 开发环境，需要 ROS2 和 C++ 工具链
- 强调性能、ROS2 消息/节点、几何约束和实时性
- 目标是让学习者理解"算法代码如何变成机器人系统的一部分"

Level 2 不是"更高级的版本"，而是工程桥接层。它的价值在于把 Python 原型连接到真实系统，而不是单纯用 C++ 重写一遍。

### Level 3: Research Extension

Level 3 是研究扩展层。

- 连接课程版本与课题、论文和进一步实现路径
- 不承诺空泛的"完整实现"，而是明确研究问题与扩展边界
- 目标是为有志于科研的学习者提供入口

## 设备策略

| 层级 | 开发环境 | 依赖 |
|------|---------|------|
| Level 1 | Python 3.10+ 运行环境 | 以具体实验的 requirements.txt 为准；可使用 venv 或 Conda 管理依赖 |
| Level 2 | Ubuntu | ROS2, C++ 编译工具链, CMake |
| Level 3 | 视研究方向而定 | 视具体课题而定 |

## 在仓库中的体现

- 每个方向的 `level-1-python/` 目录包含可运行的 Python 实验
- 每个方向的 `level-2-ros2-bridge/` 目录包含 ROS2 / C++ 桥接说明
- 每个方向的 `level-3-research/` 目录包含研究扩展方向
- `shared/ros2_interfaces/` 定义 Level 1 → Level 2 的 ROS2 消息和接口契约

## 与旧结构的关系

旧实验已归档到 `archive/legacy-experiments/`，并正在逐步迁移到方向优先结构。归档目录用于保留历史参考和迁移来源；当前学习入口统一从 `experiments/` 开始。映射关系见 `docs/curriculum/legacy-to-direction-map.md`。
