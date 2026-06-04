# Vision-Language Navigation

English: [README.en.md](./README.en.md)

## Core Question

机器人如何理解自然语言目标并朝语义相关的目标移动？

## Why This Direction Matters

此方向将语言理解、感知、记忆和导航连接在单一具身循环中。

在当前仓库中，此方向有意超前于代码迁移：Phase 1 先建立着陆页，以便后续语言 grounding 和语义导航工作能挂载到稳定的方向主页。

## Level Structure

- `level-1-python`：**当前主产品**。纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Python 3.10+ 运行环境中直接运行。包含语言目标解析与语义目标搜索。
- `level-2-ros2-bridge`：**工程桥接层**。将 Level 1 的语言导航算法接入 ROS2 / C++ / 真实机器人软件栈，面向 Ubuntu 开发环境。
- `level-3-research`：**研究扩展层**。长期 VLN 与基准导向的扩展。

## Representative Experiments

- `v01-language-goal-navigation`
- 后续从感知和垂直应用工作的语言 grounding 桥接

## Suggested Entry Point

从玩具语义搜索任务开始，再尝试更丰富的导航环境。在 Phase 1，此 README 作为可发现性锚点，因为仓库尚未包含此方向下的迁移实验。

## Research Extensions

长期指令跟随、语义地图、多模态记忆、基准级 VLN 系统。
