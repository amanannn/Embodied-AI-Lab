# ros2_interfaces

English: [README.en.md](./README.en.md)

本模块将定义教育实验与 ROS2 集成工作之间的边界。

## Intended Ownership

- ROS2 集成点的薄包装
- 发布者、订阅者或消息的共享约定
- 混合 Python 和 C++ 方向实验共用的桥接代码

## Out of Scope

以下内容不属于此处：

- 单个顶石项目或方向拥有的完整应用节点
- 单个机器人平台的硬件启动笔记
- 未在单个实验之外复用的实验本地回调或启动配置
