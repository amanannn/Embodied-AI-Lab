# ros2_interfaces — ROS2 Bridge 接口契约

English: [README.en.md](./README.en.md)

## 定位

本模块是 Level 1 → Level 2 的 ROS2 桥接接口层。它定义 Python 实验输出与 ROS2 系统之间的消息和约定，不包含完整的应用节点或硬件启动逻辑。

## 在 Python-first, ROS2-ready 路线中的角色

```
Level 1 (Python)          Level 2 (ROS2/C++)
    ↓                         ↑
    └── ros2_interfaces ──────┘
         消息定义 + 桥接约定
```

- Level 1 的 Python 实验输出标准格式的数据（CSV、JSON、numpy 数组）
- `ros2_interfaces` 定义这些数据如何映射为 ROS2 消息
- Level 2 的 C++ / ROS2 实验消费这些消息，接入真实机器人软件栈

## 当前内容

本模块当前处于接口规划阶段。未来可承接：

- 自定义 ROS2 消息定义（`.msg` 文件）
- 服务定义（`.srv` 文件）
- Python ↔ ROS2 桥接辅助脚本
- 消息与 Level 1 输出格式的映射文档

## 接口约定

### Level 1 输出格式

| 方向 | 典型输出 | 格式 |
|------|---------|------|
| 感知 | 状态估计、传感器融合 | CSV / numpy 数组 |
| SLAM 与导航 | 路径、位姿、地图 | CSV / JSON |
| 运动控制 | 轨迹、控制信号 | CSV / numpy 数组 |
| 强化学习 | 策略、动作 | numpy 数组 |

### ROS2 消息映射

候选消息定义将遵循以下原则：

- 保持与 Level 1 输出格式的直接对应关系
- 不引入 Level 1 不需要的额外字段
- 优先使用标准 ROS2 消息类型（`geometry_msgs`、`sensor_msgs`、`nav_msgs`）
- 自定义消息仅在标准类型无法覆盖时引入

## 不属于此处

- 单个顶石项目或方向拥有的完整应用节点
- 单个机器人平台的硬件启动笔记
- 未在单个实验之外复用的实验本地回调或启动配置

## 相关文档

- [Python-first, ROS2-ready 路线说明](../../docs/curriculum/python-first-ros2-ready.md)
- [旧实验到新方向结构的映射](../../docs/curriculum/legacy-to-direction-map.md)
