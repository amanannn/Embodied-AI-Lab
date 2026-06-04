# 运动控制 (Motion Control) — Level 1: Core Python Lab

English: [README.en.md](./README.en.md)

## 定位

本目录是运动控制方向的 **Level 1 入口**。Level 1 是当前主产品：纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Python 3.10+ 运行环境中直接运行。

## 规划中的实验

| 实验 | 核心问题 | 状态 |
|------|---------|------|
| `c01-pid-control-playground` | 如何通过互动调参理解 PID 的比例、积分、微分作用？ | 已完成 |
| 轨迹优化 | 如何生成平滑、可行的运动轨迹？ | 规划中 |

## 快速开始

先运行无浏览器版本，确认仿真和 JSON 输出正常：

```bash
python scripts/pid_playground.py --simulate
```

再启动本地互动页面：

```bash
python scripts/pid_playground.py --serve
```

浏览器打开脚本打印的本地地址，拖动 `Kp / Ki / Kd`、目标位置和扰动力滑块，观察轨迹、控制量和指标变化。

## 已完成实验

### `c01-pid-control-playground`

- 纯 Python PID 仿真核心：一维质量-阻尼系统、短时外力扰动、误差/控制量记录。
- 标准库 HTTP 服务：无需 FastAPI、React 或 Node.js 工程链路。
- 原生 HTML/CSS/JS 前端：滑块调参、Canvas 曲线、超调量、稳态误差、稳定时间和峰值控制量。
- 教程：[pid_playground.md](./tutorials/pid_playground.md)

## 与 Level 2 的关系

Level 1 用 Python 建立控制算法直觉。Level 2（`../level-2-ros2-bridge/`）将这些算法接入 ROS2 / C++ / 真实机器人软件栈。

## 相关文档

- [Python-first, ROS2-ready 路线说明](../../../docs/curriculum/python-first-ros2-ready.md)
- [运动控制方向页](../README.md)
