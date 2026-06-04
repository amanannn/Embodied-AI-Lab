# Perception

English: [README.en.md](./README.en.md)

## Core Question

机器人如何把原始观测转化为可用于定位、规划或理解的状态、结构与语义信息？

## Why This Direction Matters

感知是后续状态估计、规划与控制的输入层。没有可解释、可运行的感知基线，后面的闭环都缺少可信起点。

## Level Structure

- `level-1-python`：**当前主产品**。纯 Python 实现，无 ROS2 / Gazebo / Isaac / GPU 依赖，可在 Python 3.10+ 运行环境中直接运行。包含传感器仿真、噪声建模、状态估计、多传感器融合与基础视觉；学习路线见 [`level-1-python/README.md`](./level-1-python/README.md)
- `level-2-ros2-bridge`：**工程桥接层**。将 Level 1 的感知算法接入 ROS2 / C++ / 真实机器人软件栈，面向 Ubuntu 开发环境。
- `level-3-research`：**研究扩展层**。更丰富的融合、多模态感知与开放词汇方向。

## Level 1 Experiments

### Experiment 1: Sensor Fundamentals & Noise Modeling

从零搭建传感器仿真管线：理解噪声 → 模拟传感器 → 多传感器融合。

| 模块 | 内容 |
|---|---|
| `noise.py` | 四种噪声模型（高斯、偏置、漂移、离群值） |
| `sensors/` | GPS、LiDAR、IMU、里程计四类传感器仿真 |
| `fusion.py` | 多传感器加权融合 |
| `scripts/noise.py` | 噪声类型对比实验 |
| `scripts/sensors.py` | 多传感器对比 + CSV 数据输出 |
| `scripts/fusion.py` | 融合效果对比 |

### Experiment 2: Kalman Filter Family

四种滤波器完整实现：KF → EKF → UKF → PF。

| 滤波器 | 适用场景 |
|---|---|
| 线性卡尔曼滤波 (KF) | 线性系统 + 高斯噪声 |
| 扩展卡尔曼滤波 (EKF) | 非线性观测（雷达、激光雷达） |
| 无迹卡尔曼滤波 (UKF) | 强非线性，不需求导 |
| 粒子滤波 (PF) | 任意分布，多模态场景 |

### Quick Start

```bash
cd experiments/01-perception/level-1-python
pip install -r requirements.txt

# 传感器基础实验
python scripts/noise.py        # 噪声对比图
python scripts/sensors.py      # 多传感器对比 + CSV
python scripts/fusion.py       # 融合效果

# 卡尔曼滤波实验
python scripts/kf.py           # 单滤波器
python scripts/all.py          # 四合一对比
python scripts/kf_tuning.py    # 参数实验
```

详细教程见 `level-1-python/tutorials/`。

## 视觉子路线

当前 Level 1 已完成传感器仿真、状态估计与基础视觉三条主线。

当前进度：

| 实验 | 核心问题 | 硬件需求 | 状态 |
|------|---------|---------|------|
| 相机标定 | 如何从棋盘格图像得到内参和畸变系数？ | USB 摄像头 | 已落地 |
| ArUco / AprilTag 检测 | 如何实时检测标记并估计 6DoF 位姿？ | USB 摄像头 | 已落地 |
| 经典 OpenCV 视觉 | 光流、特征匹配、背景减除能做什么？ | USB 摄像头或离线样例 | 已落地 |

这些实验与传感器/滤波器主线并行，不改动现有结构。

推荐硬件配置：USB 摄像头，640×480 @ 30fps。

## Representative Experiments

- `p01-sensor-fundamentals` — 传感器仿真与噪声建模（已落地）
- `p01-state-estimation-kalman` — 四种滤波器完整实现（已落地）
- `p01-camera-calibration` — USB 摄像头棋盘格标定（已落地）
- `p01-aruco-pose` — ArUco / AprilTag 检测与单标记位姿估计（已落地）
- `p01-classic-vision` — 光流、特征匹配、背景减除基础视觉（已落地）
- `p02-vlm-grounding` — 视觉语言 grounding（规划中）
- 后续点云与多传感器融合桥接实验

## Research Extensions

多传感器融合、3D 感知、语义重建、开放词汇感知与具身场景理解。
