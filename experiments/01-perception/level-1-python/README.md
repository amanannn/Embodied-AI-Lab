# 感知 Level 1：Python 基础实验路线

English: [README.en.md](./README.en.md)

本目录是感知方向的 Level 1 入口，目标是用 Python 建立本科生可操作的感知直觉：先理解传感器误差，再理解状态估计，最后进入基础视觉。

这里的核心原则是：**先运行，先观察，再理解公式和代码**。

## 学习路线

| 实验 | 核心问题 | 运行入口 | 教程 |
|------|---------|---------|------|
| Lab 01: Sensor Noise | 传感器读数为什么不等于真值？ | `python scripts/noise.py` | `tutorials/sensors.md` |
| Lab 02: Sensor Simulation | GPS、LiDAR、IMU、里程计分别输出什么？ | `python scripts/sensors.py` | `tutorials/sensors.md` |
| Lab 03: Multi-Sensor Fusion 多传感器融合 | 多个不完美测量如何合成一个更可信状态？ | `python scripts/fusion.py` | `tutorials/sensors.md` |
| Lab 04: Kalman Filter Family | 如何用滤波器从 noisy observation 得到 state estimate？ | `python scripts/kf.py`, `python scripts/all.py` | `tutorials/kalman.md` |
| Lab 05: Camera Calibration | 如何得到 USB 摄像头内参和畸变系数？ | `python scripts/camera_calibration.py --generate-board` | `tutorials/camera_calibration.md` |
| Lab 06: ArUco / AprilTag Pose | 如何检测视觉标记并估计位姿？ | `python scripts/aruco_pose.py --generate-marker` | `tutorials/aruco_pose.md` |
| Lab 07: Classic OpenCV Vision | 光流、特征匹配、背景减除能解决什么基础视觉问题？ | `python scripts/classic_vision.py --generate-sample --mode optical-flow` | `tutorials/classic_vision.md` |

推荐学习顺序是：

```text
Lab 01 -> Lab 02 -> Lab 03 -> Lab 04 -> Lab 05 -> Lab 06 -> Lab 07
```

前四个实验建立状态估计基础，第五个实验开始进入真实 USB 摄像头视觉，第六个实验完成 marker 级 6DoF 位姿估计，第七个实验补足基础图像运动、局部对应和前景检测直觉。

## 代码模块

| 模块 | 职责 |
|------|------|
| `noise.py` | 高斯噪声、偏置、漂移、离群值等基础噪声模型 |
| `sensors/` | GPS、LiDAR、IMU、里程计仿真 |
| `fusion.py` | 时间对齐与加权融合 |
| `filters/` | KF、EKF、UKF、PF 实现 |
| `scripts/` | 可直接运行的实验入口 |
| `tutorials/` | 中英双语教程 |
| `utils/` | 轨迹生成、仿真辅助和可视化工具 |

这里的目录按工程职责组织，不等同于学习顺序。学习时先看“学习路线”，开发时再进入对应模块。

## 快速开始

```bash
pip install -r requirements.txt

python scripts/noise.py
python scripts/sensors.py
python scripts/fusion.py
python scripts/all.py
python scripts/camera_calibration.py --generate-board
python scripts/aruco_pose.py --generate-marker
python scripts/classic_vision.py --generate-sample --mode optical-flow
```

ArUco 实验需要 `cv2.aruco`，因此依赖使用 `opencv-contrib-python`。如果你的环境之前手动安装过 `opencv-python`，建议先卸载旧包再安装本目录依赖，避免两个 OpenCV 包混用。

所有生成图片、CSV、JSON 和动画默认保存到 `output/`。该目录用于实验观察，不跟踪到 git。

## 观察重点

- 噪声实验：比较高斯噪声、偏置、漂移和离群值的形态差异。
- 传感器实验：观察不同传感器的观测维度、采样率和误差特性。
- 融合实验：观察加权融合为什么通常比单一传感器更稳定。
- 滤波实验：比较 KF、EKF、UKF、PF 在不同非线性和噪声条件下的效果。
- 相机标定实验：观察棋盘格角点、重投影误差、内参矩阵和去畸变结果。
- ArUco 实验：观察 marker ID、角点、坐标轴和 `tvec_m` 的物理含义。
- 经典视觉实验：观察光流箭头、ORB 匹配线和背景减除 mask。

## 当前边界

- 本层重点是低门槛、可运行、可观察，不追求工业级性能。
- C++ 工程强化放在 `../level-2-ros2-bridge/`。
- 研究扩展放在 `../level-3-research/`。
- 当前视觉实验基于普通 USB 摄像头，不包含 RGB-D、点云或高精度 3D 重建。
