# scripts/ — 运行清单与输出说明

## 脚本列表

脚本目录是“运行入口”，不是完整学习地图。学习顺序请先看上一级 [`README.md`](../README.md)，再回到这里执行对应脚本。

| 脚本 | 用途 | 输出文件 |
|------|------|---------|
| `noise.py` | 四种噪声类型对比 | `output/noise_comparison.png` |
| `sensors.py` | 四类传感器仿真 + CSV 数据导出 | `output/sensors_*.png`, `output/sensor_data.csv` |
| `fusion.py` | 多传感器加权融合 | `output/fusion_*.png` |
| `kf.py` | 线性卡尔曼滤波 | `output/kf_static.png`, `output/kf_animated.gif` |
| `ekf.py` | 扩展卡尔曼滤波 | `output/ekf_static.png`, `output/ekf_animated.gif` |
| `ukf.py` | 无迹卡尔曼滤波 | `output/ukf_static.png`, `output/ukf_animated.gif` |
| `pf.py` | 粒子滤波 | `output/pf_static.png`, `output/pf_animated.gif` |
| `all.py` | 四合一对比 | `output/all_filters_comparison.png` |
| `kf_tuning.py` | 参数调优实验（五组对比） | `output/kf_tuning_comparison.png` |
| `camera_calibration.py` | USB 摄像头棋盘格标定 | `output/camera_calibration.json`, `output/camera_calibration/` |
| `aruco_pose.py` | ArUco / AprilTag 检测与 6DoF 位姿估计 | `output/aruco_pose/marker_*.png`, `output/aruco_pose/aruco_detection.*` |
| `classic_vision.py` | 光流、特征匹配、背景减除基础视觉 | `output/classic_vision/*.jpg`, `output/classic_vision/sample_motion.mp4` |

## 运行方式

```bash
# 环境准备
pip install -r requirements.txt

# 单独运行
python scripts/noise.py
python scripts/sensors.py
python scripts/fusion.py
python scripts/kf.py
python scripts/ekf.py
python scripts/ukf.py
python scripts/pf.py
python scripts/all.py          # 推荐：四合一对比
python scripts/kf_tuning.py    # 参数调优实验

# 使用外部数据（实验性入口）
python scripts/kf.py --data path/to/preprocessed_position_data.csv

# 视觉：相机标定
python scripts/camera_calibration.py --generate-board
python scripts/camera_calibration.py --input-dir calibration_images

# 视觉：ArUco / AprilTag
python scripts/aruco_pose.py --generate-marker
python scripts/aruco_pose.py --input-image path/to/marker_photo.jpg

# 视觉：经典 OpenCV
python scripts/classic_vision.py --generate-sample --mode optical-flow
python scripts/classic_vision.py --generate-sample --mode feature-matching
python scripts/classic_vision.py --generate-sample --mode background-subtraction
```

## 输出文件说明

所有输出保存到 `output/` 目录（不跟踪到 git）。

- `*_static.png` — 静态对比图（轨迹 + 误差 + 不确定性椭圆）
- `*_animated.gif` — 追踪动画
- `sensor_data.csv` — 传感器仿真导出数据（原始多传感器混合流，用于观察数据格式与后续预处理）
- `all_filters_comparison.png` — 四种滤波器并排对比 + RMSE 柱状图
- `camera_calibration.json` — 相机内参、畸变系数、重投影误差与有效图片清单
- `camera_calibration/` — 角点检测预览与去畸变预览图
- `aruco_pose/` — marker 图片、检测预览和位姿 JSON
- `classic_vision/` — 光流、特征匹配、背景减除预览图和样例视频
