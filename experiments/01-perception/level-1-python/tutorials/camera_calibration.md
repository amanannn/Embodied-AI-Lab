# 相机标定 —— 从普通摄像头得到机器人可用的视觉参数

English: [camera_calibration.en.md](./camera_calibration.en.md)

> 配合 `scripts/camera_calibration.py` 使用。本文面向只有 Python 和 OpenCV 基础的学习者，目标是把 USB 摄像头变成一个有内参、有畸变模型、可继续做 ArUco 位姿估计的视觉传感器。

---

## 1. 核心问题

普通摄像头只给你一张二维图片。机器人视觉需要回答更具体的问题：

- 一个像素点对应相机坐标系中的哪条光线？
- 镜头边缘为什么会弯曲，如何补偿？
- 后续做 ArUco、AprilTag、PnP 位姿估计时，相机参数从哪里来？

相机标定要估计两类参数：

| 参数 | 含义 | 用途 |
|------|------|------|
| `camera_matrix` | 焦距 `fx/fy` 与主点 `cx/cy` | 像素坐标和相机坐标之间的投影关系 |
| `dist_coeffs` | 径向/切向畸变系数 | 校正广角、廉价镜头带来的图像弯曲 |

没有相机标定，后续的 3D 位姿估计会变成“看起来能跑，但数值没有物理意义”。

---

## 2. 实验输入与输出

输入是一组不同角度拍摄的棋盘格图片。棋盘格的“内角点”数量默认为 `9 x 6`，也就是 OpenCV 要检测的黑白方块交界点。

输出文件：

| 文件 | 说明 |
|------|------|
| `output/checkerboard_9x6.png` | 自动生成的棋盘格图片 |
| `calibration_images/*.jpg` | 摄像头采集或手动放入的标定图片 |
| `output/camera_calibration.json` | 相机内参、畸变参数和误差指标 |
| `output/camera_calibration/corners_*.jpg` | 角点检测预览 |
| `output/camera_calibration/undistorted_*.jpg` | 去畸变预览 |

---

## 3. 环境准备

进入感知 Level 1 目录：

```bash
cd experiments/01-perception/level-1-python
pip install -r requirements.txt
```

如果你使用 conda：

```bash
conda activate embodied-ai
python scripts/camera_calibration.py --generate-board
```

推荐硬件：普通 USB 摄像头，`640x480 @ 30fps` 即可。这个实验不需要深度相机。

---

## 4. 生成棋盘格

先生成默认 `9 x 6` 内角点棋盘格：

```bash
python scripts/camera_calibration.py --generate-board
```

默认输出：

```text
output/checkerboard_9x6.png
```

使用方式有两种：

- 打印到纸上，并尽量贴平。
- 在另一块屏幕上打开图片，用摄像头拍摄屏幕。

更推荐打印，因为屏幕反光和摩尔纹会降低角点质量。

---

## 5. 采集图片

用摄像头采集：

```bash
python scripts/camera_calibration.py --capture --input-dir calibration_images
```

窗口打开后：

| 按键 | 作用 |
|------|------|
| `Space` 或 `s` | 保存当前帧 |
| `q` | 退出采集 |

拍摄建议：

- 至少 15 张，课程脚本默认 3 张即可运行是为了降低首次上手门槛。
- 棋盘格要覆盖画面中心、四角、边缘。
- 包含近距离、远距离、倾斜角度。
- 不要只拍正对镜头的一组图片，否则内参会不稳定。

如果你已经用手机或其他方式拍好了图片，只要把图片放入：

```text
experiments/01-perception/level-1-python/calibration_images/
```

然后直接运行离线标定。

---

## 6. 执行标定

```bash
python scripts/camera_calibration.py --input-dir calibration_images
```

脚本会依次完成：

1. 读取图片。
2. 使用 `cv2.findChessboardCorners` 检测棋盘格角点。
3. 使用 `cv2.cornerSubPix` 做亚像素级角点优化。
4. 使用 `cv2.calibrateCamera` 估计相机参数。
5. 保存 JSON、角点预览和去畸变预览。

你会看到类似输出：

```text
→ 标定结果
  有效图片: 18
  RMS 重投影误差: 0.3421 px
  平均重投影误差: 0.0478 px
  输出 JSON: output/camera_calibration.json
```

---

## 7. 如何判断结果好不好

优先看三个信号：

| 信号 | 好结果 | 常见问题 |
|------|--------|----------|
| 有效图片数量 | 15-25 张 | 只有几张，角度单一 |
| RMS 重投影误差 | 通常小于 1.0 px | 大于 2.0 px 时要检查图片质量 |
| 角点预览 | 每个内角点都贴合棋盘格 | 检测错格、模糊、反光 |

`camera_matrix` 里最重要的是：

```json
[
  [fx, 0, cx],
  [0, fy, cy],
  [0, 0, 1]
]
```

其中 `fx/fy` 是像素单位下的焦距，`cx/cy` 是图像主点。后续做 ArUco 位姿估计时，这个矩阵会直接传给 OpenCV 的 PnP 求解函数。

---

## 8. 常见问题

### 检测不到棋盘格

检查 `--board-cols` 和 `--board-rows` 是否填的是“内角点数量”，不是方块数量。默认 `9 x 6` 内角点意味着真实棋盘格有 `10 x 7` 个方块。

### 标定误差很大

通常是图片角度太单一、画面模糊、棋盘格不平、光照反光、或者拍摄时棋盘格只在画面中央。重新拍摄比调参数更有效。

### 摄像头打不开

Linux 下 `/dev/video0` 通常对应 `--camera-index 0`。如果有多个摄像头，可以尝试：

```bash
python scripts/camera_calibration.py --capture --camera-index 1
```

### GUI 窗口无法打开

可以不用 `--capture`，手动把图片放进 `calibration_images/`，再运行：

```bash
python scripts/camera_calibration.py --input-dir calibration_images
```

---

## 9. 下一步

相机标定是视觉路线的入口。完成后可以继续做：

- ArUco / AprilTag 检测与 6DoF 位姿估计。
- 单目视觉中的 PnP、投影、重投影误差理解。
- 视觉伺服和目标跟踪实验。

它对应具身智能闭环里的“看见世界之前，先知道相机自己是什么”。
