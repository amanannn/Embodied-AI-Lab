# ArUco / AprilTag 位姿估计 —— 从视觉标记理解 6DoF

English: [aruco_pose.en.md](./aruco_pose.en.md)

> 配合 `scripts/aruco_pose.py` 使用。本文默认你已经完成或至少理解 `camera_calibration.md`，因为真实位姿估计需要相机内参和畸变系数。

---

## 1. 核心问题

相机标定回答“摄像头自己是什么”。ArUco / AprilTag 位姿估计进一步回答：

- 图像里有没有一个已知视觉标记？
- 这个标记的四个角点在像素中在哪里？
- 如果知道 marker 的真实边长，它相对摄像头的 3D 位姿是多少？

这类 marker 常用于机器人实验中的快速定位、手眼标定、移动机器人 docking、桌面操作和视觉伺服。

---

## 2. 输入与输出

| 项目 | 说明 |
|------|------|
| 输入 marker | `output/aruco_pose/marker_DICT_4X4_50_0.png` |
| 输入图片 | 摄像头拍摄的 marker 图片，或实时摄像头画面 |
| 可选标定 | `output/camera_calibration.json` |
| 输出预览 | `output/aruco_pose/aruco_detection.jpg` |
| 输出数据 | `output/aruco_pose/aruco_detection.json` |

如果没有标定文件，脚本只能检测 marker 的 ID 和角点。如果提供标定文件并设置真实 `--marker-size`，脚本会估计 `rvec` 和 `tvec_m`。

---

## 3. 生成 Marker

```bash
cd experiments/01-perception/level-1-python
python scripts/aruco_pose.py --generate-marker
```

默认生成：

```text
output/aruco_pose/marker_DICT_4X4_50_0.png
```

你可以把它打印出来，或者显示在另一块屏幕上。打印时建议量一下真实边长，然后把边长填入 `--marker-size`。

---

## 4. 离线检测图片

如果你已经拍了一张 marker 图片：

```bash
python scripts/aruco_pose.py --input-image path/to/marker_photo.jpg
```

这会检测 marker 并输出预览图，但不会给出可靠的 3D 位姿。

如果已经完成相机标定：

```bash
python scripts/aruco_pose.py \
  --input-image path/to/marker_photo.jpg \
  --calibration output/camera_calibration.json \
  --marker-size 0.05
```

其中 `--marker-size 0.05` 表示 marker 真实边长为 5 cm。

---

## 5. 实时摄像头模式

```bash
python scripts/aruco_pose.py \
  --calibration output/camera_calibration.json \
  --marker-size 0.05
```

窗口打开后，把 marker 放到摄像头前。按 `q` 退出。

如果你的摄像头不是默认 `/dev/video0`，尝试：

```bash
python scripts/aruco_pose.py --camera-index 1
```

---

## 6. 代码走读

脚本核心流程：

1. `cv2.aruco.getPredefinedDictionary` 选择 marker 字典。
2. `cv2.aruco.ArucoDetector` 检测 marker 角点和 ID。
3. `cv2.aruco.estimatePoseSingleMarkers` 根据角点、相机内参和真实边长估计位姿。
4. `cv2.drawFrameAxes` 在图像上绘制相机坐标轴。
5. 保存预览图和 JSON 结果。

JSON 中最重要的是：

```json
{
  "id": 0,
  "rvec": [0.1, 0.2, 0.3],
  "tvec_m": [0.0, 0.0, 0.5]
}
```

`tvec_m` 表示 marker 原点在相机坐标系下的位置，单位是米。第三个数通常可以理解为 marker 距离相机的大致深度。

---

## 7. 常见问题

### 能检测到 ID，但没有位姿

检查是否传入了 `--calibration`，或者默认的 `output/camera_calibration.json` 是否存在。

### 位姿数值明显不对

优先检查 `--marker-size` 是否是真实边长，单位必须是米。其次检查相机标定质量。

### 检测不稳定

保证 marker 占据足够像素、边缘清晰、光照均匀。过远、模糊、反光都会导致角点不稳定。

---

## 8. 下一步

完成这个实验后，你已经有了一个基础视觉定位闭环：

```text
相机标定 -> marker 检测 -> 单目标 6DoF 位姿估计
```

这条链路可以继续扩展到视觉伺服、机器人对接、手眼标定和桌面操作。
