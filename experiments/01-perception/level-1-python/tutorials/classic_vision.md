# 经典 OpenCV 视觉 —— 光流、特征匹配与背景减除

English: [classic_vision.en.md](./classic_vision.en.md)

> 配合 `scripts/classic_vision.py` 使用。本文目标不是堆算法，而是让你用三个可运行实验理解基础视觉信号：运动、对应关系和前景。

---

## 1. 核心问题

机器人视觉不只需要“识别物体”，还经常需要更底层的问题：

- 画面中哪些点在移动？
- 两张图片中哪些局部区域对应同一个物体？
- 视频中哪些区域是新出现的前景？

这三个问题分别对应：

| 模式 | 观察对象 | 典型用途 |
|------|---------|---------|
| `optical-flow` | 点的运动轨迹 | 跟踪、视觉里程计、运动估计 |
| `feature-matching` | 两张图之间的局部特征对应 | 图像配准、定位、重建 |
| `background-subtraction` | 前景区域 | 安防、动态物体检测、场景变化 |

---

## 2. 生成样例数据

进入 Level 1 目录：

```bash
cd experiments/01-perception/level-1-python
```

生成可离线运行的样例视频和图片：

```bash
python scripts/classic_vision.py --generate-sample --mode optical-flow
```

输出在：

```text
output/classic_vision/
```

这个样例数据用于确认环境和算法链路，不代表真实摄像头效果。

---

## 3. 光流：观察运动

```bash
python scripts/classic_vision.py \
  --mode optical-flow \
  --input-video output/classic_vision/sample_motion.mp4
```

脚本使用：

- `cv2.goodFeaturesToTrack` 找可跟踪角点；
- `cv2.calcOpticalFlowPyrLK` 在两帧之间追踪这些点；
- 红色箭头显示点从上一帧到下一帧的移动。

输出：

```text
output/classic_vision/optical_flow.jpg
```

观察重点：箭头方向是否和物体运动一致，箭头长度是否反映运动幅度。

---

## 4. 特征匹配：观察对应关系

先使用样例图片：

```bash
python scripts/classic_vision.py \
  --generate-sample \
  --mode feature-matching
```

或者指定自己的两张图片：

```bash
python scripts/classic_vision.py \
  --mode feature-matching \
  --image-a image_a.jpg \
  --image-b image_b.jpg
```

脚本使用：

- `cv2.ORB_create` 提取 ORB 特征；
- `cv2.BFMatcher` 做描述子匹配；
- `cv2.drawMatches` 画出两张图片之间的匹配线。

输出：

```text
output/classic_vision/feature_matching.jpg
```

观察重点：好的匹配线应该连接同一个物体或同一个局部结构。错误匹配通常来自纹理重复、模糊或视角变化过大。

---

## 5. 背景减除：观察前景

```bash
python scripts/classic_vision.py \
  --mode background-subtraction \
  --input-video output/classic_vision/sample_motion.mp4
```

脚本使用：

- `cv2.createBackgroundSubtractorMOG2` 建立背景模型；
- 连续读取多帧；
- 输出最后一帧和前景 mask 的并排图。

输出：

```text
output/classic_vision/background_subtraction.jpg
```

观察重点：右侧 mask 中白色区域代表被认为是前景的部分。背景减除适合固定摄像头，不适合摄像头快速移动的机器人视角。

---

## 6. 摄像头模式

如果不传 `--input-video`，光流和背景减除会默认打开摄像头：

```bash
python scripts/classic_vision.py --mode optical-flow
python scripts/classic_vision.py --mode background-subtraction
```

如果默认摄像头打不开：

```bash
python scripts/classic_vision.py --mode optical-flow --camera-index 1
```

---

## 7. 与具身智能的关系

这些经典方法不是为了替代深度学习，而是提供可解释的视觉直觉：

- 光流让你理解“图像运动”和机器人运动之间的关系。
- 特征匹配让你理解定位、SLAM 和重建为什么需要稳定对应点。
- 背景减除让你理解动态物体检测和场景变化的基本思路。

完成这个实验后，感知 Level 1 的基础视觉线就从相机标定扩展到了可观察的图像运动、局部对应和前景分割。
