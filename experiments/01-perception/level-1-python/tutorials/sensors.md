# 传感器基础与噪声建模 —— 从零理解传感器数据

> 配合 `noise.py`、`sensors/`、`fusion.py` 和 `scripts/` 目录下的代码食用。本文假设你只会 Python 基础，没有任何信号处理背景。

---

## 目录

1. [问题：传感器读数为什么不等于真值？](#1-问题传感器读数为什么不等于真值)
2. [四种噪声类型（逐个拆解）](#2-四种噪声类型逐个拆解)
3. [四类传感器仿真](#3-四类传感器仿真)
4. [多传感器融合：如何合并多个不完美的测量](#4-多传感器融合如何合并多个不完美的测量)
5. [代码走读：模块结构与数据流](#5-代码走读模块结构与数据流)
6. [运行实验](#6-运行实验)
7. [下一步学什么](#7-下一步学什么)

---

## 1. 问题：传感器读数为什么不等于真值？

### 一个真实场景

你有一个机器人在直线上移动，真实位置从 0 到 10 米。你用 GPS 追踪它的位置：

- 真实位置：5.00 m
- GPS 读数：5.23 m  ← 偏了 23 厘米
- 下一次：4.87 m  ← 又偏了 13 厘米
- 再下一次：5.05 m  ← 这次接近了

**所有传感器都有误差**，这些误差有不同的特征：

| 误差类型 | 特征 | 真实例子 |
|---------|------|---------|
| 随机波动 | 每次读数在真值上下随机跳动 | GPS 定位的抖动 |
| 系统偏差 | 读数始终偏高或偏低 | IMU 加速度计的零偏 |
| 缓慢漂移 | 误差随时间缓慢累积 | 陀螺仪的积分漂移 |
| 偶发异常 | 偶尔出现大幅偏离的读数 | LiDAR 的多路径反射 |

### 结论

**不理解噪声 → 无法正确使用传感器数据。**
**不理解传感器特性 → 无法选择合适的滤波方法。**

---

## 2. 四种噪声类型（逐个拆解）

### 2.1 高斯白噪声 (Gaussian Noise)

最基础的噪声模型。每个时刻独立采样，服从正态分布 N(0, σ)。

```python
from noise import gaussian_noise
import numpy as np

timestamps = np.linspace(0, 10, 100)
noise = gaussian_noise(timestamps, std=0.5, seed=42)
```

**直觉**：传感器读数在真值上下随机波动，波动幅度由 σ 控制。

**代码位置**：`noise.py` → `gaussian_noise()`

### 2.2 常值偏置 (Constant Bias)

传感器读数始终偏移一个固定值。

```python
from noise import constant_bias

bias = constant_bias(timestamps, bias_value=1.0)
```

**直觉**：就像体重秤没归零，每次称都多显示 1 公斤。

**代码位置**：`noise.py` → `constant_bias()`

### 2.3 随机游走漂移 (Random Walk Drift)

误差随时间缓慢累积，像醉汉走路一样随机游走。

```python
from noise import random_walk_drift

drift = random_walk_drift(timestamps, drift_std=0.1)
```

**直觉**：陀螺仪积分得到的角度会随时间慢慢偏离真实值。

**代码位置**：`noise.py` → `random_walk_drift()`

### 2.4 离群值 (Outlier Noise)

偶尔出现的大幅异常读数。

```python
from noise import outlier_noise

outliers = outlier_noise(timestamps, outlier_prob=0.05, magnitude=3.0)
```

**直觉**：GPS 信号被建筑物反射时，位置会突然跳到几百米外。

**代码位置**：`noise.py` → `outlier_noise()`

### 复合噪声

真实传感器通常同时受多种噪声影响：

```python
from noise import apply_noise

noisy = apply_noise(clean_signal, timestamps, {
    "gaussian_std": 0.5,
    "bias": 0.1,
    "drift_std": 0.01,
    "outlier_prob": 0.02,
    "outlier_magnitude": 3.0,
})
```

**代码位置**：`noise.py` → `apply_noise()`

---

## 3. 四类传感器仿真

每种传感器有不同的观测维度和噪声特性。

### 3.1 GPS 传感器

- **观测内容**：2D 位置 (x, y)
- **噪声特点**：高斯噪声为主，偶发离群值
- **采样率**：1-10 Hz

```python
from sensors import GPSSensor

gps = GPSSensor(sample_rate=1.0, noise_std=0.5)
reading = gps.read(true_state=[5.0, 3.0], timestamp=1.0)
print(reading.data)  # 带噪声的位置 [x, y]
```

**代码位置**：`sensors/gps.py`

### 3.2 LiDAR 传感器

- **观测内容**：距离和角度 (range, angle)
- **噪声特点**：距离噪声小，角度噪声较大
- **采样率**：10-20 Hz

```python
from sensors import LiDARSensor

lidar = LiDARSensor(sample_rate=10.0, noise_std=0.1)
reading = lidar.read(true_state=[5.0, 3.0], timestamp=0.1)
```

**代码位置**：`sensors/lidar.py`

### 3.3 IMU 传感器

- **观测内容**：加速度 (ax, ay) 和角速度 (ω)
- **噪声特点**：高斯噪声 + 常值偏置 + 漂移
- **采样率**：100-200 Hz

```python
from sensors import IMUSensor

imu = IMUSensor(sample_rate=100.0, noise_std=0.1, bias=0.05)
reading = imu.read(true_state=[5.0, 3.0, 1.0, 0.0], timestamp=0.01)
```

**代码位置**：`sensors/imu.py`

### 3.4 里程计传感器

- **观测内容**：位移增量 (Δx, Δy)
- **噪声特点**：误差随行驶距离累积
- **采样率**：10-50 Hz

```python
from sensors import OdometrySensor

odom = OdometrySensor(sample_rate=10.0, noise_std=0.2)
reading = odom.read(true_state=[5.0, 3.0], timestamp=0.1)
```

**代码位置**：`sensors/odometry.py`

### 统一数据格式

所有传感器输出 `SensorReading`：

```python
@dataclass
class SensorReading:
    timestamp: float      # 读数时刻 [秒]
    sensor_id: str        # 传感器标识 ("gps", "lidar", "imu", "odom")
    data: np.ndarray      # 传感器数据向量
    covariance: np.ndarray # 噪声协方差矩阵
```

**代码位置**：`sensors/base.py`

---

## 4. 多传感器融合：如何合并多个不完美的测量

### 问题

你有 GPS 和里程计两个传感器测量同一位置，读数不同：
- GPS：(5.2, 3.1)
- 里程计：(4.8, 2.9)

哪个更可信？还是取平均？

### 加权融合

根据每个传感器的噪声特性分配权重：

```python
from fusion import weighted_fusion

fused = weighted_fusion(readings, weights)
```

**直觉**：噪声小的传感器权重高，噪声大的权重低。

**代码位置**：`fusion.py`

### 时间对齐

不同传感器采样率不同，需要先对齐时间戳再融合。

**代码位置**：`fusion.py` → `align_readings()`

---

## 5. 代码走读：模块结构与数据流

```
level-1-python/
├── noise.py              # 四种噪声模型 + 复合噪声
├── sensors/
│   ├── base.py           # SensorBase 基类 + SensorReading 数据格式
│   ├── gps.py            # GPS 传感器仿真
│   ├── lidar.py          # LiDAR 传感器仿真
│   ├── imu.py            # IMU 传感器仿真
│   └── odometry.py       # 里程计传感器仿真
├── fusion.py             # 多传感器加权融合
├── utils/
│   └── trajectory.py     # 轨迹生成器（直线、圆形、8 字、随机游走）
└── scripts/
    ├── noise.py          # 噪声类型对比实验
    ├── sensors.py        # 多传感器对比 + CSV 输出
    └── fusion.py         # 融合效果对比
```

### 数据流

```
真实轨迹 → 传感器仿真 → 噪声叠加 → 多传感器融合 → 可视化
    ↓           ↓           ↓           ↓           ↓
trajectory.py  sensors/    noise.py   fusion.py   visualization.py
```

---

## 6. 运行实验

### 6.1 噪声类型对比

```bash
cd experiments/01-perception/level-1-python
python scripts/noise.py
```

生成四种噪声的对比图，保存到 `output/noise_comparison.png`。

### 6.2 多传感器对比

```bash
python scripts/sensors.py
```

生成 GPS、LiDAR、IMU、里程计的观测对比图，同时输出 CSV 数据。

### 6.3 融合效果对比

```bash
python scripts/fusion.py
```

对比单传感器 vs 融合后的估计精度。

### 6.4 全部运行

```bash
python scripts/all.py
```

一次性运行所有实验。

---

## 7. 下一步学什么

完成本教程后，你已经理解了：

1. 传感器读数为什么不等于真值
2. 四种噪声类型及其物理含义
3. 四类传感器的观测特性和噪声模型
4. 如何用加权融合合并多个传感器

**下一步**：学习 [卡尔曼滤波教程](./kalman.md)，了解如何用概率方法最优地融合传感器数据。

---

## 附录：关键公式

### 高斯噪声

$$noise \sim \mathcal{N}(0, \sigma^2)$$

### 加权融合

$$\hat{x} = \frac{\sum_{i=1}^{n} w_i \cdot x_i}{\sum_{i=1}^{n} w_i}$$

其中权重 $w_i = \frac{1}{\sigma_i^2}$（噪声越小，权重越大）。
