# s02-mcl-localization — 蒙特卡洛定位

English: [mcl_localization.en.md](./mcl_localization.en.md)

## 这个实验解决什么问题？

`s01-grid-search` 假设机器人已经知道自己在地图上的位置。真实移动机器人还需要先回答一个更基础的问题：

> 在一张已知地图里，机器人现在到底在哪里？

本实验用 Monte Carlo Localization (MCL) 展示这个闭环：机器人只知道一张带信标的房间地图，运动过程中里程计会漂移，距离观测会有噪声，粒子滤波器用一群候选位姿持续修正定位结果。

## 快速运行

在 `experiments/02-slam-navigation/level-1-python/` 下运行：

```bash
python scripts/mcl_localization.py
```

如果当前环境没有图形依赖，先跑无图版本：

```bash
python scripts/mcl_localization.py --no-plot
```

也可以调节粒子数和步数：

```bash
python scripts/mcl_localization.py --particles 800 --steps 30
```

输出目录默认是：

```text
output/mcl_localization/
```

主要文件：

| 文件 | 含义 |
|------|------|
| `mcl_localization.png` | 真实轨迹、里程计轨迹、MCL 估计、粒子云和误差曲线 |
| `mcl_localization_metrics.json` | 最终误差、平均误差、里程计误差、有效粒子数 |
| `mcl_localization_samples.json` | 每一步的真实位置、估计位置和误差 |
| `mcl_localization_particles.json` | 最后一步的粒子云 |
| `mcl_localization_beacons.json` | 已知地图中的信标位置 |

## 观察重点

先看终端指标：

```text
final_error=...m
mean_error=...m
final_odometry_error=...m
effective_particle_count=...
```

重点比较：

- `final_error`：MCL 最后一步定位误差。
- `final_odometry_error`：只靠里程计累计出来的位置误差。
- `effective_particle_count`：粒子权重是否过度集中。过低说明粒子退化严重。

默认参数下，MCL 误差应该明显小于里程计误差。这个结果说明观测在持续把漂移的位姿拉回正确区域。

## 核心流程

每一步循环都做四件事：

1. `predict`：每个粒子根据控制量前进，并加入运动噪声。
2. `observe`：机器人测量自己到多个固定信标的距离。
3. `weight`：如果某个粒子预测出的距离更接近观测，它的权重更高。
4. `resample`：按权重重新抽样，让更可信的粒子留下来。

这个实验不是全局绑架定位问题，而是“有粗略初始位姿”的教学版 MCL。这样更适合本科生先理解粒子滤波闭环，再扩展到全局定位、多峰分布和复杂地图。

## 可以尝试的参数

```bash
python scripts/mcl_localization.py --particles 200 --no-plot
python scripts/mcl_localization.py --particles 1000 --no-plot
python scripts/mcl_localization.py --sensor-sigma 0.35 --no-plot
python scripts/mcl_localization.py --sensor-sigma 0.12 --no-plot
```

建议观察：

- 粒子数减少时，定位是否更不稳定？
- 传感器噪声变大时，误差是否上升？
- 有效粒子数过低时，粒子云是否集中到错误区域？

## 和 ROS2 的关系

Level 1 只保留算法直觉：粒子、权重、重采样、误差指标。进入 Level 2 后，这个实验可以桥接到 ROS2 中的 `map`、`odom`、`base_link` 坐标关系，以及激光雷达观测模型。
