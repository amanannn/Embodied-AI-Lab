# s03-ekf-slam — 扩展卡尔曼 SLAM

English: [ekf_slam.en.md](./ekf_slam.en.md)

## 这个实验解决什么问题？

MCL 解决的是“地图已知时我在哪”。SLAM 要解决的是更难的问题：

> 地图还不知道时，机器人能不能一边定位自己，一边估计地图？

本实验实现一个小规模 EKF-SLAM：机器人用 range-bearing 观测看到 3 个路标，状态向量同时包含机器人位姿和路标坐标。

```text
state = [x, y, theta, lm0_x, lm0_y, lm1_x, lm1_y, lm2_x, lm2_y]
```

这就是 SLAM 的关键直觉：定位误差会影响地图估计，地图估计又会反过来影响定位。

## 快速运行

在 `experiments/02-slam-navigation/level-1-python/` 下运行：

```bash
python scripts/ekf_slam.py
```

如果只想验证算法和 JSON 输出：

```bash
python scripts/ekf_slam.py --no-plot
```

可以调节步数和噪声：

```bash
python scripts/ekf_slam.py --steps 30 --range-sigma 0.18
```

默认输出目录：

```text
output/ekf_slam/
```

主要文件：

| 文件 | 含义 |
|------|------|
| `ekf_slam.png` | 真实轨迹、EKF-SLAM 轨迹、真实/估计路标和协方差趋势 |
| `ekf_slam_metrics.json` | 位姿误差、路标 RMSE、状态维度和协方差 trace |
| `ekf_slam_samples.json` | 每一步真实位姿、估计位姿和误差 |
| `ekf_slam_landmarks.json` | 每个路标的真实位置、估计位置和误差 |
| `ekf_slam_covariance_diagonal.json` | 最终联合协方差对角线 |

## 观察重点

终端会输出：

```text
final_pose_error=...m
mean_pose_error=...m
landmark_rmse=...m
state_size=9
covariance_trace=...
```

重点看三件事：

- `state_size=9`：状态不再只有机器人，还包含 3 个路标。
- `landmark_rmse`：地图估计是否接近真实路标。
- `covariance_trace`：联合不确定性如何随观测变化。

## 核心流程

EKF-SLAM 每一步包含：

1. `predict`：根据控制量预测机器人位姿，同时传播联合协方差。
2. `observe`：对每个路标生成距离和方位角观测。
3. `linearize`：在当前估计附近计算观测模型雅可比矩阵。
4. `update`：用卡尔曼增益同时修正机器人位姿和路标坐标。

注意：本实验为了教学稳定，默认使用已知路标关联，也就是观测知道自己来自哪个路标。真实 SLAM 还要处理 data association，这是后续研究扩展问题。

## 可以尝试的参数

```bash
python scripts/ekf_slam.py --steps 12 --no-plot
python scripts/ekf_slam.py --steps 36 --no-plot
python scripts/ekf_slam.py --range-sigma 0.25 --no-plot
python scripts/ekf_slam.py --bearing-sigma 0.08 --no-plot
```

建议观察：

- 步数增加后，路标 RMSE 是否下降？
- 观测噪声变大后，位姿误差和协方差 trace 是否上升？
- 如果只看 `x, y, theta`，为什么无法解释地图误差？

## 和 ROS2 的关系

Level 1 只强调 EKF-SLAM 的联合状态估计直觉。进入 Level 2 后，可把这里的状态和协方差概念映射到 ROS2 的 `nav_msgs/Odometry`、`geometry_msgs/PoseWithCovariance`、TF 坐标树和路标观测消息。
