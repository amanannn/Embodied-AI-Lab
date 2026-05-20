"""扩展卡尔曼滤波 Demo — 2D 匀速目标追踪。

使用雷达观测模型：传感器测量的是距离和方位角，
而非直接的 x,y 坐标。观测函数是非线性的：
    h(x) = [sqrt(px²+py²), atan2(py, px)]

EKF 用雅可比矩阵对非线性函数做局部线性化。

用法:
    python demos/demo_ekf.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from utils.simulation import (generate_ground_truth, range_bearing_observations,
                               range_bearing_to_xy, DT, STEPS, TOTAL_TIME,
                               RANGE_NOISE_STD, BEARING_NOISE_STD)
from utils.visualization import (draw_tracking_figure, draw_tracking_animation,
                                  print_metrics, COLORS)
from filters.extended_kf import ExtendedKalmanFilter

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# 滤波器参数
Q_POS = 0.01
Q_VEL = 0.05
R = np.diag([RANGE_NOISE_STD**2, BEARING_NOISE_STD**2])


def main():
    print("=" * 60)
    print("  扩展卡尔曼滤波 (EKF) — 雷达 range-bearing 观测")
    print("=" * 60)

    # 生成数据
    print("→ 生成真实轨迹 …")
    true_pos, _ = generate_ground_truth()
    print("→ 模拟雷达传感器观测 …")
    rb_obs = range_bearing_observations(true_pos)

    # 用第一个观测初始化位置（避免原点处雅可比退化）
    first_xy = range_bearing_to_xy(rb_obs[:1])[0]
    init_x = np.array([first_xy[0], first_xy[1], 0.0, 0.0])

    # 运行滤波器（内部使用原始 range-bearing 数据）
    print("→ 运行 EKF …")
    ekf = ExtendedKalmanFilter(DT, np.array([Q_POS, Q_POS, Q_VEL, Q_VEL]), R,
                               initial_x=init_x)
    for z in rb_obs:
        ekf.step(z)

    estimates = np.array(ekf.estimates)
    covariances = ekf.covariances

    # 可视化：将 range-bearing 观测转为笛卡尔坐标来画图
    obs_xy = range_bearing_to_xy(rb_obs)
    print_metrics(true_pos, obs_xy, estimates, "EKF")

    print("\n→ 渲染图片 …")
    draw_tracking_figure(
        true_pos, obs_xy, estimates, covariances,
        "Extended Kalman Filter — Radar Range-Bearing Tracking",
        OUT_DIR / "ekf_static.png",
        est_color=COLORS["ekf"],
        obs_label="Radar Obs (XY converted)")

    draw_tracking_animation(
        true_pos, obs_xy, estimates, covariances,
        "EKF — Radar Tracking Animation",
        OUT_DIR / "ekf_animated.gif",
        est_color=COLORS["ekf"],
        obs_label="Radar Obs (XY converted)",
        dt=DT, total_time=TOTAL_TIME)

    print(f"\n  [EKF 原理] 每步用雅可比矩阵在当前点'画切线'来近似非线性观测函数。")
    print("  Done. EKF demo complete.")


if __name__ == "__main__":
    main()
