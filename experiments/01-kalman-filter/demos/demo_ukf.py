"""无迹卡尔曼滤波 Demo — 2D 匀速目标追踪。

与 EKF 使用相同的雷达观测模型，但 UKF 不求导。
而是用 9 个精心选择的 sigma 点来传递概率分布：
    "撒出去 → 各走各的 → 回来汇总"

在非线性较强的场景下，UKF 的精度通常比 EKF 高一阶。

用法:
    python demos/demo_ukf.py
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
from filters.unscented_kf import UnscentedKalmanFilter

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# 滤波器参数
Q_POS = 0.01
Q_VEL = 0.05
R = np.diag([RANGE_NOISE_STD**2, BEARING_NOISE_STD**2])

# UKF 特有参数
ALPHA = 1.0   # sigma 点扩散宽度
BETA = 2.0    # 高斯先验知识（2 对高斯分布最优）
KAPPA = 0.0   # 次级缩放


def main():
    print("=" * 60)
    print("  无迹卡尔曼滤波 (UKF) — 雷达 range-bearing 观测")
    print("=" * 60)

    # 生成数据（与 EKF demo 相同的轨迹和观测）
    print("→ 生成真实轨迹 …")
    true_pos, _ = generate_ground_truth()
    print("→ 模拟雷达传感器观测 …")
    rb_obs = range_bearing_observations(true_pos)

    # 用第一个观测初始化位置
    first_xy = range_bearing_to_xy(rb_obs[:1])[0]
    init_x = np.array([first_xy[0], first_xy[1], 0.0, 0.0])

    print("→ 运行 UKF …")
    ukf = UnscentedKalmanFilter(
        DT, np.array([Q_POS, Q_POS, Q_VEL, Q_VEL]), R,
        alpha=ALPHA, beta=BETA, kappa=KAPPA, initial_x=init_x)
    for z in rb_obs:
        ukf.step(z)

    estimates = np.array(ukf.estimates)
    covariances = ukf.covariances

    obs_xy = range_bearing_to_xy(rb_obs)
    print_metrics(true_pos, obs_xy, estimates, "UKF")

    print("\n→ 渲染图片 …")
    draw_tracking_figure(
        true_pos, obs_xy, estimates, covariances,
        "Unscented Kalman Filter — Radar Range-Bearing Tracking",
        OUT_DIR / "ukf_static.png",
        est_color=COLORS["ukf"],
        obs_label="Radar Obs (XY converted)")

    draw_tracking_animation(
        true_pos, obs_xy, estimates, covariances,
        "UKF — Radar Tracking Animation",
        OUT_DIR / "ukf_animated.gif",
        est_color=COLORS["ukf"],
        obs_label="Radar Obs (XY converted)",
        dt=DT, total_time=TOTAL_TIME)

    print(f"\n  [UKF 原理] 用 {2*4+1} 个 sigma 点采样，无需雅可比矩阵，精度比 EKF 高一阶。")
    print("  Done. UKF demo complete.")


if __name__ == "__main__":
    main()
