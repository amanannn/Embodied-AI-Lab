"""线性卡尔曼滤波 Demo — 2D 匀速目标追踪。

传感器直接输出物体的 [x, y] 坐标（带高斯噪声）。
这是最经典、最简单的 KF 使用场景。

用法:
    python demos/demo_kf.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from utils.simulation import (generate_ground_truth, linear_observations,
                               DT, STEPS, TOTAL_TIME, SENSOR_NOISE)
from utils.visualization import (draw_tracking_figure, draw_tracking_animation,
                                  print_metrics, COLORS)
from filters.kalman_filter import KalmanFilter

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# 滤波器参数
Q_POS = 0.01
Q_VEL = 0.05
R = np.eye(2) * (SENSOR_NOISE ** 2)


def main():
    print("=" * 60)
    print("  线性卡尔曼滤波 (KF) — 笛卡尔位置观测")
    print("=" * 60)

    # 生成数据
    print("→ 生成真实轨迹 …")
    true_pos, _ = generate_ground_truth()
    print("→ 模拟位置传感器观测 …")
    observations = linear_observations(true_pos)

    # 运行滤波器
    print("→ 运行卡尔曼滤波器 …")
    kf = KalmanFilter(DT, np.array([Q_POS, Q_POS, Q_VEL, Q_VEL]), R)
    for z in observations:
        kf.step(z)

    estimates = np.array(kf.estimates)
    covariances = kf.covariances

    # 指标
    print_metrics(true_pos, observations, estimates, "KF")

    # 可视化
    print("\n→ 渲染图片 …")
    draw_tracking_figure(
        true_pos, observations, estimates, covariances,
        "Kalman Filter — 2D Constant Velocity Tracking",
        OUT_DIR / "kf_static.png",
        est_color=COLORS["kf"],
        obs_label="Noisy Position Obs")

    draw_tracking_animation(
        true_pos, observations, estimates, covariances,
        "Kalman Filter — Live Tracking",
        OUT_DIR / "kf_animated.gif",
        est_color=COLORS["kf"],
        obs_label="Noisy Position Obs",
        dt=DT, total_time=TOTAL_TIME)

    print("\nDone. KF demo complete.")


if __name__ == "__main__":
    main()
