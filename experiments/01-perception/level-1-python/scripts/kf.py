"""线性卡尔曼滤波 — 2D 匀速目标追踪。

传感器直接输出物体的 [x, y] 坐标（带高斯噪声）。
这是最经典、最简单的 KF 使用场景。

用法:
    python scripts/kf.py                      # 使用内置仿真数据
    python scripts/kf.py --data output/sensor_data.csv  # 使用传感器实验输出的 CSV
"""

import sys
import argparse
import csv
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


def load_csv_data(csv_path: str):
    """从 CSV 文件加载传感器数据。

    CSV 格式: timestamp, sensor_id, x, y, cov_xx, cov_xy, cov_yy

    返回:
        timestamps: (N,) 时间戳数组
        observations: (N, 2) 观测坐标 [x, y]
        covariances: (N, 2, 2) 协方差矩阵列表
    """
    timestamps = []
    observations = []
    covariances = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            observations.append([float(row["x"]), float(row["y"])])
            cov = np.array([
                [float(row["cov_xx"]), float(row["cov_xy"])],
                [float(row["cov_xy"]), float(row["cov_yy"])],
            ])
            covariances.append(cov)

    return np.array(timestamps), np.array(observations), np.array(covariances)


def main():
    parser = argparse.ArgumentParser(description="线性卡尔曼滤波 — 2D 匀速目标追踪")
    parser.add_argument("--data", type=str, default=None,
                        help="CSV 数据文件路径（由 sensors.py 生成）")
    args = parser.parse_args()

    print("=" * 60)
    print("  线性卡尔曼滤波 (KF) — 笛卡尔位置观测")
    print("=" * 60)

    if args.data:
        # 从 CSV 加载数据
        csv_path = Path(args.data)
        if not csv_path.is_file():
            print(f"[ERROR] CSV 文件不存在: {csv_path}")
            sys.exit(1)

        print(f"→ 从 CSV 加载数据: {csv_path}")
        timestamps, observations, covariances = load_csv_data(str(csv_path))

        # 使用第一个观测的时间步长估算 DT
        if len(timestamps) > 1:
            dt = np.median(np.diff(timestamps))
        else:
            dt = 0.1  # 默认 10 Hz

        # 使用 CSV 中的协方差（取平均）
        R_data = np.mean(covariances, axis=0)

        # 生成对应的真实轨迹（用于评估）
        # 注意：CSV 数据没有真实轨迹，我们用观测的平滑近似作为参考
        print("→ 无法获取真实轨迹，将使用观测平滑近似作为参考")
        from scipy.ndimage import uniform_filter1d
        true_pos = uniform_filter1d(observations, size=5, axis=0)

        total_time = timestamps[-1] - timestamps[0]
        sensor_noise = np.sqrt(np.mean([c[0, 0] for c in covariances]))

        print(f"  数据点数: {len(observations)}")
        print(f"  时间范围: {total_time:.1f} s")
        print(f"  平均采样间隔: {dt:.3f} s")
        print(f"  平均噪声标准差: {sensor_noise:.3f} m")

        # 运行滤波器
        print("\n→ 运行卡尔曼滤波器 …")
        kf = KalmanFilter(dt, np.array([Q_POS, Q_POS, Q_VEL, Q_VEL]), R_data)
        for z in observations:
            kf.step(z)

        estimates = np.array(kf.estimates)
        covariances_kf = kf.covariances

        # 指标
        print_metrics(true_pos, observations, estimates, "KF")

        # 可视化
        print("\n→ 渲染图片 …")
        draw_tracking_figure(
            true_pos, observations, estimates, covariances_kf,
            "Kalman Filter — External Data",
            OUT_DIR / "kf_external_static.png",
            est_color=COLORS["kf"],
            obs_label="CSV Position Obs")

        print("\nDone. KF script complete (external data mode).")

    else:
        # 内置仿真模式
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

        print("\nDone. KF script complete.")


if __name__ == "__main__":
    main()
