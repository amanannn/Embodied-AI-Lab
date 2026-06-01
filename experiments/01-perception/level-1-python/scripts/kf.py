"""线性卡尔曼滤波 — 2D 匀速目标追踪。

传感器直接输出物体的 [x, y] 坐标（带高斯噪声）。
这是最经典、最简单的 KF 使用场景。

用法:
    python scripts/kf.py                    # 使用内置仿真数据
    python scripts/fusion.py                # 先生成 fused_data.csv
    python scripts/kf.py --data output/fused_data.csv
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
    """从 CSV 文件加载单一位置观测流。

    CSV 格式: timestamp, sensor_id, x, y, cov_xx, cov_xy, cov_yy

    返回:
        timestamps: (N,) 时间戳数组
        observations: (N, 2) 观测坐标 [x, y]
        covariances: (N, 2, 2) 协方差矩阵列表
    """
    timestamps, observations, covariances, sensor_id = load_csv_rows(csv_path)
    return (
        np.array(timestamps, dtype=float),
        np.array(observations, dtype=float),
        np.array(covariances, dtype=float),
        sensor_id,
    )


def load_csv_rows(csv_path: str):
    """用标准库读取并校验 CSV 契约，便于无 numpy 环境测试。"""
    timestamps = []
    observations = []
    covariances = []
    sensor_ids = set()

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sensor_ids.add(row["sensor_id"])
            timestamps.append(float(row["timestamp"]))
            observations.append([float(row["x"]), float(row["y"])])
            covariances.append([
                [float(row["cov_xx"]), float(row["cov_xy"])],
                [float(row["cov_xy"]), float(row["cov_yy"])],
            ])

    if not timestamps:
        raise ValueError("CSV 文件为空，没有可用观测。")

    if len(sensor_ids) != 1:
        raise ValueError(
            "kf.py --data 只接受单一位置观测流。检测到多个 sensor_id，"
            "请先运行 scripts/fusion.py 使用 output/fused_data.csv，"
            "或自行预处理为单一笛卡尔位置观测 CSV。"
        )

    rows = sorted(zip(timestamps, observations, covariances), key=lambda item: item[0])
    deduped = []
    last_t = None
    for t, obs, cov in rows:
        if last_t is not None and abs(t - last_t) <= 1e-9:
            continue
        deduped.append((t, obs, cov))
        last_t = t

    out_timestamps = [row[0] for row in deduped]
    out_observations = [row[1] for row in deduped]
    out_covariances = [row[2] for row in deduped]
    return out_timestamps, out_observations, out_covariances, sensor_ids.pop()


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
        try:
            timestamps, observations, covariances, sensor_id = load_csv_data(str(csv_path))
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)

        # 使用第一个观测的时间步长估算 DT
        if len(timestamps) > 1:
            dt = float(np.median(np.diff(timestamps)))
        else:
            dt = 0.1  # 默认 10 Hz

        # 使用 CSV 中的协方差（取平均）
        R_data = np.mean(covariances, axis=0)

        total_time = timestamps[-1] - timestamps[0]
        sensor_noise = np.sqrt(float(np.mean(covariances[:, 0, 0])))

        print(f"  数据点数: {len(observations)}")
        print(f"  传感器流: {sensor_id}")
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

        # 可视化
        print("\n→ 渲染图片 …")
        draw_tracking_figure(
            observations, observations, estimates, covariances_kf,
            "Kalman Filter — External Data",
            OUT_DIR / "kf_external_static.png",
            est_color=COLORS["kf"],
            obs_label="CSV Position Obs",
            show_error_panel=False,
            reference_label="Observation Stream")

        obs_step = np.linalg.norm(np.diff(observations, axis=0), axis=1)
        est_step = np.linalg.norm(np.diff(estimates, axis=0), axis=1)
        print("\n→ 外部数据模式说明:")
        print("  当前没有真实轨迹，因此不报告 RMSE。")
        print(f"  平均观测步长: {np.mean(obs_step):.3f} m")
        print(f"  平均滤波步长: {np.mean(est_step):.3f} m")
        print("  请结合图像观察轨迹平滑程度与协方差椭圆变化。")

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
