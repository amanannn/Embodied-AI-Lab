"""多传感器融合实验。

读取传感器数据 CSV，执行加权融合，对比单传感器 vs 融合效果。
输出融合结果图 + 融合后的 CSV（供卡尔曼实验使用）。

用法:
    python scripts/fusion.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.trajectory import generate_trajectory
from sensors import GPSSensor, LiDARSensor
from sensors.lidar import LiDARSensor as LS
from fusion import weighted_fusion, time_align
from sensors.base import SensorReading

OUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main():
    print("=" * 60)
    print("  多传感器融合实验")
    print("=" * 60)

    # 生成轨迹
    print("→ 生成真实轨迹 …")
    ts, pos, _, _ = generate_trajectory(duration=12.0, dt=0.01,
                                         trajectory_type="figure8")

    # 生成 GPS 和 LiDAR 观测
    print("→ 生成 GPS + LiDAR 观测 …")
    gps = GPSSensor(noise_std=0.5, outlier_prob=0.02, seed=42)
    lidar = LiDARSensor(range_std=0.3, bearing_std=0.05, seed=42)

    gps_readings = gps.generate_sequence(pos, ts)
    lidar_readings = lidar.generate_sequence(pos, ts)

    # 将 LiDAR 转为笛卡尔坐标
    for r in lidar_readings:
        xy = LS.to_xy(r)
        r.data = xy
        r.covariance = np.eye(2) * r.covariance[0, 0]  # 近似

    # 按时间对齐
    print("→ 按时间戳对齐传感器数据 …")
    aligned = time_align([gps_readings, lidar_readings], tolerance=0.06)

    # 融合
    print("→ 执行加权融合 …")
    fused_times = []
    fused_data = []
    fused_covs = []

    for t, readings in sorted(aligned.items()):
        data, cov = weighted_fusion(readings)
        fused_times.append(t)
        fused_data.append(data)
        fused_covs.append(cov)

    fused_times = np.array(fused_times)
    fused_data = np.array(fused_data)

    # 误差统计
    fused_idx = [int(t / 0.01) for t in fused_times]
    fused_true = pos[fused_idx]
    fused_err = np.linalg.norm(fused_data - fused_true, axis=1)

    gps_data = np.array([r.data for r in gps_readings])
    gps_idx = [int(r.timestamp / 0.01) for r in gps_readings]
    gps_err = np.linalg.norm(gps_data - pos[gps_idx], axis=1)

    lidar_data = np.array([r.data for r in lidar_readings])
    lidar_idx = [int(r.timestamp / 0.01) for r in lidar_readings]
    lidar_err = np.linalg.norm(lidar_data - pos[lidar_idx], axis=1)

    print("\n→ 融合效果:")
    print(f"  {'Source':<10} {'RMSE':>8} {'Mean Err':>10} {'Max Err':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
    for name, err in [("GPS", gps_err), ("LiDAR", lidar_err),
                       ("Fused", fused_err)]:
        print(f"  {name:<10} {np.sqrt(np.mean(err**2)):>8.3f} "
              f"{np.mean(err):>10.3f} {np.max(err):>10.3f}")

    # 可视化
    print("\n→ 渲染融合结果图 …")
    OUT_DIR.mkdir(exist_ok=True)

    plt.style.use("dark_background")
    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(16, 7),
                                            facecolor="#0a0a0a")
    fig.suptitle("Multi-Sensor Fusion — GPS + LiDAR",
                 fontsize=18, fontweight="bold", color="#cccccc", y=0.97)

    # 左图：轨迹
    ax_traj.set_facecolor("#0a0a0a")
    ax_traj.plot(pos[:, 0], pos[:, 1], color="#00ff88", linewidth=2,
                 label="Ground Truth", alpha=0.7)
    ax_traj.scatter(gps_data[:, 0], gps_data[:, 1], color="#00ccff",
                    s=15, alpha=0.5, label="GPS")
    ax_traj.scatter(lidar_data[:, 0], lidar_data[:, 1], color="#ff8800",
                    s=3, alpha=0.3, label="LiDAR")
    ax_traj.plot(fused_data[:, 0], fused_data[:, 1], color="#ffffff",
                 linewidth=2, label="Fused")
    ax_traj.set_xlabel("X [m]", color="#aaaaaa")
    ax_traj.set_ylabel("Y [m]", color="#aaaaaa")
    ax_traj.set_title("Trajectory Comparison", color="#bbbbbb", fontsize=13)
    ax_traj.legend(loc="upper right", facecolor="#111111",
                   edgecolor="#333333", labelcolor="#cccccc")
    ax_traj.grid(True, color="#222222", alpha=0.4)
    ax_traj.set_aspect("equal")

    # 右图：误差
    ax_err.set_facecolor("#0a0a0a")
    ax_err.scatter([r.timestamp for r in gps_readings], gps_err,
                    color="#00ccff", s=8, alpha=0.5, label="GPS error")
    ax_err.scatter([r.timestamp for r in lidar_readings], lidar_err,
                    color="#ff8800", s=3, alpha=0.3, label="LiDAR error")
    ax_err.plot(fused_times, fused_err, color="#ffffff", linewidth=2,
                label="Fused error")
    ax_err.set_xlabel("Time [s]", color="#aaaaaa")
    ax_err.set_ylabel("Position Error [m]", color="#aaaaaa")
    ax_err.set_title("Error Comparison", color="#bbbbbb", fontsize=13)
    ax_err.legend(loc="upper right", facecolor="#111111",
                  edgecolor="#333333", labelcolor="#cccccc")
    ax_err.grid(True, color="#222222", alpha=0.4)

    save_path = OUT_DIR / "fusion_result.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {save_path}")

    # 写融合后的 CSV
    csv_path = OUT_DIR / "fused_data.csv"
    print(f"\n→ 写入融合数据 CSV …")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "sensor_id", "x", "y",
                          "cov_xx", "cov_xy", "cov_yy"])
        for i, t in enumerate(fused_times):
            cov = fused_covs[i]
            writer.writerow([f"{t:.3f}", "fused",
                             f"{fused_data[i, 0]:.4f}",
                             f"{fused_data[i, 1]:.4f}",
                             f"{cov[0, 0]:.6f}",
                             f"{cov[0, 1]:.6f}",
                             f"{cov[1, 1]:.6f}"])
    print(f"  [OK] {csv_path}")

    print("\nDone. Fusion experiment complete.")


if __name__ == "__main__":
    main()
