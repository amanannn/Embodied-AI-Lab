"""传感器对比实验。

对同一条轨迹分别用 GPS、LiDAR、IMU、里程计进行观测，
展示不同采样率和噪声特征下的传感器数据差异。
输出对比图 + CSV 数据文件（供卡尔曼实验使用）。

用法:
    python scripts/sensors.py
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
from sensors import GPSSensor, LiDARSensor, IMUSensor, OdometrySensor
from sensors.lidar import LiDARSensor as LS

OUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main():
    print("=" * 60)
    print("  传感器对比实验")
    print("=" * 60)

    # 生成轨迹
    print("→ 生成真实轨迹 (figure8, 12s, dt=0.01) …")
    ts, pos, vel, headings = generate_trajectory(
        duration=12.0, dt=0.01, trajectory_type="figure8")

    # 初始化四种传感器
    print("→ 初始化传感器 …")
    gps = GPSSensor(noise_std=0.5, outlier_prob=0.02, seed=42)
    lidar = LiDARSensor(range_std=0.3, bearing_std=0.05, seed=42)
    imu = IMUSensor(accel_noise_std=0.1, gyro_noise_std=0.01, seed=42)
    odom = OdometrySensor(vel_noise_std=0.05, omega_noise_std=0.02, seed=42)

    # 生成各传感器观测序列
    print("→ 生成传感器观测序列 …")
    gps_readings = gps.generate_sequence(pos, ts)
    lidar_readings = lidar.generate_sequence(pos, ts)
    imu_readings = imu.generate_sequence(pos, ts)
    odom_readings = odom.generate_sequence(pos, ts)

    # 将 LiDAR 转为笛卡尔坐标用于可视化
    lidar_xy = np.array([LS.to_xy(r) for r in lidar_readings])

    # 收集 GPS 数据
    gps_data = np.array([r.data for r in gps_readings])

    # 终端统计
    print("\n→ 传感器统计:")
    print(f"  {'Sensor':<10} {'Rate':>6} {'Readings':>9} {'X std':>8} {'Y std':>8}")
    print(f"  {'-'*10} {'-'*6} {'-'*9} {'-'*8} {'-'*8}")

    # GPS
    gps_err = gps_data - pos[[int(r.timestamp / 0.01) for r in gps_readings]]
    print(f"  {'GPS':<10} {'1 Hz':>6} {len(gps_readings):>9} "
          f"{np.std(gps_err[:, 0]):>8.3f} {np.std(gps_err[:, 1]):>8.3f}")

    # LiDAR
    lidar_idx = [int(r.timestamp / 0.01) for r in lidar_readings]
    lidar_err = lidar_xy - pos[lidar_idx]
    print(f"  {'LiDAR':<10} {'10 Hz':>6} {len(lidar_readings):>9} "
          f"{np.std(lidar_err[:, 0]):>8.3f} {np.std(lidar_err[:, 1]):>8.3f}")

    # IMU
    print(f"  {'IMU':<10} {'100 Hz':>6} {len(imu_readings):>9} "
          f"{'N/A':>8} {'N/A':>8}")

    # Odometry
    print(f"  {'Odometry':<10} {'20 Hz':>6} {len(odom_readings):>9} "
          f"{'N/A':>8} {'N/A':>8}")

    # 可视化
    print("\n→ 渲染对比图 …")
    OUT_DIR.mkdir(exist_ok=True)

    plt.style.use("dark_background")
    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(16, 7),
                                            facecolor="#0a0a0a")
    fig.suptitle("Multi-Sensor Observation Comparison",
                 fontsize=18, fontweight="bold", color="#cccccc", y=0.97)

    # 左图：轨迹 + 各传感器观测
    ax_traj.set_facecolor("#0a0a0a")
    ax_traj.plot(pos[:, 0], pos[:, 1], color="#00ff88", linewidth=2,
                 label="Ground Truth", alpha=0.7)
    ax_traj.scatter(gps_data[:, 0], gps_data[:, 1], color="#00ccff",
                    s=20, alpha=0.7, label=f"GPS ({len(gps_readings)} pts)")
    ax_traj.scatter(lidar_xy[:, 0], lidar_xy[:, 1], color="#ff8800",
                    s=3, alpha=0.3, label=f"LiDAR ({len(lidar_readings)} pts)")
    ax_traj.set_xlabel("X [m]", color="#aaaaaa")
    ax_traj.set_ylabel("Y [m]", color="#aaaaaa")
    ax_traj.set_title("Trajectory + Sensor Observations",
                       color="#bbbbbb", fontsize=13)
    ax_traj.legend(loc="upper right", facecolor="#111111",
                   edgecolor="#333333", labelcolor="#cccccc", fontsize=9)
    ax_traj.grid(True, color="#222222", alpha=0.4)
    ax_traj.set_aspect("equal")

    # 右图：GPS 和 LiDAR 的位置误差
    ax_err.set_facecolor("#0a0a0a")

    gps_time = [r.timestamp for r in gps_readings]
    gps_err_norm = np.linalg.norm(gps_err, axis=1)
    ax_err.scatter(gps_time, gps_err_norm, color="#00ccff", s=15,
                    alpha=0.7, label="GPS error")

    lidar_time = [r.timestamp for r in lidar_readings]
    lidar_err_norm = np.linalg.norm(lidar_err, axis=1)
    ax_err.scatter(lidar_time, lidar_err_norm, color="#ff8800", s=5,
                    alpha=0.4, label="LiDAR error")

    ax_err.set_xlabel("Time [s]", color="#aaaaaa")
    ax_err.set_ylabel("Position Error [m]", color="#aaaaaa")
    ax_err.set_title("Position Error Over Time",
                      color="#bbbbbb", fontsize=13)
    ax_err.legend(loc="upper right", facecolor="#111111",
                  edgecolor="#333333", labelcolor="#cccccc")
    ax_err.grid(True, color="#222222", alpha=0.4)

    save_path = OUT_DIR / "sensors_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {save_path}")

    # 写 CSV
    csv_path = OUT_DIR / "sensor_data.csv"
    print(f"\n→ 写入传感器数据 CSV …")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "sensor_id", "x", "y",
                          "cov_xx", "cov_xy", "cov_yy"])
        for r in gps_readings:
            writer.writerow([f"{r.timestamp:.3f}", r.sensor_id,
                             f"{r.data[0]:.4f}", f"{r.data[1]:.4f}",
                             f"{r.covariance[0, 0]:.6f}",
                             f"{r.covariance[0, 1]:.6f}",
                             f"{r.covariance[1, 1]:.6f}"])
        for r in lidar_readings:
            xy = LS.to_xy(r)
            # LiDAR 的协方差在极坐标，转到笛卡尔近似
            writer.writerow([f"{r.timestamp:.3f}", r.sensor_id,
                             f"{xy[0]:.4f}", f"{xy[1]:.4f}",
                             f"{r.covariance[0, 0]:.6f}",
                             "0.000000",
                             f"{r.covariance[0, 0]:.6f}"])
    print(f"  [OK] {csv_path}")

    print("\nDone. Sensor comparison complete.")


if __name__ == "__main__":
    main()
