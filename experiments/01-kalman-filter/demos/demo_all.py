"""四合一对比 Demo — 同一轨迹，四种滤波方案。

在同一条真实轨迹上，分别运行 KF、EKF、UKF、PF，
生成并排对比图和 RMSE 柱状图，终端打印汇总表格。

KF 使用线性位置观测，EKF/UKF/PF 使用雷达 range-bearing 观测。
所有滤波器在笛卡尔坐标系中计算 RMSE，确保公平对比。

用法:
    python demos/demo_all.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from utils.simulation import (generate_ground_truth, linear_observations,
                               range_bearing_observations, range_bearing_to_xy,
                               DT, STEPS, TOTAL_TIME,
                               SENSOR_NOISE, RANGE_NOISE_STD, BEARING_NOISE_STD)
from utils.visualization import (draw_comparison_figure, print_metrics)
from filters.kalman_filter import KalmanFilter
from filters.extended_kf import ExtendedKalmanFilter
from filters.unscented_kf import UnscentedKalmanFilter
from filters.particle_filter import ParticleFilter

OUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main():
    print("=" * 70)
    print("  卡尔曼滤波家族 — 四合一对比")
    print("  同一轨迹 | KF(线性观测) vs EKF/UKF/PF(雷达观测)")
    print("=" * 70)

    # ── 生成数据 ──
    print("\n→ 生成真实轨迹 …")
    true_pos, _ = generate_ground_truth()

    print("→ 模拟两类传感器 …")
    obs_linear = linear_observations(true_pos, seed=100)  # KF 用
    obs_rb = range_bearing_observations(true_pos, seed=200)  # EKF/UKF/PF 用

    results = {}

    # ── KF ──
    print("\n→ 运行 KF …")
    kf = KalmanFilter(
        DT, np.array([0.01, 0.01, 0.05, 0.05]),
        np.eye(2) * (SENSOR_NOISE ** 2))
    for z in obs_linear:
        kf.step(z)
    results["KF"] = (obs_linear, np.array(kf.estimates), kf.covariances)

    # ── EKF ──
    print("→ 运行 EKF …")
    R_rb = np.diag([RANGE_NOISE_STD**2, BEARING_NOISE_STD**2])
    init_xy = range_bearing_to_xy(obs_rb[:1])[0]
    init_x = np.array([init_xy[0], init_xy[1], 0.0, 0.0])
    ekf = ExtendedKalmanFilter(
        DT, np.array([0.01, 0.01, 0.05, 0.05]), R_rb, initial_x=init_x)
    for z in obs_rb:
        ekf.step(z)
    obs_xy_ekf = range_bearing_to_xy(obs_rb)
    results["EKF"] = (obs_xy_ekf, np.array(ekf.estimates), ekf.covariances)

    # ── UKF ──
    print("→ 运行 UKF …")
    ukf = UnscentedKalmanFilter(
        DT, np.array([0.01, 0.01, 0.05, 0.05]), R_rb,
        alpha=1.0, beta=2.0, kappa=0.0, initial_x=init_x)
    for z in obs_rb:
        ukf.step(z)
    results["UKF"] = (obs_xy_ekf, np.array(ukf.estimates), ukf.covariances)

    # ── PF ──
    print("→ 运行 PF …")
    pf = ParticleFilter(
        DT, np.array([0.05, 0.05, 0.10, 0.10]), R_rb,
        n_particles=500, initial_x=init_x)
    for z in obs_rb:
        pf.step(z)
    results["PF"] = (obs_xy_ekf, np.array(pf.estimates), pf.covariances)

    # ── 终端汇总 ──
    print("\n" + "=" * 70)
    print(f"  {'Filter':<8} {'Obs RMSE':<12} {'Est RMSE':<12} {'Reduction':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for name in ["KF", "EKF", "UKF", "PF"]:
        obs_xy, est, _ = results[name]
        obs_err = np.linalg.norm(obs_xy - true_pos, axis=1)
        est_err = np.linalg.norm(est - true_pos, axis=1)
        obs_rmse = np.sqrt(np.mean(obs_err ** 2))
        est_rmse = np.sqrt(np.mean(est_err ** 2))
        reduction = (1 - est_rmse / obs_rmse) * 100 if obs_rmse > 0 else 0
        print(f"  {name:<8} {obs_rmse:<12.3f} {est_rmse:<12.3f} {reduction:<11.1f}%")
    print("=" * 70)

    # ── 四合一对比图 ──
    print("\n→ 渲染四合一对比图 …")
    draw_comparison_figure(
        true_pos, results,
        OUT_DIR / "all_filters_comparison.png")

    print("\nDone. All demos complete. Output in output/")
    print("  - kf_static.png / kf_animated.gif")
    print("  - ekf_static.png / ekf_animated.gif")
    print("  - ukf_static.png / ukf_animated.gif")
    print("  - pf_static.png / pf_animated.gif")
    print("  - all_filters_comparison.png")


if __name__ == "__main__":
    main()
