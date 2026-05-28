"""卡尔曼滤波参数实验。

对同一轨迹运行多组不同 Q/R 参数的卡尔曼滤波器，
对比滤波效果，帮助学生理解参数选择的影响。

用法:
    python scripts/kf_tuning.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.simulation import (generate_ground_truth, linear_observations,
                               DT, STEPS, TOTAL_TIME, SENSOR_NOISE)
from filters.kalman_filter import KalmanFilter

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# ── 参数组定义 ──
# 学生可以修改这些组，或添加自己的实验组
PARAM_GROUPS = {
    "Default": {
        "Q_POS": 0.01, "Q_VEL": 0.05,
        "R_scale": 1.0,  # R = R_scale * SENSOR_NOISE^2
        "skip": 1,       # 每 skip 步取一次观测
        "desc": "平衡设定",
    },
    "Conservative": {
        "Q_POS": 0.001, "Q_VEL": 0.005,
        "R_scale": 0.5,
        "skip": 1,
        "desc": "更信任模型（Q 小）",
    },
    "Aggressive": {
        "Q_POS": 0.1, "Q_VEL": 0.5,
        "R_scale": 2.0,
        "skip": 1,
        "desc": "更信任传感器（Q 大）",
    },
    "Over-trust Sensor": {
        "Q_POS": 0.0001, "Q_VEL": 0.0001,
        "R_scale": 5.0,
        "skip": 1,
        "desc": "几乎忽略模型",
    },
    "Downsampled": {
        "Q_POS": 0.01, "Q_VEL": 0.05,
        "R_scale": 1.0,
        "skip": 5,  # 每 5 步取一次观测
        "desc": "降采样观测（1/5 频率）",
    },
}


def run_kf_with_params(true_pos, observations, dt, q_pos, q_vel, r_scale,
                        skip=1):
    """用指定参数运行卡尔曼滤波器。"""
    r = np.eye(2) * (SENSOR_NOISE ** 2) * r_scale
    kf = KalmanFilter(dt, np.array([q_pos, q_pos, q_vel, q_vel]), r)

    for i, z in enumerate(observations):
        if i % skip == 0:
            kf.step(z)
        else:
            kf.predict()  # 只预测不更新

    return np.array(kf.estimates)


def main():
    print("=" * 60)
    print("  卡尔曼滤波参数实验")
    print("=" * 60)

    # 生成数据
    print("→ 生成真实轨迹 …")
    true_pos, _ = generate_ground_truth()
    observations = linear_observations(true_pos)

    # 运行所有参数组
    results = {}
    for name, cfg in PARAM_GROUPS.items():
        print(f"→ 运行 {name} ({cfg['desc']}) …")
        est = run_kf_with_params(
            true_pos, observations, DT,
            cfg["Q_POS"], cfg["Q_VEL"], cfg["R_scale"], cfg["skip"])
        obs_err = np.linalg.norm(observations - true_pos, axis=1)
        est_err = np.linalg.norm(est - true_pos, axis=1)
        results[name] = {
            "est": est,
            "obs_rmse": np.sqrt(np.mean(obs_err ** 2)),
            "est_rmse": np.sqrt(np.mean(est_err ** 2)),
        }

    # 终端表格
    print("\n→ 参数实验结果:")
    print(f"  {'Group':<20} {'Obs RMSE':>10} {'Est RMSE':>10} "
          f"{'Reduction':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in results.items():
        reduction = (1 - r["est_rmse"] / r["obs_rmse"]) * 100
        print(f"  {name:<20} {r['obs_rmse']:>10.3f} {r['est_rmse']:>10.3f} "
              f"{reduction:>9.1f}%")

    # 可视化
    print("\n→ 渲染对比图 …")
    OUT_DIR.mkdir(exist_ok=True)

    plt.style.use("dark_background")
    n_groups = len(PARAM_GROUPS)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 6),
                              facecolor="#0a0a0a")
    if n_groups == 1:
        axes = [axes]
    fig.suptitle("Kalman Filter — Parameter Tuning Comparison",
                 fontsize=18, fontweight="bold", color="#cccccc", y=0.97)

    colors = ["#4488ff", "#00ff88", "#ff8800", "#ff4444", "#aa44ff"]

    for idx, (name, cfg) in enumerate(PARAM_GROUPS.items()):
        ax = axes[idx]
        ax.set_facecolor("#0a0a0a")

        ax.plot(true_pos[:, 0], true_pos[:, 1], color="#00ff88",
                linewidth=1.5, alpha=0.7, label="Truth")
        ax.plot(observations[:, 0], observations[:, 1], color="#ff4444",
                linewidth=0.5, alpha=0.3, label="Obs")

        est = results[name]["est"]
        est_rmse = results[name]["est_rmse"]
        ax.plot(est[:, 0], est[:, 1], color=colors[idx],
                linewidth=2, label=f"Est (RMSE={est_rmse:.3f})")

        ax.set_title(f"{name}\n{cfg['desc']}", color=colors[idx], fontsize=10)
        ax.set_xlabel("X [m]", color="#888888", fontsize=8)
        ax.set_ylabel("Y [m]", color="#888888", fontsize=8)
        ax.legend(loc="upper right", facecolor="#111111",
                  edgecolor="#333333", labelcolor="#cccccc", fontsize=7)
        ax.grid(True, color="#222222", alpha=0.3)
        ax.set_aspect("equal")
        ax.tick_params(colors="#888888", labelsize=7)

    save_path = OUT_DIR / "kf_tuning_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {save_path}")

    print("\nDone. KF tuning experiment complete.")


if __name__ == "__main__":
    main()
