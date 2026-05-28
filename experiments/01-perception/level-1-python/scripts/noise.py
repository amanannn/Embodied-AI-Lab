"""噪声类型对比实验。

对同一条轨迹信号分别叠加四种噪声类型，
生成 2x2 对比图，直观展示每种噪声的特征。

用法:
    python scripts/noise.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.trajectory import generate_trajectory
from noise import (gaussian_noise, constant_bias, random_walk_drift,
                    outlier_noise, noise_stats)

OUT_DIR = Path(__file__).resolve().parent.parent / "output"


def main():
    print("=" * 60)
    print("  噪声类型对比实验")
    print("=" * 60)

    # 生成轨迹
    print("→ 生成真实轨迹 …")
    ts, pos, _, _ = generate_trajectory(duration=12.0, dt=0.01,
                                         trajectory_type="figure8")
    clean_x = pos[:, 0]

    # 四种噪声
    print("→ 生成四种噪声 …")
    noise_gauss = gaussian_noise(ts, std=0.5, seed=42)
    noise_bias = constant_bias(ts, bias_value=1.0, warmup_time=2.0)
    noise_drift = random_walk_drift(ts, drift_std=0.05, seed=42)
    noise_outlier = outlier_noise(ts, outlier_prob=0.03,
                                   outlier_magnitude=3.0, seed=42)

    noisy_gauss = clean_x + noise_gauss
    noisy_bias = clean_x + noise_bias
    noisy_drift = clean_x + noise_drift
    noisy_outlier = clean_x + noise_outlier

    # 终端指标
    print("\n→ 噪声统计量:")
    stats = [
        ("Gaussian", noise_stats(noisy_gauss, clean_x)),
        ("Bias", noise_stats(noisy_bias, clean_x)),
        ("Drift", noise_stats(noisy_drift, clean_x)),
        ("Outlier", noise_stats(noisy_outlier, clean_x)),
    ]
    print(f"  {'Type':<10} {'Mean':>8} {'Std':>8} {'Max|err|':>10} {'Drift':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for name, s in stats:
        print(f"  {name:<10} {s['mean']:>8.3f} {s['std']:>8.3f} "
              f"{s['max_abs']:>10.3f} {s['cumulative_drift']:>8.3f}")

    # 可视化
    print("\n→ 渲染对比图 …")
    OUT_DIR.mkdir(exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#0a0a0a")
    fig.suptitle("Noise Types — Effect on Signal",
                 fontsize=18, fontweight="bold", color="#cccccc", y=0.97)

    panels = [
        (axes[0, 0], noisy_gauss, noise_gauss, "Gaussian Noise",
         "#4488ff", f"std={np.std(noise_gauss):.3f}"),
        (axes[0, 1], noisy_bias, noise_bias, "Constant Bias",
         "#ff8800", f"bias={1.0:.1f}, warmup=2s"),
        (axes[1, 0], noisy_drift, noise_drift, "Random Walk Drift",
         "#ff44ff", f"drift_std=0.05"),
        (axes[1, 1], noisy_outlier, noise_outlier, "Outliers",
         "#ff4444", f"prob=3%, mag=3.0"),
    ]

    for ax, noisy, noise_only, title, color, desc in panels:
        ax.set_facecolor("#0a0a0a")
        ax.plot(ts, clean_x, color="#00ff88", linewidth=1.5,
                label="Clean signal", alpha=0.7)
        ax.plot(ts, noisy, color=color, linewidth=0.8, alpha=0.6,
                label="Noisy signal")
        ax.set_title(f"{title}\n({desc})", color=color, fontsize=12)
        ax.set_xlabel("Time [s]", color="#aaaaaa", fontsize=9)
        ax.set_ylabel("Signal value", color="#aaaaaa", fontsize=9)
        ax.legend(loc="upper right", facecolor="#111111",
                  edgecolor="#333333", labelcolor="#cccccc", fontsize=8)
        ax.grid(True, color="#222222", alpha=0.4)

    save_path = OUT_DIR / "noise_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {save_path}")

    print("\nDone. Noise comparison complete.")


if __name__ == "__main__":
    main()
