"""粒子滤波 Demo — 2D 匀速目标追踪。

用 500 个加权粒子来近似状态的后验分布。
不像 KF/EKF/UKF 那样假设高斯分布，粒子滤波可以处理
任意形状的分布（多模态、非对称等）。

算法三步走：
    预测：  每个粒子按运动模型随机走一步
    加权：  根据传感器读数，越接近观测的粒子权重越大
    重采样：淘汰坏粒子，复制好粒子

用法:
    python demos/demo_pf.py
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
from filters.particle_filter import ParticleFilter

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# 滤波器参数
# PF 的 Q 需要比 KF/EKF/UKF 稍大，因为粒子需要充分扩散来保持多样性
Q_POS = 0.05
Q_VEL = 0.10
R = np.diag([RANGE_NOISE_STD**2, BEARING_NOISE_STD**2])

N_PARTICLES = 500  # 粒子数


def main():
    print("=" * 60)
    print("  粒子滤波 (PF) — 雷达 range-bearing 观测")
    print(f"  粒子数: {N_PARTICLES}")
    print("=" * 60)

    # 生成数据
    print("→ 生成真实轨迹 …")
    true_pos, _ = generate_ground_truth()
    print("→ 模拟雷达传感器观测 …")
    rb_obs = range_bearing_observations(true_pos)

    # 用第一个观测初始化粒子云中心
    first_xy = range_bearing_to_xy(rb_obs[:1])[0]
    init_x = np.array([first_xy[0], first_xy[1], 0.0, 0.0])

    print("→ 运行粒子滤波器 …")
    pf = ParticleFilter(
        DT, np.array([Q_POS, Q_POS, Q_VEL, Q_VEL]), R,
        n_particles=N_PARTICLES, initial_x=init_x)
    for z in rb_obs:
        pf.step(z)

    estimates = np.array(pf.estimates)
    covariances = pf.covariances

    obs_xy = range_bearing_to_xy(rb_obs)
    print_metrics(true_pos, obs_xy, estimates, "PF")

    print("\n→ 渲染图片 …")
    draw_tracking_figure(
        true_pos, obs_xy, estimates, covariances,
        "Particle Filter — Radar Range-Bearing Tracking",
        OUT_DIR / "pf_static.png",
        est_color=COLORS["pf"],
        obs_label="Radar Obs (XY converted)")

    draw_tracking_animation(
        true_pos, obs_xy, estimates, covariances,
        "PF — Radar Tracking Animation",
        OUT_DIR / "pf_animated.gif",
        est_color=COLORS["pf"],
        obs_label="Radar Obs (XY converted)",
        dt=DT, total_time=TOTAL_TIME)

    print(f"\n  [PF 原理] {N_PARTICLES} 个粒子 → 预测 → 加权 → 重采样。")
    print("  不假设高斯分布，可以处理任意分布。")
    print("  Done. PF demo complete.")


if __name__ == "__main__":
    main()
