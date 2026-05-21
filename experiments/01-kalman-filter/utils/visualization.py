"""可视化工具：深色主题的轨迹图、误差图、追踪动画和四合一对比。

所有图片输出到项目的 output/ 目录。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# ── 统一配色 ──
COLORS = {
    "true": "#00ff88",
    "obs": "#ff4444",
    "est": "#4488ff",
    "ellipse": "#4488ff",
    "grid": "#222222",
    "kf": "#4488ff",
    "ekf": "#ffaa00",
    "ukf": "#aa44ff",
    "pf": "#ff66aa",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 不确定性椭圆工具
# ═══════════════════════════════════════════════════════════════════════════════
def uncertainty_ellipse(cov_2x2, n_std=1.0):
    """从 2x2 协方差矩阵提取 1σ 不确定性椭圆的参数。

    协方差矩阵的特征值 → 椭圆轴长
    协方差矩阵的特征向量 → 椭圆方向

    1σ 椭圆 = 物体有 68% 的概率落在椭圆内。

    返回:
        (width, height, angle_degrees)
    """
    eigvals, eigvecs = np.linalg.eigh(cov_2x2)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 0))
    return width, height, angle


# ═══════════════════════════════════════════════════════════════════════════════
# 单滤波器静态图
# ═══════════════════════════════════════════════════════════════════════════════
def draw_tracking_figure(true_pos, obs, est, covs, title, save_path,
                         est_color=COLORS["est"], obs_label="Observation"):
    """绘制单个滤波器的轨迹对比 + 误差曲线图。

    左子图：真实轨迹 / 观测 / 估计轨迹 + 不确定性椭圆
    右子图：观测误差和滤波误差随时间变化

    参数:
        true_pos:  (STEPS, 2) 真实位置
        obs:       (STEPS, 2) 观测（用于可视化的笛卡尔坐标）
        est:       (STEPS, 2) 估计位置
        covs:      list[4x4]  每步的协方差矩阵
        title:     str        图标题
        save_path: Path      保存路径
        est_color: str       估计轨迹颜色
        obs_label: str       观测标签
    """
    OUT_DIR.mkdir(exist_ok=True)
    steps = len(true_pos)
    time = np.arange(steps) * 0.1

    plt.style.use("dark_background")

    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(16, 7),
                                           facecolor="#0a0a0a")
    fig.suptitle(title, fontsize=18, fontweight="bold", color="#cccccc", y=0.97)

    # ── 左子图：轨迹对比 ──
    ax_traj.set_facecolor("#0a0a0a")
    ax_traj.plot(true_pos[:, 0], true_pos[:, 1],
                 color=COLORS["true"], linewidth=2, label="Ground Truth")
    ax_traj.plot(obs[:, 0], obs[:, 1],
                 color=COLORS["obs"], linewidth=0.7, alpha=0.6,
                 label=obs_label)
    ax_traj.plot(est[:, 0], est[:, 1],
                 color=est_color, linewidth=2, label="Kalman Estimate")

    # 每隔一段距离画一个不确定性椭圆
    step_skip = max(1, steps // 18)
    for i in range(0, steps, step_skip):
        cov_2x2 = covs[i][:2, :2]
        w, h, ang = uncertainty_ellipse(cov_2x2, n_std=1.0)
        if w < 1e-6:
            continue
        e = Ellipse((est[i, 0], est[i, 1]), w, h, angle=ang,
                    facecolor=est_color, edgecolor=est_color,
                    alpha=0.08, linewidth=0.5)
        ax_traj.add_patch(e)

    ax_traj.set_xlabel("X position [m]", color="#aaaaaa")
    ax_traj.set_ylabel("Y position [m]", color="#aaaaaa")
    ax_traj.legend(loc="upper right", facecolor="#111111",
                   edgecolor="#333333", labelcolor="#cccccc")
    ax_traj.set_title("Trajectory Comparison", color="#bbbbbb", fontsize=13)
    ax_traj.grid(True, color=COLORS["grid"], alpha=0.4)
    ax_traj.set_aspect("equal")

    # ── 右子图：误差随时间变化 ──
    obs_error = np.linalg.norm(obs - true_pos, axis=1)
    kf_error = np.linalg.norm(est - true_pos, axis=1)

    ax_err.set_facecolor("#0a0a0a")
    ax_err.fill_between(time, 0, obs_error, color=COLORS["obs"],
                        alpha=0.12)
    ax_err.plot(time, obs_error, color=COLORS["obs"], linewidth=0.8,
                alpha=0.5, label="Observation Error")
    ax_err.plot(time, kf_error, color=est_color, linewidth=2,
                label="Kalman Error")

    obs_rmse = np.sqrt(np.mean(obs_error ** 2))
    kf_rmse = np.sqrt(np.mean(kf_error ** 2))
    reduction = (1 - kf_rmse / obs_rmse) * 100 if obs_rmse > 0 else 0
    ax_err.text(0.98, 0.92,
                f"Obs RMSE: {obs_rmse:.3f} m\nKF  RMSE: {kf_rmse:.3f} m\n"
                f"Reduction: {reduction:.1f}%",
                transform=ax_err.transAxes, ha="right", va="top",
                fontsize=11, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#111111",
                          edgecolor="#444444", alpha=0.9),
                color="#cccccc")

    ax_err.set_xlabel("Time [s]", color="#aaaaaa")
    ax_err.set_ylabel("Position Error [m]", color="#aaaaaa")
    ax_err.set_title("Error Comparison Over Time", color="#bbbbbb", fontsize=13)
    ax_err.legend(loc="upper right", facecolor="#111111",
                  edgecolor="#333333", labelcolor="#cccccc")
    ax_err.grid(True, color=COLORS["grid"], alpha=0.4)

    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 单滤波器动画
# ═══════════════════════════════════════════════════════════════════════════════
def draw_tracking_animation(true_pos, obs, est, covs, title, save_path,
                            est_color=COLORS["est"], obs_label="Observation",
                            dt=0.1, total_time=12.0):
    """生成追踪过程动画 GIF。

    动画展示了滤波器如何逐步接收传感器数据、修正估计、
    不确定性椭圆逐渐缩小（"信心增长"过程）。
    """
    OUT_DIR.mkdir(exist_ok=True)
    steps = len(true_pos)
    time = np.arange(steps) * dt

    plt.style.use("dark_background")

    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(16, 7),
                                           facecolor="#0a0a0a")
    fig.suptitle(title, fontsize=18, fontweight="bold", color="#cccccc", y=0.97)

    obs_err_all = np.linalg.norm(obs - true_pos, axis=1)
    kf_err_all = np.linalg.norm(est - true_pos, axis=1)
    max_err = np.max(obs_err_all) * 1.05

    (line_true,) = ax_traj.plot([], [], color=COLORS["true"], linewidth=2,
                                label="Ground Truth")
    (line_obs,) = ax_traj.plot([], [], color=COLORS["obs"], linewidth=0.6,
                               alpha=0.4, label=obs_label)
    (line_est,) = ax_traj.plot([], [], color=est_color, linewidth=2.5,
                               label="Kalman Estimate")
    (scat_true,) = ax_traj.plot([], [], "o", color=COLORS["true"],
                                markersize=8, markeredgecolor="white",
                                markeredgewidth=0.5)
    (scat_est,) = ax_traj.plot([], [], "o", color=est_color,
                               markersize=10, markeredgecolor="white",
                               markeredgewidth=0.5)
    ellipse_patch = Ellipse((0, 0), 0, 0, angle=0,
                            facecolor=est_color, edgecolor=est_color,
                            alpha=0.15, linewidth=0.5)
    ax_traj.add_patch(ellipse_patch)

    ax_traj.set_facecolor("#0a0a0a")
    ax_traj.set_xlabel("X position [m]", color="#aaaaaa")
    ax_traj.set_ylabel("Y position [m]", color="#aaaaaa")
    ax_traj.legend(loc="upper right", facecolor="#111111",
                   edgecolor="#333333", labelcolor="#cccccc")
    ax_traj.set_title("Trajectory", color="#bbbbbb", fontsize=13)
    ax_traj.grid(True, color=COLORS["grid"], alpha=0.4)
    ax_traj.set_aspect("equal")

    ax_err.set_facecolor("#0a0a0a")
    (line_obs_err,) = ax_err.plot([], [], color=COLORS["obs"], linewidth=0.6,
                                  alpha=0.4, label="Observation Error")
    (line_kf_err,) = ax_err.plot([], [], color=est_color, linewidth=2.5,
                                 label="Kalman Error")
    (scat_obs_err,) = ax_err.plot([], [], "o", color=COLORS["obs"],
                                  markersize=4, alpha=0.5)
    (scat_kf_err,) = ax_err.plot([], [], "o", color=est_color,
                                 markersize=5)
    ax_err.set_xlim(0, total_time)
    ax_err.set_ylim(0, max_err)
    ax_err.set_xlabel("Time [s]", color="#aaaaaa")
    ax_err.set_ylabel("Position Error [m]", color="#aaaaaa")
    ax_err.set_title("Error Over Time", color="#bbbbbb", fontsize=13)
    ax_err.legend(loc="upper right", facecolor="#111111",
                  edgecolor="#333333", labelcolor="#cccccc")
    ax_err.grid(True, color=COLORS["grid"], alpha=0.4)

    # 坐标范围
    all_x = np.concatenate([true_pos[:, 0], obs[:, 0], est[:, 0]])
    all_y = np.concatenate([true_pos[:, 1], obs[:, 1], est[:, 1]])
    pad_x = (all_x.max() - all_x.min()) * 0.1 + 0.5
    pad_y = (all_y.max() - all_y.min()) * 0.1 + 0.5
    ax_traj.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax_traj.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

    # 降采样帧数
    n_frames = min(steps, 150)
    stride = max(1, steps // n_frames)
    frames = list(range(0, steps, stride))
    if frames[-1] != steps - 1:
        frames.append(steps - 1)

    def animate(i):
        idx = frames[i]
        traj_slice = slice(0, idx + 1)

        line_true.set_data(true_pos[traj_slice, 0], true_pos[traj_slice, 1])
        line_obs.set_data(obs[traj_slice, 0], obs[traj_slice, 1])
        line_est.set_data(est[traj_slice, 0], est[traj_slice, 1])
        scat_true.set_data([true_pos[idx, 0]], [true_pos[idx, 1]])
        scat_est.set_data([est[idx, 0]], [est[idx, 1]])

        cov_2x2 = covs[idx][:2, :2]
        w, h, ang = uncertainty_ellipse(cov_2x2, n_std=1.0)
        ellipse_patch.set_center((est[idx, 0], est[idx, 1]))
        ellipse_patch.width = max(w, 1e-6)
        ellipse_patch.height = max(h, 1e-6)
        ellipse_patch.angle = ang

        t = time[traj_slice]
        line_obs_err.set_data(t, obs_err_all[traj_slice])
        line_kf_err.set_data(t, kf_err_all[traj_slice])
        scat_obs_err.set_data([time[idx]], [obs_err_all[idx]])
        scat_kf_err.set_data([time[idx]], [kf_err_all[idx]])

        return (line_true, line_obs, line_est, scat_true, scat_est,
                ellipse_patch, line_obs_err, line_kf_err,
                scat_obs_err, scat_kf_err)

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                   interval=80, blit=True)
    ani.save(save_path, writer="pillow", fps=12, dpi=100)
    plt.close(fig)
    print(f"  [OK] {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 终端指标输出
# ═══════════════════════════════════════════════════════════════════════════════
def print_metrics(true_pos, obs_xy, est, label="KF"):
    """在终端打印滤波效果汇总。

    参数:
        true_pos: 真实位置
        obs_xy:   笛卡尔坐标系中的观测位置
        est:      滤波器估计位置
        label:    滤波器名称
    """
    obs_err = np.linalg.norm(obs_xy - true_pos, axis=1)
    est_err = np.linalg.norm(est - true_pos, axis=1)

    obs_rmse = np.sqrt(np.mean(obs_err ** 2))
    est_rmse = np.sqrt(np.mean(est_err ** 2))
    reduction = (1 - est_rmse / obs_rmse) * 100 if obs_rmse > 0 else 0

    print(f"  [{label}] Obs RMSE: {obs_rmse:.3f} m  |  "
          f"Est RMSE: {est_rmse:.3f} m  |  "
          f"Reduction: {reduction:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# 四合一对比图
# ═══════════════════════════════════════════════════════════════════════════════
def draw_comparison_figure(true_pos, results_dict, save_path):
    """绘制四个滤波器的并排对比图 + RMSE 柱状图。

    参数:
        true_pos:      (STEPS, 2) 真实位置
        results_dict:  { "KF": (obs_xy, est, covs), ... }
                       其中 obs_xy 已经是笛卡尔坐标
        save_path:     Path 保存路径
    """
    OUT_DIR.mkdir(exist_ok=True)
    steps = len(true_pos)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 13), facecolor="#0a0a0a")
    fig.suptitle("Kalman Filter Family — Comparison on Same Trajectory",
                 fontsize=18, fontweight="bold", color="#cccccc", y=0.97)

    filter_names = ["KF", "EKF", "UKF", "PF"]
    filter_colors = [COLORS["kf"], COLORS["ekf"], COLORS["ukf"], COLORS["pf"]]

    # ── 上排 4 个轨迹子图 + 下排 RMSE 柱状图 ──
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1], hspace=0.35, wspace=0.25)

    rmse_data = {}

    for idx, name in enumerate(filter_names):
        if name not in results_dict:
            continue
        obs_xy, est, covs = results_dict[name]
        color = filter_colors[idx]

        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor("#0a0a0a")

        # 真实轨迹
        ax.plot(true_pos[:, 0], true_pos[:, 1],
                color=COLORS["true"], linewidth=1.5, label="Ground Truth", alpha=0.7)
        # 观测点（稀疏绘制，避免过度拥挤）
        skip = max(1, steps // 60)
        ax.scatter(obs_xy[::skip, 0], obs_xy[::skip, 1],
                   color=COLORS["obs"], s=3, alpha=0.4)
        # 估计轨迹
        ax.plot(est[:, 0], est[:, 1],
                color=color, linewidth=2, label=f"{name} Estimate")

        # 不确定性椭圆
        ell_step = max(1, steps // 12)
        for i in range(0, steps, ell_step):
            cov_2x2 = covs[i][:2, :2]
            w, h, ang = uncertainty_ellipse(cov_2x2, n_std=1.0)
            if w < 1e-6:
                continue
            e = Ellipse((est[i, 0], est[i, 1]), w, h, angle=ang,
                        facecolor=color, edgecolor=color,
                        alpha=0.08, linewidth=0.5)
            ax.add_patch(e)

        ax.set_title(f"{name}", color=color, fontsize=16, fontweight="bold")
        ax.set_xlabel("X [m]", color="#888888", fontsize=9)
        ax.set_ylabel("Y [m]", color="#888888", fontsize=9)
        ax.grid(True, color=COLORS["grid"], alpha=0.3)
        ax.set_aspect("equal")
        ax.tick_params(colors="#888888", labelsize=8)

        # 计算 RMSE
        est_err = np.linalg.norm(est - true_pos, axis=1)
        obs_err = np.linalg.norm(obs_xy - true_pos, axis=1)
        rmse_data[name] = {
            "est_rmse": np.sqrt(np.mean(est_err ** 2)),
            "obs_rmse": np.sqrt(np.mean(obs_err ** 2)),
        }

    # ── 下排 RMSE 柱状图 ──
    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_facecolor("#0a0a0a")

    x_pos = np.arange(len(filter_names))
    est_rmse_vals = [rmse_data[n]["est_rmse"] for n in filter_names]
    obs_rmse_vals = [rmse_data[n]["obs_rmse"] for n in filter_names]

    bar_width = 0.35
    bars_obs = ax_bar.bar(x_pos - bar_width/2, obs_rmse_vals, bar_width,
                          color=COLORS["obs"], alpha=0.6, label="Observation RMSE")
    bars_est = ax_bar.bar(x_pos + bar_width/2, est_rmse_vals, bar_width,
                          color=filter_colors, label="Estimate RMSE")

    # 标数值
    for i, (obs_v, est_v) in enumerate(zip(obs_rmse_vals, est_rmse_vals)):
        red_pct = (1 - est_v / obs_v) * 100 if obs_v > 0 else 0
        ax_bar.text(i, est_v + 0.01, f"{est_v:.3f}m",
                    ha="center", va="bottom", fontsize=10, color="#cccccc")
        ax_bar.text(i, max(obs_v, est_v) + 0.06,
                    f"-{red_pct:.0f}%", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="#00ff88")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(filter_names, fontsize=13, color="#cccccc")
    ax_bar.set_ylabel("RMSE [m]", color="#aaaaaa")
    ax_bar.set_title("Position Error Comparison", color="#bbbbbb", fontsize=13)
    ax_bar.legend(loc="upper right", facecolor="#111111",
                  edgecolor="#333333", labelcolor="#cccccc")
    ax_bar.grid(True, color=COLORS["grid"], alpha=0.3, axis="y")
    ax_bar.tick_params(colors="#888888")

    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {save_path}")
