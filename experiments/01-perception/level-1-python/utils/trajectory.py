"""工程化轨迹生成器。

支持多种轨迹形状，输出分离的时间戳、位置、速度、航向数组。
用于传感器仿真实验，替代 simulation.py 中的 generate_ground_truth()。

支持的轨迹类型:
    "straight"  — 匀速直线运动
    "circle"    — 匀速圆周运动
    "figure8"   — 八字形轨迹
    "random_walk" — 随机游走（布朗运动）
"""

import numpy as np


def generate_trajectory(duration=12.0, dt=0.01, trajectory_type="figure8",
                        seed=42):
    """生成地面真值轨迹。

    参数:
        duration:        总时长 [秒]
        dt:              时间步长 [秒]
        trajectory_type: 轨迹类型
        seed:            随机种子（仅 random_walk 使用）

    返回:
        timestamps: (N,) 时间戳 [秒]
        positions:  (N, 2) [x, y] 位置
        velocities: (N, 2) [vx, vy] 速度
        headings:   (N,) 航向角 [弧度]
    """
    n_steps = int(duration / dt) + 1
    timestamps = np.linspace(0, duration, n_steps)

    generators = {
        "straight": _straight,
        "circle": _circle,
        "figure8": _figure8,
        "random_walk": _random_walk,
    }

    if trajectory_type not in generators:
        raise ValueError(
            f"Unknown trajectory type: {trajectory_type}. "
            f"Choose from {list(generators.keys())}")

    return generators[trajectory_type](timestamps, dt, seed)


def _straight(timestamps, dt, seed):
    """匀速直线运动。"""
    n = len(timestamps)
    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))
    headings = np.zeros(n)

    vx, vy = 1.0, 0.3
    for i in range(1, n):
        pos[i] = pos[i - 1] + np.array([vx, vy]) * dt
        vel[i] = [vx, vy]
        headings[i] = np.arctan2(vy, vx)

    vel[0] = vel[1]
    headings[0] = headings[1]
    return timestamps, pos, vel, headings


def _circle(timestamps, dt, seed):
    """匀速圆周运动。"""
    n = len(timestamps)
    radius = 5.0
    omega = 0.3  # rad/s

    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))
    headings = np.zeros(n)

    for i, t in enumerate(timestamps):
        angle = omega * t
        pos[i] = [radius * np.cos(angle), radius * np.sin(angle)]
        vel[i] = [-radius * omega * np.sin(angle),
                   radius * omega * np.cos(angle)]
        headings[i] = angle + np.pi / 2

    return timestamps, pos, vel, headings


def _figure8(timestamps, dt, seed):
    """八字形轨迹（Lissajous 曲线）。"""
    n = len(timestamps)
    ax, ay = 6.0, 4.0
    wx, wy = 0.3, 0.6

    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))
    headings = np.zeros(n)

    for i, t in enumerate(timestamps):
        pos[i] = [ax * np.sin(wx * t), ay * np.sin(wy * t)]
        vel[i] = [ax * wx * np.cos(wx * t), ay * wy * np.cos(wy * t)]
        headings[i] = np.arctan2(vel[i][1], vel[i][0])

    return timestamps, pos, vel, headings


def _random_walk(timestamps, dt, seed):
    """随机游走轨迹（布朗运动）。"""
    rng = np.random.default_rng(seed)
    n = len(timestamps)
    pos = np.zeros((n, 2))
    vel = np.zeros((n, 2))
    headings = np.zeros(n)

    speed = 0.8
    for i in range(1, n):
        angle = headings[i - 1] + rng.normal(0, 0.05)
        headings[i] = angle
        vel[i] = [speed * np.cos(angle), speed * np.sin(angle)]
        pos[i] = pos[i - 1] + vel[i] * dt

    vel[0] = vel[1]
    headings[0] = headings[1]
    return timestamps, pos, vel, headings
