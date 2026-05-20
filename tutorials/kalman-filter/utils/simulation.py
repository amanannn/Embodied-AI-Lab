"""仿真数据生成：真实轨迹 + 线性/非线性观测模型。

本模块生成一条微弯的匀速运动轨迹作为"上帝视角"的地面真值，
并分别提供两类传感器观测：
  - 线性观测：直接观测位置 [x, y]（给线性 KF 用）
  - 雷达观测：观测距离+方位角 [range, bearing]（给 EKF/UKF/PF 用）
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 仿真参数
# ═══════════════════════════════════════════════════════════════════════════════
DT = 0.1            # 时间步长 [秒]
TOTAL_TIME = 12.0   # 总仿真时间 [秒]
STEPS = int(TOTAL_TIME / DT)

# 真实轨迹：从原点出发，主要向右移动，同时缓慢向下弯曲
TRUE_VX, TRUE_VY = 0.8, 0.25        # 基础速度 [米/秒]
CURVE_AX, CURVE_AY = 0.0, -0.02     # 微小向下加速度 [米/秒²]

# 线性观测（位置传感器）噪声标准差
SENSOR_NOISE = 0.45  # 米

# 雷达观测（range-bearing）噪声标准差
RANGE_NOISE_STD = 0.3       # 米
BEARING_NOISE_STD = 0.05    # 弧度（约 3°）


# ═══════════════════════════════════════════════════════════════════════════════
# 真实轨迹生成
# ═══════════════════════════════════════════════════════════════════════════════
def generate_ground_truth(dt=DT, total_time=TOTAL_TIME,
                          vx=TRUE_VX, vy=TRUE_VY,
                          ax=CURVE_AX, ay=CURVE_AY):
    """生成真实的位置序列和速度序列。

    物体以恒定加速度模型运动，形成一条微弯的曲线。
    你可以把它理解为"上帝视角"——只有仿真才知道的真实答案。

    返回:
        pos: (STEPS, 2) — [x, y] 位置
        vel: (STEPS, 2) — [vx, vy] 速度
    """
    steps = int(total_time / dt)
    pos = np.zeros((steps, 2))
    vel = np.zeros((steps, 2))

    for k in range(1, steps):
        t = k * dt
        vel[k] = [vx + ax * t, vy + ay * t]
        pos[k] = pos[k - 1] + vel[k] * dt

    return pos, vel


# ═══════════════════════════════════════════════════════════════════════════════
# 线性观测（位置传感器）
# ═══════════════════════════════════════════════════════════════════════════════
def linear_observations(true_pos, noise_std=SENSOR_NOISE, seed=42):
    """模拟位置传感器的读数 —— 在真实位置上叠加高斯噪声。

    这就像一个 GPS 接收器，每次告诉你 [x, y] 坐标，
    但每次都有随机误差。

    参数:
        true_pos: (STEPS, 2) 真实位置
        noise_std: 噪声标准差（米）
        seed: 随机种子，保证结果可复现

    返回:
        (STEPS, 2) 带噪声的观测位置
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, size=true_pos.shape)
    return true_pos + noise


# ═══════════════════════════════════════════════════════════════════════════════
# 非线性观测（雷达 / range-bearing 传感器）
# ═══════════════════════════════════════════════════════════════════════════════
def range_bearing_observations(true_pos,
                               range_std=RANGE_NOISE_STD,
                               bearing_std=BEARING_NOISE_STD,
                               seed=42):
    """模拟雷达传感器的读数 —— 测量距离和方位角，而非直接的 x,y 坐标。

    真实雷达（或激光雷达）不会告诉你"目标在 (3.2, 1.5)"，
    它告诉你"目标在 3.5 米远、25° 方向"。

    这个"从直角坐标到极坐标"的转换是非线性的（有平方和反正切），
    因此标准的线性卡尔曼滤波无法直接使用这类观测。

    参数:
        true_pos: (STEPS, 2) 真实位置
        range_std: 距离测量噪声标准差（米）
        bearing_std: 方位角测量噪声标准差（弧度）
        seed: 随机种子

    返回:
        (STEPS, 2) — 每行 [range, bearing]
    """
    rng = np.random.default_rng(seed)

    # 真值转到极坐标
    r_true = np.sqrt(true_pos[:, 0]**2 + true_pos[:, 1]**2)
    bearing_true = np.arctan2(true_pos[:, 1], true_pos[:, 0])

    # 加噪声
    r_noisy = r_true + rng.normal(0, range_std, size=len(true_pos))
    bearing_noisy = bearing_true + rng.normal(0, bearing_std, size=len(true_pos))

    return np.column_stack([r_noisy, bearing_noisy])


# ═══════════════════════════════════════════════════════════════════════════════
# 坐标转换工具
# ═══════════════════════════════════════════════════════════════════════════════
def range_bearing_to_xy(observations):
    """将 [range, bearing] 观测转回笛卡尔坐标 (x, y)。

    用于可视化 —— 我们想把雷达观测和位置估计画在同一个 x-y 平面上。

    参数:
        observations: (N, 2) — [range, bearing]

    返回:
        (N, 2) — [x, y]
    """
    r = observations[:, 0]
    bearing = observations[:, 1]
    x = r * np.cos(bearing)
    y = r * np.sin(bearing)
    return np.column_stack([x, y])
