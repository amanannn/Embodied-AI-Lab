"""噪声模型：四种基础噪声类型 + 复合噪声函数。

本模块提供传感器仿真的噪声生成工具。每种噪声函数接受时间戳数组，
返回等长的噪声数组，可直接叠加到干净信号上。

噪声类型:
    gaussian_noise   — 高斯白噪声，最基础的随机误差
    constant_bias    — 常值偏置，传感器系统性偏差
    random_walk_drift — 随机游走漂移，缓慢累积的误差
    outlier_noise    — 离群值，偶发的大幅异常读数

用法:
    from noise import apply_noise
    noisy = apply_noise(clean_signal, timestamps, {
        "gaussian_std": 0.5,
        "bias": 0.1,
        "drift_std": 0.01,
        "outlier_prob": 0.02,
        "outlier_magnitude": 3.0,
    })
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# 高斯白噪声
# ═══════════════════════════════════════════════════════════════════════════════
def gaussian_noise(timestamps, std, seed=42):
    """高斯白噪声：每个时刻独立采样 N(0, std)。

    这是最基础的传感器噪声模型。真实传感器的读数在真值上下随机波动，
    波动幅度由标准差 std 控制。

    参数:
        timestamps: (N,) 时间戳数组 [秒]
        std: 噪声标准差
        seed: 随机种子，保证可复现

    返回:
        (N,) 噪声数组
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0, std, size=len(timestamps))


# ═══════════════════════════════════════════════════════════════════════════════
# 常值偏置
# ═══════════════════════════════════════════════════════════════════════════════
def constant_bias(timestamps, bias_value, warmup_time=0.0):
    """常值偏置：传感器读数整体偏移一个固定值。

    真实场景：IMU 加速度计的零偏、GPS 的系统性位置偏差。
    偏置在传感器开机时就存在，不会随时间变化。

    warmup_time 模拟传感器预热过程：在预热期内输出 0（传感器还没准备好），
    预热结束后输出恒定偏置值。

    参数:
        timestamps: (N,) 时间戳数组 [秒]
        bias_value: 偏置大小（可正可负）
        warmup_time: 预热时间 [秒]，此期间偏置为 0

    返回:
        (N,) 偏置数组
    """
    bias = np.zeros(len(timestamps))
    bias[timestamps >= warmup_time] = bias_value
    return bias


# ═══════════════════════════════════════════════════════════════════════════════
# 随机游走漂移
# ═══════════════════════════════════════════════════════════════════════════════
def random_walk_drift(timestamps, drift_std, seed=42):
    """随机游走漂移：误差随时间缓慢累积。

    真实场景：陀螺仪积分漂移、温度变化引起的传感器慢变误差。
    每一步的漂移量 = 上一步的漂移 + N(0, drift_std)。

    漂移会随时间越走越远——这就是为什么 IMU 不能长时间单独使用。

    参数:
        timestamps: (N,) 时间戳数组 [秒]
        drift_std: 每步漂移的标准差
        seed: 随机种子

    返回:
        (N,) 漂移数组（从 0 开始累积）

    TODO: Student Exercise
        当前实现假设 timestamps 均匀采样。
        请修改为：根据相邻时间戳的实际间隔 dt 来缩放漂移步长，
        使其适用于非均匀采样的传感器数据。
    """
    rng = np.random.default_rng(seed)
    n = len(timestamps)
    steps = rng.normal(0, drift_std, size=n)
    steps[0] = 0.0  # 初始时刻无漂移
    return np.cumsum(steps)


# ═══════════════════════════════════════════════════════════════════════════════
# 离群值
# ═══════════════════════════════════════════════════════════════════════════════
def outlier_noise(timestamps, outlier_prob, outlier_magnitude, seed=42):
    """离群值噪声：以一定概率产生大幅异常读数。

    真实场景：LiDAR 的多路径反射、GPS 在城市峡谷中的多径效应、
    传感器瞬间受到电磁干扰。大部分时候读数正常，偶尔出现严重偏离。

    参数:
        timestamps: (N,) 时间戳数组 [秒]
        outlier_prob: 离群值出现概率 (0~1)
        outlier_magnitude: 离群值的最大幅度
        seed: 随机种子

    返回:
        (N,) 离群值数组（大部分为 0，少数位置有大值）
    """
    rng = np.random.default_rng(seed)
    n = len(timestamps)
    noise = np.zeros(n)

    # 决定哪些时刻出现离群值
    mask = rng.random(n) < outlier_prob

    # 离群值幅度：均匀分布在 [-magnitude, +magnitude]
    noise[mask] = rng.uniform(-outlier_magnitude, outlier_magnitude,
                               size=mask.sum())
    return noise


# ═══════════════════════════════════════════════════════════════════════════════
# 复合噪声函数
# ═══════════════════════════════════════════════════════════════════════════════
def apply_noise(clean_signal, timestamps, noise_config, seed=42):
    """将多种噪声叠加到干净信号上。

    每种噪声都是可选的——noise_config 中不存在的键会被跳过。
    所有噪声类型叠加在一起，模拟真实传感器的综合噪声特征。

    参数:
        clean_signal: (N,) 或 (N, D) 干净信号
        timestamps: (N,) 时间戳数组 [秒]
        noise_config: dict，可包含以下键:
            "gaussian_std": float — 高斯噪声标准差
            "bias": float — 常值偏置
            "warmup_time": float — 偏置预热时间
            "drift_std": float — 随机漂移标准差
            "outlier_prob": float — 离群值概率
            "outlier_magnitude": float — 离群值幅度
        seed: 随机种子

    返回:
        带噪声的信号，形状与 clean_signal 相同
    """
    signal = np.array(clean_signal, dtype=float)
    ndim = signal.ndim
    n = len(timestamps)

    # 对多维信号，每维独立加噪声
    if ndim == 1:
        noise = _build_noise_1d(timestamps, noise_config, seed)
        return signal + noise
    else:
        result = signal.copy()
        for d in range(signal.shape[1]):
            noise = _build_noise_1d(timestamps, noise_config, seed + d)
            result[:, d] += noise
        return result


def _build_noise_1d(timestamps, config, seed):
    """为一维信号构建复合噪声。"""
    n = len(timestamps)
    total = np.zeros(n)

    if "gaussian_std" in config:
        total += gaussian_noise(timestamps, config["gaussian_std"], seed=seed)

    if "bias" in config:
        warmup = config.get("warmup_time", 0.0)
        total += constant_bias(timestamps, config["bias"], warmup)

    if "drift_std" in config:
        total += random_walk_drift(timestamps, config["drift_std"],
                                    seed=seed + 1000)

    if "outlier_prob" in config and "outlier_magnitude" in config:
        total += outlier_noise(timestamps, config["outlier_prob"],
                                config["outlier_magnitude"], seed=seed + 2000)

    return total


# ═══════════════════════════════════════════════════════════════════════════════
# 噪声诊断工具
# ═══════════════════════════════════════════════════════════════════════════════
def noise_stats(noisy_signal, clean_signal=None):
    """计算噪声的基本统计量。

    如果提供 clean_signal，则从残差中分析噪声特征。
    如果不提供，则假设 noisy_signal 本身就是噪声。

    返回 dict:
        "mean": 均值（检测偏置）
        "std": 标准差（衡量噪声强度）
        "max_abs": 最大绝对值（检测离群值）
        "cumulative_drift": 首尾差值（检测漂移趋势）
    """
    if clean_signal is not None:
        residual = np.array(noisy_signal) - np.array(clean_signal)
    else:
        residual = np.array(noisy_signal)

    return {
        "mean": float(np.mean(residual)),
        "std": float(np.std(residual)),
        "max_abs": float(np.max(np.abs(residual))),
        "cumulative_drift": float(residual[-1] - residual[0]) if len(residual) > 1 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TODO: Student Exercise — 噪声诊断函数
# ═══════════════════════════════════════════════════════════════════════════════
def diagnose_noise(noisy_signal, clean_signal):
    """判断残差中包含哪些噪声类型。

    给定带噪声的信号和对应的干净信号，分析残差并判断：
    1. 是否存在显著偏置（|mean| > 阈值）
    2. 是否存在漂移趋势（首尾差 > 阈值）
    3. 是否存在离群值（max_abs > 3 * std）
    4. 高斯噪声的估计标准差

    TODO: Student Exercise
        实现这个函数。提示：
        - 用 noise_stats() 获取基本统计量
        - 偏置检测：残差均值是否显著偏离 0
        - 漂移检测：残差的首尾差是否远大于标准差
        - 离群值检测：是否有超过 3σ 的点
        - 返回一个 dict 描述诊断结果

    参数:
        noisy_signal: (N,) 带噪声的信号
        clean_signal: (N,) 干净信号（真值）

    返回:
        dict: {
            "has_bias": bool,
            "has_drift": bool,
            "has_outliers": bool,
            "gaussian_std_estimate": float,
        }
    """
    raise NotImplementedError("请自己实现这个函数")
