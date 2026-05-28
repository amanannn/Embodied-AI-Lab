"""多传感器加权融合模块。

提供最简单的多传感器数据融合方法：加权平均。
学生可以在此基础上实现更复杂的自适应融合。

核心思想:
    多个传感器同时观测同一个量，每个传感器有不同的精度。
    精度高的传感器给更大权重，精度低的给更小权重。
    加权平均后的结果比任何单个传感器都更准确。
"""

import numpy as np
from sensors.base import SensorReading


def weighted_fusion(readings, weights=None):
    """加权融合多个传感器的观测值。

    如果不指定权重，使用方差倒数自动加权：
    权重_i = 1 / trace(covariance_i)
    即噪声越小的传感器权重越大。

    参数:
        readings: list[SensorReading] — 同一时刻多个传感器的读数
        weights:  (K,) 手动指定的权重，或 None 自动计算

    返回:
        fused_data: (D,) 融合后的数据向量
        fused_cov:  (D, D) 融合后的等效协方差
    """
    if not readings:
        raise ValueError("No readings to fuse")

    k = len(readings)
    data_dim = len(readings[0].data)

    # 收集所有数据和协方差
    datas = np.array([r.data for r in readings])  # (K, D)
    covs = [r.covariance for r in readings]

    # 计算权重
    if weights is None:
        weights = np.array([1.0 / np.trace(c) for c in covs])
    weights = np.array(weights, dtype=float)

    # 归一化权重
    weights = weights / weights.sum()

    # 加权融合
    fused_data = np.zeros(data_dim)
    for i in range(k):
        fused_data += weights[i] * datas[i]

    # 融合协方差：加权平均协方差
    fused_cov = np.zeros_like(covs[0])
    for i in range(k):
        fused_cov += weights[i] ** 2 * covs[i]

    return fused_data, fused_cov


def fuse_sequence(sensor_readings_by_time):
    """对整个时间序列进行多传感器融合。

    按时间戳分组，同一时刻的多个传感器读数做加权融合。

    参数:
        sensor_readings_by_time: dict[float, list[SensorReading]]
            键为时间戳，值为该时刻的所有传感器读数列表

    返回:
        timestamps: (N,) 融合后的时间戳
        fused_data: (N, D) 融合后的数据
        fused_covs: list[(D, D)] 融合后的协方差
    """
    timestamps = sorted(sensor_readings_by_time.keys())
    fused_data_list = []
    fused_cov_list = []

    for t in timestamps:
        readings = sensor_readings_by_time[t]
        data, cov = weighted_fusion(readings)
        fused_data_list.append(data)
        fused_cov_list.append(cov)

    return (
        np.array(timestamps),
        np.array(fused_data_list),
        fused_cov_list,
    )


def time_align(readings_list, tolerance=0.05):
    """将多个传感器的读数按时间戳对齐。

    不同传感器采样率不同（GPS 1Hz, LiDAR 10Hz, ...），
    需要按时间戳匹配后才能融合。

    参数:
        readings_list: list[list[SensorReading]]
            每个元素是一个传感器的完整读数序列
        tolerance: 时间对齐容差 [秒]

    返回:
        dict[float, list[SensorReading]]
            键为对齐后的时间戳，值为该时刻匹配到的传感器读数
    """
    aligned = {}

    # 以最高频率传感器的时间戳为基准
    all_times = set()
    for readings in readings_list:
        for r in readings:
            all_times.add(r.timestamp)

    # 对每个时间戳，找各传感器最近的读数
    for t in sorted(all_times):
        matched = []
        for readings in readings_list:
            # 找最近的读数
            best = None
            best_dt = float("inf")
            for r in readings:
                dt = abs(r.timestamp - t)
                if dt < best_dt:
                    best_dt = dt
                    best = r
            if best is not None and best_dt <= tolerance:
                matched.append(best)

        if len(matched) >= 2:  # 至少两个传感器匹配到才融合
            aligned[t] = matched

    return aligned
