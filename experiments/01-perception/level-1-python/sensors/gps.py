"""GPS 位置传感器仿真。

模拟 GPS 接收器输出 [x, y] 全局坐标。
采样率 1 Hz，噪声以高斯为主，偶发离群值（多径效应）。

数据格式:
    data:       [x, y]  全局坐标 [米]
    covariance: 2×2 对角矩阵
"""

import numpy as np
from .base import SensorBase, SensorReading
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from noise import gaussian_noise, outlier_noise


class GPSSensor(SensorBase):
    """GPS 位置传感器。

    噪声特征:
        - 高斯白噪声 (std=0.5m) — 基本精度限制
        - 离群值 (prob=2%, magnitude=5m) — 多径效应

    参数:
        noise_std:       高斯噪声标准差 [米]，默认 0.5
        outlier_prob:    离群值概率，默认 0.02
        outlier_mag:     离群值最大幅度 [米]，默认 5.0
        seed:            随机种子
    """

    def __init__(self, noise_std=0.5, outlier_prob=0.02,
                 outlier_mag=5.0, seed=42):
        super().__init__("gps", sample_rate=1, noise_std=noise_std)
        self.outlier_prob = outlier_prob
        self.outlier_mag = outlier_mag
        self.rng = np.random.default_rng(seed)

    def read(self, true_state, timestamp):
        """从真实位置生成 GPS 观测。

        参数:
            true_state: [x, y] 真实位置
            timestamp:  当前时刻 [秒]

        返回:
            SensorReading
        """
        true_pos = np.array(true_state[:2], dtype=float)

        # 高斯噪声
        noise = self.rng.normal(0, self.noise_std, size=2)

        # 偶发离群值
        if self.rng.random() < self.outlier_prob:
            noise += self.rng.uniform(-self.outlier_mag, self.outlier_mag,
                                       size=2)

        data = true_pos + noise + self.bias
        cov = np.eye(2) * (self.noise_std ** 2)

        return SensorReading(
            timestamp=timestamp,
            sensor_id="gps",
            data=data,
            covariance=cov,
        )
