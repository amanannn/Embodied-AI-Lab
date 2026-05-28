"""激光雷达传感器仿真。

模拟 LiDAR 输出 [range, bearing] 极坐标观测。
采样率 10 Hz，噪声包括高斯和离群值（多路径反射）。

数据格式:
    data:       [range, bearing]  距离 [米] + 方位角 [弧度]
    covariance: 2×2 对角矩阵
"""

import numpy as np
from .base import SensorBase, SensorReading


class LiDARSensor(SensorBase):
    """激光雷达 range-bearing 传感器。

    噪声特征:
        - 距离高斯噪声 (std=0.3m)
        - 方位角高斯噪声 (std=0.05rad ≈ 3°)
        - 离群值 (prob=3%, range magnitude=2m)

    参数:
        range_std:    距离噪声标准差 [米]，默认 0.3
        bearing_std:  方位角噪声标准差 [弧度]，默认 0.05
        outlier_prob: 离群值概率，默认 0.03
        outlier_mag:  离群值距离幅度 [米]，默认 2.0
        seed:         随机种子
    """

    def __init__(self, range_std=0.3, bearing_std=0.05,
                 outlier_prob=0.03, outlier_mag=2.0, seed=42):
        super().__init__("lidar", sample_rate=10, noise_std=range_std)
        self.bearing_std = bearing_std
        self.outlier_prob = outlier_prob
        self.outlier_mag = outlier_mag
        self.rng = np.random.default_rng(seed)

    def read(self, true_state, timestamp):
        """从真实位置生成 LiDAR 观测。

        观测模型是非线性的：
            range   = sqrt(x² + y²) + noise
            bearing = atan2(y, x) + noise

        参数:
            true_state: [x, y] 真实位置（相对于传感器本体）
            timestamp:  当前时刻 [秒]

        返回:
            SensorReading (data 为 [range, bearing])
        """
        pos = np.array(true_state[:2], dtype=float)

        # 真值转极坐标
        r_true = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
        b_true = np.arctan2(pos[1], pos[0])

        # 高斯噪声
        r_noisy = r_true + self.rng.normal(0, self.noise_std)
        b_noisy = b_true + self.rng.normal(0, self.bearing_std)

        # 离群值
        if self.rng.random() < self.outlier_prob:
            r_noisy += self.rng.uniform(-self.outlier_mag, self.outlier_mag)

        data = np.array([r_noisy, b_noisy])
        cov = np.diag([self.noise_std ** 2, self.bearing_std ** 2])

        return SensorReading(
            timestamp=timestamp,
            sensor_id="lidar",
            data=data,
            covariance=cov,
        )

    @staticmethod
    def to_xy(reading):
        """将 range-bearing 观测转为笛卡尔坐标 [x, y]。

        用于可视化和融合。
        """
        r, b = reading.data
        return np.array([r * np.cos(b), r * np.sin(b)])
