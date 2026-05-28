"""IMU 惯性测量单元仿真。

模拟加速度计 + 陀螺仪输出 [ax, ay, omega]。
采样率 100 Hz，噪声包括高斯、零偏和随机漂移。

数据格式:
    data:       [ax, ay, omega]  加速度 [m/s²] + 角速度 [rad/s]
    covariance: 3×3 对角矩阵

注意: IMU 测量的是本体坐标系下的加速度和角速度，
     不是全局坐标。这里简化为 2D 平面模型。
"""

import numpy as np
from .base import SensorBase, SensorReading


class IMUSensor(SensorBase):
    """IMU 传感器（加速度计 + 陀螺仪）。

    噪声特征:
        - 加速度计: 高斯 (std=0.1 m/s²) + 零偏 + 漂移
        - 陀螺仪:   高斯 (std=0.01 rad/s) + 零偏 + 漂移

    参数:
        accel_noise_std: 加速度计高斯噪声标准差，默认 0.1
        gyro_noise_std:  陀螺仪高斯噪声标准差，默认 0.01
        accel_bias:      加速度计零偏，默认 0.05
        gyro_bias:       陀螺仪零偏，默认 0.01
        drift_std:       漂移标准差，默认 0.001
        warmup_time:     预热时间 [秒]，默认 1.0
        seed:            随机种子

    TODO: Student Exercise
        1. 修改 __init__ 参数，选择更真实的 IMU 噪声参数。
           查阅 MEMS IMU datasheet（如 MPU6050），找到典型噪声密度和零偏稳定性。
        2. 在 read() 中实现完整的噪声叠加逻辑：
           - 加速度计: true_accel + gaussian + bias + drift
           - 陀螺仪:   true_gyro + gaussian + bias + drift
           注意漂移是累积的，每步都要更新。
    """

    def __init__(self, accel_noise_std=0.1, gyro_noise_std=0.01,
                 accel_bias=0.05, gyro_bias=0.01,
                 drift_std=0.001, warmup_time=1.0, seed=42):
        super().__init__("imu", sample_rate=100,
                         noise_std=accel_noise_std)
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias = accel_bias
        self.gyro_bias = gyro_bias
        self.drift_std = drift_std
        self.warmup_time = warmup_time
        self.rng = np.random.default_rng(seed)

        # 漂移状态（累积量）
        self._drift_accel = np.zeros(2)  # [ax_drift, ay_drift]
        self._drift_gyro = 0.0           # omega_drift

    def read(self, true_state, timestamp):
        """从真实状态生成 IMU 观测。

        简化模型：true_state 是 [x, y, vx, vy]，
        我们从中提取加速度（通过差分近似）和角速度。

        实际实现中，IMU 测量的是本体坐标系下的物理量，
        这里简化为直接在全局坐标上加噪声。

        参数:
            true_state: [x, y] 或 [x, y, vx, vy] 真实状态
            timestamp:  当前时刻 [秒]

        返回:
            SensorReading (data 为 [ax, ay, omega])
        """
        # 简化：假设加速度和角速度为 0（匀速直线运动）
        # 真实场景中需要从状态差分计算
        true_accel = np.zeros(2)
        true_omega = 0.0

        # 更新漂移（随机游走）
        self._drift_accel += self.rng.normal(0, self.drift_std, size=2)
        self._drift_gyro += self.rng.normal(0, self.drift_std)

        # 判断是否过了预热期
        bias_factor = 1.0 if timestamp >= self.warmup_time else 0.0

        # 叠加噪声
        ax = (true_accel[0]
              + self.rng.normal(0, self.noise_std)
              + self.accel_bias * bias_factor
              + self._drift_accel[0])

        ay = (true_accel[1]
              + self.rng.normal(0, self.noise_std)
              + self.accel_bias * bias_factor
              + self._drift_accel[1])

        omega = (true_omega
                 + self.rng.normal(0, self.gyro_noise_std)
                 + self.gyro_bias * bias_factor
                 + self._drift_gyro)

        data = np.array([ax, ay, omega])
        cov = np.diag([self.noise_std ** 2,
                        self.noise_std ** 2,
                        self.gyro_noise_std ** 2])

        return SensorReading(
            timestamp=timestamp,
            sensor_id="imu",
            data=data,
            covariance=cov,
        )
