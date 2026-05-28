"""轮式里程计传感器仿真。

模拟编码器输出 [v, omega]（线速度 + 角速度）。
采样率 20 Hz，噪声以高斯为主，伴随累积漂移。

数据格式:
    data:       [v, omega]  线速度 [m/s] + 角速度 [rad/s]
    covariance: 2×2 对角矩阵

注意: 里程计的漂移是累积的——走得越远，误差越大。
     这是因为里程计靠轮子转数推算距离，轮胎打滑、
     地面不平等因素会导致误差不断累积。
"""

import numpy as np
from .base import SensorBase, SensorReading


class OdometrySensor(SensorBase):
    """轮式里程计传感器。

    噪声特征:
        - 高斯噪声 (速度 std=0.05 m/s, 角速度 std=0.02 rad/s)
        - 累积漂移: 距离越远误差越大

    参数:
        vel_noise_std:   线速度噪声标准差，默认 0.05
        omega_noise_std: 角速度噪声标准差，默认 0.02
        drift_rate:      漂移率（每米距离增加的漂移标准差），默认 0.01
        seed:            随机种子

    TODO: Student Exercise
        实现累积漂移逻辑：根据里程计累计行驶距离，
        按 drift_rate 成比例地增加漂移。
        提示: 在 generate_sequence() 中追踪累计距离，
        或在 read() 中维护内部状态。
    """

    def __init__(self, vel_noise_std=0.05, omega_noise_std=0.02,
                 drift_rate=0.01, seed=42):
        super().__init__("odom", sample_rate=20, noise_std=vel_noise_std)
        self.omega_noise_std = omega_noise_std
        self.drift_rate = drift_rate
        self.rng = np.random.default_rng(seed)

        # 累积漂移状态
        self._accumulated_distance = 0.0
        self._drift_v = 0.0
        self._drift_omega = 0.0
        self._last_pos = None

    def read(self, true_state, timestamp):
        """从真实状态生成里程计观测。

        里程计测量的是本体坐标系下的速度。
        这里从全局状态简化提取。

        参数:
            true_state: [x, y] 或 [x, y, vx, vy] 真实状态
            timestamp:  当前时刻 [秒]

        返回:
            SensorReading (data 为 [v, omega])
        """
        pos = np.array(true_state[:2], dtype=float)

        # 从位置差分估算速度
        if self._last_pos is not None:
            dp = pos - self._last_pos
            v_true = np.linalg.norm(dp) / self.dt
            omega_true = np.arctan2(dp[1], dp[0])
        else:
            v_true = 0.0
            omega_true = 0.0

        self._last_pos = pos.copy()

        # 累积漂移：距离越远漂移越大
        self._accumulated_distance += v_true * self.dt
        drift_magnitude = self.drift_rate * self._accumulated_distance
        self._drift_v += self.rng.normal(0, drift_magnitude * 0.1)
        self._drift_omega += self.rng.normal(0, drift_magnitude * 0.05)

        # 叠加噪声
        v = (v_true
             + self.rng.normal(0, self.noise_std)
             + self._drift_v)

        omega = (omega_true
                 + self.rng.normal(0, self.omega_noise_std)
                 + self._drift_omega)

        data = np.array([v, omega])
        cov = np.diag([self.noise_std ** 2, self.omega_noise_std ** 2])

        return SensorReading(
            timestamp=timestamp,
            sensor_id="odom",
            data=data,
            covariance=cov,
        )
