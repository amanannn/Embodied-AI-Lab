"""传感器基类与统一数据格式。

所有传感器仿真模块继承 SensorBase，输出 SensorReading。
统一格式方便后续融合、滤波和可视化。
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SensorReading:
    """一次传感器读数的统一格式。

    属性:
        timestamp:  读数时刻 [秒]
        sensor_id:  传感器标识 ("gps", "lidar", "imu", "odom")
        data:       传感器数据向量（维度取决于传感器类型）
        covariance: 该时刻的噪声协方差矩阵
    """
    timestamp: float
    sensor_id: str
    data: np.ndarray
    covariance: np.ndarray


class SensorBase:
    """传感器仿真基类。

    所有传感器共享这套接口：
    - read()  — 从真实状态生成一次带噪声的观测
    - generate_sequence() — 对整条轨迹生成观测序列

    子类只需实现 read() 方法。

    参数:
        name:       传感器名称
        sample_rate: 采样率 [Hz]
        noise_std:  主噪声标准差
        bias:       常值偏置
    """

    def __init__(self, name, sample_rate, noise_std, bias=0.0):
        self.name = name
        self.dt = 1.0 / sample_rate
        self.noise_std = noise_std
        self.bias = bias

    def read(self, true_state, timestamp):
        """从真实状态生成一次带噪声的观测。

        子类必须实现此方法。

        参数:
            true_state: 当前时刻的真实状态
            timestamp:  当前时刻 [秒]

        返回:
            SensorReading
        """
        raise NotImplementedError

    def generate_sequence(self, true_states, timestamps):
        """对整条轨迹生成观测序列。

        按照传感器的采样率对轨迹进行降采样，
        然后在每个采样时刻调用 read()。

        参数:
            true_states: (N, D) 真实状态序列
            timestamps:  (N,) 完整时间戳数组 [秒]

        返回:
            list[SensorReading] — 按采样率降采样后的观测序列
        """
        readings = []
        next_sample_time = timestamps[0]
        idx = 0

        for i, t in enumerate(timestamps):
            if t >= next_sample_time - 1e-9:
                reading = self.read(true_states[i], t)
                readings.append(reading)
                next_sample_time += self.dt

        return readings
