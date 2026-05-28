"""传感器仿真模块。

提供四种具身智能常用传感器的仿真：
    GPS     — 全局位置传感器 (1 Hz)
    LiDAR   — 激光雷达 range-bearing (10 Hz)
    IMU     — 惯性测量单元 (100 Hz)
    Odometry — 轮式里程计 (20 Hz)

所有传感器共享 SensorBase 基类和 SensorReading 数据格式。
"""

from .base import SensorBase, SensorReading
from .gps import GPSSensor
from .lidar import LiDARSensor
from .imu import IMUSensor
from .odometry import OdometrySensor

__all__ = [
    "SensorBase", "SensorReading",
    "GPSSensor", "LiDARSensor", "IMUSensor", "OdometrySensor",
]
