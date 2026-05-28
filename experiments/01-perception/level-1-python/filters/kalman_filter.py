"""线性卡尔曼滤波器 —— 2D 匀速运动模型。

状态向量:  [px, py, vx, vy]ᵀ  (4×1)
观测向量:  [px_obs, py_obs]ᵀ  (2×1)

传感器直接读取物体的 x,y 坐标（带高斯噪声），
观测模型是线性的，因此标准 KF 适用。
"""

import numpy as np


class KalmanFilter:
    """线性卡尔曼滤波器。

    每次收到一个新的传感器读数，执行一次"预测→更新"循环。
    五个卡尔曼公式分布在 predict() 和 update() 两个方法中。
    """

    def __init__(self, dt, q_diag, R, initial_x=None, initial_P=None):
        """
        参数:
            dt:     时间步长 [秒]
            q_diag: 过程噪声对角元素 [q_px, q_py, q_vx, q_vy]
            R:       测量噪声协方差矩阵 (2×2)
            initial_x: 初始状态 (4×1)，默认在原点、速度为零
            initial_P: 初始协方差 (4×4)，默认 0.5*I
        """
        self.dt = dt

        # 状态转移矩阵 F — 匀速运动模型
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        # 观测矩阵 H — 只提取位置，忽略速度
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # 过程噪声协方差 Q — 量化"运动模型有多不完美"
        self.Q = np.diag(q_diag)

        # 测量噪声协方差 R — 量化"传感器有多不准"
        self.R = np.array(R)

        # 初始状态
        self.x = (np.zeros((4, 1)) if initial_x is None
                  else np.array(initial_x).reshape(4, 1))
        self.P = (np.eye(4) * 0.5 if initial_P is None
                  else np.array(initial_P))

        # 记录历史（用于可视化）
        self.estimates = []
        self.covariances = []

    def predict(self):
        """公式 1, 2：用运动模型预测下一时刻的状态和协方差。

        公式 1: x = F · x        — 把状态推到下一时刻
        公式 2: P = F·P·Fᵀ + Q  — 不确定性传播 + 模型不完美性
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """公式 3, 4, 5：用传感器读数修正预测。

        公式 3: K = P·Hᵀ·(H·P·Hᵀ + R)⁻¹  — 卡尔曼增益
        公式 4: x = x + K·(z - H·x)       — 用观测修正状态
        公式 5: P = (I - K·H)·P           — 协方差收缩（更确定了）
        """
        z = np.array(z).reshape(2, 1)

        y = z - self.H @ self.x        # 创新（残差）
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def step(self, z):
        """一次完整的预测→更新循环。"""
        self.predict()
        self.update(z)
        self.estimates.append(self.x[:2, 0].copy())
        self.covariances.append(self.P.copy())
