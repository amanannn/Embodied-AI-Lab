"""扩展卡尔曼滤波器（EKF）—— 2D 匀速运动 + 雷达观测。

状态向量:  [px, py, vx, vy]ᵀ  (4×1)
观测向量:  [range, bearing]ᵀ  (2×1)

雷达（或激光雷达）测量的是距离和方位角，而非直接的 x,y 坐标：
    range  = sqrt(px² + py²)
    bearing = atan2(py, px)

这个观测函数是非线性的，标准 KF 无法直接使用。
EKF 的做法：在当前估计点对非线性函数做一阶泰勒展开（求雅可比矩阵），
用局部线性化后的函数替代原函数。

关键区别 vs 线性 KF：
    - 预测步骤完全相同（运动模型是线性的）
    - 更新步骤中，H 不再是常数，而是每步重新计算的雅可比矩阵 H_j
"""

import numpy as np


class ExtendedKalmanFilter:
    """扩展卡尔曼滤波器。

    用雅可比矩阵对非线性观测函数做局部线性化。
    运动模型是线性的，因此预测步骤与标准 KF 一致。
    """

    def __init__(self, dt, q_diag, R, initial_x=None, initial_P=None):
        self.dt = dt

        # 状态转移矩阵 F — 匀速运动模型（与线性 KF 相同）
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        self.Q = np.diag(q_diag)
        self.R = np.array(R)

        self.x = (np.zeros((4, 1)) if initial_x is None
                  else np.array(initial_x).reshape(4, 1))
        self.P = (np.eye(4) * 0.5 if initial_P is None
                  else np.array(initial_P))

        self.estimates = []
        self.covariances = []

    def _h(self, x):
        """非线性观测函数：状态 → 预测观测。

        h([px, py, vx, vy]) = [sqrt(px²+py²), atan2(py, px)]

        这就是雷达实际会读到的值。
        """
        px, py = x[0, 0], x[1, 0]
        r = np.sqrt(px**2 + py**2)
        r = max(r, 1e-6)   # 防止除以零
        bearing = np.arctan2(py, px)
        return np.array([[r], [bearing]])

    def _H_jacobian(self, x):
        """观测函数的雅可比矩阵。

        在 x 点处对 h(x) 求偏导：
            ∂range/∂px   = px / r     ∂range/∂py   = py / r
            ∂bearing/∂px = -py / r²   ∂bearing/∂py = px / r²
            （对速度的偏导全是 0）

        这个雅可比就是"在当前点画的切线"。
        """
        px, py = x[0, 0], x[1, 0]
        r = max(np.sqrt(px**2 + py**2), 1e-6)
        r2 = r**2

        H_j = np.zeros((2, 4))
        H_j[0, 0] = px / r      # ∂range/∂px
        H_j[0, 1] = py / r      # ∂range/∂py
        H_j[1, 0] = -py / r2    # ∂bearing/∂px
        H_j[1, 1] = px / r2     # ∂bearing/∂py
        return H_j

    def predict(self):
        """与线性 KF 完全相同的预测步骤。"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """EKF 更新步骤。

        与线性 KF 的区别：
            1. 用 h(x) 而非 H@x 计算预测观测
            2. H 是每步新算的雅可比，不是常数矩阵
            3. 方位角残差需要归一化到 [-π, π]
        """
        z = np.array(z).reshape(2, 1)

        # 预测观测（非线性函数）
        z_pred = self._h(self.x)

        # 创新（残差）—— 注意角度环绕！
        y = z - z_pred
        y[1, 0] = (y[1, 0] + np.pi) % (2 * np.pi) - np.pi

        # 当前点的雅可比矩阵
        H_j = self._H_jacobian(self.x)

        # 标准 KF 更新公式，但 H 换成了雅可比 H_j
        S = H_j @ self.P @ H_j.T + self.R
        K = self.P @ H_j.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H_j) @ self.P

    def step(self, z):
        """一次完整的预测→更新循环。"""
        self.predict()
        self.update(z)
        self.estimates.append(self.x[:2, 0].copy())
        self.covariances.append(self.P.copy())
