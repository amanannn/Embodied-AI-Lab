"""无迹卡尔曼滤波器（UKF）—— 2D 匀速运动 + 雷达观测。

状态向量:  [px, py, vx, vy]ᵀ  (4×1)
观测向量:  [range, bearing]ᵀ  (2×1)

EKF 对非线性函数做一阶泰勒近似（"在一点处画切线"），
当非线性很强时，这个近似可能很差。

UKF 用了另一种思路：不近似函数，而是近似分布。
——"与其把曲线拉直，不如撒几个点（sigma 点）去探路，
   让它们各自经过非线性函数，回来再汇总。"

这样做的好处：
    - 不需要求导（没有雅可比矩阵）
    - 精度比 EKF 高一阶（二阶 vs 一阶）
    - 对强非线性更鲁棒
"""

import numpy as np


class UnscentedKalmanFilter:
    """无迹卡尔曼滤波器。

    通过 2n+1 个确定性采样点（sigma 点）来传递概率分布。
    n=4 个状态需要 9 个 sigma 点。
    """

    def __init__(self, dt, q_diag, R,
                 alpha=1.0, beta=2.0, kappa=0.0,
                 initial_x=None, initial_P=None):
        self.dt = dt
        self.n = 4  # 状态维度

        # 状态转移矩阵（线性，与 KF 相同）
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

        # ── UKF 参数 ──
        # α: sigma 点扩散宽度（越大散得越开）
        # β: 利用高斯先验知识（β=2 对高斯分布最优）
        # κ: 次级缩放参数（通常设为 0）
        n = self.n
        self.lam = alpha**2 * (n + kappa) - n  # λ 参数

        # 预计算权重（2n+1 个 sigma 点各有权重）
        self.W_m = np.full(2 * n + 1, 0.5 / (n + self.lam))  # 均值权重
        self.W_c = np.full(2 * n + 1, 0.5 / (n + self.lam))  # 协方差权重
        self.W_m[0] = self.lam / (n + self.lam)
        self.W_c[0] = self.lam / (n + self.lam) + (1 - alpha**2 + beta)

        self.estimates = []
        self.covariances = []

    # ── 观测函数（与 EKF 相同）──
    def _h(self, x):
        """非线性观测函数：状态 → [range, bearing]"""
        x_arr = np.array(x).flatten()
        px, py = x_arr[0], x_arr[1]
        r = max(np.sqrt(px**2 + py**2), 1e-6)
        bearing = np.arctan2(py, px)
        return np.array([[r], [bearing]])

    # ── Sigma 点生成 ──
    def _sigma_points(self, x, P):
        """从均值 x 和协方差 P 生成 2n+1 个 sigma 点。

        sigma[0] = x（中心点）
        sigma[1..n]   = x + sqrt((n+λ)P) 的第 i 列
        sigma[n+1..2n] = x - sqrt((n+λ)P) 的第 i 列
        """
        n = self.n
        x_flat = np.array(x).flatten()

        # 矩阵平方根：用 Cholesky 分解
        # 如果 P 非正定（数值误差导致），用特征值分解回退
        try:
            L = np.linalg.cholesky((n + self.lam) * P)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-6)
            L = eigvecs @ np.diag(np.sqrt((n + self.lam) * eigvals))

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x_flat
        for i in range(n):
            sigmas[i + 1] = x_flat + L[:, i]
            sigmas[n + i + 1] = x_flat - L[:, i]
        return sigmas

    # ── 无迹变换 ──
    def _ut(self, sigmas, noise_cov=None):
        """对 sigma 点集做加权平均，得到变换后的均值和协方差。

        这就是 UKF 的核心："撒点 → 变换 → 汇总"
        """
        W_m = self.W_m.reshape(-1, 1)
        W_c = self.W_c.reshape(-1, 1)

        mean = np.sum(W_m * sigmas, axis=0).reshape(-1, 1)
        diff = sigmas - mean.flatten()
        cov = (diff.T @ (self.W_c[:, None] * diff))

        if noise_cov is not None:
            cov = cov + noise_cov

        return mean, cov

    # ── 预测 ──
    def predict(self):
        """通过状态转移函数传递 sigma 点。

        1. 生成 sigma 点
        2. 每个点过一遍 F（线性运动模型）
        3. 加权汇总得预测均值和协方差（加 Q）
        """
        sigmas = self._sigma_points(self.x, self.P)
        sigmas_pred = (self.F @ sigmas.T).T
        self.x, self.P = self._ut(sigmas_pred, self.Q)

    # ── 更新 ──
    def update(self, z):
        """通过观测函数传递 sigma 点。

        1. 从预测状态重新生成 sigma 点
        2. 每个点过一遍 h(x)（非线性观测函数）
        3. 汇总得预测观测的均值和协方差
        4. 算交叉协方差 → 卡尔曼增益 → 修正状态
        """
        z = np.array(z).reshape(2, 1)
        n = self.n

        # 生成新 sigma 点
        sigmas = self._sigma_points(self.x, self.P)

        # 每个 sigma 点经过观测函数
        sigmas_h = np.zeros((2 * n + 1, 2))
        for i in range(2 * n + 1):
            h_out = self._h(sigmas[i].reshape(-1, 1))
            sigmas_h[i] = h_out.flatten()

        # 预测观测的均值和协方差
        z_pred, P_zz = self._ut(sigmas_h, self.R)

        # 交叉协方差 P_xz
        diff_x = sigmas - self.x.flatten()
        diff_z = sigmas_h - z_pred.flatten()
        P_xz = diff_x.T @ (self.W_c[:, None] * diff_z)

        # 卡尔曼增益
        K = P_xz @ np.linalg.inv(P_zz)

        # 创新（注意角度环绕）
        y = z - z_pred
        y[1, 0] = (y[1, 0] + np.pi) % (2 * np.pi) - np.pi

        # 更新
        self.x = self.x + K @ y
        self.P = self.P - K @ P_zz @ K.T  # Joseph 形式，更稳定

    def step(self, z):
        """一次完整的预测→更新循环。"""
        self.predict()
        self.update(z)
        self.estimates.append(self.x[:2, 0].copy())
        self.covariances.append(self.P.copy())
