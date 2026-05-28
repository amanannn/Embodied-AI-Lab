"""粒子滤波器（PF）—— 2D 匀速运动 + 雷达观测。

状态向量:  [px, py, vx, vy]ᵀ  (4×1)
观测向量:  [range, bearing]ᵀ  (2×1)

KF/EKF/UKF 都假设状态服从高斯分布（一个"钟形曲线"），
但现实中很多情况不是高斯分布 —— 比如机器人在走廊里，
"可能在左边房间也可能在右边房间"就是双模态分布。

粒子滤波完全不假设分布形状。它用 N 个加权粒子来近似任意分布。
粒子越多，近似越精确。

算法（Bootstrap PF / SIR）：
    预测：  每个粒子按运动模型随机走一步（撒粒子）
    加权：  根据传感器读数计算每个粒子的"似然"（谁更靠谱？）
    重采样：淘汰低权重粒子，复制高权重粒子（优胜劣汰）
"""

import numpy as np


class ParticleFilter:
    """Bootstrap 粒子滤波器。

    用 N 个加权粒子近似状态的后验分布。
    不假设高斯，可以处理任意分布。
    """

    def __init__(self, dt, q_diag, R, n_particles=500,
                 initial_x=None, initial_P=None):
        self.dt = dt
        self.n_particles = n_particles

        # 状态转移矩阵 F — 匀速运动模型
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        self.Q = np.diag(q_diag)
        self.R = np.array(R)

        # 初始化粒子：从初始分布中采样
        x0 = np.zeros(4) if initial_x is None else np.array(initial_x).flatten()
        P0 = np.eye(4) * 0.5 if initial_P is None else np.array(initial_P)
        self.particles = np.random.default_rng(42).multivariate_normal(
            x0, P0, size=n_particles)

        # 初始权重均匀分配
        self.weights = np.ones(n_particles) / n_particles

        self.estimates = []
        self.covariances = []
        self.particles_history = []  # 记录粒子云（可视化用）

    def _h(self, x):
        """非线性观测函数（向量化版本）。

        输入 x: (N, 4) 或 (4,)
        输出:   (N, 2) 或 (2,) — [range, bearing]
        """
        px = x[..., 0]
        py = x[..., 1]
        r = np.sqrt(px**2 + py**2)
        r = np.maximum(r, 1e-6)
        bearing = np.arctan2(py, px)
        return np.column_stack([r, bearing]) if x.ndim > 1 else np.array([r, bearing])

    def predict(self):
        """粒子传播：每个粒子按运动模型 F 走一步，再加随机过程噪声。

        噪声是粒子滤波器保持多样性的关键 ——
        如果所有粒子都按 F 精确走，它们会迅速聚成一团，
        万一真实物体拐弯了，滤波器就跟丢了。
        """
        rng = np.random.default_rng()
        # 向量化传播：所有粒子同时过 F
        self.particles = self.particles @ self.F.T
        # 每个粒子加独立的过程噪声
        noise = rng.multivariate_normal(np.zeros(4), self.Q,
                                        size=self.n_particles)
        self.particles += noise

    def update(self, z):
        """权重更新：用传感器读数评价每个粒子的"好坏"。

        高斯似然：
            w_i = exp(-0.5 * (z - z_pred_i)ᵀ · R⁻¹ · (z - z_pred_i))

        粒子预测的观测越接近真实观测 → 权重越大。
        在对数域计算以防浮点下溢。
        """
        z = np.array(z).flatten()
        z_pred = self._h(self.particles)  # (n_particles, 2)

        # 残差（注意方位角环绕）
        dz = z - z_pred
        dz[:, 1] = (dz[:, 1] + np.pi) % (2 * np.pi) - np.pi

        # 马氏距离平方 → 对数似然
        R_inv = np.linalg.inv(self.R)
        mahal_sq = np.sum((dz @ R_inv) * dz, axis=1)  # 向量化马氏距离

        log_weights = -0.5 * mahal_sq
        log_weights -= np.max(log_weights)  # 数值稳定（防止 exp 溢出）
        weights = np.exp(log_weights)
        weights += 1e-300  # 防止全零
        self.weights = weights / np.sum(weights)

    def _systematic_resample(self):
        """系统重采样：淘汰差粒子，复制好粒子。

        想象一条线段，每段长度 = 粒子的权重。
        在 [0, 1/N] 随机选一个起点，然后每隔 1/N 取一个点，
        点落在哪段就复制哪个粒子。

        权重大的粒子占的段长 → 被多次复制。
        权重小的粒子占的段短 → 可能被淘汰。
        """
        n = self.n_particles
        # 均匀间隔，随机起点
        positions = (np.arange(n) + np.random.random()) / n

        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0

        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices]
        self.weights = np.ones(n) / n

    def _estimate(self):
        """从粒子集合估计当前状态（加权平均）和协方差。"""
        x_mean = np.average(self.particles, weights=self.weights, axis=0)
        diff = self.particles - x_mean
        cov = (self.weights[:, None] * diff).T @ diff
        return x_mean.reshape(4, 1), cov

    def step(self, z):
        """一次完整的 PF 步骤：预测 → 加权 → 估计 → 重采样。"""
        self.predict()
        self.update(z)

        x_est, P_est = self._estimate()
        self.estimates.append(x_est[:2, 0].copy())
        self.covariances.append(P_est.copy())

        # 记录粒子云（每隔几步记录一次，节省内存）
        if len(self.estimates) % 3 == 0:
            self.particles_history.append(self.particles.copy())

        self._systematic_resample()
