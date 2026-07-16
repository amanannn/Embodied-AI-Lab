# Findings: Flow Matching MNIST 项目

## 技术调研

### Flow Matching 原理
- 目标: 学习向量场 v_t(x) 将噪声分布 p_1 (高斯) 变换到数据分布 p_0 (MNIST)
- 路径: x_t = (1-t) * x_data + t * x_noise (线性插值)
- 损失: MSE(v_pred, x_noise - x_data) 即预测速度场
- 采样: dx/dt = v_t(x), 从 t=1 到 t=0 积分 (Euler/RK4)

### V100 优化策略
- FP16 混合精度: V100 Tensor Core 提供 ~2x 加速
- Batch Size 512: 充分利用 32GB 显存
- 28×28 原生尺寸: 小图在 V100 上计算量极低
- channels [64,128,256]: 参数量适中，收敛快

### 条件注入方式
- Time embedding: 正弦位置编码 + MLP
- Class embedding: Embedding lookup + 加到 time embedding
- 注入位置: U-Net 每个 ResBlock 通过 FiLM/scale-shift
