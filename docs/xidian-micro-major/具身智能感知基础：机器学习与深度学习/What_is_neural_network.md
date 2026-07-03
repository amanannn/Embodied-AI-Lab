# 什么是神经网络

> 西安电子科技大学 · 具身智能微专业
> 课程笔记

---

## 一、神经网络的三个核心问题

构建一个神经网络，本质上回答三个问题：

| 问题 | 对应组件 | 例子 |
|------|---------|------|
| 1. 选什么结构？ | **网络架构** | 几层？每层多少神经元？用什么激活函数？ |
| 2. 怎么衡量好坏？ | **损失函数** | 预测值和真实值差多少？ |
| 3. 怎么变好？ | **优化算法** | 如何调整参数让损失下降？ |

> 这三个问题的答案，构成了从简单感知机到 GPT 的所有神经网络的共同骨架。

---

## 二、前馈神经网络

### 2.1 基本概念

前馈神经网络（Feedforward Neural Network），也叫：

- **全连接神经网络**（Fully Connected Network）
- **多层感知器**（Multi-Layer Perceptron, MLP）

结构：输入层 → 隐藏层（可多层） → 输出层，信息单向流动，没有环路。

### 2.2 单层计算

一个全连接层的数学表达：

$$h = f(Wx + b)$$

| 符号 | 含义 | 形状（示例） |
|------|------|------------|
| $x$ | 输入向量 | $(d_{in}, 1)$ |
| $W$ | 权重矩阵 | $(d_{out}, d_{in})$ |
| $b$ | 偏置向量 | $(d_{out}, 1)$ |
| $f$ | 激活函数 | — |
| $h$ | 输出 | $(d_{out}, 1)$ |

```python
# 单层的完整实现（包括初始化）
layer = nn.Linear(in_features=784, out_features=256)
x = torch.randn(32, 784)       # (batch, input_dim)
h = torch.relu(layer(x))       # (batch, 256)
```

### 2.3 多层堆叠

$$h^{(1)} = f(W^{(1)}x + b^{(1)})$$
$$h^{(2)} = f(W^{(2)}h^{(1)} + b^{(2)})$$
$$\hat{y} = W^{(3)}h^{(2)} + b^{(3)}$$

> 没有激活函数的多层等价于单层——因为线性变换的复合仍是线性变换。

---

## 三、损失函数

### 3.1 为什么要定义损失函数？

损失函数将模型表现量化为一个标量。模型"有多差" = 损失有多高。训练的目标：**最小化这个值**。

### 3.2 回归任务损失

| 损失函数 | 公式 | 梯度 | 特点 |
|---------|------|------|------|
| **MSE** | $L = \frac{1}{N}\sum(y - \hat{y})^2$ | $\frac{\partial L}{\partial \hat{y}} = \frac{2}{N}(\hat{y} - y)$ | 大误差惩罚重，对小误差不敏感 |
| **MAE** | $L = \frac{1}{N}\sum|y - \hat{y}|$ | $\frac{\partial L}{\partial \hat{y}} = \pm \frac{1}{N}$ | 梯度恒定，对异常值鲁棒 |
| **Huber** | 小误差用 MSE，大误差用 MAE | — | 结合两者优点 |

**MSE vs MAE 对比**：

| 维度 | MSE | MAE |
|------|-----|-----|
| 异常值敏感度 | 高（平方惩罚） | 低（线性惩罚） |
| 最优解 | 均值 | 中位数 |
| 梯度 | 随误差增大而增大 | 恒定为 ±1 |
| 适用场景 | 数据干净 | 数据有异常值 |

### 3.3 分类任务损失

| 损失函数 | 公式 | 适用场景 |
|---------|------|---------|
| **BCE** | $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | 二分类 |
| **CE** | $L = -\sum_{c} y_c \log \hat{y}_c$ | 多分类（配合 Softmax） |

**为什么分类用交叉熵而不是 MSE？**

交叉熵配合 Softmax 的梯度为 $\hat{y} - y$，形式简洁——预测越错梯度越大，预测越对梯度越小。MSE 配合 Sigmoid/Softmax 容易出现梯度饱和（预测接近 0 或 1 时梯度趋于 0）。

---

## 四、激活函数

### 4.1 为什么需要激活函数？

$$h = W_2 \cdot f(W_1 x + b_1) + b_2$$

如果 $f(x) = x$（恒等映射），则：

$$h = W_2 W_1 x + W_2 b_1 + b_2 = W' x + b'$$

多层退化为单层。激活函数引入**非线性**，让网络能拟合任意复杂函数。

### 4.2 常见激活函数详析

| 激活函数 | 公式 | 值域 | 优点 | 缺点 |
|---------|------|------|------|------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $(0, 1)$ | 平滑，可解释为概率 | 饱和区梯度消失；输出非零中心 |
| **Tanh** | $\tanh(x)$ | $(-1, 1)$ | 零中心，梯度比 Sigmoid 大 | 仍有饱和区梯度消失 |
| **ReLU** | $f(x) = \max(0, x)$ | $[0, +\infty)$ | 计算简单，缓解梯度消失 | Dead ReLU（负半轴梯度为 0） |
| **Leaky ReLU** | $f(x) = \max(0.01x, x)$ | $(-\infty, +\infty)$ | 解决 Dead ReLU | 负半轴斜率需手动设定 |
| **GELU** | $x \cdot \Phi(x)$ | $(-\infty, +\infty)$ | Transformer 标配，平滑非线性 | 计算略复杂 |
| **Swish** | $x \cdot \sigma(x)$ | $(-\infty, +\infty)$ | 自门控，EfficientNet 使用 | 计算开销 |

#### Dead ReLU 问题

```python
# 当一个神经元对所有输入都输出 0 时，梯度永远为 0，该神经元"死亡"
x = torch.randn(1000, 1) * 0.1 - 5   # 输入集中在负半轴
relu = nn.ReLU()
print(f'非零输出比例: {(relu(x) > 0).float().mean():.1%}')  # 可能接近 0%
```

**解决**：Leaky ReLU / PReLU（负半轴给一个小斜率）、合理初始化（He 初始化）。

### 4.3 激活函数选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| 隐藏层（CNN） | ReLU / Leaky ReLU | 简单高效 |
| 隐藏层（Transformer） | GELU | 平滑，与 Attention 配合好 |
| 二分类输出 | Sigmoid | 输出 (0,1) 可解释为概率 |
| 多分类输出 | Softmax | 输出和为 1 的分布 |
| 回归输出 | 无（恒等） | 输出可以是任意实数 |

---

## 五、前向传播与反向传播

### 5.1 训练全流程

```
随机初始化参数
    ↓
┌─ 前向传播：输入 → 各层计算 → 输出预测值
│       ↓
│   计算损失：用损失函数衡量预测与真实值的差距
│       ↓
│   反向传播：链式法则求导，逐层计算梯度
│       ↓
│   更新参数：w = w - η × ∂loss/∂w
│       ↓
└─ 回到前向传播（重复直到收敛）
```

### 5.2 反向传播的数学本质——链式法则

假设一个简单的二层网络：

$$z^{(1)} = W^{(1)}x$$
$$h = f(z^{(1)})$$
$$z^{(2)} = W^{(2)}h$$
$$\hat{y} = z^{(2)}$$
$$L = \frac{1}{2}(\hat{y} - y)^2$$

对 $W^{(2)}$ 求梯度：

$$\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial W^{(2)}}$$

$$= (\hat{y} - y) \cdot 1 \cdot h^T$$

对 $W^{(1)}$ 求梯度（跨过激活函数）：

$$\frac{\partial L}{\partial W^{(1)}} = (\hat{y} - y) \cdot W^{(2)T} \cdot f'(z^{(1)}) \cdot x^T$$

> **链式法则的本质**：复合函数的导数 = 各层局部导数的乘积。反向传播就是按"从输出到输入"的顺序逐层计算这些局部导数。

### 5.3 PyTorch 中的自动求导

```python
x = torch.tensor([1.0, 2.0], requires_grad=False)
W = torch.tensor([[0.5, -0.3], [0.1, 0.8]], requires_grad=True)

# 前向
z = W @ x                       # (2,)
loss = (z - torch.tensor([1.0, 0.0])).pow(2).sum()

# 反向——一行代码搞定全部梯度计算
loss.backward()
print(W.grad)                    # ∂loss/∂W 已自动计算
# tensor([[ 0.2000,  0.4000],
#         [-0.5200, -1.0400]])
```

---

## 六、梯度下降法

### 6.1 核心公式

$$w_{new} = w_{old} - \eta \cdot \nabla_w L$$

### 6.2 学习率的影响

| 学习率 η | 效果 | 表现 |
|----------|------|------|
| 太小 ($10^{-5}$) | 收敛极慢 | 训练几万步 loss 几乎不动 |
| 适中 ($10^{-3}$) | 稳定下降 | Loss 平滑递减，最终收敛 |
| 偏大 ($10^{-1}$) | 震荡 | Loss 忽高忽低，可能不收敛 |
| 太大 ($1$) | 发散 | Loss 爆炸，NaN |

### 6.3 梯度下降的三种变体

| 变体 | 每步使用的数据量 | 梯度方向 | 计算量 | 跳出局部极小 |
|------|----------------|---------|--------|------------|
| **BGD** | 全部数据 | 最准确 | 最大 | 困难 |
| **SGD** | 1 个样本 | 噪声最大 | 最小 | 容易 |
| **MBGD** | 1 个 batch（32~256） | 有一定噪声 | 适中 | 较容易 |

```python
# MBGD 的 PyTorch 实现
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# DataLoader 自动将数据切分为 batch，每个 batch 就是一次 MBGD 更新
```

> 实践中最常用的是 **MBGD（Mini-Batch Gradient Descent）**——平衡了计算效率和梯度准确性。

---

## 七、参数初始化

### 7.1 为什么初始化重要？

不好的初始化会导致：
- **梯度消失**：权重太小 → 信号逐层衰减 → 深层几乎无梯度
- **梯度爆炸**：权重太大 → 信号逐层放大 → 梯度溢出（NaN）

### 7.2 常用初始化方法

| 方法 | 公式 | 适用激活函数 |
|------|------|------------|
| **Xavier** | $W \sim U[-\frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}]$ | Sigmoid, Tanh |
| **He** | $W \sim N(0, \sqrt{\frac{2}{n_{in}}})$ | ReLU 及变体 |

```python
# PyTorch 默认使用 He 初始化（对 ReLU 友好）
conv = nn.Conv2d(3, 64, 3)
# 也可手动指定
nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
```

> **直觉**：He 初始化让 ReLU 激活后的方差保持稳定——$n_{in}$ 越大，每个权重越小，保证输入的方差不随层数爆炸或衰减。

---

## 八、正则化：防止过拟合

| 方法 | 原理 | PyTorch |
|------|------|---------|
| **L1 正则** | 惩罚权重绝对值和 → 稀疏权重 | `weight_decay` (需手动) |
| **L2 正则** | 惩罚权重平方和 → 权重衰减 | `optimizer(weight_decay=1e-4)` |
| **Dropout** | 随机丢弃神经元，强制冗余学习 | `nn.Dropout(p=0.5)` |
| **BatchNorm** | 层间归一化，稳定训练 | `nn.BatchNorm2d(64)` |
| **早停** | 验证集 loss 不下降时停止 | 手动实现 |
| **数据增强** | 扩充训练数据多样性 | `torchvision.transforms` |

### L1 vs L2 正则化

| 正则化 | 惩罚项 | 效果 |
|--------|--------|------|
| L1 | $\lambda\sum\|w_i\|$ | 产生稀疏权重（很多权重为 0），特征选择 |
| L2 | $\lambda\sum w_i^2$ | 权重整体缩小但不为零，数值稳定 |

$$L_{total} = L_{original} + \lambda \cdot R(W)$$

---

## 九、总结：从零到一的训练清单

- [ ] 定义网络结构（几层、多大、什么激活函数）
- [ ] 选择权重初始化方法（ReLU→He，Sigmoid→Xavier）
- [ ] 选择损失函数（回归→MSE，分类→CrossEntropy）
- [ ] 选择优化器（Adam 入门首选）
- [ ] 设置学习率（$10^{-3}$ 起步）
- [ ] 训练循环：forward → loss → zero_grad → backward → step
- [ ] 监控 Train Loss / Val Loss，判断过拟合/欠拟合
- [ ] 调参：学习率、batch_size、网络深度/宽度、正则化

---

*相关笔记：[什么是卷积网络](./What_is_Convolution_network.md)、[从传统图像处理到深度学习](./From_traditional_image_processing_to_deep_learning.md)、[常用卷积算子](./Commonly_used_convolution_operators.md)、[如何构建和训练神经网络（上）](../具身智能认知进阶：多模态大模型与语义理解/How_to_Build_and_Train_a_Neural_Network_p1.md)*
