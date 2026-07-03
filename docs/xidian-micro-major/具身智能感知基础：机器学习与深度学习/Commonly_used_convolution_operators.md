# 常用卷积算子

> 西安电子科技大学 · 具身智能微专业
> 课程笔记

---

## 一、什么是卷积算子？

卷积算子是 CNN 中执行**特征提取和表示学习**的基本运算单元。

传统 $3 \times 3$ 卷积虽然通用，但存在三个痛点：

| 痛点 | 问题 | 催生的变体 |
|------|------|-----------|
| 参数量大 | $C_{in} \times C_{out} \times 9$，深层可达百万 | 分组卷积、深度可分离卷积 |
| 感受野固定 | 每层只能看到 $3 \times 3$ | 空洞卷积 |
| 形状固定 | 只适用于规则网格 | 可变形卷积 |

---

## 二、卷积变体全景

| 算子 | 核心作用 | 参数量（相对标准卷积） | 关键应用 |
|------|---------|---------------------|---------|
| 1×1 卷积 | 通道维度变换 | $1/9$ | 瓶颈结构、特征融合 |
| 池化 | 降采样 + 不变性 | 0 | 所有 CNN |
| 转置卷积 | 上采样 | 同标准卷积 | 语义分割、GAN |
| 分组卷积 | 降低参数量 | $1/g$ | ResNeXt |
| 深度可分离卷积 | 极致轻量化 | $\approx 1/C_{out}$ | MobileNet |
| 空洞卷积 | 扩大感受野 | 同标准卷积 | DeepLab |
| 可变形卷积 | 适应不规则形状 | ≈ 标准卷积 | 检测、分割 |

---

## 三、各算子详解

### 3.1 1×1 卷积

感受野为 $1 \times 1$，但不要小看它——它是 CNN 中的"瑞士军刀"。

#### 数学本质

对每个像素位置的 $C_{in}$ 维向量做线性变换：

$$y_{i,j} = W \cdot x_{i,j} + b$$

其中 $W \in \mathbb{R}^{C_{out} \times C_{in}}$，完全等价于对每个位置独立应用全连接层。

#### 核心用途

| 用途 | 输入通道 | 输出通道 | 效果 |
|------|---------|---------|------|
| 降维 | 512 | 256 | 减少后续层参数量 |
| 升维 | 256 | 512 | 增加特征容量 |
| 跨通道信息融合 | 任意 | 任意 | 不同通道特征交互 |

```python
# ResNet 瓶颈结构：1×1降维 → 3×3卷积 → 1×1升维
bottleneck = nn.Sequential(
    nn.Conv2d(256, 64, 1),   # 降维：256→64
    nn.Conv2d(64, 64, 3, padding=1),
    nn.Conv2d(64, 256, 1),   # 升维：64→256
)
# 参数量：256×64 + 64×64×9 + 64×256 = 69,632
# 直接3×3：256×256×9 = 589,824  (8.5倍)
```

### 3.2 池化层（Pooling）

#### 三种池化方式

| 池化方法 | 计算方式 | 输出值 | 特点 |
|---------|---------|--------|------|
| **最大池化**（Max） | $\max(x_{i:i+k, j:j+k})$ | 邻域最大值 | 保留纹理/边缘，对微小位移不变 |
| **平均池化**（Avg） | $\frac{1}{k^2}\sum x$ | 邻域均值 | 平滑，保留背景信息，全局信息 |
| **自适应池化**（Adaptive） | 自动计算 kernel/stride | — | 输出固定尺寸，不依赖输入 |

```python
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
adaptive = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化 → (B, C, 1, 1)
```

#### 输出尺寸公式

$$H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1$$

#### 池化的三大作用

| 作用 | 机制 | 重要性 |
|------|------|--------|
| **降维** | 减小空间尺寸 | 降低后续层计算量 |
| **增大感受野** | 缩小特征图 → 每个像素"看到"的输入范围更大 | 深层特征更全局 |
| **平移不变性** | 微小平移不影响最大池化结果 | 提升鲁棒性 |

### 3.3 转置卷积（Transposed Convolution）

#### 为什么叫"转置"？

标准卷积（矩阵形式）：$y = C x$（$C$ 是卷积矩阵）
转置卷积：$y = C^T x$

不是求逆，而是"形状上的逆操作"——输入和输出尺寸互换。

#### 输出尺寸公式

$$H_{out} = (H_{in} - 1) \times S - 2P + K$$

#### 棋盘效应（Checkerboard Artifact）

转置卷积最常见的陷阱：

```text
原因：当 kernel_size 不能被 stride 整除时，输出图像出现
      重叠不均匀的"棋盘格"伪影。

解决：
1. kernel_size 设为 stride 的整数倍
2. 用 上采样（插值）+ 标准卷积 替代转置卷积
   nn.Upsample(scale_factor=2) → nn.Conv2d(...)
```

```python
# 转置卷积
deconv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

# 更好的上采样方案（避免棋盘效应）
upsample = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    nn.Conv2d(64, 32, kernel_size=3, padding=1)
)
```

### 3.4 分组卷积（Group Convolution）

#### 原理

将输入通道分为 $g$ 组，每组独立卷积，输出拼接：

```text
输入 (24通道)           输出 (48通道)
┌──────┐  conv  ┌──────┐
│ 1-8  │ ────→  │ 1-16 │  组1：8→16
├──────┤  conv  ├──────┤
│ 9-16 │ ────→  │17-32│  组2：8→16
├──────┤  conv  ├──────┤
│17-24 │ ────→  │33-48│  组3：8→16
└──────┘        └──────┘
```

#### 参数量分析

| 类型 | $C_{in}$ | $C_{out}$ | 参数量 | 公式 |
|------|---------|---------|--------|------|
| 标准卷积 | 24 | 48 | 10,368 | $3 \times 3 \times 24 \times 48$ |
| 分组卷积 ($g=3$) | 24 | 48 | 3,456 | $g \times 3 \times 3 \times 8 \times 16$ |

参数量减少为原来的 $1/g$，同时各组之间**不共享信息**。

```python
# 分组卷积
conv = nn.Conv2d(24, 48, kernel_size=3, groups=3)

# 极端情况：g = C_in（深度卷积）
depthwise = nn.Conv2d(24, 24, kernel_size=3, groups=24)
```

### 3.5 深度可分离卷积（Depthwise Separable Convolution）

#### 两步分解

| 步骤 | 操作 | 输入→输出 | 参数量 | 作用 |
|------|------|---------|--------|------|
| **Depthwise** | 逐通道 $3 \times 3$ 卷积 | $C_{in}$ → $C_{in}$ | $C_{in} \times 9$ | 空间特征提取 |
| **Pointwise** | $1 \times 1$ 卷积 | $C_{in}$ → $C_{out}$ | $C_{in} \times C_{out}$ | 通道信息融合 |

#### 参数量对比（$C_{in}=24, C_{out}=48$）

| 方法 | 参数量 | 计算 |
|------|--------|------|
| 标准 $3 \times 3$ 卷积 | 10,368 | $3 \times 3 \times 24 \times 48$ |
| 深度可分离卷积 | 1,368 | $3 \times 3 \times 24 + 24 \times 48$ |
| **缩减比例** | **86.8%** | — |

```python
# 深度可分离卷积
depthwise = nn.Conv2d(24, 24, kernel_size=3, groups=24)
pointwise = nn.Conv2d(24, 48, kernel_size=1)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(c_in, c_in, kernel_size,
                                   groups=c_in, padding=kernel_size//2)
        self.pointwise = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
```

> MobileNet V1 完全基于深度可分离卷积构建，以标准卷积 1/9 的计算量达到接近的精度。

### 3.6 空洞卷积（Dilated / Atrous Convolution）

#### 原理

在卷积核元素之间插入"空洞"（0），等效感受野计算公式：

$$K_{eff} = K + (K - 1) \times (d - 1)$$

| dilation | 等效感受野（K=3） | 可视化 |
|----------|-----------------|--------|
| 1 | $3 \times 3$ | 紧邻 9 个像素 |
| 2 | $5 \times 5$ | 隔一个取一个 |
| 3 | $7 \times 7$ | 隔两个取一个 |
| 4 | $9 \times 9$ | 隔三个取一个 |

#### 为什么不用更大的卷积核？

| 方案 | 参数量（3通道→64通道） | 感受野 |
|------|---------------------|--------|
| $3 \times 3$ dilation=1 | $3 \times 3 \times 3 \times 64 = 1,728$ | 3 |
| $5 \times 5$ dilation=1 | $5 \times 5 \times 3 \times 64 = 4,800$ | 5 |
| $3 \times 3$ dilation=2 | $3 \times 3 \times 3 \times 64 = 1,728$ | 5 |

> 空洞卷积以**相同的参数量**获得**更大的感受野**——DeepLab 系列的核心技巧。

#### 多尺度空洞卷积（ASPP）

```python
# ASPP：用多组不同 dilation 的卷积并行处理
class ASPP(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 1)              # 1×1 基准
        self.conv6 = nn.Conv2d(c_in, c_out, 3, dilation=6, padding=6)
        self.conv12 = nn.Conv2d(c_in, c_out, 3, dilation=12, padding=12)
        self.conv18 = nn.Conv2d(c_in, c_out, 3, dilation=18, padding=18)
        self.pool = nn.AdaptiveAvgPool2d(1)                  # 全局上下文

    def forward(self, x):
        return self.conv1(x) + self.conv6(x) + self.conv12(x) + self.conv18(x) + ...
```

### 3.7 可变形卷积（Deformable Convolution, DCN）

#### 动机

标准卷积的采样点是规则的矩形网格：

```text
标准 3×3 采样点          DCN 3×3 采样点（偏移后）
●  ●  ●                  ●      ●
                          ●  ●
●  ●  ●                      ●
                              ●  ●
●  ●  ●                  ●
```

对于不规则形状的目标（弯曲的文字、变形的物体），规则采样无法对齐。

#### 实现流程

```
输入 → 卷积学习偏移量 Δp → 双线性插值 → 可变形卷积 → 输出
```

| 步骤 | 操作 |
|------|------|
| 1 | 额外卷积从输入学习每个采样点的 $(Δx, Δy)$ 偏移 |
| 2 | 偏移后的采样位置通常不是整数 → 双线性插值 |
| 3 | 用插值后的值执行标准卷积 |

> DCN 让感受野"学会适应"目标形状——检测和分割任务中提升显著。

---

## 四、输出尺寸速查

| 算子 | 输出尺寸公式 |
|------|------------|
| 标准卷积 | $H_{out} = \lfloor \frac{H_{in} + 2P - K}{S} \rfloor + 1$ |
| 池化 | 同上 |
| 转置卷积 | $H_{out} = (H_{in} - 1)S - 2P + K$ |
| 空洞卷积 | $K_{eff} = K + (K-1)(d-1)$，代入标准卷积公式 |

---

## 五、延伸：神经网络架构搜索（NAS）

除了人工设计卷积算子，**神经网络架构搜索（NAS）** 让算法自动搜索最优网络结构。

| 方法 | 搜索空间 | 搜索策略 | 代表工作 |
|------|---------|---------|---------|
| 强化学习 | 层类型、连接 | RNN 控制器 + Policy Gradient | NASNet |
| 进化算法 | 拓扑结构 | 变异 + 交叉 + 选择 | AmoebaNet |
| 可微分 | 连续松弛 | 梯度下降 | DARTS |
| 权重共享 | 子网络采样 | 超网络训练 | ENAS, OFA |

> NAS 的核心三要素：**搜索空间**（能搜什么）→ **搜索策略**（怎么搜）→ **性能评估策略**（怎么判断好坏）。

---

*相关笔记：[什么是卷积网络](./What_is_Convolution_network.md)、[从传统图像处理到深度学习](./From_traditional_image_processing_to_deep_learning.md)*
