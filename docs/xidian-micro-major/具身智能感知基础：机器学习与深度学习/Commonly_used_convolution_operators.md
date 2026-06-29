# 常用卷积算子

> 西安电子科技大学 · 具身智能微专业
> 课程笔记

## 一、什么是卷积算子？

卷积算子是 CNN 中执行**特征提取和表示学习**的基本运算单元。

传统 3×3 卷积的问题：功能单一、参数量偏大、感受野较小。各种卷积变体通过改变计算方式，在不同维度上优化性能和效率。

## 二、卷积变体全景

| 算子 | 核心作用 | 关键特性 |
|------|---------|---------|
| 1×1 卷积 | 通道维度变换 | 降维/升维，不改变空间尺寸 |
| 池化 | 降采样 | 降低分辨率，增大感受野 |
| 转置卷积 | 上采样 | 从小尺寸恢复大分辨率 |
| 分组卷积 | 降低参数量 | 分组独立卷积，减少 1/g 参数 |
| 深度可分离卷积 | 轻量化 | 逐通道卷积 + 1×1 投影 |
| 空洞卷积 | 扩大感受野 | 不增加参数，引入空洞率 |
| 可变形卷积 | 适应不规则形状 | 学习感受野的偏移量 |

## 三、各算子详解

### 3.1 1×1 卷积

感受野为 1×1 的卷积核，操作与常规卷积完全相同。**核心用途：通道数的升降维**。

```python
conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
```

### 3.2 池化层（Pooling）

改变输入特征的空间尺寸，主要用于**降采样**。

```python
pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

| 池化方法 | 计算方式 | 特点 |
|---------|---------|------|
| 最大池化（Max Pooling） | 取邻域最大值 | 保留最显著特征 |
| 平均池化（Avg Pooling） | 取邻域平均值 | 平滑，保留背景信息 |
| 自适应池化（Adaptive） | 自动调整参数 | 输出固定尺寸，不依赖输入大小 |

池化的作用：

- **特征降维**：减小特征图尺寸
- **保持特征不变性**：轻微平移不影响池化结果
- **减少过拟合**：降低参数量和计算量

> 深度学习的三种状态：**拟合**（训练良好）、**欠拟合**（模型太简单，数据规律都没学到）、**过拟合**（训练集上表现好，测试集上表现差）。

### 3.3 转置卷积（Transposed Convolution）

也叫反卷积（Deconvolution）。**目的：上采样**——从较小尺寸的输入得到大分辨率输出。运算等价于先对输入做 padding 再做常规卷积。

```python
conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
```

### 3.4 分组卷积（Group Convolution）

输入和输出通道被划分为 $g$ 个组，每个组的输出只与对应组内的输入相连。参数量仅为不分组时的 $1/g$。

```python
conv = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, groups=3)
```

**参数量对比**：

- 常规 3×3 卷积：$3 \times 3 \times 24 \times 48 = 10368$
- 分组卷积（$g=3$）：$3 \times (3 \times 3 \times 8 \times 16) = 3456$

> 分组卷积是 ResNeXt 和 MobileNet 等高效网络的基础组件。

### 3.5 深度可分离卷积（Depthwise Separable Convolution）

由两部分组成：

1. **深度卷积**（Depthwise）：每个通道独立做卷积（$g$ = 输入通道数）
2. **逐点卷积**（Pointwise）：1×1 卷积将深度卷积的输出投影到新特征图

```python
depthwise_conv = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, groups=24)
pointwise_conv  = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=1)
```

**优势**：大幅降低参数量和计算量，同时实现信息的跨通道交互与整合。MobileNet 系列的核心思想。

### 3.6 空洞卷积（Dilated / Atrous Convolution）

在原始卷积核的基础上引入**空洞率（Dilation Rate）**，在核元素之间插入空隙。

```python
conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2)
```

- dilation=1：标准 3×3 卷积（感受野 3×3）
- dilation=2：3×3 卷积感受野等效为 5×5
- dilation=3：3×3 卷积感受野等效为 7×7

> **不增加参数，获得更大感受野**——在语义分割（DeepLab）中广泛应用。

### 3.7 可变形卷积（Deformable Convolution）

对标准卷积进行局部形变，适应不规则形状的目标。

**流程**：

```
输入张量 → 卷积计算偏移量 → 双线性插值 → 可变形卷积 → 输出张量
```

**双线性插值**：当目标像素坐标不是整数（如 (3.7, 5.2)）时，用周围 4 个最近邻整数像素的加权平均来合成该位置的值。本质是在两个方向分别做线性插值。

**优势**：适应不规则结构，减少网格偏差（grid bias），对几何形变更鲁棒。

## 四、延伸：神经网络架构搜索（NAS）

除了人工设计卷积算子，还有**神经网络架构搜索（Neural Architecture Search）**——让算法自动搜索最优的网络结构。这是 AutoML 的核心方向之一。

---

*笔记状态：初稿完成，每个算子配有 PyTorch 代码示例。*

*相关笔记：[什么是卷积网络](./What_is_Convolution_network.md)、[从传统图像处理到深度学习](./From_traditional_image_processing_to_deep_learning.md)*
