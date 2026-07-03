# 什么是注意力机制

> 西安电子科技大学 · 具身智能微专业
> 课程笔记

---

## 一、为什么需要注意力

人脑通过注意力机制来缓解信息超载问题——在海量信息中聚焦关键部分，忽略无关内容。深度学习中的注意力机制灵感正来源于此。

在深度学习中，"信息超载"同样存在：

| 场景 | 超载来源 | 注意力解决 |
|------|---------|-----------|
| 图像分类 | 224×224×3 = 150,528 个像素，并非所有像素都重要 | 聚焦目标区域 |
| 机器翻译 | 源语言序列长度 N，每步需要理解全局上下文 | 动态选择关键 token |
| 视频理解 | 帧数 × 空间分辨率，信息量爆炸 | 关注关键帧和区域 |

> 注意力机制的核心思想：不是所有信息都同等重要。让模型学会"关注哪里"和"关注什么"，**动态分配有限的计算资源到最相关的信息上**。

### 1.1 注意力 vs 传统方法

| 对比维度 | CNN 全连接 | RNN 隐藏状态 | 注意力机制 |
|---------|-----------|------------|-----------|
| 感受野 | 局部（单层）→ 全局（深层） | 逐步衰减 | 一次性全局 |
| 长程依赖 | 需要多层堆叠 | 受限于序列长度 | 直接建立任意距离的联系 |
| 并行化 | 天然并行 | 串行 | 天然并行 |
| 可解释性 | 弱 | 弱 | 可视化注意力权重 |

---

## 二、按域分类

按照注意力关注的维度，分为三类：

| 类型 | 关注域 | 解决的问题 | 代表工作 |
|------|--------|-----------|---------|
| 通道注意力 | Channel | 哪些通道更重要？ | SE-Net, SK-Net |
| 空间注意力 | Spatial | 哪些位置更重要？ | STN, DCN, Non-local |
| 时间/自注意力 | Temporal/Token | 哪些时刻/元素更相关？ | Transformer, GPT |

---

## 三、通道注意力

### 3.1 动机——为什么通道重要程度不同？

CNN 中每个卷积核提取一种特定模式：

| 通道 | 可能提取的特征 | 对"人脸识别"的重要程度 | 对"场景分类"的重要程度 |
|------|-------------|-------------------|-------------------|
| Ch-1 | 边缘检测 | ★★☆ | ★★★ |
| Ch-2 | 纹理检测 | ★☆☆ | ★★★ |
| Ch-3 | 眼睛形状 | ★★★ | ★☆☆ |
| Ch-4 | 噪点 | ★☆☆ | ★☆☆ |

> 同一个通道的激活图，对不同的下游任务价值完全不同。通道注意力就是让网络自己学会"这个通道在当前任务中值多少权重"。

### 3.2 SE Block（Squeeze-and-Excitation）

SE Block（Hu et al., 2018）是最经典的通道注意力实现：

#### 数学描述

给定输入特征图 $X \in \mathbb{R}^{C \times H \times W}$：

**Step 1 — Squeeze（通道汇聚）**：
$$z_c = \frac{1}{H \times W}\sum_{i=1}^{H}\sum_{j=1}^{W} X_c(i, j)$$

将每个通道的空间信息压缩为一个标量 $z_c$，得到 $z \in \mathbb{R}^{C}$。

**Step 2 — Excitation（权重激活）**：
$$s = \sigma(W_2 \cdot \delta(W_1 \cdot z))$$

其中 $W_1 \in \mathbb{R}^{\frac{C}{r} \times C}$，$W_2 \in \mathbb{R}^{C \times \frac{C}{r}}$，$r$ 是缩减率（通常 16），$\delta$ 是 ReLU，$\sigma$ 是 Sigmoid。

两个 FC 层构成**瓶颈结构**——先压缩再扩展，既减少参数又增加非线性。

**Step 3 — Scale（注意力添加）**：
$$\tilde{X}_c = s_c \cdot X_c$$

| 步骤 | 操作 | 输入形状 | 输出形状 | 参数量 |
|------|------|---------|---------|--------|
| 1. Squeeze | 全局平均池化 | $C \times H \times W$ | $C \times 1 \times 1$ | 0 |
| 2. Excitation | FC1→ReLU→FC2→Sigmoid | $C$ | $C$ | $2C^2/r$ |
| 3. Scale | 逐通道点乘 | — | $C \times H \times W$ | 0 |

#### 完整实现

```python
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Args:
        channels: 输入通道数
        reduction: 瓶颈缩减率（默认 16）
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        # Squeeze: (B, C, H, W) → (B, C, 1, 1) → (B, C)
        z = self.squeeze(x).view(b, c)
        # Excitation: (B, C) → (B, C) → (B, C, 1, 1)
        s = self.excitation(z).view(b, c, 1, 1)
        # Scale: 逐通道加权
        return x * s
```

#### 为什么用两个 FC 层而不是一个？

| 设计 | 参数量 | 非线性 | 表达能力 |
|------|--------|--------|---------|
| 单 FC | $C^2$ | 无（Sigmoid 前无激活） | 弱 |
| 双 FC（瓶颈） | $2C^2/r$ | ReLU + Sigmoid | 强 |

> 瓶颈结构用 $r=16$ 时参数量仅为单 FC 的 $2/16 = 12.5\%$，同时 ReLU 引入的非线性让网络能学习更复杂的通道间关系。

### 3.3 SE 变体

| 变体 | 改进点 | 论文 |
|------|--------|------|
| SE（原版） | 全局平均池化 + FC 瓶颈 | Hu et al., 2018 |
| SE-Var | 仅用 Sigmoid 学习的标量参数，无 Squeeze | — |
| SK-Net | 多尺度卷积核 + 动态选择 | Li et al., 2019 |
| ECA-Net | 用 1D 卷积替代 FC，自适应 kernel size | Wang et al., 2020 |

### 3.4 应用

| 模型 | 如何引入 | 效果 |
|------|---------|------|
| **SENet** | 在 ResNet 每个残差块后插入 SE | ImageNet top-5 error 降低 25% |
| **MobileNet V3** | 在轻量化瓶颈中嵌入 SE | 同等延迟下精度提升 |
| **EfficientNet** | 系统性使用 SE 实现高效扩展 | 减少 8.4× 参数的同时保持 SOTA |

---

## 四、空间注意力

### 4.1 动机——为什么位置重要程度不同？

```text
输入图像 (H × W)：
┌─────────────────────┐
│  天空（70% 面积）     │  ← 分类不重要
│    ┌──────┐          │
│    │ 人脸  │          │  ← 分类关键
│    └──────┘          │
│  草地               │  ← 分类不重要
└─────────────────────┘
```

CNN 对图像所有位置**等权处理**。空间注意力打破这种平等性——模型应该把更多"注意力预算"分配给关键区域。

### 4.2 实现原理

空间注意力的标准流程（来自 CBAM 的空间分支）：

| 步骤 | 操作 | 形状变化 |
|------|------|---------|
| 1 | 沿通道轴分别做平均池化和最大池化 | $(C,H,W) \rightarrow (1,H,W) \times 2$ |
| 2 | 拼接双池化结果，7×7 卷积 + Sigmoid | $(2,H,W) \rightarrow (1,H,W)$ |
| 3 | 逐元素乘回原特征图 | $(1,H,W) \odot (C,H,W) \rightarrow (C,H,W)$ |

**为什么同时用平均池化和最大池化？**

| 池化类型 | 捕捉的信息 | 
|---------|-----------|
| 平均池化 | 区域整体激活强度——"这里平均响应有多强？" |
| 最大池化 | 最显著特征——"这里有没有突出的模式？" |

两者互补，组合使用比单用任一种都好。

#### 实现

```python
class SpatialAttention(nn.Module):
    """CBAM 空间注意力分支"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # 双池化
        avg_out = torch.mean(x, dim=1, keepdim=True)    # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        # 拼接 + 卷积 + 激活
        pooled = torch.cat([avg_out, max_out], dim=1)   # (B, 2, H, W)
        weight = torch.sigmoid(self.conv(pooled))       # (B, 1, H, W)
        return x * weight
```

### 4.3 应用

| 模型 | 空间注意力方式 | 特点 |
|------|-------------|------|
| **STN** | 学习全局空间变换（仿射/透视） | 端到端可微，"学会看哪里" |
| **DCN** | 学习每个卷积核位置的偏移量 | 自适应感受野形状 |
| **Non-local** | 计算任意两个空间位置的相似度 | 捕获长程空间依赖 |
| **CBAM** | 通道 + 空间串行注意力 | 即插即用的轻量模块 |

---

## 五、CBAM：通道 + 空间融合

CBAM（Convolutional Block Attention Module）将通道注意力和空间注意力**串行组合**：

$$F' = M_c(F) \otimes F$$
$$F'' = M_s(F') \otimes F'$$

| 阶段 | 模块 | 关注维度 | 权重形状 |
|------|------|---------|---------|
| 第一级 | 通道注意力 $M_c$ | 哪个通道重要 | $(C, 1, 1)$ |
| 第二级 | 空间注意力 $M_s$ | 哪个位置重要 | $(1, H, W)$ |

```python
class CBAM(nn.Module):
    """通道注意力 + 空间注意力的串行组合"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = SEBlock(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)    # 先通道
        x = self.spatial_att(x)    # 再空间
        return x
```

> CBAM 的设计理念：先确定"看哪个通道"，再确定"看通道里的哪个位置"——两级过滤，逐步聚焦。

---

## 六、时间注意力与 Scaled Dot-Product Attention

### 6.1 动机

序列数据（文本、语音、视频帧）中，当前位置的理解可能依赖于过去或未来的其他位置：

```text
"我 今天 去 银行 存钱"
         ↑
    这个"银行"指什么？
    需要结合"存钱"才能确定 → 不是河边的"银行"
```

时间注意力让每个位置**直接**与整个序列的所有位置交互，不受距离限制。

### 6.2 QKV 机制的直觉理解

QKV 机制的设计灵感来自**信息检索**：

| 矩阵 | 全称 | 来源 | 直觉类比 | 形状 |
|------|------|------|---------|------|
| $Q$ | Query | 输入 × $W^Q$ | "我想查什么？" | $(N, d_k)$ |
| $K$ | Key | 输入 × $W^K$ | "每个元素的标签" | $(N, d_k)$ |
| $V$ | Value | 输入 × $W^V$ | "每个元素的内容" | $(N, d_v)$ |

**类比**：你在图书馆（序列）查找一本书——
1. 你有一个查询需求 $Q$（"机器学习"）
2. 每本书有标签 $K$（书名、关键词）
3. $Q$ 与每本书的 $K$ 计算相似度 → 得到哪本书最相关（注意力权重）
4. 根据权重，从每本书的 $V$（实际内容）中加权提取信息

### 6.3 数学推导

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Step 1：计算相似度矩阵**

$$S = QK^T \in \mathbb{R}^{N \times N}$$

其中 $S_{ij}$ 表示第 $i$ 个位置的 Query 与第 $j$ 个位置的 Key 的相似度。

**Step 2：缩放 + Softmax 归一化**

$$A = softmax\left(\frac{S}{\sqrt{d_k}}\right)$$

得到注意力权重矩阵，每行和为 1。

**Step 3：加权聚合**

$$O = AV \in \mathbb{R}^{N \times d_v}$$

每个位置的输出是所有位置 Value 的加权和。

#### 为什么要除以 $\sqrt{d_k}$？

假设 $Q$ 和 $K$ 的每个元素独立同分布，均值为 0，方差为 1：

$$QK^T = \sum_{i=1}^{d_k} q_i k_i$$

点积的均值为 0，方差为 $d_k$。当 $d_k$ 很大时（如 512），点积的数值范围会达到几十到几百，导致 softmax 进入饱和区（梯度趋于 0）。

除以 $\sqrt{d_k}$ 将方差归一化为 1，保持梯度稳定。

#### 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, n_heads, seq_len, d_k)
        mask: (batch, 1, seq_len, seq_len) or None
        """
        # Step 1 & 2: 相似度 + 缩放
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 可选：mask 掉未来位置（decoder 自回归）或 padding
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 2: Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Step 3: 加权聚合
        output = torch.matmul(attn_weights, V)

        return output, attn_weights
```

### 6.4 复杂度分析

| 组件 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| $QK^T$ 矩阵乘法 | $O(N^2 d_k)$ | $O(N^2)$ |
| Softmax | $O(N^2)$ | $O(N^2)$ |
| $A V$ 矩阵乘法 | $O(N^2 d_v)$ | $O(N d_v)$ |

瓶颈在于 $O(N^2)$ —— 这也是长序列处理的根本挑战，催生了 Sparse Attention、Flash Attention 等优化方向。

---

## 七、多头注意力（Multi-Head Attention）

### 7.1 动机——单头不够

单头注意力只能学到一种"关注模式"。多头注意力让模型同时从**多个不同子空间**去关注信息，类似于 CNN 中使用多个卷积核提取不同特征。

$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O$$

$$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

### 7.2 实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # 将 QKV 的映射合并为一个大矩阵，提高效率
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, x, mask=None):
        B, N, D = x.shape

        # 一次性计算 Q、K、V
        qkv = self.qkv_proj(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(attn_output), attn_weights
```

### 7.3 每个头学到了什么？

| 头 | 可能关注的关系 | 示例 |
|----|-------------|------|
| Head 1 | 语法依赖 | 主语 → 谓语 |
| Head 2 | 指代关系 | 代词 → 先行词 |
| Head 3 | 局部上下文 | 相邻词 |
| Head 4 | 长程语义 | 段落首尾 |

> 多头机制是 Transformer 成功的关键——不同头自然分化出不同的"关注策略"，无需人工设计。

---

## 八、自注意力与其他注意力的关系

### 8.1 按 QKV 来源分类

| 注意力类型 | Q 来源 | K 来源 | V 来源 | 应用场景 |
|-----------|--------|--------|--------|---------|
| **自注意力** | 自身序列 | 自身序列 | 自身序列 | Encoder、GPT |
| **交叉注意力** | 目标序列 | 源序列 | 源序列 | Decoder（翻译） |
| **掩码注意力** | 自身序列 | 自身序列（mask 未来） | 自身序列 | GPT/Decoder 自回归 |

### 8.2 自注意力的核心优势

| 能力 | 说明 |
|------|------|
| **全局依赖** | 任意两个位置直接建立联系，无需逐层传递 |
| **动态权重** | 权重由输入内容决定，而非固定（vs 卷积核） |
| **并行计算** | 序列所有位置同时计算，不依赖前一步结果（vs RNN） |
| **可解释性** | 注意力权重可视化，直观展示模型的"关注点" |
| **长度灵活** | 处理变长序列，无固定输入限制 |

### 8.3 自注意力的局限

| 局限 | 原因 | 解决方案 |
|------|------|---------|
| $O(N^2)$ 复杂度 | 每个位置关注所有位置 | Sparse/Linformer/FlashAttention |
| 缺乏位置信息 | 全连接无序 | Position Encoding/Embedding |
| 大量数据需求 | 无归纳偏置（不像 CNN 有局部性先验） | 预训练 + 微调范式 |

---

## 九、注意力机制的演进路线

```text
2014 — Bahdanau Attention（RNN 翻译对齐）
  │
2015 — Luong Attention（全局 + 局部注意力变体）
  │
2017 — Self-Attention + Multi-Head Attention（Transformer 诞生）
  │
2018 — SE-Net（通道注意力在 CV 爆发）
  │     CBAM（通道 + 空间融合）
  │
2019 — Sparse Attention（Sparse Transformer, Longformer）
  │
2020 — Vision Transformer (ViT)（纯注意力做图像分类）
  │     ECA-Net（高效通道注意力）
  │
2021 — Swin Transformer（分层注意力，CV 统治级表现）
  │
2022 — Flash Attention（IO 感知的快速注意力）
  │
2023 — Flash Attention v2 / Ring Attention（更长序列）
```

---

## 十、总结对比

| 维度 | 通道注意力 | 空间注意力 | CBAM | 自注意力 |
|------|-----------|-----------|------|---------|
| 关注维度 | 通道 | 空间位置 | 通道 → 空间 | Token 间关系 |
| 权重形状 | $(C, 1, 1)$ | $(1, H, W)$ | 两者 | $(N, N)$ |
| 核心操作 | 全局池化 + FC | 双池化 + 卷积 | SE + Spatial | QK^T + Softmax |
| 复杂度 | $O(C^2/r)$ | $O(HW)$ | $O(C^2/r + HW)$ | $O(N^2 d)$ |
| 代表应用 | EfficientNet | STN, DCN | 即插即用 | GPT, ViT |
| 适用数据 | 图像 | 图像 | 图像 | 序列/图像 |

---

*相关笔记：[什么是神经网络](../具身智能感知基础：机器学习与深度学习/What_is_neural_network.md)、[什么是卷积网络](../具身智能感知基础：机器学习与深度学习/What_is_Convolution_network.md)、[Computer Vision & Image Super-Resolution](../具身智能感知基础：机器学习与深度学习/Computer_Vision_&_Image_Super-resolution.md)*
