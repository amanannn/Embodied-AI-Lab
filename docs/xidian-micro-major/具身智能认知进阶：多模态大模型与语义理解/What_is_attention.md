# 什么是注意力机制

> 西安电子科技大学 · 具身智能微专业
> 课程笔记

---

## 一、为什么需要注意力

人脑通过注意力机制来缓解信息超载问题——在海量信息中聚焦关键部分，忽略无关内容。深度学习中的注意力机制灵感正来源于此。

> 注意力机制的核心思想：不是所有信息都同等重要。让模型学会"关注哪里"和"关注什么"。

---

## 二、按域分类

按照注意力关注的维度，分为三类：

| 类型 | 关注域 | 解决的问题 |
|------|--------|-----------|
| 通道注意力 | Channel | 哪些通道更重要？ |
| 空间注意力 | Spatial | 哪些位置更重要？ |
| 时间注意力 | Temporal | 哪些时刻更相关？ |

---

## 三、通道注意力

### 3.1 动机

不同的通道提取了不同的特征（边缘、纹理、颜色等），它们对当前任务的重要程度不同。通道注意力让网络自适应地学习每个通道的权重。

### 3.2 SE Block（Squeeze-and-Excitation）

以 SE Block 为例，通道注意力的实现分三步：

| 步骤 | 操作 | 输入 | 输出 |
|------|------|------|------|
| 1. 通道汇聚（Squeeze） | 全局平均池化 | $C \times H \times W$ | $C \times 1 \times 1$ |
| 2. 权重激活（Excitation） | FC1 → Swish → FC2 → Sigmoid | $C$ | $C$（学习后的权重） |
| 3. 注意力添加（Scale） | 特征图 × 通道权重（逐通道点乘） | — | 加权后的特征图 |

```python
# SE Block 伪代码
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        self.squeeze = nn.AdaptiveAvgPool2d(1)        # 步骤1
        self.excitation = nn.Sequential(               # 步骤2
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)                 # (B, C)
        w = self.excitation(w).view(b, c, 1, 1)        # (B, C, 1, 1)
        return x * w                                    # 步骤3: 逐通道加权
```

> 由于权重是向量形式（每个通道一个标量），因此采用全连接层（线性层）来学习权重参数。

### 3.3 应用

| 模型 | 说明 |
|------|------|
| MobileNet V3 | 在轻量化网络中引入 SE 模块 |
| EfficientNet | 系统性使用 SE 来实现高效扩展 |

---

## 四、空间注意力

### 4.1 动机

不同位置的信息重要程度不同——图像中目标所在的区域比背景更重要。空间注意力让网络关注关键位置，忽略无关区域，降低计算成本。

### 4.2 实现

| 步骤 | 操作 |
|------|------|
| 1 | 沿通道轴分别做平均池化和最大池化，得到两个 $1 \times H \times W$ 的二维平面图 |
| 2 | 将两个特征图拼接（concat），送入卷积层，再通过激活函数得到空间注意力权重矩阵 |
| 3 | 将权重矩阵与输入特征图逐元素相乘，得到空间注意力特征图 |

```python
# 空间注意力伪代码
class SpatialAttention(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, H, W)
        concat = torch.cat([avg_out, max_out], dim=1)   # (B, 2, H, W)
        weight = torch.sigmoid(self.conv(concat))       # (B, 1, H, W)
        return x * weight
```

### 4.3 应用

| 模型 | 说明 |
|------|------|
| STN（Spatial Transformer Network） | 学习空间变换参数 |
| DCN（Deformable Convolution Network） | 学习卷积核的偏移量 |

---

## 五、时间注意力

### 5.1 动机

在序列数据中，后续输入与前面的输入存在依赖关系。时间注意力的目的：**让每个时刻的输入都能与整个序列建立联系**。

### 5.2 QKV 机制

时间注意力的核心是 Query-Key-Value 三元组：

| 步骤 | 操作 |
|------|------|
| 1 | 由输入序列经过线性映射得到 Q、K、V 三个矩阵 |
| 2 | Q 与 K 的转置做矩阵乘法，得到注意力权重矩阵 |
| 3 | 权重矩阵与 V 做点乘，得到加权输出 |

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| 矩阵 | 含义 | 类比 |
|------|------|------|
| Q（Query） | 查询 | "我在找什么？" |
| K（Key） | 键 | "我是什么？" |
| V（Value） | 值 | "我包含什么信息？" |

> 除以 $\sqrt{d_k}$ 的作用是缩放，防止点积过大导致 softmax 梯度消失。

### 5.3 应用

基于时间注意力的 GLTR 模块等。

---

## 六、自注意力机制

### 6.1 核心思想

自注意力（Self-Attention）是注意力机制的特例：**Q、K、V 都来自同一个输入序列**。

### 6.2 计算步骤

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | Query 和 Key 进行相似度计算 | 得到注意力权值 |
| 2 | 权值归一化（Softmax） | 得到可直接使用的权重 |
| 3 | 权重与 Value 加权求和 | 得到最终输出 |

### 6.3 关键意义

> 自注意力机制使模型在处理一个序列时，能计算序列中每个元素与其他所有元素的关联度。这些权重反映了元素之间的相互关系——在语言模型中，它们可以反映词与词之间的语义关联度。

自注意力是 Transformer 架构的核心引擎，也是现代大语言模型（GPT、BERT 等）的基础计算单元。

---

## 七、三类注意力对比

| 维度 | 通道注意力 | 空间注意力 | 时间/自注意力 |
|------|-----------|-----------|-------------|
| 关注什么 | 哪个通道重要 | 哪个位置重要 | 哪个时刻/元素相关 |
| 权重形状 | $C \times 1 \times 1$ | $1 \times H \times W$ | $T \times T$（注意力矩阵） |
| 典型实现 | SE Block | CBAM 空间分支 | Scaled Dot-Product |
| 代表应用 | EfficientNet | STN, DCN | Transformer, GPT |

---

*相关笔记：[什么是神经网络](../具身智能感知基础：机器学习与深度学习/What_is_neural_network.md)、[什么是卷积网络](../具身智能感知基础：机器学习与深度学习/What_is_Convolution_network.md)*
