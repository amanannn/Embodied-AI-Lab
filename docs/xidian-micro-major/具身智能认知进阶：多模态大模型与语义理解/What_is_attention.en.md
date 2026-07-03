# What is Attention Mechanism

> Xidian University · Embodied AI Micro-Major
> Course Notes

---

## I. Why Attention

The human brain uses attention to cope with information overload — focusing on the key parts among massive information while ignoring the irrelevant. The attention mechanism in deep learning is inspired by this.

> The core idea of attention: not all information is equally important. Let the model learn "where to look" and "what to attend to."

---

## II. Classification by Domain

Attention mechanisms are classified into three types based on the domain they attend to:

| Type | Domain | Problem Solved |
|------|--------|---------------|
| Channel Attention | Channel | Which channels are more important? |
| Spatial Attention | Spatial | Which positions are more important? |
| Temporal Attention | Temporal | Which moments are more relevant? |

---

## III. Channel Attention

### 3.1 Motivation

Different channels extract different features (edges, textures, colors, etc.), and their importance varies across tasks. Channel attention enables the network to adaptively learn the weight of each channel.

### 3.2 SE Block (Squeeze-and-Excitation)

Taking SE Block as an example, channel attention is implemented in three steps:

| Step | Operation | Input | Output |
|------|-----------|-------|--------|
| 1. Squeeze | Global Average Pooling | $C \times H \times W$ | $C \times 1 \times 1$ |
| 2. Excitation | FC1 → Swish → FC2 → Sigmoid | $C$ | $C$ (learned weights) |
| 3. Scale | Feature map × channel weights (per-channel multiply) | — | Weighted feature map |

```python
# SE Block pseudocode
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        self.squeeze = nn.AdaptiveAvgPool2d(1)        # Step 1
        self.excitation = nn.Sequential(               # Step 2
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)                 # (B, C)
        w = self.excitation(w).view(b, c, 1, 1)        # (B, C, 1, 1)
        return x * w                                    # Step 3: per-channel weighting
```

> Since the weights are in vector form (one scalar per channel), fully connected layers (linear layers) are used to learn the weight parameters.

### 3.3 Applications

| Model | Description |
|-------|-------------|
| MobileNet V3 | Introduced SE modules in lightweight networks |
| EfficientNet | Systematically used SE for efficient scaling |

---

## IV. Spatial Attention

### 4.1 Motivation

Information at different positions has different importance — the region containing the target object in an image is more important than the background. Spatial attention enables the network to focus on key positions and ignore irrelevant areas, reducing computational cost.

### 4.2 Implementation

| Step | Operation |
|------|-----------|
| 1 | Apply average pooling and max pooling along the channel axis, producing two $1 \times H \times W$ 2D maps |
| 2 | Concatenate the two maps, pass through a convolution layer, then an activation function to obtain the spatial attention weight matrix |
| 3 | Element-wise multiply the weight matrix with the input feature map to get the final spatial attention feature map |

```python
# Spatial Attention pseudocode
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

### 4.3 Applications

| Model | Description |
|-------|-------------|
| STN (Spatial Transformer Network) | Learns spatial transformation parameters |
| DCN (Deformable Convolution Network) | Learns offsets for convolution kernels |

---

## V. Temporal Attention

### 5.1 Motivation

In sequential data, later inputs have dependencies on earlier inputs. The goal of temporal attention: **enable each moment's input to establish connections with the entire sequence**.

### 5.2 QKV Mechanism

The core of temporal attention is the Query-Key-Value triplet:

| Step | Operation |
|------|-----------|
| 1 | Linearly project the input sequence to obtain Q, K, V matrices |
| 2 | Matrix multiply Q with the transpose of K to get the attention weight matrix |
| 3 | Dot product the weight matrix with V to get the weighted output |

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Matrix | Meaning | Analogy |
|--------|---------|---------|
| Q (Query) | Query | "What am I looking for?" |
| K (Key) | Key | "What am I?" |
| V (Value) | Value | "What information do I contain?" |

> Dividing by $\sqrt{d_k}$ scales the dot products, preventing vanishing gradients in softmax when the dot products are too large.

### 5.3 Applications

GLTR modules based on temporal attention, and more.

---

## VI. Self-Attention

### 6.1 Core Idea

Self-Attention is a special case of attention: **Q, K, and V all come from the same input sequence**.

### 6.2 Computation Steps

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | Compute similarity between Query and Key | Obtain attention scores |
| 2 | Normalize scores (Softmax) | Obtain usable weights |
| 3 | Weighted sum with Value | Obtain final output |

### 6.3 Key Significance

> Self-attention enables the model, when processing a sequence, to compute the relevance between each element and every other element in the sequence. These weights reflect the inter-element relationships — in language models, they can reflect the semantic relatedness between words.

Self-attention is the core engine of the Transformer architecture and the fundamental computation unit of modern large language models (GPT, BERT, etc.).

---

## VII. Comparison of Three Attention Types

| Dimension | Channel Attention | Spatial Attention | Temporal / Self-Attention |
|-----------|-------------------|-------------------|--------------------------|
| What to attend to | Which channel matters | Which position matters | Which moment/element is relevant |
| Weight shape | $C \times 1 \times 1$ | $1 \times H \times W$ | $T \times T$ (attention matrix) |
| Typical implementation | SE Block | CBAM spatial branch | Scaled Dot-Product |
| Representative application | EfficientNet | STN, DCN | Transformer, GPT |

---

*Related notes: [What is a Neural Network](../具身智能感知基础：机器学习与深度学习/What_is_neural_network.md), [What is a Convolutional Network](../具身智能感知基础：机器学习与深度学习/What_is_Convolution_network.md)*
