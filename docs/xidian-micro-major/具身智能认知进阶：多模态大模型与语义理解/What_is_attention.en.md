# What is Attention Mechanism

> Xidian University · Embodied AI Micro-Major
> Course Notes

---

## I. Why Attention

The human brain uses attention to cope with information overload — focusing on key parts among massive information while ignoring the irrelevant. The attention mechanism in deep learning draws direct inspiration from this.

In deep learning, "information overload" is everywhere:

| Scenario | Overload Source | Attention Solution |
|----------|----------------|-------------------|
| Image Classification | 224×224×3 = 150,528 pixels, not all relevant | Focus on target region |
| Machine Translation | Source sequence length N, each step needs global context | Dynamically select key tokens |
| Video Understanding | Frames × spatial resolution, information explosion | Attend to key frames and regions |

> The core idea of attention: **not all information is equally important**. Let the model learn "where to look" and "what to attend to" by dynamically allocating limited computational resources to the most relevant information.

### 1.1 Attention vs Traditional Methods

| Dimension | CNN (Fully Connected) | RNN (Hidden State) | Attention Mechanism |
|-----------|----------------------|-------------------|-------------------|
| Receptive Field | Local (single layer) → Global (deep) | Gradually decays | One-shot global |
| Long-range Dependencies | Requires many layers | Limited by sequence length | Direct connections at any distance |
| Parallelization | Naturally parallel | Sequential | Naturally parallel |
| Interpretability | Weak | Weak | Visualizable attention weights |

---

## II. Classification by Domain

Attention mechanisms are classified into three types based on the domain they attend to:

| Type | Domain | Problem Solved | Representative Work |
|------|--------|---------------|-------------------|
| Channel Attention | Channel | Which channels are more important? | SE-Net, SK-Net |
| Spatial Attention | Spatial | Which positions are more important? | STN, DCN, Non-local |
| Temporal / Self-Attention | Temporal/Token | Which moments/elements are more relevant? | Transformer, GPT |

---

## III. Channel Attention

### 3.1 Motivation — Why Do Channels Have Different Importance?

In a CNN, each convolution kernel extracts a specific pattern:

| Channel | Possible Feature | Importance for "Face Recognition" | Importance for "Scene Classification" |
|---------|-----------------|----------------------------------|--------------------------------------|
| Ch-1 | Edge detection | ★★☆ | ★★★ |
| Ch-2 | Texture detection | ★☆☆ | ★★★ |
| Ch-3 | Eye shape | ★★★ | ★☆☆ |
| Ch-4 | Noise | ★☆☆ | ★☆☆ |

> The same channel's activation map has completely different value for different downstream tasks. Channel attention lets the network learn "how much weight does this channel deserve for the current task?"

### 3.2 SE Block (Squeeze-and-Excitation)

SE Block (Hu et al., 2018) is the most classic channel attention implementation:

#### Mathematical Description

Given input feature map $X \in \mathbb{R}^{C \times H \times W}$:

**Step 1 — Squeeze**:
$$z_c = \frac{1}{H \times W}\sum_{i=1}^{H}\sum_{j=1}^{W} X_c(i, j)$$

Compresses the spatial information of each channel into a single scalar $z_c$, yielding $z \in \mathbb{R}^{C}$.

**Step 2 — Excitation**:
$$s = \sigma(W_2 \cdot \delta(W_1 \cdot z))$$

Where $W_1 \in \mathbb{R}^{\frac{C}{r} \times C}$, $W_2 \in \mathbb{R}^{C \times \frac{C}{r}}$, $r$ is the reduction ratio (typically 16), $\delta$ is ReLU, $\sigma$ is Sigmoid.

The two FC layers form a **bottleneck structure** — compress first, then expand, reducing parameters while adding non-linearity.

**Step 3 — Scale**:
$$\tilde{X}_c = s_c \cdot X_c$$

| Step | Operation | Input Shape | Output Shape | Parameters |
|------|-----------|-------------|-------------|------------|
| 1. Squeeze | Global Average Pooling | $C \times H \times W$ | $C \times 1 \times 1$ | 0 |
| 2. Excitation | FC1→ReLU→FC2→Sigmoid | $C$ | $C$ | $2C^2/r$ |
| 3. Scale | Per-channel multiply | — | $C \times H \times W$ | 0 |

#### Full Implementation

```python
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    Args:
        channels: number of input channels
        reduction: bottleneck reduction ratio (default 16)
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
        # Scale: per-channel weighting
        return x * s
```

#### Why Two FC Layers Instead of One?

| Design | Parameters | Non-linearity | Expressiveness |
|--------|-----------|--------------|----------------|
| Single FC | $C^2$ | None (no activation before Sigmoid) | Weak |
| Dual FC (bottleneck) | $2C^2/r$ | ReLU + Sigmoid | Strong |

> With $r=16$, the bottleneck uses only $2/16 = 12.5\%$ of the parameters of a single FC, while the ReLU non-linearity lets the network learn more complex inter-channel relationships.

### 3.3 SE Variants

| Variant | Improvement | Paper |
|---------|------------|-------|
| SE (original) | Global Average Pooling + FC bottleneck | Hu et al., 2018 |
| SE-Var | Learned scalar parameters with Sigmoid only, no Squeeze | — |
| SK-Net | Multi-scale kernels + dynamic selection | Li et al., 2019 |
| ECA-Net | 1D convolution replaces FC, adaptive kernel size | Wang et al., 2020 |

### 3.4 Applications

| Model | How SE is Used | Effect |
|-------|---------------|--------|
| **SENet** | Insert SE after each residual block in ResNet | ImageNet top-5 error reduced by 25% |
| **MobileNet V3** | Embed SE in lightweight bottlenecks | Higher accuracy at the same latency |
| **EfficientNet** | Systematic SE usage for efficient scaling | 8.4× fewer parameters while maintaining SOTA |

---

## IV. Spatial Attention

### 4.1 Motivation — Why Do Positions Have Different Importance?

```text
Input Image (H × W):
┌─────────────────────────┐
│  Sky (70% of area)      │  ← Not important for classification
│    ┌──────────┐         │
│    │  Face    │         │  ← Critical for classification
│    └──────────┘         │
│  Grass                  │  ← Not important for classification
└─────────────────────────┘
```

CNNs treat all image positions **equally**. Spatial attention breaks this equality — the model should allocate more "attention budget" to critical regions.

### 4.2 Implementation Principle

The standard spatial attention pipeline (from CBAM's spatial branch):

| Step | Operation | Shape Transformation |
|------|-----------|---------------------|
| 1 | Apply both average and max pooling along the channel axis | $(C,H,W) \rightarrow (1,H,W) \times 2$ |
| 2 | Concatenate dual pooling results, 7×7 conv + Sigmoid | $(2,H,W) \rightarrow (1,H,W)$ |
| 3 | Element-wise multiply back to original feature map | $(1,H,W) \odot (C,H,W) \rightarrow (C,H,W)$ |

**Why use both average and max pooling?**

| Pooling Type | Information Captured |
|-------------|---------------------|
| Average Pooling | Overall regional activation intensity — "How strong is the average response here?" |
| Max Pooling | Most salient features — "Is there any prominent pattern here?" |

The two are complementary; using both together outperforms either alone.

#### Implementation

```python
class SpatialAttention(nn.Module):
    """CBAM spatial attention branch"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # Dual pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)    # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        # Concat + conv + activation
        pooled = torch.cat([avg_out, max_out], dim=1)   # (B, 2, H, W)
        weight = torch.sigmoid(self.conv(pooled))       # (B, 1, H, W)
        return x * weight
```

### 4.3 Applications

| Model | Spatial Attention Approach | Characteristics |
|-------|--------------------------|-----------------|
| **STN** | Learns global spatial transformation (affine/perspective) | End-to-end differentiable, "learns where to look" |
| **DCN** | Learns per-kernel-position offsets | Adaptive receptive field shape |
| **Non-local** | Computes pairwise similarity between any two spatial positions | Captures long-range spatial dependencies |
| **CBAM** | Channel + spatial serial attention | Lightweight plug-and-play module |

---

## V. CBAM: Channel + Spatial Fusion

CBAM (Convolutional Block Attention Module) combines channel and spatial attention **in series**:

$$F' = M_c(F) \otimes F$$
$$F'' = M_s(F') \otimes F'$$

| Stage | Module | Attention Dimension | Weight Shape |
|-------|--------|-------------------|-------------|
| First | Channel Attention $M_c$ | Which channel matters | $(C, 1, 1)$ |
| Second | Spatial Attention $M_s$ | Which position matters | $(1, H, W)$ |

```python
class CBAM(nn.Module):
    """Serial combination of channel + spatial attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = SEBlock(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)    # Channel first
        x = self.spatial_att(x)    # Then spatial
        return x
```

> CBAM's design philosophy: first decide "which channel to look at," then decide "where in that channel to look" — two-stage filtering, progressively focusing.

---

## VI. Temporal Attention & Scaled Dot-Product Attention

### 6.1 Motivation

In sequential data (text, speech, video frames), understanding the current position may depend on other positions in the past or future:

```text
"I went to the bank to deposit money"
                    ↑
    This "bank" — what does it mean?
    Must combine with "deposit money" to determine → it's a financial bank, not a river bank
```

Temporal attention lets each position **directly** interact with all positions in the entire sequence, unrestricted by distance.

### 6.2 Intuitive Understanding of the QKV Mechanism

The QKV mechanism is inspired by **information retrieval**:

| Matrix | Full Name | Source | Intuitive Analogy | Shape |
|--------|-----------|--------|-------------------|-------|
| $Q$ | Query | Input × $W^Q$ | "What am I looking for?" | $(N, d_k)$ |
| $K$ | Key | Input × $W^K$ | "Tag for each element" | $(N, d_k)$ |
| $V$ | Value | Input × $W^V$ | "Content of each element" | $(N, d_v)$ |

**Analogy**: You're in a library (the sequence) looking for a book —
1. You have a query $Q$ ("machine learning")
2. Each book has tags $K$ (title, keywords)
3. $Q$ computes similarity with each book's $K$ → determines which book is most relevant (attention weights)
4. Based on the weights, extract information from each book's $V$ (actual content) weighted by relevance

### 6.3 Mathematical Derivation

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Step 1: Compute similarity matrix**

$$S = QK^T \in \mathbb{R}^{N \times N}$$

Where $S_{ij}$ represents the similarity between position $i$'s Query and position $j$'s Key.

**Step 2: Scale + Softmax normalization**

$$A = softmax\left(\frac{S}{\sqrt{d_k}}\right)$$

Produces the attention weight matrix, with each row summing to 1.

**Step 3: Weighted aggregation**

$$O = AV \in \mathbb{R}^{N \times d_v}$$

Each position's output is the weighted sum of all positions' Values.

#### Why Divide by $\sqrt{d_k}$?

Assume each element of $Q$ and $K$ is i.i.d. with mean 0 and variance 1:

$$QK^T = \sum_{i=1}^{d_k} q_i k_i$$

The dot product has mean 0 and variance $d_k$. When $d_k$ is large (e.g., 512), the dot product values can range from tens to hundreds, pushing softmax into its saturation region (gradients approach 0).

Dividing by $\sqrt{d_k}$ normalizes the variance back to 1, keeping gradients stable.

#### Full Implementation

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
        # Step 1 & 2: Similarity + scaling
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Optional: mask out future positions (decoder autoregressive) or padding
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 2: Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Step 3: Weighted aggregation
        output = torch.matmul(attn_weights, V)

        return output, attn_weights
```

### 6.4 Complexity Analysis

| Component | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| $QK^T$ matrix multiply | $O(N^2 d_k)$ | $O(N^2)$ |
| Softmax | $O(N^2)$ | $O(N^2)$ |
| $A V$ matrix multiply | $O(N^2 d_v)$ | $O(N d_v)$ |

The bottleneck is $O(N^2)$ — this is the fundamental challenge for long sequences, driving optimizations like Sparse Attention and Flash Attention.

---

## VII. Multi-Head Attention

### 7.1 Motivation — One Head Is Not Enough

Single-head attention can only learn one type of "attention pattern." Multi-head attention lets the model simultaneously attend from **multiple different subspaces**, analogous to using multiple convolution kernels to extract different features in CNNs.

$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O$$

$$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

### 7.2 Implementation

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # Combine QKV projections into one large matrix for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, x, mask=None):
        B, N, D = x.shape

        # Compute Q, K, V in one go
        qkv = self.qkv_proj(x)  # (B, N, 3*D)
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Attention computation
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(attn_output), attn_weights
```

### 7.3 What Does Each Head Learn?

| Head | Possible Focus | Example |
|------|---------------|---------|
| Head 1 | Syntactic dependencies | Subject → Predicate |
| Head 2 | Coreference | Pronoun → Antecedent |
| Head 3 | Local context | Adjacent words |
| Head 4 | Long-range semantics | Paragraph opening and closing |

> The multi-head mechanism is key to Transformer's success — different heads naturally differentiate into distinct "attention strategies" without manual design.

---

## VIII. Self-Attention and Its Relationship to Other Attention Types

### 8.1 Classification by QKV Source

| Attention Type | Q Source | K Source | V Source | Application |
|---------------|----------|----------|----------|-------------|
| **Self-Attention** | Own sequence | Own sequence | Own sequence | Encoder, GPT |
| **Cross-Attention** | Target sequence | Source sequence | Source sequence | Decoder (translation) |
| **Masked Attention** | Own sequence | Own sequence (mask future) | Own sequence | GPT/Decoder autoregressive |

### 8.2 Core Advantages of Self-Attention

| Capability | Description |
|-----------|-------------|
| **Global Dependencies** | Any two positions directly connected, no layer-by-layer propagation needed |
| **Dynamic Weights** | Weights determined by input content, not fixed (vs. convolution kernels) |
| **Parallel Computation** | All positions computed simultaneously, no dependency on previous results (vs. RNN) |
| **Interpretability** | Attention weights are visualizable, intuitively showing the model's "focus" |
| **Length Flexibility** | Handles variable-length sequences with no fixed input constraint |

### 8.3 Limitations of Self-Attention

| Limitation | Cause | Solution |
|-----------|-------|----------|
| $O(N^2)$ complexity | Every position attends to every position | Sparse / Linformer / Flash Attention |
| Lack of position information | Fully connected, order-agnostic | Positional Encoding / Embedding |
| Large data requirements | No inductive bias (unlike CNN's locality prior) | Pretrain + fine-tune paradigm |

---

## IX. Evolution of Attention Mechanisms

```text
2014 — Bahdanau Attention (RNN translation alignment)
  │
2015 — Luong Attention (global + local attention variants)
  │
2017 — Self-Attention + Multi-Head Attention (Transformer is born)
  │
2018 — SE-Net (channel attention explodes in CV)
  │     CBAM (channel + spatial fusion)
  │
2019 — Sparse Attention (Sparse Transformer, Longformer)
  │
2020 — Vision Transformer (ViT) (pure attention for image classification)
  │     ECA-Net (efficient channel attention)
  │
2021 — Swin Transformer (hierarchical attention, dominant CV performance)
  │
2022 — Flash Attention (IO-aware fast attention)
  │
2023 — Flash Attention v2 / Ring Attention (longer sequences)
```

---

## X. Summary Comparison

| Dimension | Channel Attention | Spatial Attention | CBAM | Self-Attention |
|-----------|-------------------|-------------------|------|---------------|
| Attention Domain | Channel | Spatial position | Channel → Spatial | Token relationships |
| Weight Shape | $(C, 1, 1)$ | $(1, H, W)$ | Both | $(N, N)$ |
| Core Operation | Global pooling + FC | Dual pooling + Conv | SE + Spatial | QK^T + Softmax |
| Complexity | $O(C^2/r)$ | $O(HW)$ | $O(C^2/r + HW)$ | $O(N^2 d)$ |
| Representative Application | EfficientNet | STN, DCN | Plug-and-play | GPT, ViT |
| Applicable Data | Image | Image | Image | Sequence/Image |

---

*Related notes: [What is a Neural Network](../具身智能感知基础：机器学习与深度学习/What_is_neural_network.md), [What is a Convolutional Network](../具身智能感知基础：机器学习与深度学习/What_is_Convolution_network.md), [Computer Vision & Image Super-Resolution](../具身智能感知基础：机器学习与深度学习/Computer_Vision_&_Image_Super-resolution.md)*
