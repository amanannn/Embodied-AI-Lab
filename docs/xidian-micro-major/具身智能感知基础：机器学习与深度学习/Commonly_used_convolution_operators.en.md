# Commonly Used Convolution Operators

> Xidian University · Embodied Intelligence Micro-Major
> Course Notes

---

## 1. What Is a Convolution Operator?

A convolution operator is the basic computational unit in CNNs that performs **feature extraction and representation learning**.

Although the traditional $3 \times 3$ convolution is versatile, it suffers from three pain points:

| Pain Point | Problem | Resulting Variants |
|------------|---------|-------------------|
| Large number of parameters | $C_{in} \times C_{out} \times 9$, can reach millions in deep layers | Group convolution, depthwise separable convolution |
| Fixed receptive field | Each layer can only see $3 \times 3$ | Dilated convolution |
| Fixed shape | Only applicable to regular grids | Deformable convolution |

---

## 2. Panorama of Convolution Variants

| Operator | Core Function | Parameter Count (Relative to Standard Convolution) | Key Applications |
|----------|--------------|----------------------------------------------------|-----------------|
| 1×1 Convolution | Channel dimension transformation | $1/9$ | Bottleneck structure, feature fusion |
| Pooling | Downsampling + invariance | 0 | All CNNs |
| Transposed Convolution | Upsampling | Same as standard convolution | Semantic segmentation, GAN |
| Group Convolution | Reduce parameter count | $1/g$ | ResNeXt |
| Depthwise Separable Convolution | Extreme lightweighting | $\approx 1/C_{out}$ | MobileNet |
| Dilated Convolution | Expand receptive field | Same as standard convolution | DeepLab |
| Deformable Convolution | Adapt to irregular shapes | ≈ standard convolution | Detection, segmentation |

---

## 3. Detailed Explanation of Each Operator

### 3.1 1×1 Convolution

Its receptive field is $1 \times 1$, but do not underestimate it — it is the "Swiss Army knife" of CNNs.

#### Mathematical Essence

It performs a linear transformation on the $C_{in}$-dimensional vector at each pixel position:

$$y_{i,j} = W \cdot x_{i,j} + b$$

where $W \in \mathbb{R}^{C_{out} \times C_{in}}$, which is completely equivalent to applying a fully connected layer independently at each position.

#### Core Uses

| Use Case | Input Channels | Output Channels | Effect |
|----------|---------------|----------------|--------|
| Dimensionality reduction | 512 | 256 | Reduces parameter count of subsequent layers |
| Dimensionality increase | 256 | 512 | Increases feature capacity |
| Cross-channel information fusion | Any | Any | Interaction between features of different channels |

```python
# ResNet Bottleneck: 1×1 reduction → 3×3 convolution → 1×1 expansion
bottleneck = nn.Sequential(
    nn.Conv2d(256, 64, 1),   # Reduction: 256→64
    nn.Conv2d(64, 64, 3, padding=1),
    nn.Conv2d(64, 256, 1),   # Expansion: 64→256
)
# Parameter count: 256×64 + 64×64×9 + 64×256 = 69,632
# Direct 3×3: 256×256×9 = 589,824  (8.5×)
```

### 3.2 Pooling Layer

#### Three Pooling Methods

| Pooling Method | Computation | Output Value | Characteristics |
|---------------|------------|--------------|-----------------|
| **Max Pooling** | $\max(x_{i:i+k, j:j+k})$ | Maximum value in neighborhood | Preserves textures/edges, invariant to small displacements |
| **Average Pooling** | $\frac{1}{k^2}\sum x$ | Mean of neighborhood | Smooth, preserves background information, global information |
| **Adaptive Pooling** | Automatically computes kernel/stride | — | Fixed output size, independent of input |

```python
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
adaptive = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling → (B, C, 1, 1)
```

#### Output Size Formula

$$H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1$$

#### Three Major Functions of Pooling

| Function | Mechanism | Importance |
|----------|-----------|------------|
| **Dimensionality Reduction** | Reduces spatial size | Lowers computational cost of subsequent layers |
| **Increasing Receptive Field** | Shrinks feature map → each pixel "sees" a larger input range | Deeper features become more global |
| **Translation Invariance** | Small translations do not affect max pooling results | Improves robustness |

### 3.3 Transposed Convolution

#### Why Is It Called "Transposed"?

Standard convolution (matrix form): $y = C x$ ($C$ is the convolution matrix)
Transposed convolution: $y = C^T x$

It is not an inverse operation, but rather the "inverse operation in shape" — the input and output dimensions are swapped.

#### Output Size Formula

$$H_{out} = (H_{in} - 1) \times S - 2P + K$$

#### Checkerboard Artifact

The most common pitfall of transposed convolution:

```text
Cause: When kernel_size is not divisible by stride, the output image
       exhibits unevenly overlapping "checkerboard" artifacts.

Solutions:
1. Set kernel_size to an integer multiple of stride
2. Replace transposed convolution with upsampling (interpolation) + standard convolution
   nn.Upsample(scale_factor=2) → nn.Conv2d(...)
```

```python
# Transposed convolution
deconv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

# Better upsampling scheme (avoids checkerboard artifacts)
upsample = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    nn.Conv2d(64, 32, kernel_size=3, padding=1)
)
```

### 3.4 Group Convolution

#### Principle

Input channels are divided into $g$ groups, each group is convolved independently, and the outputs are concatenated:

```text
Input (24 channels)          Output (48 channels)
┌──────┐  conv  ┌──────┐
│ 1-8  │ ────→  │ 1-16 │   Group 1: 8→16
├──────┤  conv  ├──────┤
│ 9-16 │ ────→  │17-32│   Group 2: 8→16
├──────┤  conv  ├──────┤
│17-24 │ ────→  │33-48│   Group 3: 8→16
└──────┘        └──────┘
```

#### Parameter Count Analysis

| Type | $C_{in}$ | $C_{out}$ | Parameter Count | Formula |
|------|---------|----------|-----------------|---------|
| Standard Convolution | 24 | 48 | 10,368 | $3 \times 3 \times 24 \times 48$ |
| Group Convolution ($g=3$) | 24 | 48 | 3,456 | $g \times 3 \times 3 \times 8 \times 16$ |

The parameter count is reduced to $1/g$ of the original, while groups **do not share information** with each other.

```python
# Group convolution
conv = nn.Conv2d(24, 48, kernel_size=3, groups=3)

# Extreme case: g = C_in (depthwise convolution)
depthwise = nn.Conv2d(24, 24, kernel_size=3, groups=24)
```

### 3.5 Depthwise Separable Convolution

#### Two-Step Decomposition

| Step | Operation | Input→Output | Parameter Count | Function |
|------|-----------|-------------|-----------------|----------|
| **Depthwise** | Per-channel $3 \times 3$ convolution | $C_{in}$ → $C_{in}$ | $C_{in} \times 9$ | Spatial feature extraction |
| **Pointwise** | $1 \times 1$ convolution | $C_{in}$ → $C_{out}$ | $C_{in} \times C_{out}$ | Channel information fusion |

#### Parameter Count Comparison ($C_{in}=24, C_{out}=48$)

| Method | Parameter Count | Computation |
|--------|----------------|-------------|
| Standard $3 \times 3$ Convolution | 10,368 | $3 \times 3 \times 24 \times 48$ |
| Depthwise Separable Convolution | 1,368 | $3 \times 3 \times 24 + 24 \times 48$ |
| **Reduction Ratio** | **86.8%** | — |

```python
# Depthwise separable convolution
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

> MobileNet V1 is entirely built upon depthwise separable convolutions, achieving accuracy close to that of standard convolutions with only 1/9 of the computational cost.

### 3.6 Dilated / Atrous Convolution

#### Principle

"Holes" (zeros) are inserted between kernel elements. The effective receptive field is calculated as:

$$K_{eff} = K + (K - 1) \times (d - 1)$$

| dilation | Effective Receptive Field (K=3) | Visualization |
|----------|-------------------------------|---------------|
| 1 | $3 \times 3$ | Adjacent 9 pixels |
| 2 | $5 \times 5$ | Every other pixel |
| 3 | $7 \times 7$ | Two pixels between samples |
| 4 | $9 \times 9$ | Three pixels between samples |

#### Why Not Use Larger Kernels?

| Scheme | Parameter Count (3 channels→64 channels) | Receptive Field |
|--------|-----------------------------------------|-----------------|
| $3 \times 3$ dilation=1 | $3 \times 3 \times 3 \times 64 = 1,728$ | 3 |
| $5 \times 5$ dilation=1 | $5 \times 5 \times 3 \times 64 = 4,800$ | 5 |
| $3 \times 3$ dilation=2 | $3 \times 3 \times 3 \times 64 = 1,728$ | 5 |

> Dilated convolution achieves a **larger receptive field** with the **same number of parameters** — the core technique of the DeepLab series.

#### Multi-Scale Dilated Convolution (ASPP)

```python
# ASPP: Process in parallel with multiple groups of different dilation rates
class ASPP(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 1)              # 1×1 baseline
        self.conv6 = nn.Conv2d(c_in, c_out, 3, dilation=6, padding=6)
        self.conv12 = nn.Conv2d(c_in, c_out, 3, dilation=12, padding=12)
        self.conv18 = nn.Conv2d(c_in, c_out, 3, dilation=18, padding=18)
        self.pool = nn.AdaptiveAvgPool2d(1)                  # Global context

    def forward(self, x):
        return self.conv1(x) + self.conv6(x) + self.conv12(x) + self.conv18(x) + ...
```

### 3.7 Deformable Convolution (DCN)

#### Motivation

The sampling points of standard convolution form a regular rectangular grid:

```text
Standard 3×3 Sampling Points    DCN 3×3 Sampling Points (after offset)
●  ●  ●                         ●      ●
                                 ●  ●
●  ●  ●                             ●
                                     ●  ●
●  ●  ●                         ●
```

For objects with irregular shapes (curved text, deformed objects), regular sampling cannot align properly.

#### Implementation Pipeline

```
Input → Learn offsets Δp via convolution → Bilinear interpolation → Deformable convolution → Output
```

| Step | Operation |
|------|-----------|
| 1 | An additional convolution learns the $(Δx, Δy)$ offset for each sampling point from the input |
| 2 | Offset sampling positions are usually not integers → bilinear interpolation |
| 3 | Perform standard convolution using interpolated values |

> DCN allows the receptive field to "learn to adapt" to object shapes — significantly improving performance in detection and segmentation tasks.

---

## 4. Output Size Quick Reference

| Operator | Output Size Formula |
|----------|-------------------|
| Standard Convolution | $H_{out} = \lfloor \frac{H_{in} + 2P - K}{S} \rfloor + 1$ |
| Pooling | Same as above |
| Transposed Convolution | $H_{out} = (H_{in} - 1)S - 2P + K$ |
| Dilated Convolution | $K_{eff} = K + (K-1)(d-1)$, then substitute into the standard convolution formula |

---

## 5. Extension: Neural Architecture Search (NAS)

In addition to manually designing convolution operators, **Neural Architecture Search (NAS)** allows algorithms to automatically search for optimal network structures.

| Method | Search Space | Search Strategy | Representative Work |
|--------|-------------|-----------------|---------------------|
| Reinforcement Learning | Layer types, connections | RNN controller + Policy Gradient | NASNet |
| Evolutionary Algorithm | Topological structures | Mutation + crossover + selection | AmoebaNet |
| Differentiable | Continuous relaxation | Gradient descent | DARTS |
| Weight Sharing | Sub-network sampling | Hypernetwork training | ENAS, OFA |

> The three core elements of NAS: **Search Space** (what can be searched) → **Search Strategy** (how to search) → **Performance Evaluation Strategy** (how to judge quality).

---

*Related Notes: [What Is a Convolutional Network](./What_is_Convolution_network.md), [From Traditional Image Processing to Deep Learning](./From_traditional_image_processing_to_deep_learning.md)*
