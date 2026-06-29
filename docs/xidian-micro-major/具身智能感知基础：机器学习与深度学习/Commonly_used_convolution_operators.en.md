# Commonly Used Convolution Operators

> Xidian University · Embodied Intelligence Micro-Major
> Course Notes

## 1. What is a Convolution Operator?

A convolution operator is the fundamental computational unit in CNNs for **feature extraction and representation learning**.

Traditional 3×3 convolution has drawbacks: limited functionality, relatively large parameter count, and small receptive field. Various convolution variants optimize performance and efficiency across different dimensions by altering the computation pattern.

## 2. Operator Overview

| Operator | Core Role | Key Feature |
|----------|-----------|-------------|
| 1×1 Convolution | Channel dimension transformation | Reduce/expand channels; spatial size unchanged |
| Pooling | Downsampling | Reduce resolution, enlarge receptive field |
| Transposed Convolution | Upsampling | Recover large resolution from small input |
| Group Convolution | Reduce parameters | Independent group-wise convolution; 1/g parameters |
| Depthwise Separable | Lightweight design | Per-channel convolution + 1×1 projection |
| Dilated Convolution | Enlarge receptive field | Introduce dilation rate without extra parameters |
| Deformable Convolution | Handle irregular shapes | Learn offsets for receptive field |

## 3. Detailed Explanations

### 3.1 1×1 Convolution

A convolution kernel with a 1×1 receptive field. Operation is identical to regular convolution. **Core use: channel dimensionality reduction or expansion**.

```python
conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
```

### 3.2 Pooling

Changes the spatial size of input features, primarily for **downsampling**.

```python
pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

| Pooling Method | Computation | Characteristics |
|---------------|-------------|----------------|
| Max Pooling | Take the maximum in the neighborhood | Preserves the most salient features |
| Average Pooling | Take the mean of the neighborhood | Smoother; preserves background information |
| Adaptive Pooling | Automatically adjust parameters | Fixed output size regardless of input dimensions |

Benefits of pooling:

- **Feature dimensionality reduction**: Shrink feature map size.
- **Preserve feature invariance**: Minor translations do not affect pooling results.
- **Reduce overfitting**: Lower parameter and computation counts.

> Three states of deep learning: **Fitting** (training well), **Underfitting** (model too simple; fails to learn data patterns), **Overfitting** (performs well on training set, poorly on test set).

### 3.3 Transposed Convolution

Also known as deconvolution. **Purpose: upsampling** — obtaining a large-resolution output from a smaller input. Mathematically equivalent to padding the input first, then applying a regular convolution.

```python
conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
```

### 3.4 Group Convolution

Input and output channels are divided into $g$ groups. Each group's output channels connect only to the corresponding group's input channels. Parameter count is reduced to $1/g$ of the ungrouped version.

```python
conv = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, groups=3)
```

**Parameter count comparison**:

- Standard 3×3 convolution: $3 \times 3 \times 24 \times 48 = 10{,}368$
- Group convolution ($g=3$): $3 \times (3 \times 3 \times 8 \times 16) = 3{,}456$

> Group convolution is a foundational building block of efficient networks like ResNeXt and MobileNet.

### 3.5 Depthwise Separable Convolution

Composed of two stages:

1. **Depthwise Convolution**: Each channel is convolved independently ($g$ = number of input channels)
2. **Pointwise Convolution**: 1×1 convolution projects depthwise outputs to a new feature map

```python
depthwise_conv = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, groups=24)
pointwise_conv  = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=1)
```

**Advantage**: Drastically reduces parameter count and computation while enabling cross-channel information interaction and integration. This is the core idea behind the MobileNet family.

### 3.6 Dilated (Atrous) Convolution

Introduces a **dilation rate** hyperparameter, inserting gaps between kernel elements in the original convolution kernel.

```python
conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2)
```

- dilation=1: standard 3×3 convolution (receptive field 3×3)
- dilation=2: 3×3 convolution with effective receptive field of 5×5
- dilation=3: 3×3 convolution with effective receptive field of 7×7

> **Larger receptive field without extra parameters** — widely used in semantic segmentation (e.g., DeepLab).

### 3.7 Deformable Convolution

Locally deforms the standard convolution to adapt to irregular object shapes.

**Pipeline**:

```
Input Tensor → Conv computes offsets → Bilinear Interpolation → Deformable Conv → Output Tensor
```

**Bilinear Interpolation**: When the target pixel coordinate is not an integer (e.g., (3.7, 5.2)), the value is synthesized as a weighted average of the 4 nearest integer pixels. It is essentially linear interpolation performed in two directions on a 2D grid.

**Benefits**: Adapts to irregular structures, reduces grid bias, and is more robust to geometric deformations.

## 4. Beyond Manual Design: Neural Architecture Search (NAS)

Beyond hand-designed convolution operators, there is **Neural Architecture Search (NAS)** — letting algorithms automatically search for the optimal network structure. This is a core direction in AutoML.

---

*Note status: First draft complete. Each operator includes a PyTorch code example.*

*Related notes: [What is a Convolutional Network](./What_is_Convolution_network.en.md), [From Traditional Image Processing to Deep Learning](./From_traditional_image_processing_to_deep_learning.en.md)*
