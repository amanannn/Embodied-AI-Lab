# What is a Convolutional Network (CNN)

> Xidian University · Embodied Intelligence Micro-Major
> Course Notes

## 1. Starting from 1D Convolution

### 1.1 Discrete 1D Convolution

Convolution is the core operation in signal processing. The discrete 1D convolution is defined as:

$$y[t] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[t-k]$$

Intuitively: **the kernel $h$ slides over the input signal $x$, performing element-wise multiplication and summation at each position**.

### 1.2 The Meaning of Convolution

> Convolution extracts feature information that captures a certain characteristic of the signal.

Different kernels extract different features — edges, textures, corners, or higher-level semantic patterns.

## 2. 2D Convolution

Images are 2D signals, and convolution kernels are also 2D:

$$\text{output}(i,j) = \sum_{m}\sum_{n} \text{input}(i+m, j+n) \cdot \text{kernel}(m, n)$$

| Hyperparameter | Meaning |
|----------------|---------|
| Kernel Size | The neighborhood size involved in each computation, e.g., 3×3, 5×5 |
| Stride | The number of pixels the kernel moves each step |

From **hand-designed kernels** (Sobel, Laplacian) to **learnable convolution kernels** — this is the core leap from traditional image processing to deep learning.

## 3. Why Convolution Instead of Fully Connected for Images?

If we flatten an image into a 1D vector and feed it to a fully connected network:

| Problems of FC Networks | How CNNs Solve Them |
|-------------------------|---------------------|
| Exploding parameter count (224×224×3 ≈ 150k input neurons) | Shared kernel parameters — count independent of input size |
| Loss of spatial information (neighborhood relationships flattened away) | Convolution naturally preserves 2D spatial structure |
| No translation invariance | Kernel slides across the entire image — insensitive to translation |

Core advantages of CNNs:

- **Sparse Connectivity**: Each neuron connects only to a local region (local features); deeper layers compose global features.
- **Parameter Sharing**: The same kernel is reused across the entire image.
- **Translation Invariance**: A feature is detected regardless of where it appears.

## 4. Classic CNN Architecture Evolution

### 4.1 LeNet-5 (1998)

The pioneering architecture for handwritten digit recognition. Structure: Conv → Pool → Conv → Pool → FC → Output.

### 4.2 AlexNet (2012)

ImageNet competition winner, bringing deep CNNs into the mainstream. Key improvements: ReLU activation, Dropout, GPU training.

### 4.3 VGG (2014)

Core insight: **stack multiple small kernels (3×3) instead of one large kernel**.

> Two stacked 3×3 convolutions = receptive field of a single 5×5 convolution, but with fewer parameters and more nonlinearity.

### 4.4 GoogLeNet / Inception (2014)

Introduced the **Inception module**: run multiple kernel sizes (1×1, 3×3, 5×5) and pooling in parallel within the same layer, fusing multi-scale feature information.

```
Input → ┬─ 1×1 Conv ───→
        ├─ 3×3 Conv ───→  Concatenate → Output
        ├─ 5×5 Conv ───→
        └─ 3×3 MaxPool →
```

### 4.5 ResNet (2015)

**Core problem**: As networks deepen, vanishing/exploding gradients make training difficult.

| Problem | Symptom |
|---------|---------|
| Vanishing Gradient | Shallow-layer gradients approach 0; parameters barely update |
| Exploding Gradient | Shallow-layer gradients explode; updates oscillate wildly |

**Residual Block**:

$$y = \mathcal{F}(x) + x$$

> Learning 100 → 101 is hard, but learning 0 → 1 is easy. The skip connection lets the network learn only the "residual."

Effects: solved deep network training, accelerated convergence, enabled shallow feature reuse.

## 5. Applications of CNNs

| Level | Task | Representative Methods |
|-------|------|----------------------|
| Low-Level | Super-resolution | SR3 |
| Low-Level | Denoising | FastDVDnet |
| High-Level | Object Detection | YOLO |
| High-Level | Semantic Segmentation | DeepLab |

## 6. CNN Computation Example

Take a simple network:

```
Input: 3×3×1 (H×W×C)
  ↓ Conv layer: 64 filters of 3×3×1 → W[64, 1, 3, 3], B[64]
Output feature map: 3×3×64
```

More generally (with padding to preserve dimensions):

```
Input [1, 5, 5] (C=1, H=5, W=5)
  ↓ Conv1: in=1, out=16, kernel=3, padding=1 → W[16, 1, 3, 3], B[16]
Output [16, 5, 5]
  ↓ Conv2: in=16, out=N, kernel=3, padding=1 → W[N, 16, 3, 3], B[N]
Output [N, 5, 5]
```

> Feature map dimension convention: [channels, height, width]

**Key takeaway**: For large 2D matrices, small-kernel CNNs vastly outperform fully connected networks — controllable parameter count, preserved spatial structure, hierarchical features.

---

*Note status: First draft complete. Pending step-by-step computation examples for 1D and 2D convolution.*

*Related notes: [Commonly Used Convolution Operators](./Commonly_used_convolution_operators.en.md), [What is a Neural Network](./What_is_neural_network.en.md)*
