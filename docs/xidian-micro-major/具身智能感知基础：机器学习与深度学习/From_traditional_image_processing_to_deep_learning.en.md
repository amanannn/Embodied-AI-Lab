# From Traditional Image Processing to Deep Learning

> Xidian University · Embodied Intelligence Micro-Major
> Course Notes

## 1. Digital Images: Where Everything Starts

A digital image is essentially a **large 2D array** — each element is a pixel value.

| Image Type | Pixel Range | Channels | Bit Depth |
|-----------|-------------|----------|-----------|
| Binary | 0 or 1 | 1 | 1 bit |
| Grayscale | 0–255 | 1 | 8 bit |
| RGB Color | (R, G, B) each 0–255 | 3 | 24 bit |

## 2. The Traditional Image Processing Pipeline

A typical traditional image processing workflow:

```
Raw Image → Enhancement/Denoising → Edge Detection → Segmentation → Feature Extraction → Downstream Task
```

### 2.1 Image Enhancement

Enhancement aims to **amplify useful information and suppress noise**.

| Method | Principle | Typical Use |
|--------|-----------|-------------|
| Histogram Equalization | Redistribute pixel intensities to stretch contrast | Correcting under/over-exposed images |
| Spatial Filtering | Weighted operations directly on pixel neighborhoods | Denoising, smoothing, sharpening |
| Frequency-Domain Filtering | Operate in frequency domain after Fourier Transform | Removing periodic noise |

### 2.2 Denoising and Smoothing

Noise is unwanted random variation in an image. Common noise types:

| Noise Type | Characteristics | Source |
|-----------|----------------|--------|
| Salt & Pepper | Random black/white pixels | Sensor defects, transmission errors |
| Gaussian | Pixel values perturbed by a normal distribution | Sensor thermal noise, low-light conditions |
| Poisson | Signal-dependent shot noise | Photon counting statistics |

Smoothing filters suppress noise at the cost of detail loss (edge blurring).

| Filter | Principle | Characteristics |
|--------|-----------|----------------|
| Mean Filter | Average of neighboring pixels | Simple, but blurs edges heavily |
| Gaussian Filter | Weighted average (closer pixels matter more) | Preserves more edges than mean |
| Median Filter | Median of neighboring pixels | Best for salt & pepper noise |
| Bilateral Filter | Considers both spatial distance and intensity difference | Denoises while preserving edges |

### 2.3 Edge Detection (Sharpening)

Edges are regions where pixel values change sharply — fundamental cues for image understanding.

| Operator | Direction | Kernel Example |
|----------|-----------|----------------|
| Sobel | Horizontal edges | $G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$ |
| Sobel | Vertical edges | $G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$ |
| Laplacian | Isotropic | $\begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$ |

- **Sobel**: First-order derivative — detects edge direction and strength.
- **Laplacian**: Second-order derivative — responds strongly to intensity changes, but is noise-sensitive.

### 2.4 Image Segmentation

Partitioning an image into meaningful regions.

| Method | Principle | Limitation |
|--------|-----------|------------|
| Thresholding | Separate foreground/background by intensity | Sensitive to lighting changes |
| Edge-Based | Close edge contours into regions | Fails when edges are discontinuous |

### 2.5 Feature Extraction

Good features should be **representative** and **discriminative** — similar for same-class instances, distinct across classes.

| Feature | Full Name | Characteristics |
|---------|-----------|----------------|
| SIFT | Scale-Invariant Feature Transform | Scale- and rotation-invariant; computationally expensive |
| HOG | Histogram of Oriented Gradients | Robust to illumination and geometric changes; widely used for pedestrian detection |

## 3. The Core Mechanism: Filters and Convolution

Both traditional image processing and deep learning share the same fundamental operation — **convolution**:

$$\text{output}(i,j) = \sum_{m}\sum_{n} \text{input}(i+m, j+n) \cdot \text{kernel}(m, n)$$

A filter (convolution kernel) slides across the entire image, performing element-wise multiplication and summation at each position.

## 4. From Traditional to Deep: The Fundamental Difference

| Dimension | Traditional Image Processing | Deep Learning (CNN) |
|-----------|------------------------------|---------------------|
| **Filter Origin** | Hand-designed, based on mathematical/physical intuition | Learned automatically from data |
| **Design Cost** | High — requires expert redesign per task | Low — unified framework, swap the data |
| **Feature Hierarchy** | Single layer, shallow features (edges, corners) | Multi-layer: edges → textures → parts → semantics |
| **Generalization** | Task-specific; re-tuning required for new scenarios | Learns universal representations from large samples |
| **Interpretability** | High — every step has clear physical meaning | Low — learned kernels are "black box" representations |

**The core shift**:

> In traditional methods, every filter is hand-designed and manually computed.  
> In deep learning, convolution kernels are learned automatically through **forward propagation + backpropagation + gradient descent** — given large amounts of data, the network learns deep, hierarchical feature representations.

This is why deep learning revolutionized computer vision after ImageNet: hand-crafted features (SIFT, HOG) have an inherent ceiling, while data-driven feature learning does not.

## 5. From the Course Perspective

What this lecture means for embodied intelligence:

- Robots capture digital images through cameras → images must be processed to become useful information.
- Traditional methods are lightweight and interpretable — suitable for resource-constrained embedded scenarios.
- Deep learning methods are powerful and form the core of modern perception systems.
- These two paths are **not in opposition** — the underlying principles of filters are **shared**. Understanding traditional methods is essential to understanding why CNNs work.

---

*Note status: First draft complete. Pending experiment code cross-references.*

*Related notes: [Embodied Intelligence Perception Fundamentals: Machine Vision and Deep Learning](./具身智能感知基础：机器视觉与深度学习.en.md), [What is a Neural Network](./What_is_neural_network.en.md), [What is a Convolutional Network](./What_is_Convolution_network.en.md), [Commonly Used Convolution Operators](./Commonly_used_convolution_operators.en.md)*
