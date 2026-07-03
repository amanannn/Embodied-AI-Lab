# Computer Vision & Image Super-Resolution

> Xidian University · Embodied AI Micro-Major
> Course Notes

---

## I. Neural Network Optimization Overview

Before diving into super-resolution, let's review the three dimensions of neural network optimization:

| Dimension | Content | Representatives |
|-----------|---------|-----------------|
| Architecture | Network topology | Fully Connected, CNN, Transformer, Mamba |
| Paradigm | Training and generation strategies | GAN, Diffusion |
| Module | Reusable computation units | Pooling, 1×1 Conv, DCN, Residual, Attention |

> These three dimensions are not isolated — an excellent model is typically a combination of good architecture, suitable paradigm, and well-designed modules.

---

## II. Image Super-Resolution Problem Definition

### 2.1 Basic Concept

The goal of Image Super-Resolution (SR): **restore a high-resolution image from a low-resolution input**.

$$I_{SR} = f_\theta(I_{LR})$$

Where $I_{LR}$ is the low-resolution input, $f_\theta$ is the super-resolution network, and $I_{SR}$ is the reconstructed high-resolution image.

### 2.2 Overall Framework

```
LR Image (input) → Super-Resolution Network → SR Image (output)
                                                    ↓
                                              Compute Loss (MSE / L1 / SSIM)
                                                    ↓
HR Image (ground truth) ←——— Backpropagation ←———
```

> Development has largely focused on optimizing network structure. From three-layer convolutions to Transformers, it is essentially about designing a better $f_\theta$.

---

## III. Evaluation Metrics

### 3.1 Objective Metrics

| Metric | Full Name | Meaning | Range |
|--------|-----------|---------|-------|
| MSE | Mean Squared Error | Per-pixel squared error | Lower is better |
| PSNR | Peak Signal-to-Noise Ratio | Peak signal-to-noise ratio | Higher is better, 0 ~ +∞ dB |
| SSIM | Structural Similarity | Structural similarity | 0 ~ 1, higher is better |

$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)$$

The lower the MSE, the higher the PSNR, indicating less difference from the original image.

### 3.2 SSIM vs PSNR

SSIM mimics human perception by focusing on **edge and texture similarity**:

| Metric | Focus | Alignment with Human Vision |
|--------|-------|---------------------------|
| PSNR | Pixel-level brightness differences | Moderate |
| SSIM | Edge and texture structure | Better |

> Humans are insensitive to absolute pixel brightness and color, but highly sensitive to edge and texture positions. SSIM aligns better with human visual intuition than PSNR.

### 3.3 Subjective Metrics

| Metric | Full Name | Method |
|--------|-----------|--------|
| MOS | Mean Opinion Score | Human raters score test samples |

---

## IV. Loss Functions

| Loss Function | Characteristics | Use Case |
|--------------|-----------------|----------|
| MSE Loss | Optimizes PSNR, may over-smooth | Pursuing numerical metrics |
| L1 Loss | More robust to outliers, preserves more high frequencies | General choice |
| SSIM Loss | Directly optimizes structural similarity | Pursuing perceptual quality |

---

## V. Development History

The evolution of super-resolution network architectures moves from "simple and shallow" to "deep attention and generative paradigms."

### 5.1 Founding Era: SRCNN to VDSR

| Model | Year | Core Innovation | Key Technique |
|-------|------|----------------|---------------|
| **SRCNN** | 2015 | First CNN-based SR model | Three-layer conv: patch extraction → nonlinear mapping → reconstruction |
| **FSRCNN** | 2016 | Accelerated SRCNN | Smaller kernels replace large ones; transposed convolution upsampling |
| **VDSR** | 2016 | Deeper networks | Global residual connection enables deep network training |

**Interpretation of SRCNN's Three Layers:**

| Layer | Function | Description |
|-------|----------|-------------|
| Layer 1 | Patch extraction & feature representation | Maps input image to high-dimensional feature space |
| Layer 2 | Nonlinear feature mapping | Learns nonlinear LR-to-HR relationships |
| Layer 3 | Feature reconstruction | Maps high-dimensional features back to image space |

### 5.2 Deepening Era: The Power of Residual Learning

| Model | Core Innovation |
|-------|----------------|
| **LapSRN** | Pyramid architecture, simultaneously outputs ×2, ×4, ×8 SR results |
| **EDSR** | Local + global residual connections, removes BN layers, further deepens the network |

> EDSR discovered that Batch Normalization is harmful in super-resolution — it erases dynamic range information from images. This finding influenced the design of all subsequent SR models.

### 5.3 Attention Mechanism Era

| Model | Core Innovation |
|-------|----------------|
| **RCAN** | First to introduce channel attention, enabling the network to adaptively optimize inter-channel relationships |

Channel attention: gives higher weight to important channels while suppressing irrelevant ones.

### 5.4 Architectural Revolution: Transformers & Generative Paradigms

| Model | Architecture | Core Innovation |
|-------|-------------|----------------|
| **SwinIR** | Transformer | First to introduce Swin Transformer to image SR |
| **SRGAN** | GAN | First to use GANs for super-resolution, pursuing perceptual quality |
| **SR3** | Diffusion | Diffusion model for super-resolution |

---

## VI. Key Insights

> The history of image super-resolution essentially answers three questions:
>
> 1. **How to go deeper?** — Residual connections (VDSR → EDSR)
> 2. **How to focus on what matters?** — Attention mechanisms (RCAN → SwinIR)
> 3. **How to be more realistic?** — Generative methods (SRGAN → SR3 → Diffusion)

---

*Related notes: [What is a Convolutional Network](./What_is_Convolution_network.md), [Commonly Used Convolution Operators](./Commonly_used_convolution_operators.md)*
