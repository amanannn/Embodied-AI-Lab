# How to Build and Train a Neural Network (Part 2): CNN Practice and Generative Adversarial Networks

> Xidian University · Embodied Intelligence Micro-Major
> Course Notes

---

## 1. CNN Principles Review

### 1.1 How Does a Machine "See" an Image?

```text
What humans see:               What computers see:
┌──────────┐            [[[ 0,  0,255],
│   🌞    │               [255,128, 0],
│   ☁️    │               [ 0,255,128],
│  🏠 🌲  │               ...
└──────────┘             [128, 0,255]]]
                         Shape: (3, H, W), Value range: [0, 255]
```

The CNN convolution kernel slides over this numerical matrix, detecting local patterns — edges, corners, textures, and higher-level semantics.

### 1.2 Convolution Output Size Calculation

$$H_{out} = \left\lfloor\frac{H_{in} + 2P - K}{S}\right\rfloor + 1$$

| Parameter | Meaning | Common Values |
|-----------|---------|---------------|
| $K$ | kernel_size | 3, 5, 7 |
| $P$ | padding | $K/2$ floor (to keep spatial size unchanged) |
| $S$ | stride | 1 (feature extraction), 2 (downsampling) |

```python
# "same" padding: output size = input size / stride
# kernel_size=3 → requires padding=1
# kernel_size=5 → requires padding=2
conv_same = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
# Input (B, 64, 32, 32) → Output (B, 128, 32, 32)  spatial size unchanged
```

### 1.3 Core Principles of Building a CNN

1. **Channel count**: Spatial size halves layer by layer (pooling), channel count doubles layer by layer — preserving total information
2. **Receptive field**: Shallow layers see details (edges/textures), deep layers see the global picture (semantics/objects)
3. **Skip connections**: The core of ResNet — allowing gradients to flow directly through deep layers

---

## 2. Data Preparation

### 2.1 DataLoader Parameters

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,      # Samples per batch
    shuffle=True,        # Shuffle for training, not for testing
    num_workers=2,       # Multi-process loading
    pin_memory=True,     # Accelerate data transfer for GPU training
    drop_last=True       # Drop the last incomplete batch (BN requires batch≥2)
)
```

| Parameter | Impact | Suggestion |
|-----------|--------|------------|
| `batch_size` | Large → stable gradients but more memory; Small → more noise but better generalization | 32/64/128 (adjust based on VRAM) |
| `num_workers` | Too few → GPU waits for data; Too many → CPU memory overflow | 4~8 (typically half the number of CPU cores) |
| `pin_memory` | Accelerates CPU→GPU transfer | Enable for GPU training |

### 2.2 Data Preprocessing Pipeline

```python
# Training augmentation pipeline
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),        # Random crop (CIFAR standard)
    transforms.RandomHorizontalFlip(p=0.5),      # Random horizontal flip
    transforms.ColorJitter(0.1, 0.1, 0.1),       # Color jitter
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))  # CIFAR-10 statistics
])

# No augmentation for testing
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
```

> `Normalize(mean, std)` standardizes data to $N(0,1)$ — inputs with mean 0 and variance 1 are most friendly for gradient propagation.

---

## 3. Network Construction

### 3.1 Convolutional Layer Parameter Selection

```python
# Standard convolution block
conv_block = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Spatial size unchanged
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),                                # Spatial size halved
)
```

**Parameter count calculation**:
$$Params_{conv} = K \times K \times C_{in} \times C_{out} + C_{out}$$

```python
# Quick parameter count for all layers
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Trainable parameters: {count_params(model):,}')
```

### 3.2 Fully Connected Layers

```python
# FC layers are parameter-heavy — use with caution
fc = nn.Linear(320, 10)
# params = 320 × 10 + 10 = 3,210
```

> Convolution parameter count is independent of input size (weight sharing), while fully connected parameter count grows linearly with input size — this is why global average pooling (GAP) is typically used before the CNN classification head to eliminate spatial dimensions.

### 3.3 Complete MNIST Network (Layer-by-Layer Analysis)

```python
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)      # No padding
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 1×28×28→10×12×12
        x = self.pool(F.relu(self.conv2(x)))   # 10×12×12→20×4×4
        x = x.view(x.size(0), -1)              # 20×4×4→320
        return self.fc(x)                       # 320→10
```

| Layer | Input Shape | Output Shape | Parameters | Calculation |
|-------|-------------|--------------|------------|-------------|
| Conv1 (1→10, k=5) | (1, 28, 28) | (10, 24, 24) | 260 | 1×10×5×5 + 10 |
| Pool (k=2) | (10, 24, 24) | (10, 12, 12) | 0 | — |
| Conv2 (10→20, k=5) | (10, 12, 12) | (20, 8, 8) | 5,020 | 10×20×5×5 + 20 |
| Pool (k=2) | (20, 8, 8) | (20, 4, 4) | 0 | — |
| Flatten | (20, 4, 4) | (320) | 0 | — |
| FC (320→10) | (320) | (10) | 3,210 | 320×10 + 10 |
| **Total** | | | **8,490** | — |

---

## 4. Training Loop (Complete Example)

```python
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()               # 1. Clear previous gradients
        outputs = model(inputs)             # 2. Forward pass
        loss = criterion(outputs, target)   # 3. Compute loss
        loss.backward()                     # 4. Backward pass
        optimizer.step()                    # 5. Update weights

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(f'[Epoch {epoch}, Batch {batch_idx+1:5d}] '
                  f'loss: {running_loss / 300:.3f}')
            running_loss = 0.0
```

---

## 5. Deep Learning Hardware

### 5.1 Precision Requirements for Training vs Inference

| Stage | Precision Requirement | Reason | Typical Format |
|-------|----------------------|--------|----------------|
| **Training** | **FP32** and above | Gradient differences are tiny; accumulated errors are fatal | FP32, TF32 |
| **Fine-tuning** | FP16/FP32 mixed | Some critical layers retain FP32 | Mixed Precision |
| **Inference** | FP16 or INT8 | Relatively lower precision requirement | FP16, INT8, INT4 |

### 5.2 Mixed Precision Training

```python
# PyTorch Automatic Mixed Precision (AMP) — nearly free speed boost
scaler = torch.amp.GradScaler()

with torch.amp.autocast('cuda'):
    outputs = model(inputs)           # Automatically computed in FP16
    loss = criterion(outputs, target)

scaler.scale(loss).backward()         # Scale up loss to prevent gradient underflow
scaler.step(optimizer)                # Automatically unscale + update
scaler.update()
```

### 5.3 GPU Key Parameters

| Parameter | Meaning | Impact on Deep Learning |
|-----------|---------|------------------------|
| **CUDA Cores** | General parallel computing units | General matrix operations |
| **Tensor Cores** | Matrix multiplication dedicated accelerators | **Workhorse for training and inference** |
| **VRAM Capacity** | Determines max model + batch_size | Primary bottleneck for OOM |
| **VRAM Bandwidth** | Data read/write speed | Bottleneck for large model inference |
| **FLOPS** | Theoretical peak compute power | Upper bound reference; actual utilization 30-70% |

### 5.4 Deployment Optimization Techniques

| Technique | Principle | Effect | Tools |
|-----------|-----------|--------|-------|
| **Quantization** | FP32 → INT8/FP16 | Model shrinks 4×, speedup 2-4× | PyTorch Quantization |
| **Pruning** | Remove unimportant weights/channels | Reduce parameters by 30-90% | `torch.nn.utils.prune` |
| **Distillation** | Large model teaches small model | Small model approaches large model accuracy | Custom training |
| **TensorRT** | Operator fusion + kernel tuning | Inference speedup 2-5× | NVIDIA TensorRT |
| **ONNX** | Cross-framework intermediate representation | Export once, deploy everywhere | `torch.onnx.export` |

---

## 6. Generative Adversarial Networks (GAN)

### 6.1 What is a Generative Model?

**Discriminative model** $P(y|x)$: Given input, determine the category → classification, detection
**Generative model** $P(x)$ or $P(x|y)$: Learn data distribution, generate new samples → image generation, style transfer

### 6.2 Core Idea of GAN

> **The game between counterfeiter and police**:
>
> - **Generator G** (counterfeiter): Generates fake samples from random noise, trying to pass them off as real
> - **Discriminator D** (police): Determines whether the input is a real sample or a fake one
> - The two are trained alternately, and eventually G generates fake samples convincing enough to fool D

### 6.3 Mathematical Formulation

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim P_{data}}[\log D(x)] + \mathbb{E}_{z \sim P_z}[\log(1 - D(G(z)))]$$

| Term | Meaning | Who Optimizes |
|------|---------|---------------|
| $\mathbb{E}[\log D(x)]$ | The closer D's judgment on **real samples** is to 1, the better | D maximizes |
| $\mathbb{E}[\log(1-D(G(z)))]$ | The closer D's judgment on **fake samples** is to 0, the better | D maximizes; G minimizes (making D misclassify) |

### 6.4 GAN Training Procedure

```python
# One GAN training step
for real_images in dataloader:
    batch_size = real_images.size(0)

    # ── Train Discriminator D ──
    # Real samples: D should output close to 1
    d_real = D(real_images)
    d_real_loss = F.binary_cross_entropy(d_real, torch.ones(batch_size, 1))

    # Fake samples: D should output close to 0
    z = torch.randn(batch_size, latent_dim)
    fake_images = G(z)
    d_fake = D(fake_images.detach())  # detach! Do not backpropagate gradients to G
    d_fake_loss = F.binary_cross_entropy(d_fake, torch.zeros(batch_size, 1))

    d_loss = d_real_loss + d_fake_loss
    d_optimizer.zero_grad(); d_loss.backward(); d_optimizer.step()

    # ── Train Generator G ──
    # G wants D to classify fake samples as real
    z = torch.randn(batch_size, latent_dim)
    fake_images = G(z)
    d_output = D(fake_images)
    g_loss = F.binary_cross_entropy(d_output, torch.ones(batch_size, 1))

    g_optimizer.zero_grad(); g_loss.backward(); g_optimizer.step()
```

### 6.5 Common Problems in GAN Training and Their Solutions

| Problem | Manifestation | Cause | Solution |
|---------|---------------|-------|----------|
| **Mode Collapse** | G generates only a few types of samples | G finds a "shortcut" to fool D | Minibatch Discrimination, WGAN |
| **Training Instability** | Loss oscillates violently | Imbalance between D and G | Adjust D/G update ratio, spectral normalization |
| **Vanishing Gradients** | G loss does not decrease | D is too strong, G has no gradient signal | LSGAN (least squares loss replaces BCE) |
| **Difficult to Evaluate** | Lack of objective metrics | Subjective generation quality | FID, IS (Inception Score) |

#### Improvements in WGAN

WGAN uses the **Wasserstein distance** instead of JS divergence, fundamentally alleviating training instability:

$$W(P_r, P_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)]$$

Core changes: Remove Sigmoid from D, use weight clipping (or gradient penalty), replace Adam with RMSprop.

### 6.6 The GAN Family

| Variant | Full Name | Core Improvement | Representative Application |
|---------|-----------|-----------------|---------------------------|
| **DCGAN** | Deep Convolutional GAN | CNN replaces fully connected + BatchNorm | Image generation backbone architecture |
| **cGAN** | Conditional GAN | Add conditional labels to control generation | Class-conditional generation |
| **Pix2Pix** | — | Paired data + U-Net + PatchGAN | Image translation (sketch→photo) |
| **CycleGAN** | — | Cycle consistency loss, no paired data needed | Style transfer (horse→zebra) |
| **StyleGAN** | — | Style vectors control generation attributes | High-resolution face generation |
| **SRGAN** | Super-Resolution GAN | Perceptual loss + adversarial loss | Image super-resolution |
| **BigGAN** | — | Large batches + large model + spectral normalization | High-quality ImageNet generation |

### 6.7 GAN vs Diffusion

| Dimension | GAN | Diffusion Model |
|-----------|-----|-----------------|
| Generation method | One step from noise to image | Gradual denoising (multiple steps) |
| Training difficulty | High (game-theoretic balance) | Relatively low (regression task) |
| Generation quality | High (good details) | Very high (strong diversity) |
| Inference speed | Fast (single step) | Slow (10-1000 steps) |
| Mode coverage | May collapse | Full coverage (good diversity) |

---

*Related notes: [Part 1: Training Loop & Loss Functions](./How_to_Build_and_Train_a_Neural_Network_p1.md), [What is a Convolutional Network](../具身智能感知基础：机器学习与深度学习/What_is_Convolution_network.md), [Commonly Used Convolution Operators](../具身智能感知基础：机器学习与深度学习/Commonly_used_convolution_operators.md)*
