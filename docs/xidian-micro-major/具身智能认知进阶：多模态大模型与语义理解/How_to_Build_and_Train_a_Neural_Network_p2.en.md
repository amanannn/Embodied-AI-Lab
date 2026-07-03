# How to Build and Train a Neural Network (Part 2): CNN Hands-on and GANs

> Xidian University · Embodied Intelligence Micro-Major
> Course Notes

## 1. CNN Principles Recap

### 1.1 How Does a Machine Read an Image?

A grayscale image = an $H \times W$ 2D array, where each element is a pixel value. CNN kernels slide over this array to extract local features.

### 1.2 Key Points for Building a CNN

1. **Convolutional Layer**: Extract features — only need to specify channels; spatial dimensions are auto-derived from `kernel_size` and `stride`.
2. **Pooling Layer**: Downsample to reduce feature map size.
3. **Fully Connected Layer**: Map convolutional features to classification outputs.

## 2. Data Preparation

### 2.1 DataLoader

```python
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
```

| Parameter | Meaning |
|-----------|---------|
| `dataset` | Dataset object |
| `batch_size` | Number of samples per batch |
| `shuffle` | Whether to shuffle data (`True` for training, `False` for testing) |
| `num_workers` | Number of subprocesses for data loading |

### 2.2 Complete MNIST Data Loading

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./dataset/mnist', train=True,
    download=True, transform=transform
)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(
    root='./dataset/mnist', train=False,
    download=True, transform=transform
)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
```

> `Normalize((0.1307,), (0.3081,))` uses MNIST's global mean and standard deviation. After preprocessing, data follows a standard normal distribution, which benefits training.

## 3. Network Construction

### 3.1 Convolutional Layer

```python
conv_layer = torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1)
```

> In code, only channels matter — output spatial dimensions are auto-derived from `kernel_size` and `stride`.

### 3.2 Fully Connected Layer

```python
self.fc = torch.nn.Linear(320, 10)
```

Parameters: $320 \times 10 = 3200$. For comparison, a single 3×3 conv kernel has far fewer and is shared across the entire image. **Convolution uses dramatically fewer parameters than FC while preserving spatial structure.**

### 3.3 Complete Network Design (MNIST Example)

```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)      # Flatten → feed to FC layer
        x = self.fc(x)
        return x

model = Net()
```

| Layer | Input | Output | Notes |
|-------|-------|--------|-------|
| Conv1 | 1×28×28 | 10×24×24 | 5×5 kernel, stride 1 |
| Pooling | 10×24×24 | 10×12×12 | 2×2 max pooling |
| Conv2 | 10×12×12 | 20×8×8 | 5×5 kernel, stride 1 |
| Pooling | 20×8×8 | 20×4×4 | 2×2 max pooling |
| Flatten | 20×4×4 | 320 | Flatten to 1D |
| FC | 320 | 10 | Fully connected → 10 classes |

## 4. Loss Function and Optimizer

### 4.1 Selection Rationale

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

- **Cross-Entropy Loss**: MNIST is multi-class classification; Softmax + CrossEntropyLoss yields a clean gradient form and fast convergence.
- **SGD (Mini-Batch Gradient Descent, MBGD)**: Uses a small batch to update gradients — balancing BGD's stability with SGD's computational efficiency.

| Parameter | Meaning |
|-----------|---------|
| `model.parameters()` | Model parameters to optimize (weights, biases, etc.) |
| `lr` | Learning Rate |

## 5. Training Loop

```python
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data

        optimizer.zero_grad()           # 1. Clear gradients
        outputs = model(inputs)          # 2. Forward
        loss = criterion(outputs, target) # 3. Compute loss
        loss.backward()                  # 4. Backward
        optimizer.step()                 # 5. Update parameters

        running_loss += loss.item()
        if batch_idx % 300 == 299:       # Print every 300 batches
            print(f'[{epoch+1}, {batch_idx+1:5d}] loss: {running_loss/300:.3f}')
            running_loss = 0.0
```

For training log interpretation, see Part 1 §3.3 — the key is to observe the relative trends of Train Loss and Val Loss.

## 6. Deep Learning Hardware

### 6.1 Precision Requirements: Training vs. Inference

| Phase | Precision Needed | Reason |
|-------|-----------------|--------|
| **Training** | FP32 and above | Gradient updates are tiny; high precision prevents accumulated errors |
| **Inference** | FP16 works; INT8 possible | Precision requirements lower; benefits embedded deployment |

| Precision | Bits | Typical Use |
|-----------|------|-------------|
| FP32 | 32-bit float | Training workhorse |
| FP16 | 16-bit float (half precision) | Inference acceleration |
| INT8 | 8-bit integer | Embedded / mobile inference |

### 6.2 Key GPU Parameters

| Parameter | Meaning |
|-----------|---------|
| CUDA Cores | General-purpose parallel compute units |
| Tensor Cores | Matrix multiplication accelerators — critical for deep learning |
| VRAM Capacity | Determines maximum model and batch size |
| FLOPS | Floating-point operations per second — theoretical compute |
| OPS | Operations per second (includes integer ops) |

### 6.3 Network Optimization and Acceleration

| Technique | Principle | Effect |
|-----------|-----------|--------|
| **Quantization** | Convert FP32 weights to INT8 / FP16 | Reduces model size and inference latency |
| **Pruning** | Remove unimportant connections or channels | Reduces parameter count and computation |
| **TensorRT** | NVIDIA inference optimization engine | Operator fusion, precision calibration, auto kernel tuning |

## 7. Generative Adversarial Networks (GANs)

### 7.1 What is a Generative Model?

A generative model learns the **distribution** of data, with the goal of producing new samples that "look like" real data.

> Analogy — **counterfeit money**: The Generator is the counterfeiter; the Discriminator is the police. The counterfeiter improves to fool the police; the police improves to detect fakes. Through this game, the fakes eventually become indistinguishable from the real thing.

### 7.2 GAN Loss Function

The GAN loss is special because it formulates a **game**:

$$\min_G \max_D \; \mathbb{E}_{x \sim P_{data}}[\log D(x)] + \mathbb{E}_{z \sim P_z}[\log(1 - D(G(z)))]$$

- $D$ (Discriminator): maximize accuracy of distinguishing real from fake.
- $G$ (Generator): minimize the probability that the Discriminator correctly identifies its output as fake.

### 7.3 Strengths and Weaknesses of GANs

| Strengths | Weaknesses |
|-----------|------------|
| High generation quality, rich detail | Unstable training; prone to mode collapse |
| No need to explicitly define data distribution | Generator and Discriminator must be carefully balanced |
| Unsupervised learning — no labeling needed | Difficult to evaluate generation quality |

> **The Generator and Discriminator must stay balanced** — if one becomes too strong, the other cannot learn.

### 7.4 The GAN Family

| Variant | Full Name | Improvement |
|---------|-----------|-------------|
| cGAN | Conditional GAN | Adds conditional information to control generation |
| DCGAN | Deep Convolutional GAN | Replaces FC layers with CNNs; improves stability |
| Pix2Pix | — | Supervised image translation with paired data |
| CycleGAN | — | Unsupervised image translation without paired data |

---

*Note status: Part 2 — first draft complete. See Part 1 at [How to Build and Train a Neural Network (Part 1)](./How_to_Build_and_Train_a_Neural_Network_p1.en.md).*

*Related notes: [What is a Convolutional Network](../具身智能感知基础：机器学习与深度学习/What_is_Convolution_network.en.md), [Commonly Used Convolution Operators](../具身智能感知基础：机器学习与深度学习/Commonly_used_convolution_operators.en.md)*
