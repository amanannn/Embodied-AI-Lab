# 如何构建和训练神经网络（下）：CNN 实战与生成对抗网络

> 西安电子科技大学 · 具身智能微专业
> 课程笔记

---

## 一、CNN 原理回顾

### 1.1 机器如何"看"一张图片？

```text
人类看到的：              计算机看到的：
┌──────────┐            [[[ 0,  0,255],
│   🌞    │               [255,128, 0],
│   ☁️    │               [ 0,255,128],
│  🏠 🌲  │               ...
└──────────┘             [128, 0,255]]]
                        形状: (3, H, W)，值域: [0, 255]
```

CNN 的卷积核在这张数值矩阵上滑动，检测局部模式——边缘、角点、纹理、更高级的语义。

### 1.2 卷积输出尺寸计算

$$H_{out} = \left\lfloor\frac{H_{in} + 2P - K}{S}\right\rfloor + 1$$

| 参数 | 含义 | 常用值 |
|------|------|--------|
| $K$ | kernel_size | 3, 5, 7 |
| $P$ | padding | $K/2$ 向下取整（保持尺寸不变） |
| $S$ | stride | 1（特征提取）, 2（降采样） |

```python
# "same" padding：输出尺寸 = 输入尺寸 / stride
# kernel_size=3 → 需要 padding=1
# kernel_size=5 → 需要 padding=2
conv_same = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
# 输入 (B, 64, 32, 32) → 输出 (B, 128, 32, 32)  尺寸不变
```

### 1.3 搭建 CNN 的核心原则

1. **通道数**：空间尺寸逐层减半（池化），通道数逐层翻倍——保持总信息量
2. **感受野**：浅层看细节（边缘/纹理），深层看全局（语义/物体）
3. **跳跃连接**：ResNet 的核心——让梯度直接流过深层

---

## 二、数据准备

### 2.1 DataLoader 参数

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,      # 每批样本数
    shuffle=True,        # 训练集打乱，测试集不打乱
    num_workers=2,       # 多进程加载
    pin_memory=True,     # GPU 训练时加速数据传输
    drop_last=True       # 丢弃最后的不完整 batch（BN 要求 batch≥2）
)
```

| 参数 | 影响 | 建议 |
|------|------|------|
| `batch_size` | 大→梯度稳定但显存多；小→噪声大但泛化好 | 32/64/128（按显存调整） |
| `num_workers` | 太少→GPU 等数据；太多→CPU 内存爆 | 4~8（通常 CPU 核心数一半） |
| `pin_memory` | 加速 CPU→GPU 传输 | GPU 训练时开 |

### 2.2 数据预处理管道

```python
# 训练时的增强管道
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),        # 随机裁剪（CIFAR 标配）
    transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
    transforms.ColorJitter(0.1, 0.1, 0.1),       # 颜色扰动
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))  # CIFAR-10 统计量
])

# 测试时不做增强
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
```

> `Normalize(mean, std)` 将数据标准化为 $N(0,1)$——均值为 0 方差为 1 的输入对梯度传播最友好。

---

## 三、网络构建

### 3.1 卷积层的参数选择

```python
# 标准卷积块
conv_block = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 空间尺寸不变
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.Conv2d(128, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),                                # 空间尺寸减半
)
```

**参数量计算**：
$$Params_{conv} = K \times K \times C_{in} \times C_{out} + C_{out}$$

```python
# 快速计算所有层的参数量
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'可训练参数: {count_params(model):,}')
```

### 3.2 全连接层

```python
# FC 层是参数量大户——需谨慎使用
fc = nn.Linear(320, 10)
# params = 320 × 10 + 10 = 3,210
```

> 卷积的参数量与输入尺寸无关（权重共享），全连接的参数量随输入尺寸线性增长——这就是为什么 CNN 分类头之前通常用全局平均池化（GAP）抹掉空间维度。

### 3.3 MNIST 完整网络（逐层分析）

```python
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)      # 无 padding
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 1×28×28→10×12×12
        x = self.pool(F.relu(self.conv2(x)))   # 10×12×12→20×4×4
        x = x.view(x.size(0), -1)              # 20×4×4→320
        return self.fc(x)                       # 320→10
```

| 层 | 输入形状 | 输出形状 | 参数量 | 计算 |
|----|---------|---------|--------|------|
| Conv1 (1→10, k=5) | (1, 28, 28) | (10, 24, 24) | 260 | 1×10×5×5 + 10 |
| Pool (k=2) | (10, 24, 24) | (10, 12, 12) | 0 | — |
| Conv2 (10→20, k=5) | (10, 12, 12) | (20, 8, 8) | 5,020 | 10×20×5×5 + 20 |
| Pool (k=2) | (20, 8, 8) | (20, 4, 4) | 0 | — |
| Flatten | (20, 4, 4) | (320) | 0 | — |
| FC (320→10) | (320) | (10) | 3,210 | 320×10 + 10 |
| **总计** | | | **8,490** | — |

---

## 四、训练循环（完整示例）

```python
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device)

        optimizer.zero_grad()               # 1. 清空上一轮梯度
        outputs = model(inputs)             # 2. 前向
        loss = criterion(outputs, target)   # 3. 损失
        loss.backward()                     # 4. 反向
        optimizer.step()                    # 5. 更新

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print(f'[Epoch {epoch}, Batch {batch_idx+1:5d}] '
                  f'loss: {running_loss / 300:.3f}')
            running_loss = 0.0
```

---

## 五、深度学习硬件

### 5.1 训练 vs 推理的精度需求

| 阶段 | 精度需求 | 原因 | 典型格式 |
|------|---------|------|---------|
| **训练** | **FP32** 及以上 | 梯度微小，累积误差致命 | FP32, TF32 |
| **微调** | FP16/FP32 混合 | 部分关键层保留 FP32 | Mixed Precision |
| **推理** | FP16 或 INT8 | 精度要求相对低 | FP16, INT8, INT4 |

### 5.2 混合精度训练

```python
# PyTorch 自动混合精度（AMP）——几乎免费的速度提升
scaler = torch.amp.GradScaler()

with torch.amp.autocast('cuda'):
    outputs = model(inputs)           # 自动用 FP16 计算
    loss = criterion(outputs, target)

scaler.scale(loss).backward()         # 放大 loss 防止梯度下溢
scaler.step(optimizer)                # 自动 unscale + 更新
scaler.update()
```

### 5.3 GPU 关键参数

| 参数 | 含义 | 对深度学习的影响 |
|------|------|---------------|
| **CUDA 核心** | 通用并行计算单元 | 一般的矩阵运算 |
| **Tensor 核心** | 矩阵乘法专用加速器 | **训练和推理的主力** |
| **显存容量** | 决定最大模型+batch_size | OOM 的主要瓶颈 |
| **显存带宽** | 数据读写速度 | 大模型推理的瓶颈 |
| **FLOPS** | 理论峰值算力 | 上限参考，实际利用率 30-70% |

### 5.4 部署优化技术

| 技术 | 原理 | 效果 | 工具 |
|------|------|------|------|
| **量化** | FP32 → INT8/FP16 | 模型缩小 4×，加速 2-4× | PyTorch Quantization |
| **剪枝** | 移除不重要的权重/通道 | 减少 30-90% 参数 | `torch.nn.utils.prune` |
| **蒸馏** | 大模型教小模型 | 小模型达到接近大模型精度 | 自定义训练 |
| **TensorRT** | 算子融合 + 内核调优 | 推理加速 2-5× | NVIDIA TensorRT |
| **ONNX** | 跨框架中间表示 | 一次导出，多平台部署 | `torch.onnx.export` |

---

## 六、生成对抗网络（GAN）

### 6.1 什么是生成模型？

**判别模型** $P(y|x)$：给定输入，判断类别 → 分类、检测
**生成模型** $P(x)$ 或 $P(x|y)$：学习数据分布，生成新样本 → 图像生成、风格迁移

### 6.2 GAN 的核心思想

> **造假者和警察的博弈**：
> 
> - **生成器 G**（造假者）：从随机噪声生成假样本，试图以假乱真
> - **判别器 D**（警察）：判断输入是真样本还是假样本
> - 两者交替训练，最终 G 生成的假样本足以骗过 D

### 6.3 数学形式

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim P_{data}}[\log D(x)] + \mathbb{E}_{z \sim P_z}[\log(1 - D(G(z)))]$$

| 项 | 含义 | 谁在优化 |
|----|------|---------|
| $\mathbb{E}[\log D(x)]$ | 判别器对**真样本**的判断越接近 1 越好 | D 最大化 |
| $\mathbb{E}[\log(1-D(G(z)))]$ | 判别器对**假样本**的判断越接近 0 越好 | D 最大化；G 最小化（让 D 判错） |

### 6.4 GAN 的训练流程

```python
# GAN 的一个训练 step
for real_images in dataloader:
    batch_size = real_images.size(0)

    # ── 训练判别器 D ──
    # 真样本：D 应输出接近 1
    d_real = D(real_images)
    d_real_loss = F.binary_cross_entropy(d_real, torch.ones(batch_size, 1))

    # 假样本：D 应输出接近 0
    z = torch.randn(batch_size, latent_dim)
    fake_images = G(z)
    d_fake = D(fake_images.detach())  # detach! 不反传梯度到 G
    d_fake_loss = F.binary_cross_entropy(d_fake, torch.zeros(batch_size, 1))

    d_loss = d_real_loss + d_fake_loss
    d_optimizer.zero_grad(); d_loss.backward(); d_optimizer.step()

    # ── 训练生成器 G ──
    # G 希望 D 把假样本判为真
    z = torch.randn(batch_size, latent_dim)
    fake_images = G(z)
    d_output = D(fake_images)
    g_loss = F.binary_cross_entropy(d_output, torch.ones(batch_size, 1))

    g_optimizer.zero_grad(); g_loss.backward(); g_optimizer.step()
```

### 6.5 GAN 训练的常见问题与解决

| 问题 | 表现 | 原因 | 解决 |
|------|------|------|------|
| **模式崩塌** | G 只生成少数几种样本 | G 找到了能骗过 D 的"捷径" | Minibatch Discrimination, WGAN |
| **训练不稳定** | Loss 剧烈震荡 | D 和 G 不平衡 | 调整 D/G 更新比例，谱归一化 |
| **梯度消失** | G loss 不降 | D 太强，G 无梯度信号 | LSGAN（最小二乘损失替代 BCE） |
| **难以评估** | 缺乏客观指标 | 生成质量主观 | FID, IS（Inception Score） |

#### WGAN 的改进

WGAN 用 **Wasserstein 距离** 替代 JS 散度，从根本上缓解训练不稳定：

$$W(P_r, P_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{x \sim P_g}[f(x)]$$

核心改动：D 去掉 Sigmoid、使用权重裁剪（或梯度惩罚）、用 RMSprop 替代 Adam。

### 6.6 GAN 家族

| 变体 | 全称 | 核心改进 | 代表应用 |
|------|------|---------|---------|
| **DCGAN** | Deep Convolutional GAN | CNN 替代全连接 + BatchNorm | 图像生成基础架构 |
| **cGAN** | Conditional GAN | 加入条件标签控制生成 | 类别可控生成 |
| **Pix2Pix** | — | 成对数据 + U-Net + PatchGAN | 图像翻译（素描→照片） |
| **CycleGAN** | — | 循环一致性损失，无需成对数据 | 风格迁移（马→斑马） |
| **StyleGAN** | — | 风格向量控制生成属性 | 高清人脸生成 |
| **SRGAN** | Super-Resolution GAN | 感知损失 + 对抗损失 | 图像超分辨率 |
| **BigGAN** | — | 大批量 + 大模型 + 谱归一化 | 高质量 ImageNet 生成 |

### 6.7 GAN vs Diffusion

| 维度 | GAN | Diffusion Model |
|------|-----|-----------------|
| 生成方式 | 一步从噪声到图像 | 逐步去噪（多步） |
| 训练难度 | 高（博弈平衡） | 较低（回归任务） |
| 生成质量 | 高（细节好） | 很高（多样性强） |
| 推理速度 | 快（单步） | 慢（10-1000 步） |
| 模式覆盖 | 可能崩塌 | 覆盖完整（多样性好） |

---

*相关笔记：[Part 1: 训练流程与损失函数](./How_to_Build_and_Train_a_Neural_Network_p1.md)、[什么是卷积网络](../具身智能感知基础：机器学习与深度学习/What_is_Convolution_network.md)、[常用卷积算子](../具身智能感知基础：机器学习与深度学习/Commonly_used_convolution_operators.md)*
