# 如何构建和训练神经网络（下）：CNN 实战与生成对抗网络

> 西安电子科技大学 · 具身智能微专业
> 课程笔记

## 一、CNN 原理回顾

### 1.1 机器如何读取一张图片？

一张灰度图像 = 一个 $H \times W$ 的二维数组，每个元素是像素值。CNN 的卷积核在这个数组上滑动，提取局部特征。

### 1.2 搭建 CNN 的要点

1. **卷积层**：提取特征——只需关心通道数，尺寸由 `kernel_size` 和 `stride` 自动计算
2. **池化层**：降采样，降低特征图尺寸
3. **全连接层**：卷积特征 → 分类输出

## 二、数据准备

### 2.1 DataLoader

```python
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
```

| 参数 | 含义 |
|------|------|
| `dataset` | 数据集对象 |
| `batch_size` | 每批加载的样本数 |
| `shuffle` | 是否打乱数据（训练集 `True`，测试集 `False`） |
| `num_workers` | 数据加载的子进程数 |

### 2.2 MNIST 数据集完整加载

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

> `Normalize((0.1307,), (0.3081,))` 是 MNIST 数据集的全局均值和标准差，预处理后数据服从标准正态分布，利于训练。

## 三、网络构建

### 3.1 卷积层

```python
conv_layer = torch.nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, stride=1)
```

> 代码中只需考虑通道数，输出特征图尺寸由 `kernel_size` 和 `stride` 自动推导。

### 3.2 全连接层

```python
self.fc = torch.nn.Linear(320, 10)
```

参数量：$320 \times 10 = 3200$，远超一个 3×3 卷积核（$3 \times 3 \times C_{in} \times C_{out}$ 典型也在几千，但共享到整张图）。直观理解：**卷积比全连接参数少得多，且保留空间结构**。

### 3.3 完整网络设计（MNIST 示例）

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
        x = x.view(batch_size, -1)      # 展平 → 送入全连接层
        x = self.fc(x)
        return x

model = Net()
```

| 层 | 输入 | 输出 | 说明 |
|----|------|------|------|
| Conv1 | 1×28×28 | 10×24×24 | 5×5 卷积，步长 1 |
| Pooling | 10×24×24 | 10×12×12 | 2×2 最大池化 |
| Conv2 | 10×12×12 | 20×8×8 | 5×5 卷积，步长 1 |
| Pooling | 20×8×8 | 20×4×4 | 2×2 最大池化 |
| Flatten | 20×4×4 | 320 | 展平为一维 |
| FC | 320 | 10 | 全连接 → 10 类输出 |

## 四、损失函数与优化器

### 4.1 选择依据

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

- **交叉熵损失**：MNIST 是多分类任务，输出层 Softmax + CrossEntropyLoss 组合梯度好，收敛快
- **SGD（小批量梯度下降，MBGD）**：使用一个 batch 的数据更新梯度——兼顾批量梯度下降的稳定性和随机梯度下降的计算效率

| 参数 | 含义 |
|------|------|
| `model.parameters()` | 要优化的模型参数（权重、偏置等） |
| `lr` | 学习率（Learning Rate） |

## 五、训练循环

```python
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data

        optimizer.zero_grad()           # 1. 清空梯度
        outputs = model(inputs)          # 2. 前向传播
        loss = criterion(outputs, target) # 3. 计算损失
        loss.backward()                  # 4. 反向传播
        optimizer.step()                 # 5. 更新参数

        running_loss += loss.item()
        if batch_idx % 300 == 299:       # 每 300 个 batch 输出一次
            print(f'[{epoch+1}, {batch_idx+1:5d}] loss: {running_loss/300:.3f}')
            running_loss = 0.0
```

训练日志解读参见上篇 §3.3——关键观察 Train Loss 与 Val Loss 的相对趋势。

## 六、深度学习硬件

### 6.1 训练 vs 推理的精度需求

| 阶段 | 精度需求 | 原因 |
|------|---------|------|
| **训练** | FP32 及以上 | 梯度更新微小，需要高精度避免累积误差 |
| **推理** | FP16 即可，甚至 INT8 | 精度要求相对低，利于部署在嵌入式设备 |

| 精度 | 位数 | 典型场景 |
|------|------|---------|
| FP32 | 32 bit 浮点 | 训练主力 |
| FP16 | 16 bit 浮点（半精度） | 推理加速 |
| INT8 | 8 bit 整数 | 嵌入式/移动端推理 |

### 6.2 GPU 关键参数

| 参数 | 含义 |
|------|------|
| CUDA 核心 | 通用并行计算单元 |
| Tensor 核心 | 专为矩阵乘法加速设计，深度学习关键 |
| 显存容量 | 决定能放多大的模型和 batch |
| FLOPS | 每秒浮点运算次数，衡量理论算力 |
| OPS | 每秒运算次数（含整数运算） |

### 6.3 网络优化加速

| 技术 | 原理 | 效果 |
|------|------|------|
| **量化（Quantization）** | 将 FP32 权重转为 INT8/FP16 | 降低模型体积和推理延迟 |
| **剪枝（Pruning）** | 移除不重要的连接或通道 | 减少参数量和计算量 |
| **TensorRT** | NVIDIA 推理优化引擎 | 算子融合、精度校准、内核自动调优 |

## 七、生成对抗网络（GAN）

### 7.1 什么是生成模型？

生成模型学习数据的**分布**，目标是生成与真实数据"长得像"的新样本。

> 思想类比——**造假币**：生成器是造假者，判别器是警察。造假者不断精进以骗过警察，警察不断学习以识破假币。两者博弈，最终假币以假乱真。

### 7.2 GAN 的损失函数

GAN 的 Loss 之所以特殊，在于它是一个**博弈问题**：

$$\min_G \max_D \; \mathbb{E}_{x \sim P_{data}}[\log D(x)] + \mathbb{E}_{z \sim P_z}[\log(1 - D(G(z)))]$$

- $D$（判别器）：最大化识别真假的准确率
- $G$（生成器）：最小化判别器识别自己为假的概率

### 7.3 GAN 的优劣

| 优势 | 缺陷 |
|------|------|
| 生成质量高，细节丰富 | 训练不稳定，模式崩塌（mode collapse） |
| 不需要显式定义数据分布 | 生成器和判别器需要精心平衡 |
| 无监督学习，不需标注 | 难以评估生成质量 |

> **生成器和判别器必须平衡**——一方过强导致另一方无法学习。

### 7.4 GAN 家族

| 变体 | 全称 | 改进点 |
|------|------|--------|
| cGAN | Conditional GAN | 加入条件信息，控制生成内容 |
| DCGAN | Deep Convolutional GAN | 用 CNN 替代全连接，提升稳定性 |
| Pix2Pix | — | 成对数据的有监督图像翻译 |
| CycleGAN | — | 无成对数据的无监督图像翻译 |

---

*笔记状态：下篇初稿完成。上篇见 [How to Build and Train a Neural Network (Part 1)](./How_to_Build_and_Train_a_Neural_Network_p1.md)。*

*相关笔记：[什么是卷积网络](../具身智能感知基础：机器学习与深度学习/What_is_Convolution_network.md)、[常用卷积算子](../具身智能感知基础：机器学习与深度学习/Commonly_used_convolution_operators.md)*
