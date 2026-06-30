# MNIST 实验手册——第一次实践环节

> 西安电子科技大学 · 具身智能微专业
> 课程笔记 · 实践环节

---

## 一、实验目标

这是整个微专业的**第一个动手实验**。目标是跑通深度学习项目的完整流程：

1. 搭建 Python + PyTorch 开发环境
2. 加载并探索 MNIST 手写数字数据集
3. 从零构建卷积神经网络 (CNN)
4. 完成训练、评估、可视化
5. 保存模型并独立推理

> 这个实验不要求你理解 CNN 的每个细节——先跑通全流程，建立直觉，理论会在后续课程中展开。

---

## 二、环境准备

### 2.1 依赖安装

```bash
pip install torch torchvision matplotlib numpy
```

或者使用项目附带的 `requirements.txt`：

```bash
cd code
pip install -r requirements.txt
```

### 2.2 依赖说明

| 库 | 版本 | 作用 |
|----|------|------|
| `torch` | ≥2.0.0 | 深度学习框架 |
| `torchvision` | ≥0.15.0 | 数据集下载 + 图像预处理 |
| `matplotlib` | ≥3.7.0 | 可视化：训练曲线、图像显示 |
| `numpy` | ≥1.24.0 | 数值计算辅助 |

### 2.3 验证安装

```python
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

print(f"PyTorch 版本: {torch.__version__}")
print(f"Torchvision 版本: {torchvision.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

# 快速张量测试
x = torch.randn(2, 3)
print(f"张量创建成功，形状: {x.shape}")
```

---

## 三、MNIST 数据集

### 3.1 数据集简介

MNIST（Modified National Institute of Standards and Technology）是深度学习领域的 "Hello World"：

| 属性 | 值 |
|------|-----|
| 内容 | 0-9 手写数字灰度图像 |
| 图像尺寸 | 28 × 28 像素 |
| 训练集 | 60,000 张 |
| 测试集 | 10,000 张 |
| 类别数 | 10 |
| 像素值范围 | 0 ~ 255 |

### 3.2 加载数据

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 预处理管道
transform = transforms.Compose([
    transforms.ToTensor(),                          # PIL → Tensor, [0,255] → [0,1]
    transforms.Normalize((0.1307,), (0.3081,))     # 标准化
])

# 下载并加载
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
```

### 3.3 数据探索

开始训练前，先看看数据长什么样：

```python
import matplotlib.pyplot as plt

# 显示前 16 张图像
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# 统计类别分布
labels = [train_dataset[i][1] for i in range(len(train_dataset))]
for digit in range(10):
    print(f"数字 {digit}: {labels.count(digit)} 张")
```

> 在开始建模之前，先「看」一眼数据——这是深度学习中最重要的习惯。确认数据的形状、分布和内容，避免在错误的数据上浪费时间。

---

## 四、构建 CNN 模型

### 4.1 网络结构

```python
import torch.nn as nn

class MNIST_CNN(nn.Module):
    """
    MNIST 手写数字识别 CNN

    输入: (batch, 1, 28, 28)
    输出: (batch, 10) —— 10 个类别的 logits
    """
    def __init__(self):
        super().__init__()

        # 卷积层 1: 1→32 通道，3×3 卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 卷积层 2: 32→64 通道，3×3 卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层: 2×2 最大池化
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)   # 7×7 是两次池化后的尺寸
        self.fc2 = nn.Linear(128, 10)            # 10 个类别输出
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积块 1
        x = self.relu(self.conv1(x))    # (1,28,28) → (32,28,28)
        x = self.pool(x)                 # (32,28,28) → (32,14,14)

        # 卷积块 2
        x = self.relu(self.conv2(x))    # (32,14,14) → (64,14,14)
        x = self.pool(x)                 # (64,14,14) → (64,7,7)

        # 展平 + 分类
        x = x.view(x.size(0), -1)       # (64,7,7) → (64*7*7)
        x = self.relu(self.fc1(x))      # (64*7*7) → (128)
        x = self.dropout(x)              # Dropout 防过拟合
        x = self.fc2(x)                  # (128) → (10)
        return x
```

### 4.2 尺寸计算

| 层 | 输入尺寸 | 输出尺寸 | 参数量 |
|----|---------|---------|--------|
| Conv1 (1→32, k=3, p=1) | 1×28×28 | 32×28×28 | 32×(1×9+1)=320 |
| MaxPool (k=2) | 32×28×28 | 32×14×14 | 0 |
| Conv2 (32→64, k=3, p=1) | 32×14×14 | 64×14×14 | 64×(32×9+1)=18,496 |
| MaxPool (k=2) | 64×14×14 | 64×7×7 | 0 |
| FC1 (64×7×7→128) | 3136 | 128 | 3136×128+128=401,536 |
| Dropout | — | — | 0 |
| FC2 (128→10) | 128 | 10 | 128×10+10=1,290 |

卷积输出尺寸公式：

$$H_{out} = \frac{H_{in} + 2P - K}{S} + 1$$

其中 $P$ 为 padding，$K$ 为 kernel size，$S$ 为 stride。

---

## 五、训练模型

### 5.1 配置超参数

```python
# 超参数
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

# 模型、损失函数、优化器
model = MNIST_CNN()
criterion = nn.CrossEntropyLoss()           # 多分类交叉熵
optimizer = torch.optim.Adam(               # Adam 优化器
    model.parameters(), lr=LEARNING_RATE
)
```

### 5.2 训练循环

```python
def train(model, train_loader, criterion, optimizer, epochs):
    """完整训练循环"""
    model.train()
    losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # 1. 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 2. 反向传播
            optimizer.zero_grad()   # 清零梯度（否则会累积）
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新权重

            # 3. 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            losses.append(loss.item())

        # 每轮打印
        acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

    return losses

losses = train(model, train_loader, criterion, optimizer, EPOCHS)
```

### 5.3 训练要点

| 要点 | 说明 |
|------|------|
| `model.train()` | 启用训练模式（Dropout、BatchNorm 行为不同） |
| `optimizer.zero_grad()` | **必须**在每步前清零梯度，否则梯度会累积 |
| `loss.backward()` | 反向传播，计算所有参数的梯度 |
| `optimizer.step()` | 使用梯度更新参数 |
| `outputs.max(1)` | 取 logits 最大值的索引作为预测类别 |

> 训练循环的三行核心代码——`zero_grad()`、`loss.backward()`、`optimizer.step()`——是所有 PyTorch 训练的标准范式。理解这三步就理解了深度学习训练的引擎。

---

## 六、评估模型

### 6.1 测试集评估

```python
def evaluate(model, test_loader):
    """在测试集上评估模型"""
    model.eval()    # 评估模式
    correct = 0
    total = 0

    with torch.no_grad():    # 关闭梯度计算，节省内存
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc

test_acc = evaluate(model, test_loader)
```

| 评估模式 | `model.train()` | `model.eval()` |
|---------|----------------|----------------|
| Dropout | 激活 | 关闭 |
| BatchNorm | 使用批次统计量 | 使用全局统计量 |
| 梯度 | 需要 | `torch.no_grad()` 关闭 |

### 6.2 可视化预测

```python
def visualize_predictions(model, test_loader, num_images=16):
    """可视化模型预测结果"""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_images], labels[:num_images]

    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap='gray')
        color = 'green' if predicted[i] == labels[i] else 'red'
        ax.set_title(f'Pred: {predicted[i]} (True: {labels[i]})',
                     color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

visualize_predictions(model, test_loader)
```

### 6.3 训练曲线

```python
plt.figure(figsize=(10, 4))
plt.plot(losses, alpha=0.3, color='blue', label='Batch Loss')
# 平滑曲线
smoothed = np.convolve(losses, np.ones(50)/50, mode='valid')
plt.plot(range(49, len(losses)), smoothed, color='blue', linewidth=2,
         label='Smoothed (window=50)')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 七、保存与加载模型

### 7.1 保存模型

```python
# 方法一：保存完整模型（简单但文件大）
torch.save(model, 'mnist_model_full.pth')

# 方法二：只保存参数（推荐）
torch.save(model.state_dict(), 'mnist_model.pth')
print("模型已保存")
```

### 7.2 加载模型进行推理

```python
# 重建模型结构 + 加载参数
model = MNIST_CNN()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()
print("模型已加载，准备推理")

# 单张推理
image, label = test_dataset[0]
with torch.no_grad():
    output = model(image.unsqueeze(0))   # 添加 batch 维度
    prob = output.softmax(dim=1)          # 转换为概率
    pred = prob.argmax(dim=1).item()
    confidence = prob[0, pred].item()

print(f"预测: {pred}, 真实: {label}, 置信度: {confidence:.2%}")
```

---

## 八、实验检查清单

完成以下每一项，你对深度学习实践就有了完整的直觉：

- [ ] 环境搭建成功，`import torch` 无报错
- [ ] MNIST 数据集下载并可视化
- [ ] CNN 模型定义并打印 `model` 查看结构
- [ ] 训练 5 个 epoch，loss 稳定下降
- [ ] 测试集准确率 > 95%
- [ ] 可视化预测结果，观察哪些数字容易混淆
- [ ] 绘制训练曲线，观察 loss 下降趋势
- [ ] 保存模型参数文件
- [ ] 独立加载模型完成单张推理

---

## 九、常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `CUDA out of memory` | 显存不足 | 减小 batch_size，或切换到 CPU |
| `shape mismatch` | 全连接层输入尺寸算错 | 检查两次池化后的特征图尺寸 |
| 准确率不升 | 学习率不当或数据未打乱 | 调小学习率，确认 `shuffle=True` |
| loss 不下降 | 梯度未清零 | 确认 `optimizer.zero_grad()` 在循环中 |
| 下载 MNIST 失败 | 网络问题 | 手动下载放入 `data/` 目录 |

> 遇到报错不要怕。深度学习的大部分时间都在和 shape、dtype 和维度打交道——这是正常的，也是成长的一部分。

---

*相关笔记：[Python 快速入门](./01-Python快速入门.md)*
