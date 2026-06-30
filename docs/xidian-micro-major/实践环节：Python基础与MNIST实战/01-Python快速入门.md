# Python 快速入门——面向深度学习实践

> 西安电子科技大学 · 具身智能微专业
> 课程笔记 · 实践环节

---

## 一、为什么学 Python

Python 是深度学习领域的事实标准语言。NumPy 提供高效的数值计算，Matplotlib 让数据可视化变得简单，PyTorch 则让神经网络的构建和训练像搭积木一样直观。

> 深度学习不是学语法，而是在数值计算、可视化和模型训练中反复实践。Python 是完成这一切的载体。

---

## 二、Python 基础速览

### 2.1 变量与数据类型

Python 是动态类型语言，变量无需声明类型：

```python
x = 10           # int
y = 3.14         # float
name = "MNIST"   # str
flag = True      # bool
```

| 类型      | 示例                | 说明           |
| --------- | ------------------- | -------------- |
| `int`   | `42`              | 整数，无溢出   |
| `float` | `3.14`, `1e-3`  | 浮点数         |
| `str`   | `"hello"`         | 不可变字符串   |
| `bool`  | `True`, `False` | 布尔值         |
| `list`  | `[1, 2, 3]`       | 可变有序集合   |
| `tuple` | `(1, 2, 3)`       | 不可变有序集合 |
| `dict`  | `{"a": 1}`        | 键值对         |
| `set`   | `{1, 2, 3}`       | 无序不重复集合 |

### 2.2 列表与切片

```python
nums = [0, 1, 2, 3, 4, 5]
nums[0]      # 0 —— 第一个元素
nums[-1]     # 5 —— 最后一个元素
nums[1:4]    # [1, 2, 3] —— 切片，含头不含尾
nums[:3]     # [0, 1, 2] —— 从头开始
nums[::2]    # [0, 2, 4] —— 步长为 2
```

列表推导式是 Python 的标志性语法：

```python
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
evens = [x for x in range(20) if x % 2 == 0]
```

### 2.3 字典

```python
config = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 5
}
config["optimizer"] = "Adam"      # 添加键值
config.get("dropout", 0.5)        # 安全访问，键不存在时返回默认值 0.5
```

### 2.4 控制流

```python
# 条件判断
if accuracy > 0.95:
    print("模型表现优秀")
elif accuracy > 0.85:
    print("模型可以接受")
else:
    print("需要继续调参")

# for 循环 —— 深度学习中最常用的结构
for epoch in range(5):
    print(f"Epoch {epoch + 1}")

# 遍历列表
losses = [0.8, 0.5, 0.3, 0.2]
for i, loss in enumerate(losses):
    print(f"Step {i}: loss = {loss:.4f}")
```

### 2.5 函数

```python
def compute_accuracy(predictions, labels):
    """计算分类准确率"""
    correct = (predictions == labels).sum()
    total = len(labels)
    return correct / total

# 带默认参数的函数
def create_optimizer(params, lr=0.001, weight_decay=0.0):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
```

### 2.6 类

```python
class MetricsTracker:
    """训练指标追踪器"""
    def __init__(self):
        self.losses = []
        self.accuracies = []

    def update(self, loss, acc):
        self.losses.append(loss)
        self.accuracies.append(acc)

    def get_latest(self):
        return self.losses[-1], self.accuracies[-1]
```

---

## 三、NumPy —— 数值计算基础

NumPy 的核心是 `ndarray`——多维数组，它是 PyTorch Tensor 的前身。

### 3.1 创建数组

```python
import numpy as np

a = np.array([1, 2, 3, 4])              # 从列表创建
b = np.zeros((3, 4))                     # 全零矩阵
c = np.ones((2, 3))                      # 全一矩阵
d = np.eye(3)                            # 单位矩阵
e = np.arange(0, 10, 0.5)               # 等差序列
f = np.random.randn(100)                 # 标准正态分布
g = np.random.randint(0, 10, (3, 3))    # 随机整数矩阵
```

### 3.2 数组属性与索引

```python
arr = np.random.randn(32, 3, 28, 28)     # (batch, channel, height, width)
arr.shape    # (32, 3, 28, 28)
arr.dtype    # float64
arr.ndim     # 4

# 切片
arr[0]           # 第一个样本，shape (3, 28, 28)
arr[0, 0]        # 第一个样本的第一个通道
arr[:10]         # 前 10 个样本
arr[:, :, :14]   # 所有样本的上半部分
```

### 3.3 数学运算

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b        # array([5, 7, 9]) —— 逐元素加法
a * b        # array([4, 10, 18]) —— 逐元素乘法
np.dot(a, b) # 32 —— 点积
a @ b        # 32 —— 矩阵乘法运算符（Python 3.5+）
```

| 操作              | NumPy                      | 说明                     |
| ----------------- | -------------------------- | ------------------------ |
| 逐元素加/减/乘/除 | `+`, `-`, `*`, `/` | 形状相同或可广播         |
| 矩阵乘法          | `@` 或 `np.dot()`      | 遵循线性代数规则         |
| 求和              | `np.sum(arr, axis=0)`    | axis=0 按列，axis=1 按行 |
| 均值              | `np.mean(arr)`           | 全局或按轴               |
| 标准差            | `np.std(arr)`            | 全局或按轴               |
| 转置              | `arr.T`                  | 交换维度                 |

### 3.4 广播机制

广播允许不同形状的数组进行运算：

```python
# (3, 4) + (4,) → (3, 4) —— 向量广播到矩阵的每一行
matrix = np.ones((3, 4))
bias = np.array([1, 2, 3, 4])
result = matrix + bias      # shape 仍为 (3, 4)

# (32, 1, 28, 28) * (1, 3, 1, 1) → (32, 3, 28, 28)
```

> 广播是深度学习中批次归一化、特征缩放等操作的基础。理解广播原理能帮你避免大量不必要的维度变换。

### 3.5 线性代数

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A @ B                     # 矩阵乘法
np.linalg.inv(A)          # 矩阵求逆
np.linalg.eig(A)          # 特征值与特征向量
np.linalg.norm(A)         # 范数
```

---

## 四、Matplotlib —— 数据可视化

### 4.1 基本绘图

```python
import matplotlib.pyplot as plt

# 折线图 —— 最常用于绘制训练曲线
losses = [0.8, 0.5, 0.3, 0.2, 0.15, 0.12]
plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curve')
plt.legend()
plt.grid(True)
plt.show()
```

### 4.2 常用图表类型

| 图表类型 | 函数               | 典型用途                     |
| -------- | ------------------ | ---------------------------- |
| 折线图   | `plt.plot()`     | 训练曲线、loss/accuracy 趋势 |
| 散点图   | `plt.scatter()`  | 数据分布、特征可视化         |
| 柱状图   | `plt.bar()`      | 类别对比、模型性能比较       |
| 直方图   | `plt.hist()`     | 权重分布、梯度分布           |
| 热力图   | `plt.imshow()`   | 图像显示、混淆矩阵、注意力图 |
| 子图     | `plt.subplots()` | 多图对比展示                 |

### 4.3 显示图像

```python
# 显示 MNIST 图像
image = dataset[0][0].reshape(28, 28)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title('MNIST Sample')
plt.show()

# 批量显示
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    img, label = dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### 4.4 保存图像

```python
plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
```

---

## 五、PyTorch 基础 —— 深度学习核心

### 5.1 张量

PyTorch 的张量 (`Tensor`) 与 NumPy 数组类似，但可以在 GPU 上加速运算，并支持自动求导。

```python
import torch

# 创建张量
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.zeros(3, 4)
z = torch.randn(32, 1, 28, 28)   # 典型的图像批次

# 与 NumPy 互转
arr = x.numpy()                    # Tensor → NumPy
tensor = torch.from_numpy(arr)     # NumPy → Tensor

# GPU 支持
if torch.cuda.is_available():
    x = x.cuda()
```

| 操作     | PyTorch                         | NumPy 等效                |
| -------- | ------------------------------- | ------------------------- |
| 创建全零 | `torch.zeros(3, 4)`           | `np.zeros((3, 4))`      |
| 随机正态 | `torch.randn(3, 4)`           | `np.random.randn(3, 4)` |
| 形状     | `x.shape`                     | `arr.shape`             |
| 矩阵乘法 | `x @ y` 或 `torch.mm(x, y)` | `arr @ brr`             |
| 求和     | `x.sum()`                     | `arr.sum()`             |
| 变形     | `x.view(-1, 28*28)`           | `arr.reshape(-1, 784)`  |

### 5.2 自动求导

自动求导是深度学习框架的核心能力——只需定义前向计算，框架自动计算所有梯度。

```python
# 创建需要梯度的张量
w = torch.randn(3, requires_grad=True)
x = torch.randn(3)

# 前向计算
loss = (w * x).sum()     # 任意计算图

# 反向传播 —— 自动计算 ∂loss/∂w
loss.backward()
print(w.grad)             # 梯度已就绪
```

> 传统方法中，每个模型的梯度公式都要手推手写。PyTorch 的 autograd 引擎自动追踪计算图并反向传播，让你专注于模型设计而非求导。

### 5.3 构建神经网络

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)    # 展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

| 组件     | 类                                        | 作用                |
| -------- | ----------------------------------------- | ------------------- |
| 卷积层   | `nn.Conv2d`                             | 提取空间特征        |
| 池化层   | `nn.MaxPool2d`                          | 降采样，减少计算量  |
| 全连接层 | `nn.Linear`                             | 综合特征做分类/回归 |
| 激活函数 | `nn.ReLU`, `nn.Sigmoid`               | 引入非线性          |
| 正则化   | `nn.Dropout`                            | 防止过拟合          |
| 损失函数 | `nn.CrossEntropyLoss`, `nn.MSELoss`   | 衡量预测误差        |
| 优化器   | `torch.optim.Adam`, `torch.optim.SGD` | 更新参数            |

### 5.4 训练循环

```python
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()   # 清零梯度
        loss.backward()         # 计算梯度
        optimizer.step()        # 更新参数

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")
```

### 5.5 数据加载

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

| 概念           | 说明                                                 |
| -------------- | ---------------------------------------------------- |
| `Dataset`    | 单个数据集的封装，实现`__getitem__` 和 `__len__` |
| `DataLoader` | 批量加载、打乱、多线程读取                           |
| `transforms` | 数据预处理和增强管道                                 |
| `batch_size` | 每次送入模型的样本数                                 |

---

## 六、从 Python 到深度学习实践

上面五节覆盖了进入 MNIST 实验所需的全部基础：

1. **Python 语法** —— 变量、控制流、函数、类
2. **NumPy** —— 数组操作、广播、线性代数
3. **Matplotlib** —— 曲线绘制、图像显示、画布保存
4. **PyTorch 张量** —— 创建、运算、GPU 加速
5. **PyTorch 建模** —— nn.Module、损失函数、优化器、训练循环

> 当你开始 MNIST 实验时，这些内容会自然串联起来：用 NumPy 理解数据 → 用 DataLoader 加载数据 → 用 nn.Module 定义网络 → 用训练循环迭代优化 → 用 Matplotlib 可视化结果。

---

*相关笔记：[MNIST 实验手册](./02-MNIST实验手册.md)*