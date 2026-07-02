# Python Quick Start — For Deep Learning Practice

> Xidian University · Embodied AI Micro-Major
> Course Notes · Practice Module

---

## I. Why Python

Python is the de facto standard language for deep learning. NumPy provides efficient numerical computation, Matplotlib makes data visualization straightforward, and PyTorch makes building and training neural networks as intuitive as snapping building blocks together.

> Deep learning is not about learning syntax — it's about iterative practice across numerical computation, visualization, and model training. Python is the vehicle that carries all of this.

---

## II. Python Basics at a Glance

### 2.1 Variables and Data Types

Python is dynamically typed — no need to declare variable types:

```python
x = 10           # int
y = 3.14         # float
name = "MNIST"   # str
flag = True      # bool
```

| Type | Example | Notes |
|------|---------|-------|
| `int` | `42` | Integer, no overflow |
| `float` | `3.14`, `1e-3` | Floating point |
| `str` | `"hello"` | Immutable string |
| `bool` | `True`, `False` | Boolean |
| `list` | `[1, 2, 3]` | Mutable ordered collection |
| `tuple` | `(1, 2, 3)` | Immutable ordered collection |
| `dict` | `{"a": 1}` | Key-value pairs |
| `set` | `{1, 2, 3}` | Unordered unique elements |

### 2.2 Lists and Slicing

```python
nums = [0, 1, 2, 3, 4, 5]
nums[0]      # 0 — first element
nums[-1]     # 5 — last element
nums[1:4]    # [1, 2, 3] — slice, start inclusive, end exclusive
nums[:3]     # [0, 1, 2] — from the beginning
nums[::2]    # [0, 2, 4] — stride of 2
```

List comprehensions are Python's signature syntax:

```python
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
evens = [x for x in range(20) if x % 2 == 0]
```

### 2.3 Dictionaries

```python
config = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 5
}
config["optimizer"] = "Adam"      # Add key
config.get("dropout", 0.5)        # Safe access with default
```

### 2.4 Control Flow

```python
# Conditionals
if accuracy > 0.95:
    print("Excellent performance")
elif accuracy > 0.85:
    print("Acceptable performance")
else:
    print("Need further tuning")

# for loops — the most common structure in deep learning
for epoch in range(5):
    print(f"Epoch {epoch + 1}")

# Iterating over lists
losses = [0.8, 0.5, 0.3, 0.2]
for i, loss in enumerate(losses):
    print(f"Step {i}: loss = {loss:.4f}")
```

### 2.5 Functions

```python
def compute_accuracy(predictions, labels):
    """Compute classification accuracy"""
    correct = (predictions == labels).sum()
    total = len(labels)
    return correct / total

# Function with default parameters
def create_optimizer(params, lr=0.001, weight_decay=0.0):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
```

### 2.6 Classes

```python
class MetricsTracker:
    """Training metrics tracker"""
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

## III. NumPy — Numerical Computation Foundation

The core of NumPy is the `ndarray` — a multi-dimensional array, the precursor to PyTorch Tensors.

### 3.1 Creating Arrays

```python
import numpy as np

a = np.array([1, 2, 3, 4])              # From list
b = np.zeros((3, 4))                     # All zeros
c = np.ones((2, 3))                      # All ones
d = np.eye(3)                            # Identity matrix
e = np.arange(0, 10, 0.5)               # Evenly spaced sequence
f = np.random.randn(100)                 # Standard normal distribution
g = np.random.randint(0, 10, (3, 3))    # Random integer matrix
```

### 3.2 Array Attributes and Indexing

```python
arr = np.random.randn(32, 3, 28, 28)     # (batch, channel, height, width)
arr.shape    # (32, 3, 28, 28)
arr.dtype    # float64
arr.ndim     # 4

# Slicing
arr[0]           # First sample, shape (3, 28, 28)
arr[0, 0]        # First sample, first channel
arr[:10]         # First 10 samples
arr[:, :, :14]   # Top half of all samples
```

### 3.3 Mathematical Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

a + b        # array([5, 7, 9]) — element-wise addition
a * b        # array([4, 10, 18]) — element-wise multiplication
np.dot(a, b) # 32 — dot product
a @ b        # 32 — matrix multiplication operator (Python 3.5+)
```

| Operation | NumPy | Notes |
|-----------|-------|-------|
| Element-wise +/−/×/÷ | `+`, `-`, `*`, `/` | Same shape or broadcastable |
| Matrix multiplication | `@` or `np.dot()` | Follows linear algebra rules |
| Sum | `np.sum(arr, axis=0)` | axis=0 columns, axis=1 rows |
| Mean | `np.mean(arr)` | Global or per-axis |
| Std | `np.std(arr)` | Global or per-axis |
| Transpose | `arr.T` | Swap dimensions |

### 3.4 Broadcasting

Broadcasting allows arrays of different shapes to be operated on together:

```python
# (3, 4) + (4,) → (3, 4) — vector broadcast to each row of the matrix
matrix = np.ones((3, 4))
bias = np.array([1, 2, 3, 4])
result = matrix + bias      # shape remains (3, 4)

# (32, 1, 28, 28) * (1, 3, 1, 1) → (32, 3, 28, 28)
```

> Broadcasting is the foundation of batch normalization, feature scaling, and many other deep learning operations. Understanding it helps you avoid unnecessary dimension reshuffling.

### 3.5 Linear Algebra

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

A @ B                     # Matrix multiplication
np.linalg.inv(A)          # Matrix inverse
np.linalg.eig(A)          # Eigenvalues and eigenvectors
np.linalg.norm(A)         # Norm
```

---

## IV. Matplotlib — Data Visualization

### 4.1 Basic Plotting

```python
import matplotlib.pyplot as plt

# Line plot — most commonly used for training curves
losses = [0.8, 0.5, 0.3, 0.2, 0.15, 0.12]
plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curve')
plt.legend()
plt.grid(True)
plt.show()
```

### 4.2 Common Chart Types

| Chart Type | Function | Typical Use |
|-----------|---------|------------|
| Line | `plt.plot()` | Training curves, loss/accuracy trends |
| Scatter | `plt.scatter()` | Data distribution, feature visualization |
| Bar | `plt.bar()` | Category comparison, model performance |
| Histogram | `plt.hist()` | Weight distribution, gradient distribution |
| Heatmap | `plt.imshow()` | Image display, confusion matrix, attention maps |
| Subplots | `plt.subplots()` | Multi-chart comparison |

### 4.3 Displaying Images

```python
# Display a single MNIST image
image = dataset[0][0].reshape(28, 28)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title('MNIST Sample')
plt.show()

# Batch display
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    img, label = dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### 4.4 Saving Figures

```python
plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
```

---

## V. PyTorch Basics — The Core of Deep Learning

### 5.1 Tensors

PyTorch Tensors are similar to NumPy arrays but can be accelerated on GPU and support automatic differentiation.

```python
import torch

# Creating tensors
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.zeros(3, 4)
z = torch.randn(32, 1, 28, 28)   # Typical image batch

# Converting to/from NumPy
arr = x.numpy()                    # Tensor → NumPy
tensor = torch.from_numpy(arr)     # NumPy → Tensor

# GPU support
if torch.cuda.is_available():
    x = x.cuda()
```

| Operation | PyTorch | NumPy Equivalent |
|-----------|---------|-----------------|
| All zeros | `torch.zeros(3, 4)` | `np.zeros((3, 4))` |
| Random normal | `torch.randn(3, 4)` | `np.random.randn(3, 4)` |
| Shape | `x.shape` | `arr.shape` |
| Matrix multiply | `x @ y` or `torch.mm(x, y)` | `arr @ brr` |
| Sum | `x.sum()` | `arr.sum()` |
| Reshape | `x.view(-1, 28*28)` | `arr.reshape(-1, 784)` |

### 5.2 Automatic Differentiation

Autograd is the core capability of deep learning frameworks — define the forward computation and the framework automatically computes all gradients.

```python
# Create tensors that require gradients
w = torch.randn(3, requires_grad=True)
x = torch.randn(3)

# Forward computation
loss = (w * x).sum()     # Arbitrary computation graph

# Backpropagation — automatically computes ∂loss/∂w
loss.backward()
print(w.grad)             # Gradients ready
```

> Traditionally, gradient formulas for each model had to be derived and coded by hand. PyTorch's autograd engine automatically tracks the computation graph and backpropagates, letting you focus on model design rather than calculus.

### 5.3 Building Neural Networks

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
        x = x.view(x.size(0), -1)    # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

| Component | Class | Purpose |
|-----------|-------|---------|
| Convolution | `nn.Conv2d` | Extract spatial features |
| Pooling | `nn.MaxPool2d` | Downsampling, reduce computation |
| Fully Connected | `nn.Linear` | Combine features for classification/regression |
| Activation | `nn.ReLU`, `nn.Sigmoid` | Introduce non-linearity |
| Regularization | `nn.Dropout` | Prevent overfitting |
| Loss | `nn.CrossEntropyLoss`, `nn.MSELoss` | Measure prediction error |
| Optimizer | `torch.optim.Adam`, `torch.optim.SGD` | Update parameters |

### 5.4 Training Loop

```python
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()   # Clear gradients
        loss.backward()         # Compute gradients
        optimizer.step()        # Update parameters

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")
```

### 5.5 Data Loading

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

| Concept | Description |
|---------|-------------|
| `Dataset` | Encapsulates a single dataset, implements `__getitem__` and `__len__` |
| `DataLoader` | Batch loading, shuffling, multi-threaded reading |
| `transforms` | Data preprocessing and augmentation pipeline |
| `batch_size` | Number of samples fed to the model at once |

---

## VI. From Python to Deep Learning Practice

The five sections above cover everything you need before starting the MNIST experiment:

1. **Python syntax** — variables, control flow, functions, classes
2. **NumPy** — array operations, broadcasting, linear algebra
3. **Matplotlib** — curve plotting, image display, figure saving
4. **PyTorch Tensors** — creation, operations, GPU acceleration
5. **PyTorch Modeling** — nn.Module, loss functions, optimizers, training loop

> When you start the MNIST experiment, all of this comes together: understand data with NumPy → load data with DataLoader → define the network with nn.Module → optimize with the training loop → visualize results with Matplotlib.

---

*Related notes: [MNIST Experiment Manual](./02-MNIST实验手册.en.md)*
