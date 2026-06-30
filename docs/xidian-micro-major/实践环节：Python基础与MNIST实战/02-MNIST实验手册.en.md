# MNIST Experiment Manual — First Practice Session

> Xidian University · Embodied AI Micro-Major
> Course Notes · Practice Module

---

## I. Experiment Objectives

This is the **first hands-on experiment** of the entire micro-major. The goal is to run through the complete deep learning project pipeline:

1. Set up Python + PyTorch development environment
2. Load and explore the MNIST handwritten digit dataset
3. Build a Convolutional Neural Network (CNN) from scratch
4. Complete training, evaluation, and visualization
5. Save the model and perform independent inference

> You don't need to understand every detail of CNNs in this experiment — run through the full pipeline first, build intuition, theory will unfold in subsequent courses.

---

## II. Environment Setup

### 2.1 Install Dependencies

```bash
pip install torch torchvision matplotlib numpy
```

Or use the bundled `requirements.txt`:

```bash
cd code
pip install -r requirements.txt
```

### 2.2 Dependency Overview

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Deep learning framework |
| `torchvision` | ≥0.15.0 | Dataset download + image preprocessing |
| `matplotlib` | ≥3.7.0 | Visualization: training curves, image display |
| `numpy` | ≥1.24.0 | Numerical computation utilities |

### 2.3 Verify Installation

```python
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Quick tensor test
x = torch.randn(2, 3)
print(f"Tensor created successfully, shape: {x.shape}")
```

---

## III. MNIST Dataset

### 3.1 Dataset Overview

MNIST (Modified National Institute of Standards and Technology) is the "Hello World" of deep learning:

| Attribute | Value |
|-----------|-------|
| Content | 0-9 handwritten digit grayscale images |
| Image Size | 28 × 28 pixels |
| Training Set | 60,000 images |
| Test Set | 10,000 images |
| Classes | 10 |
| Pixel Range | 0 ~ 255 |

### 3.2 Loading Data

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.ToTensor(),                          # PIL → Tensor, [0,255] → [0,1]
    transforms.Normalize((0.1307,), (0.3081,))     # Normalize
])

# Download and load
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
```

### 3.3 Data Exploration

Before training, look at what the data actually looks like:

```python
import matplotlib.pyplot as plt

# Display first 16 images
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# Check class distribution
labels = [train_dataset[i][1] for i in range(len(train_dataset))]
for digit in range(10):
    print(f"Digit {digit}: {labels.count(digit)} images")
```

> Before you start modeling, "look" at the data — this is the most important habit in deep learning. Verify shape, distribution, and content to avoid wasting time on bad data.

---

## IV. Building a CNN Model

### 4.1 Network Architecture

```python
import torch.nn as nn

class MNIST_CNN(nn.Module):
    """
    MNIST handwritten digit recognition CNN

    Input:  (batch, 1, 28, 28)
    Output: (batch, 10) — logits for 10 classes
    """
    def __init__(self):
        super().__init__()

        # Convolution 1: 1→32 channels, 3×3 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Convolution 2: 32→64 channels, 3×3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Pooling: 2×2 max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)   # 7×7 is the size after two pooling ops
        self.fc2 = nn.Linear(128, 10)            # 10 class outputs
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv block 1
        x = self.relu(self.conv1(x))    # (1,28,28) → (32,28,28)
        x = self.pool(x)                 # (32,28,28) → (32,14,14)

        # Conv block 2
        x = self.relu(self.conv2(x))    # (32,14,14) → (64,14,14)
        x = self.pool(x)                 # (64,14,14) → (64,7,7)

        # Flatten + classify
        x = x.view(x.size(0), -1)       # (64,7,7) → (64*7*7)
        x = self.relu(self.fc1(x))      # (64*7*7) → (128)
        x = self.dropout(x)              # Dropout prevents overfitting
        x = self.fc2(x)                  # (128) → (10)
        return x
```

### 4.2 Shape Computation

| Layer | Input Shape | Output Shape | Parameters |
|-------|------------|-------------|------------|
| Conv1 (1→32, k=3, p=1) | 1×28×28 | 32×28×28 | 32×(1×9+1)=320 |
| MaxPool (k=2) | 32×28×28 | 32×14×14 | 0 |
| Conv2 (32→64, k=3, p=1) | 32×14×14 | 64×14×14 | 64×(32×9+1)=18,496 |
| MaxPool (k=2) | 64×14×14 | 64×7×7 | 0 |
| FC1 (64×7×7→128) | 3136 | 128 | 3136×128+128=401,536 |
| Dropout | — | — | 0 |
| FC2 (128→10) | 128 | 10 | 128×10+10=1,290 |

Convolution output size formula:

$$H_{out} = \frac{H_{in} + 2P - K}{S} + 1$$

Where $P$ is padding, $K$ is kernel size, $S$ is stride.

---

## V. Training the Model

### 5.1 Configure Hyperparameters

```python
# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

# Model, loss function, optimizer
model = MNIST_CNN()
criterion = nn.CrossEntropyLoss()           # Multi-class cross-entropy
optimizer = torch.optim.Adam(               # Adam optimizer
    model.parameters(), lr=LEARNING_RATE
)
```

### 5.2 Training Loop

```python
def train(model, train_loader, criterion, optimizer, epochs):
    """Complete training loop"""
    model.train()
    losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # 1. Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 2. Backward pass
            optimizer.zero_grad()   # Clear gradients (or they accumulate!)
            loss.backward()         # Compute gradients
            optimizer.step()        # Update weights

            # 3. Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            losses.append(loss.item())

        # Print per epoch
        acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")

    return losses

losses = train(model, train_loader, criterion, optimizer, EPOCHS)
```

### 5.3 Training Key Points

| Point | Explanation |
|-------|-------------|
| `model.train()` | Enable training mode (Dropout and BatchNorm behave differently) |
| `optimizer.zero_grad()` | **Must** clear gradients before each step, or they accumulate |
| `loss.backward()` | Backpropagation, compute gradients for all parameters |
| `optimizer.step()` | Update parameters using gradients |
| `outputs.max(1)` | Take the index of the max logit as the predicted class |

> The three core lines of the training loop — `zero_grad()`, `loss.backward()`, `optimizer.step()` — are the standard paradigm for all PyTorch training. Understand these three steps and you understand the engine of deep learning training.

---

## VI. Evaluating the Model

### 6.1 Test Set Evaluation

```python
def evaluate(model, test_loader):
    """Evaluate model on test set"""
    model.eval()    # Evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():    # Disable gradient computation, save memory
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

| Mode | `model.train()` | `model.eval()` |
|------|----------------|----------------|
| Dropout | Active | Disabled |
| BatchNorm | Uses batch statistics | Uses running statistics |
| Gradients | Required | `torch.no_grad()` disables them |

### 6.2 Visualizing Predictions

```python
def visualize_predictions(model, test_loader, num_images=16):
    """Visualize model predictions"""
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

### 6.3 Training Curve

```python
plt.figure(figsize=(10, 4))
plt.plot(losses, alpha=0.3, color='blue', label='Batch Loss')
# Smoothing
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

## VII. Saving and Loading Models

### 7.1 Saving

```python
# Method 1: Save the entire model (simple but large file)
torch.save(model, 'mnist_model_full.pth')

# Method 2: Save only parameters (recommended)
torch.save(model.state_dict(), 'mnist_model.pth')
print("Model saved")
```

### 7.2 Loading for Inference

```python
# Reconstruct model structure + load parameters
model = MNIST_CNN()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()
print("Model loaded, ready for inference")

# Single-image inference
image, label = test_dataset[0]
with torch.no_grad():
    output = model(image.unsqueeze(0))   # Add batch dimension
    prob = output.softmax(dim=1)          # Convert to probabilities
    pred = prob.argmax(dim=1).item()
    confidence = prob[0, pred].item()

print(f"Predicted: {pred}, True: {label}, Confidence: {confidence:.2%}")
```

---

## VIII. Experiment Checklist

Complete each item below and you'll have solid intuition for deep learning practice:

- [ ] Environment set up, `import torch` runs without error
- [ ] MNIST dataset downloaded and visualized
- [ ] CNN model defined, printed `model` to inspect structure
- [ ] Trained for 5 epochs, loss decreasing steadily
- [ ] Test accuracy > 95%
- [ ] Visualized predictions, observed which digits are easily confused
- [ ] Plotted training curve, observed loss descent trend
- [ ] Saved model parameter file
- [ ] Independently loaded model for single-image inference

---

## IX. Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `CUDA out of memory` | Insufficient GPU memory | Reduce batch_size, or switch to CPU |
| `shape mismatch` | FC layer input size miscalculated | Recheck feature map dimensions after two pooling layers |
| Accuracy stagnates | Learning rate issue or unshuffled data | Lower learning rate, confirm `shuffle=True` |
| Loss won't decrease | Gradients not zeroed | Confirm `optimizer.zero_grad()` inside the loop |
| MNIST download fails | Network issue | Manually download and place in `data/` directory |

> Don't be afraid of errors. Most of your time in deep learning is spent dealing with shapes, dtypes, and dimensions — this is normal, and it's part of the growth.

---

*Related notes: [Python Quick Start](./01-Python快速入门.en.md)*
