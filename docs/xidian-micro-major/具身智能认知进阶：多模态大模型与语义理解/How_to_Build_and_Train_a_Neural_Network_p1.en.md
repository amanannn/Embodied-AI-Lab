# How to Build and Train a Neural Network (Part 1): Training Pipeline & Loss Functions

> Xidian University · Embodied AI Micro-major
> Course Notes

---

## 1. Dataset

### 1.1 Common Benchmark Datasets

| Dataset | Content | Image Size | Scale | Usage |
|---------|---------|-----------|-------|-------|
| MNIST | Handwritten digits (grayscale) | 28×28×1 | 60k train + 10k test | Entry-level classification |
| Fashion-MNIST | Clothing (grayscale) | 28×28×1 | 60k + 10k | MNIST replacement (harder) |
| CIFAR-10 | Natural images (color) | 32×32×3 | 50k + 10k | General classification |
| CIFAR-100 | Natural images (fine-grained) | 32×32×3 | 50k + 10k | Fine-grained classification |
| ImageNet | Natural images (color) | 224×224×3 | 1.28M + 50k | Large-scale classification benchmark |
| COCO | Natural scenes | Variable | 330k | Detection / Segmentation / Captioning |
| DIV2K | Super-resolution | 2K resolution | 800 + 100 | Super-resolution |

### 1.2 The Three Subsets of a Dataset

| Subset | Role | Frequency of Use | Whom It Affects |
|--------|------|-----------------|-----------------|
| **Training set** (Train) | Updates model parameters | Every epoch | Model weights |
| **Validation set** (Val) | Tune hyperparameters, monitor generalization, early stopping | Every epoch evaluation | Hyperparameter selection |
| **Test set** (Test) | Final performance evaluation | **Used only once** | Paper-reported metrics |

#### Typical Split Ratios

| Data Size | Train : Val : Test |
|-----------|-------------------|
| < 10k | 60 : 20 : 20 |
| 10k~100k | 80 : 10 : 10 |
| > 100k | 90 : 5 : 5 |
| > 1M | 98 : 1 : 1 |

### 1.3 Training Pipeline Closed Loop

```
Train the model on the training set
    ↓
Evaluate the model on the validation set (no weight update)
    ↓
Adjust hyperparameters based on validation performance (lr, batch_size, depth...)
    ↓
Repeat training, save the best model on validation set
    ↓
Confirm final performance on the test set (only once!)
```

> Learning analogy: **Practice problems (Train) → Mock exam (Val) → Correction → Repeat → College entrance exam (Test)**. Repeatedly tuning hyperparameters on the mock exam and then cheating on the college entrance exam (using the test set as the validation set) is the most common mistake in research.

---

## 2. Training Loop

### 2.1 Five Operations in One Training Step

```python
outputs = model(inputs)             # 1. Forward: input → output
loss = criterion(outputs, target)   # 2. Loss: prediction vs ground truth
optimizer.zero_grad()               # 3. Zero grad: clear gradients from previous step
loss.backward()                     # 4. Backward: compute gradients for each parameter
optimizer.step()                    # 5. Step: gradient descent updates weights
```

| Step | Method | Input | Output | Common Mistake |
|------|--------|-------|--------|----------------|
| Forward | `model(x)` | Data batch | Predictions | Forgetting `model.train()/eval()` |
| Loss | `criterion(out, y)` | Predictions + labels | Scalar | Wrong loss function for classification/regression |
| Zero Grad | `optimizer.zero_grad()` | — | — | **Forgetting to zero leads to gradient accumulation** |
| Backward | `loss.backward()` | Scalar | Gradients | `retain_graph` needed for multiple backward passes |
| Step | `optimizer.step()` | — | Updated weights | Order with zero_grad must not be swapped |

> The order `zero_grad()` → `backward()` → `step()` must never be wrong — this is the "heartbeat" of all PyTorch training.

### 2.2 Complete Epoch Loop

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total



def validate(model, loader, criterion, device):
    model.eval()  # Disable Dropout/BatchNorm training behavior
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():  # No gradient computation, saves memory and speeds up
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total
```

### 2.3 `model.train()` vs `model.eval()`

| Component | During `train()` | During `eval()` | Why |
|-----------|-----------------|-----------------|-----|
| Dropout | **Active**, randomly drops | Disabled, all neurons participate | Eval needs deterministic output |
| BatchNorm | Uses **current batch** statistics | Uses **global statistics accumulated during training** | Eval batch may be too small |

```python
# Classic forget-to-write model.eval() bug
model.train()
# ... training code ...
# Directly start validation ← BUG! Dropout still active, inconsistent validation results each time
val_acc = validate(model, val_loader)  # Unreliable results
```

---

## 3. Loss Functions in Depth

### 3.1 Mathematical Derivation of Cross-Entropy

Multi-class cross-entropy (with Softmax):

$$L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

where $\hat{y}_c = \frac{e^{z_c}}{\sum_j e^{z_j}}$ (Softmax).

**Combined gradient**:

$$\frac{\partial L}{\partial z_c} = \hat{y}_c - y_c$$

This form is extremely elegant — the gradient is simply "prediction minus ground truth." This is why classification tasks almost universally use CrossEntropyLoss.

### 3.2 Interpreting Training Logs

```text
Epoch 1: Train Loss 0.85, Train Acc 70.2% | Val Loss 0.62, Val Acc 78.1%
Epoch 2: Train Loss 0.45, Train Acc 84.3% | Val Loss 0.48, Val Acc 83.5%
Epoch 3: Train Loss 0.32, Train Acc 89.1% | Val Loss 0.40, Val Acc 86.2%
Epoch 4: Train Loss 0.21, Train Acc 93.5% | Val Loss 0.38, Val Acc 87.0%
Epoch 5: Train Loss 0.15, Train Acc 96.2% | Val Loss 0.42, Val Acc 86.5%  ← Overfitting signal!
Epoch 6: Train Loss 0.09, Train Acc 97.8% | Val Loss 0.47, Val Acc 85.8%  ← Continues worsening
```

| State | Train Loss | Val Loss | Train-Val Gap | Action |
|-------|-----------|---------|---------------|--------|
| **Normal training** | ↓↓ | ↓↓ | Small and stable | Continue |
| **Overfitting** | ↓↓ | ↑ | Starts widening | Early stopping / Regularization / Dropout |
| **Underfitting** | High → stops decreasing | High → stops decreasing | Small | Increase model capacity |
| **Not converging** | Oscillating | Oscillating | Large | Reduce learning rate |

---

## 4. Overfitting and Underfitting

### 4.1 Mathematical Perspective on Overfitting

Overfitting is essentially the model memorizing **noise** in the training set rather than **signal**:

$$Error = Bias^2 + Variance + IrreducibleError$$

| Situation | Bias | Variance | Behavior |
|-----------|------|----------|----------|
| Underfitting | High | Low | Poor training and validation |
| Just right | Low | Low | Good training and validation |
| Overfitting | Low | High | Good training, poor validation |

### 4.2 Anti-Overfitting Toolbox

| Method | Principle | Strength | Implementation |
|--------|-----------|----------|----------------|
| **Data Augmentation** | Expand training distribution | ★★★ | Rotation / Crop / Flip / Color |
| **Dropout** | Randomly drop neurons | ★★★ | `nn.Dropout(p=0.5)` |
| **L2 Regularization** | Penalize large weights | ★★☆ | `optimizer(weight_decay=1e-4)` |
| **Early Stopping** | Stop when Val loss plateaus | ★★☆ | Save best checkpoint |
| **BatchNorm** | Layer-wise normalization, implicit regularization | ★☆☆ | `nn.BatchNorm2d` |
| **Simplify Model** | Reduce layers/neurons | ★☆☆ | Occam's razor |

#### How Dropout Works

```python
# Training phase
# Randomly set neuron outputs to 0 with probability p
# Equivalent to training a "sub-network" at each iteration
x = torch.randn(100, 256)
dropout = nn.Dropout(p=0.5)
out_train = dropout(x)  # ~50% of values are 0

# Testing phase: Dropout automatically disabled — all neurons participate, output scaled by (1-p)
model.eval()
out_test = model(x)  # No dropout
```

> Intuition behind Dropout: No neuron is allowed to "lazily rely" on other neurons — every neuron must learn useful features on its own, because its colleagues might get "fired" at any moment.

### 4.3 Solutions for Underfitting

| Method | Applicable Scenario |
|--------|---------------------|
| Increase depth/width | Model is too simple |
| Switch to a stronger architecture | Task is complex |
| Add features | Insufficient input information |
| Reduce regularization | Constraints are too strong |
| Train longer | Not yet converged |

---

## 5. Optimizers

### 5.1 SGD Family

| Optimizer | Update Rule | Features |
|-----------|-------------|----------|
| **SGD** | $w_{t+1} = w_t - \eta g_t$ | Baseline |
| **SGD+Momentum** | $v_t = \beta v_{t-1} + g_t$; $w_{t+1} = w_t - \eta v_t$ | Inertia acceleration, escapes saddle points |
| **SGD+Nesterov** | First step with momentum, then compute gradient at that point | "Look-ahead" gradient, more stable |

**Intuition behind Momentum**: A ball rolling down a hill — influenced not only by the current slope (gradient) but also by its previous velocity (momentum).

$$v_t = \beta v_{t-1} + (1 - \beta)g_t$$

### 5.2 Adam Family

| Optimizer | Key Improvement | Use Case |
|-----------|-----------------|----------|
| **Adam** | Adaptive learning rate (first + second moment estimates) | **Default choice for beginners** |
| **AdamW** | Adam + decoupled weight decay | Modern default recommendation |
| **RMSprop** | Adaptive learning rate (second moment only) | RNN training |

#### Mathematics of Adam

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t \quad \text{(First moment — mean of gradients)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2 \quad \text{(Second moment — variance of gradients)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(Bias correction)}$$
$$w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### 5.3 How to Choose an Optimizer

| Scenario | Recommendation | Initial Learning Rate |
|----------|----------------|----------------------|
| Rapid prototyping | Adam / AdamW | $10^{-3}$ |
| CV classification/detection | AdamW + Cosine Schedule | $10^{-3}$ |
| NLP Transformer | AdamW + Warmup | $10^{-4}$ |
| GAN | Adam ($\beta_1=0.5$) | $2\times10^{-4}$ |
| Pursuit of extreme reproducibility | SGD+Momentum | $10^{-1}$ (needs careful tuning) |

```python
# The most commonly used trio
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()
```

---

## 6. Learning Rate Scheduling

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| **Step** | Multiply lr by γ every N epochs | Simple and crude |
| **Cosine** | Cosine curve decays from initial to near zero | CV standard |
| **Warmup** | Linearly increase lr for first N steps, then normal scheduling | Transformer |
| **ReduceLROnPlateau** | Automatically reduce lr when Val loss plateaus | Automatic mode |

```python
# Warmup + Cosine combination (Transformer standard)
def warmup_cosine(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps         # Linear increase
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)
```

---

*Related notes: [What is a Neural Network](../具身智能感知基础：机器学习与深度学习/What_is_neural_network.md), [Part 2: CNN in Practice & GAN](./How_to_Build_and_Train_a_Neural_Network_p2.md)*
