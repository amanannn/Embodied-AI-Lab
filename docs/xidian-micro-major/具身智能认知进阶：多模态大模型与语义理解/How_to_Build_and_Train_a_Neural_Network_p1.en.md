# How to Build and Train a Neural Network (Part 1): Training Pipeline and Loss Functions

> Xidian University · Embodied Intelligence Micro-Major
> Course Notes

## 1. Datasets

### 1.1 Common Benchmark Datasets

| Dataset | Content | Scale | Use Case |
|---------|---------|-------|----------|
| MNIST | Handwritten digits (grayscale 28×28) | 70k | Entry-level classification |
| CIFAR-10 | Natural images (color 32×32) | 60k | General-purpose classification |

### 1.2 The Three Subsets

| Subset | Purpose | Usage |
|--------|---------|-------|
| **Training Set** | Train model parameters | Forward + backward propagation + weight updates |
| **Validation Set** | Tune hyperparameters, monitor generalization | Forward propagation only; no weight updates |
| **Test Set** | Final evaluation | Used only once to confirm the model's true performance |

### 1.3 Training Pipeline

```
Train model on training set
    ↓
Evaluate on validation set
    ↓
Adjust hyperparameters based on validation performance
    ↓
Repeat training; select the model with the best validation performance
    ↓
Confirm final performance on test set
```

> Analogy for the learning process: **Do problems → Get graded → Correct mistakes**

## 2. The Training Loop

A single training step involves four operations:

```python
outputs = model(inputs)          # 1. Forward: input → output
loss = criterion(outputs, target) # 2. Compute loss: prediction vs. ground truth
optimizer.zero_grad()             # 3. Clear previous gradients (prevent accumulation)
loss.backward()                   # 4. Backward: compute gradient for each parameter
optimizer.step()                  # 5. Update: actually modify network weights
```

| Step | Method | Purpose |
|------|--------|---------|
| Forward | `model(inputs)` | Input passes through layers to produce predictions |
| Loss | `criterion(outputs, target)` | Measure the gap between predictions and ground truth |
| Zero Grad | `optimizer.zero_grad()` | Clear accumulated gradients from the previous round |
| Backward | `loss.backward()` | Chain-rule differentiation; compute how each parameter should change |
| Step | `optimizer.step()` | Update weights based on gradients and learning rate |

## 3. Loss Functions

### 3.1 What is a Loss Function?

> Loss Function = The "scoring rubric" for model learning

```
Prediction → Compare ← Ground Truth
                  ↓
        Error Magnitude = Loss
```

A loss function quantifies model performance as a scalar — the goal of training is to minimize this value.

### 3.2 How to Choose a Loss Function?

Core principle: **the loss function must align with the task objective**.

| Task | Recommended Loss | Reason |
|------|-----------------|--------|
| Regression | MSE (Mean Squared Error) | Heavily penalizes large errors |
| Regression (with outliers) | MAE (Mean Absolute Error) | More robust to outliers |
| Binary classification | BCE (Binary Cross-Entropy) | Probabilistic output; smooth gradients |
| Multi-class classification | Cross-Entropy | Paired with Softmax; stable, fast-converging gradients |

> Why does MNIST use Cross-Entropy? Multi-class classification + Softmax output layer — the combined gradient has a clean form and converges quickly.

### 3.3 How to Interpret Training Logs?

Decreasing loss does not guarantee good generalization. Three typical states:

| State | Train Loss | Val Loss | Accuracy | Meaning |
|-------|-----------|---------|----------|---------|
| **Normal Training** | ↓ Decreasing | ↓ Decreasing | ↑ Increasing | Model is learning and generalizing well |
| **Overfitting** | ↓ Still decreasing | ↑ Rising instead | Getting worse | Model is memorizing the training set |
| **Underfitting** | Very high | Very high | Low | Model lacks expressive capacity |

## 4. Overfitting and Underfitting

### 4.1 Overfitting

**Causes**:

- Training data too homogeneous; insufficient samples
- Excessive noise in training data
- Model too complex

**Solutions**:

| Method | Description |
|--------|-------------|
| Data Augmentation | Rotation, cropping, flipping, color perturbation |
| Expand Dataset | Collect more real data |
| Regularization (L1 / L2) | Constrain weights from becoming too large |
| Dropout | Randomly drop neurons; force redundant learning |
| Simplify Model | Occam's Razor — prefer simpler, effective models |

### 4.2 Underfitting

**Causes**: Too few feature dimensions; model too simple to capture patterns in the training data.

**Solutions**:

- Increase feature dimensionality
- Increase model complexity (more layers, more neurons)
- Refine training data and features (remove noisy features)

## 5. Optimizers

### 5.1 Batch Gradient Descent (BGD)

Uses the entire training dataset to compute the gradient:

$$w_{new} = w_{old} - \eta \cdot \nabla L_{\text{all data}}$$

| Aspect | Characteristic |
|--------|---------------|
| Gradient direction | Accurate |
| Computation cost | High (iterates over all data per step) |
| Weakness | When gradient is zero (saddle point / local minimum), weights stop updating — gradient descent fails |

### 5.2 Stochastic Gradient Descent (SGD)

Uses one or a small batch of samples to compute the gradient at each step:

$$w_{new} = w_{old} - \eta \cdot \nabla L_{\text{single or small batch}}$$

| Aspect | Characteristic |
|--------|---------------|
| Gradient direction | Noisy (fluctuates) |
| Computation cost | Low |
| Advantage | Noise helps escape saddle points and local minima |
| Convergence | Advances amid fluctuations; eventually converges |

> **Partial derivatives determine the climbing direction; the learning rate determines the step size.**

---

*Note status: Part 1 — first draft complete. Part 2 now covers CNN hands-on, hardware, and GANs. See [Part 2](./How_to_Build_and_Train_a_Neural_Network_p2.en.md).*

*Related notes: [What is a Neural Network](../具身智能感知基础：机器学习与深度学习/What_is_neural_network.en.md), [What is a Convolutional Network](../具身智能感知基础：机器学习与深度学习/What_is_Convolution_network.en.md), [Part 2: CNN Hands-on & GANs](./How_to_Build_and_Train_a_Neural_Network_p2.en.md)*
