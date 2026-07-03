# What is a Neural Network

> Xidian University · Embodied Intelligence Micro-Major
> Course Notes

---

## 1. Three Core Questions of Neural Networks

Building a neural network essentially answers three questions:

| Question | Corresponding Component | Example |
|----------|------------------------|---------|
| 1. What structure to choose? | **Network Architecture** | How many layers? How many neurons per layer? What activation function? |
| 2. How to measure performance? | **Loss Function** | How far is the prediction from the ground truth? |
| 3. How to improve? | **Optimization Algorithm** | How to adjust parameters to reduce the loss? |

> The answers to these three questions form the common skeleton of all neural networks, from the simple perceptron to GPT.

---

## 2. Feedforward Neural Network

### 2.1 Basic Concepts

Feedforward Neural Network, also known as:

- **Fully Connected Network**
- **Multi-Layer Perceptron (MLP)**

Structure: Input Layer &rarr; Hidden Layer(s) (can be multiple) &rarr; Output Layer, with information flowing unidirectionally and no loops.

### 2.2 Single Layer Computation

The mathematical expression of a fully connected layer:

$$h = f(Wx + b)$$

| Symbol | Meaning | Shape (Example) |
|--------|---------|-----------------|
| $x$ | Input vector | $(d_{in}, 1)$ |
| $W$ | Weight matrix | $(d_{out}, d_{in})$ |
| $b$ | Bias vector | $(d_{out}, 1)$ |
| $f$ | Activation function | &mdash; |
| $h$ | Output | $(d_{out}, 1)$ |

```python
# Full implementation of a single layer (including initialization)
layer = nn.Linear(in_features=784, out_features=256)
x = torch.randn(32, 784)       # (batch, input_dim)
h = torch.relu(layer(x))       # (batch, 256)
```

### 2.3 Stacking Multiple Layers

$$h^{(1)} = f(W^{(1)}x + b^{(1)})$$
$$h^{(2)} = f(W^{(2)}h^{(1)} + b^{(2)})$$
$$\hat{y} = W^{(3)}h^{(2)} + b^{(3)}$$

> Multiple layers without activation functions are equivalent to a single layer &mdash; because the composition of linear transformations is still a linear transformation.

---

## 3. Loss Functions

### 3.1 Why Define a Loss Function?

The loss function quantifies model performance into a single scalar. "How bad the model is" = how high the loss is. The goal of training: **minimize this value**.

### 3.2 Regression Task Losses

| Loss Function | Formula | Gradient | Characteristics |
|---------------|---------|----------|-----------------|
| **MSE** | $L = \frac{1}{N}\sum(y - \hat{y})^2$ | $\frac{\partial L}{\partial \hat{y}} = \frac{2}{N}(\hat{y} - y)$ | Heavy penalty on large errors, insensitive to small errors |
| **MAE** | $L = \frac{1}{N}\sum|y - \hat{y}|$ | $\frac{\partial L}{\partial \hat{y}} = \pm \frac{1}{N}$ | Constant gradient, robust to outliers |
| **Huber** | MSE for small errors, MAE for large errors | &mdash; | Combines advantages of both |

**MSE vs MAE Comparison**:

| Aspect | MSE | MAE |
|--------|-----|-----|
| Outlier Sensitivity | High (quadratic penalty) | Low (linear penalty) |
| Optimal Solution | Mean | Median |
| Gradient | Increases with error | Constant at &pm;1 |
| Use Case | Clean data | Data with outliers |

### 3.3 Classification Task Losses

| Loss Function | Formula | Use Case |
|---------------|---------|----------|
| **BCE** | $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | Binary classification |
| **CE** | $L = -\sum_{c} y_c \log \hat{y}_c$ | Multi-class classification (with Softmax) |

**Why use cross-entropy instead of MSE for classification?**

Cross-entropy combined with Softmax has a gradient of $\hat{y} - y$, which is elegant in form &mdash; the more wrong the prediction, the larger the gradient; the more correct the prediction, the smaller the gradient. MSE combined with Sigmoid/Softmax is prone to gradient saturation (gradient approaches 0 when predictions are close to 0 or 1).

---

## 4. Activation Functions

### 4.1 Why Are Activation Functions Needed?

$$h = W_2 \cdot f(W_1 x + b_1) + b_2$$

If $f(x) = x$ (identity mapping), then:

$$h = W_2 W_1 x + W_2 b_1 + b_2 = W' x + b'$$

Multiple layers degenerate into a single layer. Activation functions introduce **nonlinearity**, allowing the network to fit arbitrarily complex functions.

### 4.2 Common Activation Functions in Detail

| Activation Function | Formula | Range | Advantages | Disadvantages |
|---------------------|---------|-------|------------|---------------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $(0, 1)$ | Smooth, interpretable as probability | Vanishing gradient in saturation region; output not zero-centered |
| **Tanh** | $\tanh(x)$ | $(-1, 1)$ | Zero-centered, larger gradient than Sigmoid | Still has vanishing gradient in saturation region |
| **ReLU** | $f(x) = \max(0, x)$ | $[0, +\infty)$ | Simple computation, mitigates vanishing gradient | Dead ReLU (gradient is 0 on negative half-axis) |
| **Leaky ReLU** | $f(x) = \max(0.01x, x)$ | $(-\infty, +\infty)$ | Solves Dead ReLU | Negative slope needs manual setting |
| **GELU** | $x \cdot \Phi(x)$ | $(-\infty, +\infty)$ | Standard in Transformers, smooth nonlinearity | Slightly more complex computation |
| **Swish** | $x \cdot \sigma(x)$ | $(-\infty, +\infty)$ | Self-gating, used in EfficientNet | Computational cost |

#### The Dead ReLU Problem

```python
# When a neuron outputs 0 for all inputs, the gradient is permanently 0 — the neuron "dies"
x = torch.randn(1000, 1) * 0.1 - 5   # inputs concentrated on the negative half-axis
relu = nn.ReLU()
print(f'Ratio of non-zero outputs: {(relu(x) > 0).float().mean():.1%}')  # may be close to 0%
```

**Solutions**: Leaky ReLU / PReLU (give a small slope on the negative half-axis), proper initialization (He initialization).

### 4.3 Activation Function Selection Guide

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| Hidden layers (CNN) | ReLU / Leaky ReLU | Simple and efficient |
| Hidden layers (Transformer) | GELU | Smooth, works well with Attention |
| Binary classification output | Sigmoid | Output (0,1) interpretable as probability |
| Multi-class classification output | Softmax | Outputs a distribution summing to 1 |
| Regression output | None (identity) | Output can be any real number |

---

## 5. Forward Propagation and Backpropagation

### 5.1 Full Training Process

```
Randomly initialize parameters
    |
    |&rarr; Forward Propagation: input &rarr; layer-by-layer computation &rarr; output predictions
    |       |
    |       |&rarr; Compute loss: measure the gap between predictions and ground truth using the loss function
    |       |
    |       |&rarr; Backpropagation: apply the chain rule to compute gradients layer by layer
    |       |
    |       |&rarr; Update parameters: w = w - &eta; &times; &part;loss/&part;w
    |       |
    |&larr; Return to forward propagation (repeat until convergence)
```

### 5.2 The Mathematical Essence of Backpropagation &mdash; The Chain Rule

Consider a simple two-layer network:

$$z^{(1)} = W^{(1)}x$$
$$h = f(z^{(1)})$$
$$z^{(2)} = W^{(2)}h$$
$$\hat{y} = z^{(2)}$$
$$L = \frac{1}{2}(\hat{y} - y)^2$$

Computing the gradient with respect to $W^{(2)}$:

$$\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial W^{(2)}}$$

$$= (\hat{y} - y) \cdot 1 \cdot h^T$$

Computing the gradient with respect to $W^{(1)}$ (across the activation function):

$$\frac{\partial L}{\partial W^{(1)}} = (\hat{y} - y) \cdot W^{(2)T} \cdot f'(z^{(1)}) \cdot x^T$$

> **The essence of the chain rule**: The derivative of a composite function equals the product of the local derivatives at each layer. Backpropagation computes these local derivatives in order from "output to input", layer by layer.

### 5.3 Automatic Differentiation in PyTorch

```python
x = torch.tensor([1.0, 2.0], requires_grad=False)
W = torch.tensor([[0.5, -0.3], [0.1, 0.8]], requires_grad=True)

# Forward
z = W @ x                       # (2,)
loss = (z - torch.tensor([1.0, 0.0])).pow(2).sum()

# Backward — one line does all gradient computation
loss.backward()
print(W.grad)                    # &part;loss/&part;W is automatically computed
# tensor([[ 0.2000,  0.4000],
#         [-0.5200, -1.0400]])
```

---

## 6. Gradient Descent

### 6.1 Core Formula

$$w_{new} = w_{old} - \eta \cdot \nabla_w L$$

### 6.2 The Effect of Learning Rate

| Learning Rate &eta; | Effect | Behavior |
|---------------------|--------|----------|
| Too small ($10^{-5}$) | Extremely slow convergence | Loss barely changes after tens of thousands of steps |
| Moderate ($10^{-3}$) | Stable descent | Loss decreases smoothly, eventually converges |
| Too large ($10^{-1}$) | Oscillation | Loss fluctuates wildly, may not converge |
| Way too large ($1$) | Divergence | Loss explodes, NaN |

### 6.3 Three Variants of Gradient Descent

| Variant | Data Used Per Step | Gradient Direction | Computation | Escaping Local Minima |
|---------|--------------------|--------------------|-------------|-----------------------|
| **BGD** | Entire dataset | Most accurate | Largest | Difficult |
| **SGD** | 1 sample | Noisiest | Smallest | Easy |
| **MBGD** | 1 batch (32&ndash;256) | Some noise | Moderate | Relatively easy |

```python
# MBGD implementation in PyTorch
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# DataLoader automatically splits data into batches; each batch is one MBGD update
```

> In practice, **MBGD (Mini-Batch Gradient Descent)** is the most commonly used &mdash; it balances computational efficiency and gradient accuracy.

---

## 7. Parameter Initialization

### 7.1 Why Is Initialization Important?

Poor initialization can lead to:
- **Vanishing gradients**: Weights too small &rarr; signal decays layer by layer &rarr; almost no gradient in deep layers
- **Exploding gradients**: Weights too large &rarr; signal amplifies layer by layer &rarr; gradient overflow (NaN)

### 7.2 Common Initialization Methods

| Method | Formula | Suitable Activation Functions |
|--------|---------|-----------------------------|
| **Xavier** | $W \sim U[-\frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}]$ | Sigmoid, Tanh |
| **He** | $W \sim N(0, \sqrt{\frac{2}{n_{in}}})$ | ReLU and its variants |

```python
# PyTorch uses He initialization by default (friendly to ReLU)
conv = nn.Conv2d(3, 64, 3)
# Can also be specified manually
nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
```

> **Intuition**: He initialization keeps the variance stable after ReLU activation &mdash; the larger $n_{in}$ is, the smaller each weight becomes, ensuring that the input variance does not explode or decay across layers.

---

## 8. Regularization: Preventing Overfitting

| Method | Principle | PyTorch |
|--------|-----------|---------|
| **L1 Regularization** | Penalizes sum of absolute weights &rarr; sparse weights | `weight_decay` (manually) |
| **L2 Regularization** | Penalizes sum of squared weights &rarr; weight decay | `optimizer(weight_decay=1e-4)` |
| **Dropout** | Randomly drops neurons, forces redundant learning | `nn.Dropout(p=0.5)` |
| **BatchNorm** | Normalizes between layers, stabilizes training | `nn.BatchNorm2d(64)` |
| **Early Stopping** | Stop when validation loss stops decreasing | Manual implementation |
| **Data Augmentation** | Expands training data diversity | `torchvision.transforms` |

### L1 vs L2 Regularization

| Regularization | Penalty Term | Effect |
|----------------|--------------|--------|
| L1 | $\lambda\sum\|w_i\|$ | Produces sparse weights (many weights become 0), feature selection |
| L2 | $\lambda\sum w_i^2$ | Shrinks weights overall but not to zero, numerical stability |

$$L_{total} = L_{original} + \lambda \cdot R(W)$$

---

## 9. Summary: Training Checklist from Scratch

- [ ] Define the network structure (how many layers, how large, what activation function)
- [ ] Choose the weight initialization method (ReLU &rarr; He, Sigmoid &rarr; Xavier)
- [ ] Choose the loss function (regression &rarr; MSE, classification &rarr; CrossEntropy)
- [ ] Choose the optimizer (Adam is the go-to first choice)
- [ ] Set the learning rate (start at $10^{-3}$)
- [ ] Training loop: forward &rarr; loss &rarr; zero_grad &rarr; backward &rarr; step
- [ ] Monitor Train Loss / Val Loss, determine overfitting / underfitting
- [ ] Hyperparameter tuning: learning rate, batch_size, network depth/width, regularization

---

*Related notes: [What is a Convolutional Network](./What_is_Convolutional_Network.en.md), [From Traditional Image Processing to Deep Learning](./From_traditional_image_processing_to_deep_learning.en.md), [Commonly Used Convolution Operators](./Commonly_used_convolution_operators.en.md), [How to Build and Train a Neural Network (Part 1)](../具身智能认知进阶：多模态大模型与语义理解/How_to_Build_and_Train_a_Neural_Network_p1.en.md)*
