# What is a Neural Network

> Xidian University · Embodied Intelligence Micro-Major
> Course Notes

## 1. Three Core Questions

Building a neural network fundamentally answers three questions:

1. **Define the Network**: What structure? How many layers? How many neurons per layer?
2. **Loss Function**: How to measure the gap between predictions and ground truth?
3. **Network Optimization**: How to adjust parameters to minimize the loss?

## 2. Feedforward Neural Networks

A Feedforward Neural Network (FNN) is also known as:

- **Fully Connected Network**
- **Multi-Layer Perceptron** (MLP)

Structure: Input Layer → Hidden Layers (one or more) → Output Layer. Information flows in one direction — no cycles.

## 3. Loss Functions

The loss function measures the gap between model predictions and true values:

| Loss Function | Full Name | Use Case |
|--------------|-----------|----------|
| MSE | Mean Squared Error | Regression tasks |
| MAE | Mean Absolute Error | Regression, more robust to outliers |
| BCE | Binary Cross-Entropy | Binary classification |

## 4. Gradient Descent

Once a loss function is chosen, how do we find the optimal parameters $w$ that minimize the loss?

**Core idea**: Update parameters in the opposite direction of the gradient (descend the gradient).

$$w_{new} = w_{old} - \eta \cdot \nabla L$$

- $\eta$: **Learning rate** — controls the step size of each update
- Too large → oscillation, may skip the optimum
- Too small → slow convergence, may get stuck in local optima

## 5. Activation Functions

### Why Activation Functions?

If every layer is a linear transformation, stacking multiple layers is still just a linear function — network depth becomes meaningless. Activation functions introduce **nonlinearity**, enabling the network to approximate arbitrarily complex functions.

### Properties of a Good Activation Function

1. Continuous and differentiable nonlinear function
2. The function and its derivative should be computationally simple
3. The derivative's range should be in a reasonable interval (to avoid vanishing or exploding gradients)

### Common Activation Functions

| Activation | Formula | Characteristics |
|-----------|---------|----------------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | Output in (0,1); prone to vanishing gradients |
| TanH | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | Output in (-1,1); zero-centered |
| ReLU | $f(x) = \max(0, x)$ | Computationally simple; alleviates vanishing gradients; currently the most widely used |

## 6. Forward and Backward Propagation

### Training Loop

```
Randomly initialize parameters
    ↓
Forward propagation (input → layer-wise computation → output prediction)
    ↓
Compute loss (evaluate predictions with loss function)
    ↓
Backward propagation (apply chain rule to compute gradients of each parameter)
    ↓
Update parameters: w = w − η × ∂loss/∂w
    ↓
Next iteration (repeat until convergence)
```

### The Key to Backpropagation

- The core is the **Chain Rule**: the derivative of a composite function = the product of the derivatives of its constituent functions.
- Gradients propagate layer by layer, from the output layer back toward the input.
- The gradient at each layer = its local gradient × the gradient passed down from upstream.

---

*Note status: Second lecture, first draft. Pending concrete computation examples.*

*Related notes: [Embodied Intelligence Perception Fundamentals: Machine Vision and Deep Learning](./具身智能感知基础：机器视觉与深度学习.en.md), [From Traditional Image Processing to Deep Learning](./From_traditional_image_processing_to_deep_learning.en.md), [What is a Convolutional Network](./What_is_Convolution_network.en.md), [Commonly Used Convolution Operators](./Commonly_used_convolution_operators.en.md)*
