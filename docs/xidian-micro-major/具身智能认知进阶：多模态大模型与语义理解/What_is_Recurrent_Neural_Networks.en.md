# What is a Recurrent Neural Network (RNN)

> Xidian University · Embodied Intelligence Micro-Major
> Course Notes

---

## I. Why Do We Need RNNs

Traditional fully connected networks and CNNs assume that inputs are **independent** — each image, each sample is processed individually, with no connection between them. However, many tasks naturally involve **temporal dependencies**:

| Task | Input | Temporal Relationship |
|------|------|---------|
| Language Model | Text sequence | "Today" → "weather" → "is good" has an order |
| Speech Recognition | Audio frames | Consecutive frames are continuous; order cannot be shuffled |
| Video Understanding | Frame sequence | Motion unfolds over time |
| Time Series Prediction | Historical sequence | Past values influence the future |

> RNNs, through **neurons with self-feedback**, can process temporal data of arbitrary length — this is a key step from "static perception" toward "dynamic understanding."

---

## II. Basic Structure of an RNN

### 2.1 Macroscopic Structure

```
Input Layer → Hidden Layer (with Delay Unit) → Output Layer
                   ↑_____↓
                 Feedback Connection
```

| Component | Function | CNN Analogy |
|------|------|---------|
| **Input Layer** | Receives the current input $x_t$ | FC input |
| **Hidden Layer** | Stores **activation state** (historical information) | Feature maps |
| **Delay Unit** | Passes the hidden state from the previous time step to the current one | ❌ CNN does not have this |
| **Output Layer** | Produces the output $h_t$ at the current time step | FC output |

### 2.2 Unfolding in Time

The core of an RNN — **the same network weights are reused at different time steps**:

```text
t=1        t=2        t=3        t=4
x₁ →┌─┐   x₂ →┌─┐   x₃ →┌─┐   x₄ →┌─┐
    │RNN│─h₁─→│RNN│─h₂─→│RNN│─h₃─→│RNN│─h₄→ ...
    └───┘     └───┘     └───┘     └───┘
```

$$h_t = f(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

| Symbol | Meaning | Shape |
|------|------|------|
| $x_t$ | Input at time $t$ | $(d_{in}, 1)$ |
| $h_{t-1}$ | Hidden state at time $t-1$ | $(d_h, 1)$ |
| $W_{xh}$ | Input → Hidden weights | $(d_h, d_{in})$ |
| $W_{hh}$ | Hidden → Hidden weights (temporal connection) | $(d_h, d_h)$ |
| $h_t$ | Hidden state (output) at time $t$ | $(d_h, 1)$ |

> Key insight: **$W_{xh}$ and $W_{hh}$ are shared across all time steps** — no matter how long the sequence is, the number of parameters stays the same. This allows RNNs to handle variable-length sequences.

---

## III. RNN Forward Propagation

```python
# Single-step forward propagation of an RNN
def rnn_step(x_t, h_prev, W_xh, W_hh, b_h):
    """
    x_t:    current input (d_in,)
    h_prev: previous hidden state (d_h,)
    """
    # Input transformation + historical state transformation
    h_t = torch.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
    return h_t

# Forward propagation for the entire sequence
def rnn_forward(X, h0, W_xh, W_hh, b_h):
    """
    X: input sequence (seq_len, batch, d_in)
    Returns hidden states for all time steps
    """
    h = h0
    outputs = []
    for t in range(len(X)):
        h = rnn_step(X[t], h, W_xh, W_hh, b_h)
        outputs.append(h)
    return torch.stack(outputs)
```

---

## IV. RNN Backpropagation: BPTT

### 4.1 Why It Is Called "Backpropagation Through Time"

BPTT (Backpropagation Through Time) — after unfolding the RNN in time, it resembles a multi-layer feedforward network with as many layers as the sequence length.

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)$$

When computing the gradient with respect to $W_{hh}$, the gradient must **propagate from the last time step back to the first**:

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}}$$

And $\frac{\partial h_t}{\partial W_{hh}}$ in turn depends on $h_{t-1}$, which depends on $h_{t-2}$... forming a long chain.

### 4.2 Vanishing and Exploding Gradients

| Problem | Cause | Manifestation | Solution |
|------|------|------|------|
| **Vanishing Gradient** | Activation function gradient < 1 (e.g., tanh), chain multiplication approaches 0 | Early time steps learn nothing | LSTM, GRU, Gradient Clipping |
| **Exploding Gradient** | Weights too large, gradient grows exponentially through chain multiplication | Loss → NaN | Gradient Clipping |

```python
# Gradient clipping — the simplest defense against explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

> In theory, RNNs can handle sequences of arbitrary length, but in practice, due to vanishing gradients, the effective memory of a plain RNN is only about 10-20 steps.

---

## V. RNN in PyTorch

```python
import torch.nn as nn

# Built-in RNN layer
rnn = nn.RNN(
    input_size=128,      # Input feature dimension d_in
    hidden_size=256,     # Hidden layer dimension d_h
    num_layers=2,        # Number of stacked layers
    batch_first=True,    # (batch, seq, feature) format
    bidirectional=False  # Whether bidirectional
)

# Forward propagation
x = torch.randn(32, 50, 128)   # (batch, seq_len, input_size)
output, h_n = rnn(x)            # output: (32, 50, 256)
                                # h_n:    (2, 32, 256) — final layer hidden state
```

| Parameter | Description |
|------|------|
| `input_size` | Dimensionality of the input vector at each time step |
| `hidden_size` | Dimensionality of the hidden state vector |
| `num_layers` | Number of stacked RNN layers |
| `bidirectional` | Bidirectional RNN — processes the sequence both forward and backward |
| `batch_first` | Recommended to set to True |

---

## VI. Limitations of RNNs

> When sequences are long, plain RNNs suffer from vanishing gradients and lose long-range dependencies.

### 6.1 Summary of Limitations

| Limitation | Cause | Impact |
|------|------|------|
| Short-term memory | Vanishing gradient | Can only remember about 10-20 steps of context |
| Sequential computation | $h_t$ depends on $h_{t-1}$ | Cannot parallelize; slow training |
| Capacity constraint | Single hidden state $h$ | Difficulty storing multiple types of information simultaneously |
| No selectivity | Treats all inputs equally at every time step | Cannot "forget" irrelevant information |

### 6.2 Solutions

| Solution | Core Idea | Representative |
|------|---------|------|
| LSTM | Gating mechanism + cell state | Selective memory/forgetting |
| GRU | Simplified LSTM (two gates) | Fewer parameters, similar performance |
| Bidirectional RNN | Looks at both past and future context | BiLSTM |
| Attention Mechanism | Direct connections across arbitrary distances | Transformer replaced RNNs |

---

## VII. RNN Application Scenarios

| Scenario | Input | Output | Typical Architecture |
|------|------|------|---------|
| Text Classification | Word sequence | Category | RNN → Last output → FC |
| Sequence Labeling (NER) | Word sequence | Label per word | RNN → Output at each step → FC |
| Machine Translation | Source language sequence | Target language sequence | Encoder RNN → Decoder RNN |
| Time Series Prediction | Historical value sequence | Future values | RNN → FC |
| Speech Recognition | Audio feature sequence | Text sequence | Multi-layer BiLSTM |

---

## VIII. RNN vs CNN vs Transformer

| Dimension | RNN | CNN | Transformer |
|------|-----|-----|-------------|
| Temporal Modeling | ✅ Naturally suited | ⚠️ Requires 1D convolution | ✅ Positional encoding |
| Long-range Dependencies | ❌ Vanishing gradient | ⚠️ Requires many layers | ✅ Self-attention |
| Parallel Computation | ❌ Sequential | ✅ Parallel | ✅ Parallel |
| Parameter Count | Low (shared weights) | Low | High (QKV projections) |
| Training Speed | Slow | Fast | Fast (short sequences) |

---

*Related notes: [What is Attention](./What_is_attention.en.md), [What is a Convolutional Network](../具身智能感知基础：机器学习与深度学习/What_is_Convolution_network.en.md)*
