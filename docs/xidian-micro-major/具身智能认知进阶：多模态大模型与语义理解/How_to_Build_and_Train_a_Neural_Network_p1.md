# 如何构建和训练神经网络（上）：训练流程与损失函数

> 西安电子科技大学 · 具身智能微专业
> 课程笔记

---

## 一、数据集

### 1.1 常见基准数据集

| 数据集 | 内容 | 图像尺寸 | 规模 | 用途 |
|--------|------|---------|------|------|
| MNIST | 手写数字（灰度） | 28×28×1 | 60k train + 10k test | 入门分类 |
| Fashion-MNIST | 服饰（灰度） | 28×28×1 | 60k + 10k | MNIST 替代（更难） |
| CIFAR-10 | 自然图像（彩色） | 32×32×3 | 50k + 10k | 通用分类 |
| CIFAR-100 | 自然图像（细粒度） | 32×32×3 | 50k + 10k | 细粒度分类 |
| ImageNet | 自然图像（彩色） | 224×224×3 | 1.28M + 50k | 大规模分类基准 |
| COCO | 自然场景 | 不定 | 330k | 检测/分割/描述 |
| DIV2K | 超分辨率 | 2K 分辨率 | 800 + 100 | 超分辨率 |

### 1.2 数据集的三个子集

| 子集 | 作用 | 使用频率 | 影响谁 |
|------|------|---------|--------|
| **训练集**（Train） | 更新模型参数 | 每个 epoch 都用 | 模型权重 |
| **验证集**（Val） | 调超参数、监控泛化、早停 | 每个 epoch 评估 | 超参数选择 |
| **测试集**（Test） | 最终性能评估 | **只用一次** | 论文报告指标 |

#### 典型划分比例

| 数据规模 | 训练 : 验证 : 测试 |
|---------|------------------|
| < 1 万 | 60 : 20 : 20 |
| 1~10 万 | 80 : 10 : 10 |
| > 10 万 | 90 : 5 : 5 |
| > 100 万 | 98 : 1 : 1 |

### 1.3 训练流程闭环

```
使用训练集训练模型
    ↓
使用验证集评估模型（不更新权重）
    ↓
根据验证集效果调整超参数（lr、batch_size、层数...）
    ↓
重复训练，记录验证集最佳模型
    ↓
使用测试集确认最终效果（仅一次！）
```

> 类比学习过程：**做题（Train）→ 模拟考（Val）→ 订正 → 重复 → 高考（Test）**。在模拟考上反复调参然后去高考作弊（把测试集当验证集用）是科研中最常见的错误。

---

## 二、训练循环

### 2.1 一个训练 step 的五个操作

```python
outputs = model(inputs)             # 1. Forward：输入 → 输出
loss = criterion(outputs, target)   # 2. Loss：预测 vs 真实值
optimizer.zero_grad()               # 3. 清零：清除上一轮梯度
loss.backward()                     # 4. Backward：计算每个参数的梯度
optimizer.step()                    # 5. 更新：梯度下降修改权重
```

| 步骤 | 方法 | 输入 | 输出 | 常见错误 |
|------|------|------|------|---------|
| Forward | `model(x)` | 数据 batch | 预测值 | 忘记 `model.train()/eval()` |
| Loss | `criterion(out, y)` | 预测+标签 | 标量 | 分类/回归用错损失 |
| Zero Grad | `optimizer.zero_grad()` | — | — | **忘记清零导致梯度累积** |
| Backward | `loss.backward()` | 标量 | 梯度 | 多次 backward 需 `retain_graph` |
| Step | `optimizer.step()` | — | 更新后权重 | 和 zero_grad 顺序不能颠倒 |

> `zero_grad()` → `backward()` → `step()` 的顺序绝对不能错——这是所有 PyTorch 训练的"心跳"。

### 2.2 完整的 epoch 循环

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
    model.eval()  # 关闭 Dropout/BatchNorm 的训练行为
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():  # 不计算梯度，省内存加速
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

| 组件 | `train()` 时 | `eval()` 时 | 为什么 |
|------|------------|-----------|--------|
| Dropout | **激活**，随机丢弃 | 关闭，所有神经元参与 | eval 需要确定性输出 |
| BatchNorm | 使用**当前 batch** 的统计量 | 使用**训练期累积的全局**统计量 | eval 时 batch 可能很小 |

```python
# 经典的忘写 model.eval() bug
model.train()
# ... 训练代码 ...
# 直接开始验证 ← BUG! Dropout 还在工作，每次验证结果不一致
val_acc = validate(model, val_loader)  # 结果不可靠
```

---

## 三、损失函数深入

### 3.1 交叉熵的数学推导

多分类交叉熵（配合 Softmax）：

$$L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

其中 $\hat{y}_c = \frac{e^{z_c}}{\sum_j e^{z_j}}$（Softmax）。

**组合梯度**：

$$\frac{\partial L}{\partial z_c} = \hat{y}_c - y_c$$

这个形式极其简洁——梯度就是"预测值减真实值"。这也是为什么分类任务几乎统一使用 CrossEntropyLoss。

### 3.2 训练日志解读

```text
Epoch 1: Train Loss 0.85, Train Acc 70.2% | Val Loss 0.62, Val Acc 78.1%
Epoch 2: Train Loss 0.45, Train Acc 84.3% | Val Loss 0.48, Val Acc 83.5%
Epoch 3: Train Loss 0.32, Train Acc 89.1% | Val Loss 0.40, Val Acc 86.2%
Epoch 4: Train Loss 0.21, Train Acc 93.5% | Val Loss 0.38, Val Acc 87.0%
Epoch 5: Train Loss 0.15, Train Acc 96.2% | Val Loss 0.42, Val Acc 86.5%  ← 过拟合信号!
Epoch 6: Train Loss 0.09, Train Acc 97.8% | Val Loss 0.47, Val Acc 85.8%  ← 继续恶化
```

| 状态 | Train Loss | Val Loss | Train-Val Gap | 行动 |
|------|-----------|---------|--------------|------|
| **正常训练** | ↓↓ | ↓↓ | 小且稳定 | 继续 |
| **过拟合** | ↓↓ | ↑ | 开始扩大 | 早停/正则化/Dropout |
| **欠拟合** | 高→不怎么降 | 高→不怎么降 | 小 | 增加模型容量 |
| **不收敛** | 震荡 | 震荡 | 大 | 降低学习率 |

---

## 四、过拟合与欠拟合

### 4.1 过拟合的数学视角

过拟合本质是模型记住了训练集中的**噪声**而非**信号**：

$$Error = Bias^2 + Variance + IrreducibleError$$

| 情况 | Bias | Variance | 表现 |
|------|------|---------|------|
| 欠拟合 | 高 | 低 | 训练和验证都差 |
| 刚好 | 低 | 低 | 训练和验证都好 |
| 过拟合 | 低 | 高 | 训练好，验证差 |

### 4.2 防过拟合武器库

| 方法 | 原理 | 强度 | 实现 |
|------|------|------|------|
| **数据增强** | 扩充训练分布 | ★★★ | 旋转/裁剪/翻转/颜色 |
| **Dropout** | 随机丢弃神经元 | ★★★ | `nn.Dropout(p=0.5)` |
| **L2 正则** | 惩罚大权重 | ★★☆ | `optimizer(weight_decay=1e-4)` |
| **早停** | Val loss 不降即停 | ★★☆ | 保存最佳 checkpoint |
| **BatchNorm** | 层间归一化，隐式正则 | ★☆☆ | `nn.BatchNorm2d` |
| **简化模型** | 减少层数/神经元 | ★☆☆ | 奥卡姆剃刀 |

#### Dropout 的工作原理

```python
# 训练时
# 以概率 p 随机将神经元输出置 0
# 等价于每次迭代训练一个"子网络"
x = torch.randn(100, 256)
dropout = nn.Dropout(p=0.5)
out_train = dropout(x)  # 约 50% 的值为 0

# 测试时 Dropout 自动关闭——所有神经元参与，输出乘以 (1-p)
model.eval()
out_test = model(x)  # 无丢弃
```

> Dropout 的直觉：不让任何一个神经元"偷懒依赖"其他神经元——每个神经元都必须自己学到有用的特征，因为同事随时可能被"开除"。

### 4.3 欠拟合的解决

| 方法 | 适用场景 |
|------|---------|
| 增加层数/宽度 | 模型太简单 |
| 换更强的架构 | 任务复杂 |
| 添加特征 | 输入信息不足 |
| 减小正则化 | 约束太强 |
| 训练更久 | 还没收敛 |

---

## 五、优化器

### 5.1 SGD 系列

| 优化器 | 更新规则 | 特点 |
|--------|---------|------|
| **SGD** | $w_{t+1} = w_t - \eta g_t$ | 基础款 |
| **SGD+Momentum** | $v_t = \beta v_{t-1} + g_t$; $w_{t+1} = w_t - \eta v_t$ | 惯性加速，冲出鞍点 |
| **SGD+Nesterov** | 先按动量走一步，再在那点算梯度 | "前瞻"梯度，更稳定 |

**Momentum 的直觉**：球滚下山——不仅受当前坡度（梯度）影响，还受之前速度（动量）影响。

$$v_t = \beta v_{t-1} + (1 - \beta)g_t$$

### 5.2 Adam 系列

| 优化器 | 关键改进 | 适用场景 |
|--------|---------|---------|
| **Adam** | 自适应学习率（一阶 + 二阶矩估计） | **入门首选** |
| **AdamW** | Adam + 解耦权重衰减 | 现代默认推荐 |
| **RMSprop** | 自适应学习率（仅二阶矩） | RNN 训练 |

#### Adam 的数学原理

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t \quad \text{（一阶矩——梯度均值）}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2 \quad \text{（二阶矩——梯度方差）}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{（偏差校正）}$$
$$w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### 5.3 如何选择优化器

| 场景 | 推荐 | 学习率起点 |
|------|------|-----------|
| 快速原型 | Adam / AdamW | $10^{-3}$ |
| CV 分类/检测 | AdamW + Cosine Schedule | $10^{-3}$ |
| NLP Transformer | AdamW + Warmup | $10^{-4}$ |
| GAN | Adam (β1=0.5) | $2\times10^{-4}$ |
| 追求极致复现 | SGD+Momentum | $10^{-1}$（需精细调参） |

```python
# 最常用的三件套
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()
```

---

## 六、学习率调度

| 策略 | 行为 | 适用 |
|------|------|------|
| **Step** | 每 N 个 epoch 将 lr 乘以 γ | 简单粗暴 |
| **Cosine** | 余弦曲线从初始衰减到接近 0 | CV 标配 |
| **Warmup** | 前 N 步线性增加，之后正常调度 | Transformer |
| **ReduceLROnPlateau** | Val loss 不降时自动降 lr | 自动挡 |

```python
# Warmup + Cosine 组合（Transformer 标配）
def warmup_cosine(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps         # 线性增加
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)
```

---

*相关笔记：[什么是神经网络](../具身智能感知基础：机器学习与深度学习/What_is_neural_network.md)、[Part 2: CNN 实战与 GAN](./How_to_Build_and_Train_a_Neural_Network_p2.md)*
