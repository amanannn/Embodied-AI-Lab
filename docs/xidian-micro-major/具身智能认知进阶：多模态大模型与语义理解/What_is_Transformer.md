# 什么是 Transformer

> 西安电子科技大学 · 具身智能微专业
> 课程笔记

---

## 一、为什么需要 Transformer

### 1.1 序列建模的瓶颈

| 架构 | 优势 | 致命缺陷 |
|------|------|---------|
| **RNN / LSTM / GRU** | 自然处理变长序列 | 顺序计算 → 无法并行化；长程依赖未根除，梯度在长链传播中衰减或爆炸 |
| **CNN 序列模型（ByteNet / ConvS2S）** | 可并行 | 捕获长距离依赖需堆叠多层或扩大卷积核，缺乏全局感受野 |

> Attention Is All You Need——完全摒弃循环与卷积，仅靠注意力机制建模全局依赖，实现 $O(1)$ 路径长度与高度并行化。

### 1.2 硬件与数据时代契机

| 驱动力 | 说明 |
|--------|------|
| **GPU/TPU 矩阵算力** | 亟需能充分利用硬件并行性的架构——Transformer 的自注意力本质是矩阵乘法 |
| **互联网数据激增** | 模型容量与训练效率成为制约，Transformer 的扩展性完美契合大数据时代 |

---

## 二、Transformer 核心设计

> **设计哲学**：以自注意力为核心，通过残差连接、层归一化、位置编码与前馈网络的模块化组合，构建既能捕获全局上下文、又具备良好优化特性的深度网络。

### 2.1 整体架构范式

| 架构范式 | 说明 | 代表模型 |
|---------|------|---------|
| **Encoder-Decoder** | 原始设计，Encoder 提取源端表示，Decoder 自回归生成，Cross-Attention 桥接 | 原始 Transformer, T5 |
| **Encoder-only** | 双向编码，专注理解任务 | BERT, RoBERTa, DeBERTa |
| **Decoder-only** | 单向自回归，统一生成与理解，Scaling Law 验证持续扩展潜力 | GPT 系列, LLaMA, Qwen |

> 堆叠式同质层——N 个相同 Block 串联，每层参数独立，这种"重复模块"设计简化了工程实现也利于理论分析。

### 2.2 自注意力机制——核心引擎

#### 缩放点积注意力

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| 组件 | 含义 | 形状 |
|------|------|------|
| $Q$（Query） | 查询向量——"我在找什么？" | $(N, d_k)$ |
| $K$（Key） | 键向量——"我是什么？" | $(N, d_k)$ |
| $V$（Value） | 值向量——"我包含什么信息？" | $(N, d_v)$ |
| $\sqrt{d_k}$ | 缩放因子——防止点积过大导致 Softmax 梯度饱和 | 标量 |

> 复杂度 $O(n^2 d)$ 是长序列的主要瓶颈，催生了稀疏注意力、FlashAttention、Mamba 等高效方案。

#### 多头注意力

将 Q/K/V 投影到 $h$ 个子空间并行计算，再拼接线性变换：

$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$$
$$head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

| 特性 | 说明 |
|------|------|
| **不同子空间** | 不同头关注语法、指代、语义角色等不同层面 |
| **参数量** | 与单头全维度相当，无额外开销（$h \times \frac{d}{h} \times \frac{d}{h} \approx d^2$） |
| **掩码机制** | Decoder 使用因果掩码（预测第 $t$ 个 token 只能看 $\le t$ 位置），Encoder 无掩码 |

### 2.3 前馈神经网络——知识存储

$$FFN(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

| 维度 | 说明 |
|------|------|
| **内层维度** | 通常 $4 \times d_{model}$，提供非线性变换与记忆存储 |
| **逐位置独立** | 每个 token 的 FFN 计算相互独立，天然并行 |
| **现代改进** | SwiGLU / GeGLU 等门控激活替代 ReLU；MoE 将 FFN 稀疏化以扩大容量 |

> FFN 是 Transformer 的"知识存储"主体——注意力负责"查信息"，FFN 负责"存知识"。

### 2.4 残差连接与层归一化

| 归一化策略 | 公式 | 特点 |
|-----------|------|------|
| **Post-Norm**（原始论文） | $LayerNorm(x + Sublayer(x))$ | 理论上限高但训练不稳定，需精细调学习率 |
| **Pre-Norm**（主流实践） | $x + Sublayer(LayerNorm(x))$ | 梯度流更平稳，训练更鲁棒，已成大模型标配 |

| 变体 | 改进 |
|------|------|
| **RMSNorm** | 去除均值中心化，减少计算量，效果相当 |
| **DeepNorm** | 针对超深网络调整初始化与归一化策略 |

> 残差流是 Transformer 的"信息高速公路"——子层仅对其做增量修改，保障百层以上网络可训练。

### 2.5 位置编码

| 类型 | 代表 | 特点 |
|------|------|------|
| **绝对位置编码** | 正弦余弦（原始）→ 可学习嵌入 | 简单，但泛化性差，难以外推至训练未见长度 |
| **相对位置编码** | T5 Bias, ALiBi | 直接建模 token 间相对距离，天然支持长度外推 |
| **旋转位置编码（RoPE）** | LLaMA, Qwen 标配 | 将位置注入 Q/K 向量，注意力分数自然依赖相对位置 |

> RoPE 已成为 LLM 事实标准，衍生出 YaRN、NTK-Aware Scaled RoPE 等外推增强方案。

### 2.6 Tokenization 与输入输出

| 组件 | 说明 |
|------|------|
| **Tokenization** | BPE / WordPiece / Unigram 平衡词汇大小与未登录词；SentencePiece 支持语言无关 |
| **Embedding** | Token Embedding + Position Encoding 相加，按 $\sqrt{d_{model}}$ 缩放匹配位置编码量级 |
| **输出头** | Linear + Softmax 预测下一 token；部分模型 Weight Tying 共享输入 Embedding 与输出投影 |

---

## 三、Transformer 的演变

### 3.1 架构分化与专精

| 分支 | 演进路线 | 核心预训练目标 |
|------|---------|-------------|
| **Encoder-only** | BERT → RoBERTa → DeBERTa | MLM, RTD |
| **Decoder-only** | GPT → GPT-3/4 → LLaMA → Qwen | 自回归 Next-Token Prediction |
| **Encoder-Decoder** | T5 → BART → mBART | Span Corruption, DAE |

### 3.2 效率与长上下文优化

| 方向 | 代表方法 | 核心思路 |
|------|---------|---------|
| **线性/稀疏注意力** | Linformer, Sparse Transformer | 降低 $O(n^2)$ → $O(n \log n)$ 或 $O(nk)$ |
| **状态空间模型（SSM）** | Mamba, RWKV | RNN-like 状态压缩，近线性复杂度逼近 Transformer |
| **KV Cache 优化** | MQA, GQA | 减少 KV 头数节省显存 |
| **系统/算子优化** | FlashAttention, PagedAttention | 从 IO 与内存层面加速推理 |

### 3.3 规模化与系统工程

| 维度 | 技术 |
|------|------|
| **分布式训练** | Megatron-LM, DeepSpeed, FSDP——3D 并行（数据+张量+流水线）成标配 |
| **对齐与安全** | RLHF → DPO → Constitutional AI——有用性、诚实性、无害性 |
| **多模态扩展** | ViT（Patch 化）→ CLIP → Flamingo → GPT-4V——视觉-语言统一 |

---

## 四、深度讨论

### 4.1 理论理解与可解释性

| 议题 | 现状 |
|------|------|
| **归纳偏置之争** | Transformer 是否真"无偏置"？注意力结构、位置编码、FFN 形式均隐含强先验 |
| **内部机制** | Mechanistic Interpretability 发现归纳头、复制电路等组件，但整体仍近黑箱 |
| **优化动力学** | 为何 Pre-Norm 更易训练？残差流如何影响损失景观？理论滞后于实践 |

### 4.2 局限性与替代路径

| 局限 | 缓解手段 | 权衡 |
|------|---------|------|
| $O(n^2)$ 复杂度 | SSM、线性注意力、稀疏注意力 | 性能 vs 效率的平衡 |
| KV Cache 推理成本 | 推测解码、量化、蒸馏 | 速度 vs 质量的取舍 |
| Scaling Law 可持续性 | 合成数据、高效架构 | 数据质量与碳足迹 |

### 4.3 未来方向

| 方向 | 探索 |
|------|------|
| **架构创新** | Transformer + SSM 混合模型、动态计算、自适应深度 |
| **超越 Next-Token Prediction** | 世界模型、规划能力、符号推理——需要新目标函数？ |
| **具身与交互** | 实时感知-决策闭环对架构提出新要求 |
| **小模型复兴** | 蒸馏 + 专项微调——边缘设备与隐私场景 |

---

*相关笔记：[什么是注意力](./What_is_attention.md)、[什么是循环神经网络](./What_is_Recurrent_Neural_Networks.md)、[多模态大模型预训练专题](./多模态大模型预训练—微调—后训练专题研究.md)*
