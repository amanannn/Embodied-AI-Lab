# 🎨 Flow Matching 条件生成手写数字

> 基于 Flow Matching 的 MNIST 条件生成项目 | V100 优化 | PyTorch 2.5 + Accelerate
>
> 完整闭环：数据加载 → 条件训练 → 推理生成 → Web 服务

---

## 📁 项目结构

```
flow_matching_mnist_v100/
├── data_loader.py          # ① MNIST 数据加载与预处理
├── models.py               # ② ConditionalUNet + DiT 模型定义
├── train.py                # ③ Flow Matching 训练 (Accelerate + FP16)
├── generate.py             # ④ 条件采样生成 (Euler + RK4)
├── server.py               # ⑤ Flask Web 推理服务
├── checkpoints/            # 训练 checkpoint 存储
├── generated/              # 生成样例图存储
├── data/                   # MNIST 数据集（自动下载）
├── README.md               # 本文档
├── task_plan.md            # 项目规划
└── findings.md             # 技术调研笔记
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活 conda 环境
conda activate <your_env>

# 确认关键库已安装
pip install torch torchvision accelerate flask tqdm datasets

# 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 2. 本地自测（CPU 亦可）

```bash
# 测试数据加载
python data_loader.py

# 测试模型前向传播
python models.py
```

### 3. 训练模型

```bash
# 标准启动（U-Net, FP16, batch=512, epochs=100）
accelerate launch train.py --epochs 100 --batch_size 512

# 使用 DiT 模型
accelerate launch train.py --model_type dit --epochs 100 --batch_size 256

# 从 checkpoint 恢复训练
accelerate launch train.py --resume checkpoints/ckpt_epoch_0050.pt

# 自定义配置
accelerate launch train.py --epochs 200 --batch_size 256 --lr 5e-4 --sample_every 5
```

**预计训练时间（V100-SXM2-32GB）：**
| 模型 | Batch Size | 每 Epoch 耗时 | 100 Epochs 总计 |
|------|-----------|--------------|----------------|
| U-Net | 512 | ~3 秒 | ~5 分钟 |
| DiT | 256 | ~8 秒 | ~13 分钟 |

### 4. 推理生成

```bash
# 生成单个数字 "7"，使用 Euler 100 步
python generate.py --checkpoint checkpoints/ckpt_epoch_0100.pt --digit 7 --method euler --steps 100

# 使用 RK4 50 步（质量更高）
python generate.py --checkpoint checkpoints/ckpt_epoch_0100.pt --digit 7 --method rk4 --steps 50

# 生成 0~9 全部数字网格图
python generate.py --checkpoint checkpoints/ckpt_epoch_0100.pt --digit all --method rk4
```

### 5. 启动 Web 服务

```bash
# 启动 Flask 服务（默认端口 5000）
python server.py --checkpoint checkpoints/ckpt_epoch_0100.pt

# 自定义端口
python server.py --checkpoint checkpoints/ckpt_epoch_0100.pt --port 8080
```

浏览器访问 `http://<服务器IP>:5000`，即可在网页上选择数字并实时生成。

---

## 📊 实时监控

### GPU 监控

```bash
# 每秒刷新 GPU 状态
gpustat -i 1

# 持续监控并记录
gpustat -i 1 --show-pid | tee gpu_log.txt
```

### 关键指标

| 指标 | 正常范围 | 异常信号 |
|------|---------|---------|
| GPU 利用率 | 85~100% | <60% 说明 CPU 瓶颈（增大 batch_size） |
| 显存占用 | 2~8 GB | >30GB 接近上限（减小 batch_size） |
| 温度 | 50~75°C | >85°C 需检查散热 |
| 功耗 | 150~250W | <100W 说明 GPU 未充分利用 |

---

## 🔧 常见问题与修复

### 1. FP16 溢出（Loss 突然变为 NaN）

**原因：** 梯度在 FP16 下超出表示范围（±65504）

**修复方案：**
```bash
# 方案 A: 降低学习率
python train.py --lr 5e-4

# 方案 B: 增大梯度裁剪（train.py 默认 clip=1.0）
# 编辑 train.py: accelerator.clip_grad_norm_(model.parameters(), max_norm=0.5)

# 方案 C: 在 Accelerator 初始化时设置（更保守的损失缩放）
# accelerator = Accelerator(mixed_precision="fp16", kwargs_handlers=[...])
```

### 2. DataLoader Worker 死锁

**现象：** 训练卡住不动，GPU 利用率为 0%

**修复方案：**
```bash
# 减少 worker 数量
python train.py --num_workers 0   # 单进程加载，最稳定

# 或设置环境变量
export OMP_NUM_THREADS=1
```

### 3. 显存不足 (OOM)

```bash
# 降低 batch size
python train.py --batch_size 256

# DiT 模型更省显存
python train.py --model_type dit --batch_size 256
```

### 4. 生成质量差（模糊或不清晰）

```bash
# 增加采样步数
python generate.py --checkpoint ckpt.pt --digit 5 --steps 200

# 使用 RK4 方法
python generate.py --checkpoint ckpt.pt --digit 5 --method rk4 --steps 100

# 确认训练是否收敛（loss 应降至 <0.02）
```

---

## 🏗️ 架构详解

### 数据流全景

```
┌──────────┐    ┌───────────────────────┐    ┌──────────────┐
│  MNIST   │───▶│  Accelerate FP16 Train │───▶│  Checkpoint   │
│ 28×28×1  │    │  ┌─────────────────┐  │    │  .pt 文件     │
│ [-1, 1]  │    │  │ Flow Matching   │  │    └──────┬───────┘
└──────────┘    │  │ x_t = (1-t)x + tz│  │           │
                │  │ v = z - x        │  │           ▼
                │  │ MSE(v_pred, v)   │  │    ┌──────────────┐
                │  └─────────────────┘  │    │  generate.py  │
                └───────────────────────┘    │  Euler / RK4  │
                                             └──────┬───────┘
                                                    │
                    ┌───────────────────────────────┘
                    ▼
            ┌──────────────┐     ┌──────────────┐
            │   PNG 图片    │     │  Flask 服务   │
            │ generated/    │     │  POST /generate│
            └──────────────┘     └──────┬───────┘
                                        │
                                        ▼
                                ┌──────────────┐
                                │   Web 前端    │
                                │  HTML + JS    │
                                │  base64 展示  │
                                └──────────────┘
```

### Flow Matching 原理

```
目标: 学习速度场 v_t(x)，将噪声分布变换为数据分布

训练:
  t ~ U(0, 1)                           # 随机时间步
  x_t = (1-t)·x_real + t·z             # 线性插值路径
  v_target = z - x_real                 # 真实速度场
  loss = MSE(model(x_t, t, label), v_target)

推理 (ODE 积分):
  x(1) ~ N(0, I)                        # 从纯噪声开始
  dx/dt = v_t(x)                        # 沿速度场演化
  x(0) = 积分结果                       # 到达数据分布
```

### 模型对比

| 特性 | Conditional U-Net | DiT (Diffusion Transformer) |
|------|-------------------|---------------------------|
| 架构类型 | CNN 编码器-解码器 | Vision Transformer |
| 参数量 | ~3.5M | ~8M |
| 训练速度 | 快 (~3s/epoch) | 较慢 (~8s/epoch) |
| 条件注入 | FiLM (scale-shift) | adaLN (自适应 LayerNorm) |
| 归纳偏置 | 强（卷积局部性） | 弱（全局自注意力） |
| 适用场景 | 小规模生成 | 可扩展到大分辨率 |

---

## 📝 模块说明

### ① data_loader.py
| 项目 | 内容 |
|------|------|
| **作用** | 加载 MNIST 数据集，归一化至 [-1,1]，返回 DataLoader |
| **关键函数** | `get_dataloaders(batch_size, num_workers, data_dir)` |
| **输入** | batch_size=512, num_workers=4, data_dir="./data" |
| **输出** | (train_loader, test_loader)，每个 batch: images[B,1,28,28], labels[B] |

### ② models.py
| 项目 | 内容 |
|------|------|
| **作用** | 定义 ConditionalUNet 和 DiT，支持时间/类别双重条件注入 |
| **关键类** | `ConditionalUNet`, `DiT`, `TimestepEmbedding`, `ResBlock`, `DiTBlock` |
| **关键函数** | `create_model(model_type)` — 模型工厂函数 |
| **输入** | x[B,1,28,28], t[B]∈[0,1], label[B]∈{0,...,9} |
| **输出** | v_pred[B,1,28,28] — 预测的速度场 |

### ③ train.py
| 项目 | 内容 |
|------|------|
| **作用** | Flow Matching 训练循环，Accelerate + FP16 混合精度 |
| **关键函数** | `flow_matching_loss()`, `train_epoch()`, `generate_samples()`, `main()` |
| **输入** | MNIST DataLoader + 命令行参数 |
| **输出** | checkpoints/ckpt_epoch_*.pt, generated/sample_epoch_*.png |

### ④ generate.py
| 项目 | 内容 |
|------|------|
| **作用** | 加载 checkpoint，通过 ODE 积分从噪声生成手写数字 |
| **关键函数** | `sample_euler()`, `sample_rk4()`, `generate()`, `generate_all()` |
| **输入** | checkpoint 路径 + digit (0~9 或 'all') + 采样方法/步数 |
| **输出** | generated/digit_{label}_{method}_{steps}steps.png |

### ⑤ server.py
| 项目 | 内容 |
|------|------|
| **作用** | Flask Web 推理服务 + 内联前端页面 |
| **关键函数** | `load_model()`, `generate_single()`, `api_generate()` |
| **输入** | POST /generate: {"digit": 5, "method": "rk4", "steps": 50} |
| **输出** | JSON: {"success": true, "image_base64": "..."} |

---

## 🎤 30 秒答辩话术

> "本项目实现了基于 **Flow Matching** 的条件手写数字生成系统。
>
> **核心原理**方面，Flow Matching 通过学习一个**速度场**来连接噪声分布和数据分布——训练时在噪声和数据之间做线性插值，让模型预测从噪声指向数据的速度方向；推理时从纯噪声出发，沿学到的速度场进行 ODE 积分，逐步演化成目标图像。
>
> **工程实践**方面，我使用 **PyTorch 2.5 + HuggingFace Accelerate** 框架，在 **Tesla V100 32GB** 上启用了 FP16 混合精度训练，充分发挥 Tensor Core 的算力优势。骨干网络实现了 U-Net 和 DiT 两种架构，通过 FiLM 和自适应 LayerNorm 注入时间步和类别条件。
>
> **完整闭环**包括：数据自动加载预处理 → Accelerate 分布式训练 → Checkpoint 持久化 → 类别条件采样（Euler/RK4）→ Flask Web 服务 → 浏览器端实时交互。整个项目可在 **5 分钟**内完成 100 轮训练收敛，生成清晰可辨的手写数字。"

---

## 📦 依赖清单

```
torch >= 2.5.0
torchvision
accelerate
flask
tqdm
datasets          # 可选（本项目的 data_loader 使用 torchvision）
```

---

## 📄 License

MIT — 仅供学习与研究使用。
