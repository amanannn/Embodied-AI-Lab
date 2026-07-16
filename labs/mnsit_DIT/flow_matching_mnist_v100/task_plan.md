# Task Plan: Flow Matching 条件生成手写数字

## 项目目标
基于 Flow Matching 实现条件生成 MNIST 手写数字，完整闭环：数据加载 → 训练 → 推理 → Web 服务。

## 阶段划分

### Phase 1: 架构设计与项目骨架 ✅
- [x] 确定目录结构
- [x] 确定骨干网络: Conditional U-Net (主) + DiT (可选扩展)
- [x] 确定超参: lr=1e-3, batch=512, epochs=100, FP16

### Phase 2: 核心代码实现
- [ ] data_loader.py — MNIST 数据加载，归一化至 [-1, 1]
- [ ] models.py — ConditionalUNet + DiT (可选)
- [ ] train.py — Flow Matching 训练循环 (Accelerate + FP16)
- [ ] generate.py — 条件采样生成 (Euler/RK4)
- [ ] server.py — Flask Web 服务 (+ 内联 HTML 前端)

### Phase 3: 运行指南
- [ ] README.md — 完整运行指南 + 监控 + 故障排除

### Phase 4: 进阶扩展（训练跑通后）
- [ ] DiT 模型实现
- [ ] UNet vs DiT 对比

### Phase 5: 文档与答辩准备
- [ ] 每个文件三段式中文说明
- [ ] 流程图
- [ ] 30秒答辩话术

## 技术决策
| 决策 | 选择 | 理由 |
|------|------|------|
| 骨干网络 | Conditional U-Net | 成熟稳定，训练快 |
| 图像尺寸 | 28×28 (原生) | V100 无压力，避免信息丢失 |
| 混合精度 | FP16 | V100 Tensor Core 加速 |
| Batch Size | 512 | V100 32GB 轻松容纳 |
| 训练框架 | Accelerate | 简洁，支持多GPU扩展 |
| 推理采样 | Euler + RK4 | 教学对比 |
| Web 框架 | Flask | 轻量，无额外依赖 |
