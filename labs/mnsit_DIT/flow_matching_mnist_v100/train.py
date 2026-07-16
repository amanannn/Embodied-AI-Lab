"""
【Flow Matching 训练模块 - train.py】

【作用】
    使用 Accelerate + FP16 混合精度训练条件 Flow Matching 模型。
    训练循环：采样 t ~ U(0,1)，构建插值 x_t，预测速度场 v，
    MSE 损失优化。每 epoch 保存 checkpoint，每 10 epoch 生成样例。

【关键函数】
    train_epoch()        - 单 epoch 训练循环
    generate_samples()   - 训练中生成条件样本（快速 Euler 采样）
    main()               - 主入口：参数解析 → 数据加载 → 训练

【输入输出】
    输入:  MNIST 数据集 (28×28 灰度图, 10 类)
          命令行参数: --epochs, --batch_size, --lr, --model_type
    输出:  checkpoints/ckpt_epoch_*.pt  (模型权重 + 优化器状态)
           generated/sample_epoch_*.png  (条件生成样例图)

【启动命令】
    accelerate launch train.py --epochs 100 --batch_size 512
    accelerate launch train.py --model_type dit --epochs 100 --batch_size 256
"""

import os
import argparse
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from accelerate import Accelerator
from tqdm import tqdm

from data_loader import get_dataloaders
from models import create_model


# ============================================================
# Flow Matching 损失计算
# ============================================================
def flow_matching_loss(
    model: torch.nn.Module,
    x_real: torch.Tensor,
    label: torch.Tensor,
) -> torch.Tensor:
    """
    计算 Flow Matching 损失。

    原理:
        1. 从标准高斯采样噪声 z ~ N(0, I)
        2. 从均匀分布采样时间步 t ~ U(0, 1)
        3. 线性插值: x_t = (1-t)·x_real + t·z
        4. 真实速度场: v_target = z - x_real
        5. 模型预测: v_pred = model(x_t, t, label)
        6. 损失: MSE(v_pred, v_target)

    Args:
        model:  条件生成模型
        x_real: [B, 1, 28, 28] 真实 MNIST 图像
        label:  [B] 类别标签

    Returns:
        loss: 标量损失值
    """
    B = x_real.shape[0]
    device = x_real.device

    # 1. 采样噪声 z ~ N(0, I)
    z = torch.randn_like(x_real)

    # 2. 采样时间步 t ~ U(0, 1)（每个样本独立采样）
    t = torch.rand(B, device=device)

    # 3. 线性插值路径: x_t = (1-t)·x_real + t·z
    #    注意: t 需要 reshape 为 [B, 1, 1, 1] 以广播
    t_reshaped = t.view(B, 1, 1, 1)
    x_t = (1 - t_reshaped) * x_real + t_reshaped * z

    # 4. 真实速度场（方向: 从噪声指向数据）
    v_target = z - x_real
    # 注: 也可使用 x_real - z，这仅改变速度方向符号，
    # 采样时需相应调整 ODE 方向，此处采用论文常用的 z - x_real

    # 5. 模型预测速度场
    v_pred = model(x_t, t, label)

    # 6. MSE 损失
    loss = F.mse_loss(v_pred, v_target)

    return loss


# ============================================================
# 训练中生成条件样本（快速 Euler 采样）
# ============================================================
@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    num_steps: int = 50,                # Euler 积分步数（训练中快速预览）
    device: str = "cuda",
) -> torch.Tensor:
    """
    为 10 个数字 (0~9) 各生成一张条件样本。

    使用 Euler 方法对 ODE dx/dt = v_t(x) 从 t=1 到 t=0 积分。

    Args:
        model:     训练中的模型
        num_steps: 积分步数（越大质量越高，越慢）
        device:    计算设备

    Returns:
        grid: [1, 3, H, W] 10 张样本组成的网格图（可直接 save_image）
    """
    model.eval()

    B = 10  # 数字 0~9
    # 从纯噪声开始 (t=1)
    x = torch.randn(B, 1, 28, 28, device=device)
    labels = torch.arange(0, 10, device=device, dtype=torch.long)

    dt = -1.0 / num_steps  # 反向积分: 1 → 0

    for step in range(num_steps):
        t = torch.full((B,), 1.0 + step * dt, device=device)  # 当前时间步
        v = model(x, t, labels)                                # 预测速度场
        x = x + dt * v                                         # Euler 步进
        # 不进行 clamp，保持 ODE 自然演化

    model.train()

    # 将像素值裁剪到 [-1, 1] 并转为 [0, 1] 用于保存
    x = x.clamp(-1, 1)
    x = (x + 1) / 2  # [-1,1] → [0,1]

    # 拼成网格
    grid = make_grid(x, nrow=5, padding=2)
    return grid


# ============================================================
# 单 epoch 训练
# ============================================================
def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    total_epochs: int,
) -> float:
    """
    执行一个 epoch 的 Flow Matching 训练。

    Args:
        model:      模型（已被 accelerator.prepare 包装）
        dataloader: 数据加载器
        optimizer:  优化器
        accelerator: HuggingFace Accelerator 实例
        epoch:      当前 epoch 编号
        total_epochs: 总 epoch 数

    Returns:
        avg_loss: 本 epoch 平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # 进度条（仅在主进程显示）
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:3d}/{total_epochs}",
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )

    for images, labels in pbar:
        # Flow Matching 不使用数据增强中的噪声——干净图像直接作为 x_real
        # images 已经是归一化到 [-1, 1] 的 MNIST 图像

        optimizer.zero_grad(set_to_none=True)  # set_to_none 更高效

        # 计算 Flow Matching 损失
        loss = flow_matching_loss(model, images, labels)

        # 反向传播（Accelerator 自动处理 FP16 缩放）
        accelerator.backward(loss)

        # 梯度裁剪（防止 FP16 溢出）
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 统计损失
        total_loss += loss.detach().item()
        num_batches += 1

        # 更新进度条
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# ============================================================
# 主训练函数
# ============================================================
def main():
    # ---- 命令行参数 ----
    parser = argparse.ArgumentParser(
        description="Flow Matching 条件生成 MNIST | V100 优化版"
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=512, help="批次大小 (V100 推荐 512)")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--model_type", type=str, default="unet",
                        choices=["unet", "dit"], help="模型类型: unet 或 dit")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 子进程数")
    parser.add_argument("--data_dir", type=str, default="./data", help="MNIST 数据目录")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Checkpoint 保存目录")
    parser.add_argument("--generated_dir", type=str, default="./generated",
                        help="生成样例保存目录")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    parser.add_argument("--sample_steps", type=int, default=50,
                        help="训练中采样 Euler 步数")
    parser.add_argument("--sample_every", type=int, default=10,
                        help="每隔多少 epoch 保存样例图")
    args = parser.parse_args()

    # ---- 初始化 Accelerator ----
    accelerator = Accelerator(
        mixed_precision="fp16",            # V100 Tensor Core FP16 加速
        log_with="tensorboard",            # 可选：TensorBoard 日志
        project_dir=args.checkpoint_dir,
    )

    # 打印训练配置（仅主进程）
    if accelerator.is_main_process:
        print("=" * 60)
        print("🚀 Flow Matching 条件生成 MNIST 训练")
        print("=" * 60)
        print(f"   模型类型:     {args.model_type.upper()}")
        print(f"   训练轮数:     {args.epochs}")
        print(f"   批次大小:     {args.batch_size}")
        print(f"   学习率:       {args.lr}")
        print(f"   混合精度:     FP16 (V100 Tensor Core)")
        print(f"   设备:         {accelerator.device}")
        print(f"   进程数:       {accelerator.num_processes}")
        print("=" * 60)

    # ---- 创建目录 ----
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.generated_dir, exist_ok=True)

    # ---- 数据加载 ----
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
    )

    # ---- 模型创建 ----
    model = create_model(args.model_type)
    n_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        print(f"\n📐 模型参数量: {n_params:,} (~{n_params/1e6:.1f}M)")

    # ---- 优化器 ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # ---- 学习率调度器（余弦退火）----
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ---- Accelerator 包装（模型/优化器/数据加载器/调度器）----
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # ---- 恢复训练（可选）----
    start_epoch = 1
    if args.resume:
        if accelerator.is_main_process:
            print(f"\n📂 从 checkpoint 恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Accelerator 加载状态
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if accelerator.is_main_process:
            print(f"   恢复到 Epoch {start_epoch}")

    # ---- 训练循环 ----
    if accelerator.is_main_process:
        print(f"\n🎯 开始训练 ({args.epochs} epochs)\n")

    for epoch in range(start_epoch, args.epochs + 1):
        # 单 epoch 训练
        avg_loss = train_epoch(
            model, train_loader, optimizer, accelerator,
            epoch=epoch, total_epochs=args.epochs,
        )

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ---- 日志输出（主进程）----
        if accelerator.is_main_process:
            print(f"   ✅ Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")

        # ---- 保存 checkpoint（每 epoch，主进程）----
        if accelerator.is_main_process:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_epoch_{epoch:04d}.pt")
            # 获取未包装的模型状态
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save({
                "epoch": epoch,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
                "model_type": args.model_type,
            }, ckpt_path)
            print(f"   💾 Checkpoint 已保存: {ckpt_path}")

        # ---- 生成样例图（每 sample_every epoch，主进程）----
        if accelerator.is_main_process and (epoch % args.sample_every == 0):
            unwrapped_model = accelerator.unwrap_model(model)
            sample_grid = generate_samples(
                unwrapped_model,
                num_steps=args.sample_steps,
                device=accelerator.device,
            )
            sample_path = os.path.join(args.generated_dir, f"sample_epoch_{epoch:04d}.png")
            save_image(sample_grid, sample_path)
            print(f"   🖼️  样例图已保存: {sample_path}")

        # 同步所有进程
        accelerator.wait_for_everyone()

    # ---- 训练完成 ----
    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("🎉 训练完成！")
        print(f"   Checkpoint 目录: {args.checkpoint_dir}/")
        print(f"   样例图目录:     {args.generated_dir}/")
        print(f"   启动推理:       python generate.py --checkpoint checkpoints/ckpt_epoch_0100.pt --digit 5")
        print(f"   启动 Web 服务:  python server.py --checkpoint checkpoints/ckpt_epoch_0100.pt")
        print("=" * 60)


if __name__ == "__main__":
    main()
