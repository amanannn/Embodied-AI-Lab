"""
【条件样本生成模块 - generate.py】

【作用】
    加载训练好的 Flow Matching checkpoint，根据指定的数字标签 (0~9)
    通过求解 ODE dx/dt = v_t(x) 从噪声生成手写数字图像。
    支持 Euler 和 RK4 两种数值积分方法。

【关键函数】
    sample_euler()  - Euler 一阶 ODE 积分采样
    sample_rk4()    - RK4 四阶 Runge-Kutta 积分采样（更精确）
    generate()      - 高级接口：加载 ckpt → 采样 → 保存 PNG

【输入输出】
    输入:  checkpoint 路径 + 数字标签 (0~9) + 积分步数/方法
    输出:  generated/digit_{label}_{method}.png  (28×28 灰度 PNG)

【使用示例】
    # 生成数字 "7"，使用 Euler 方法，100 步
    python generate.py --checkpoint checkpoints/ckpt_epoch_0100.pt --digit 7 --method euler --steps 100

    # 批量生成 0~9
    python generate.py --checkpoint checkpoints/ckpt_epoch_0100.pt --digit all --method rk4 --steps 50
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from models import create_model


# ============================================================
# Euler 一阶采样
# ============================================================
@torch.no_grad()
def sample_euler(
    model: torch.nn.Module,
    label: torch.Tensor,
    num_steps: int = 100,
    device: str = "cuda",
) -> torch.Tensor:
    """
    使用 Euler 方法求解 ODE: dx/dt = v_t(x)，从 t=1 反向积分到 t=0。

    原理:
        已知 x(1) ~ N(0, I)（纯噪声），希望得到 x(0)（数据样本）。
        Euler 迭代: x_{t - Δt} = x_t + Δt · v_t(x_t)
        其中 Δt = -1/num_steps（负号表示反向积分）。

    Args:
        model:     训练好的条件模型
        label:     [B] 目标数字标签
        num_steps: ODE 积分步数（越多越精确，默认 100）
        device:    计算设备

    Returns:
        images: [B, 1, 28, 28] 生成的图像（值域 [-1, 1]）
    """
    B = label.shape[0]
    model.eval()

    # 初始状态: t=1 时的纯噪声 x ~ N(0, I)
    x = torch.randn(B, 1, 28, 28, device=device)

    # 步长: 从 t=1 到 t=0，共 num_steps 步
    dt = -1.0 / num_steps

    for step in range(num_steps):
        t_current = 1.0 + step * dt                           # 当前时间步
        t_tensor = torch.full((B,), t_current, device=device)
        v = model(x, t_tensor, label)                          # 预测速度场
        x = x + dt * v                                         # Euler 步进

    model.train()  # 恢复训练模式（如果在训练中调用）
    return x


# ============================================================
# RK4 四阶采样（更精确）
# ============================================================
@torch.no_grad()
def sample_rk4(
    model: torch.nn.Module,
    label: torch.Tensor,
    num_steps: int = 50,
    device: str = "cuda",
) -> torch.Tensor:
    """
    使用 4 阶 Runge-Kutta 方法求解 ODE: dx/dt = v_t(x)。

    RK4 每步计算 4 次速度场评估，精度远高于 Euler，
    通常用一半的步数即可达到同等或更好的质量。

    Args:
        model:     训练好的条件模型
        label:     [B] 目标数字标签
        num_steps: ODE 积分步数（默认 50，约等价于 Euler 100 步）
        device:    计算设备

    Returns:
        images: [B, 1, 28, 28] 生成的图像（值域 [-1, 1]）
    """
    B = label.shape[0]
    model.eval()

    x = torch.randn(B, 1, 28, 28, device=device)
    dt = -1.0 / num_steps

    for step in range(num_steps):
        t_current = 1.0 + step * dt
        t_half = t_current + dt / 2
        t_next = t_current + dt

        # RK4 四个斜率
        # k1 = v(t, x)
        k1 = model(x, torch.full((B,), t_current, device=device), label)
        # k2 = v(t + dt/2, x + dt/2 * k1)
        k2 = model(x + dt / 2 * k1, torch.full((B,), t_half, device=device), label)
        # k3 = v(t + dt/2, x + dt/2 * k2)
        k3 = model(x + dt / 2 * k2, torch.full((B,), t_half, device=device), label)
        # k4 = v(t + dt, x + dt * k3)
        k4 = model(x + dt * k3, torch.full((B,), t_next, device=device), label)

        # RK4 加权更新
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    model.train()
    return x


# ============================================================
# 高级生成接口
# ============================================================
def generate(
    checkpoint_path: str,
    digit: int,
    num_steps: int = 100,
    method: str = "euler",
    output_dir: str = "./generated",
    device: str = "cuda",
) -> str:
    """
    加载 checkpoint，生成指定数字的图像，保存为 PNG。

    Args:
        checkpoint_path: .pt checkpoint 文件路径
        digit:           目标数字 0~9
        num_steps:       ODE 积分步数
        method:          积分方法 "euler" 或 "rk4"
        output_dir:      输出目录
        device:          计算设备

    Returns:
        output_path: 生成的 PNG 文件路径
    """
    # 1. 加载 checkpoint
    print(f"📂 加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # 获取模型类型（checkpoint 中记录了 model_type）
    model_type = ckpt.get("model_type", "unet")
    print(f"   模型类型: {model_type.upper()}")

    # 2. 创建模型并加载权重
    model = create_model(model_type).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"   训练轮数: Epoch {ckpt.get('epoch', '?')}, Loss: {ckpt.get('loss', '?'):.6f}")

    # 3. 准备输入
    label = torch.tensor([digit], dtype=torch.long, device=device)

    # 4. 采样生成
    print(f"🎨 生成数字 '{digit}' | 方法: {method.upper()} | 步数: {num_steps}")
    sample_fn = sample_rk4 if method.lower() == "rk4" else sample_euler
    image = sample_fn(model, label, num_steps=num_steps, device=device)

    # 5. 后处理: [-1, 1] → [0, 1]
    image = image.clamp(-1, 1)
    image = (image + 1) / 2

    # 6. 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"digit_{digit}_{method}_{num_steps}steps.png")
    save_image(image, output_path)
    print(f"✅ 图像已保存: {output_path}")

    return output_path


# ============================================================
# 批量生成（0~9 全部数字）
# ============================================================
def generate_all(
    checkpoint_path: str,
    num_steps: int = 100,
    method: str = "euler",
    output_dir: str = "./generated",
    device: str = "cuda",
) -> str:
    """
    生成 0~9 全部数字的网格图。

    Returns:
        output_path: 网格图文件路径
    """
    print(f"📂 加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = ckpt.get("model_type", "unet")

    model = create_model(model_type).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    labels = torch.arange(0, 10, device=device, dtype=torch.long)

    print(f"🎨 生成数字 0~9 | 方法: {method.upper()} | 步数: {num_steps}")
    sample_fn = sample_rk4 if method.lower() == "rk4" else sample_euler
    images = sample_fn(model, labels, num_steps=num_steps, device=device)

    # 后处理
    images = images.clamp(-1, 1)
    images = (images + 1) / 2

    # 拼成网格
    grid = make_grid(images, nrow=5, padding=2)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"all_digits_{method}_{num_steps}steps.png")
    save_image(grid, output_path)
    print(f"✅ 网格图已保存: {output_path}")

    return output_path


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flow Matching 条件生成 MNIST 数字图像"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint 文件路径")
    parser.add_argument("--digit", type=str, default="5",
                        help="目标数字 0~9，或 'all' 生成全部")
    parser.add_argument("--steps", type=int, default=100, help="ODE 积分步数")
    parser.add_argument("--method", type=str, default="euler",
                        choices=["euler", "rk4"], help="采样方法")
    parser.add_argument("--output_dir", type=str, default="./generated", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    args = parser.parse_args()

    if args.digit.lower() == "all":
        generate_all(
            checkpoint_path=args.checkpoint,
            num_steps=args.steps,
            method=args.method,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        generate(
            checkpoint_path=args.checkpoint,
            digit=int(args.digit),
            num_steps=args.steps,
            method=args.method,
            output_dir=args.output_dir,
            device=args.device,
        )
