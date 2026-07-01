"""
图像超分辨率训练脚本
========================
使用轻量级 CNN 模型学习从低分辨率图像到高分辨率图像的映射。
训练数据使用 torchvision 的 CIFAR-10 数据集（如无法下载，会生成合成数据）。
输入：低分辨率图像（下采样再上采样得到的模糊版本）
输出：超分辨率重建图像（与原始高分辨率图像尽可能接近）

用法：
    python train_sr.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# 1. 定义超分辨率网络
# ---------------------------
class SRCNN(nn.Module):
    """
    轻量级超分辨率卷积神经网络。
    结构：三个卷积层，激活函数使用 ReLU。
    输入和输出尺寸相同，网络学习的是残差（高频细节）。
    """
    def __init__(self):
        super(SRCNN, self).__init__()
        # 第一层：从 3 通道输入提取 64 个特征图，卷积核 5x5
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        # 第二层：特征映射，卷积核 3x3
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 第三层：重建为 3 通道输出，卷积核 5x5
        self.conv3 = nn.Conv2d(64, 3, kernel_size=5, padding=2)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        前向传播。
        输入 x：低分辨率图像 [B, 3, H, W]
        输出  ：超分辨率图像 [B, 3, H, W] = 输入 + 残差
        """
        identity = x  # 保存输入，用于残差连接
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        # 残差学习：输出 = 输入 + 预测的残差
        out = identity + out
        return out


# ---------------------------
# 2. 数据准备
# ---------------------------
def get_cifar10_dataloader(batch_size=64, root="./data"):
    """
    使用 CIFAR-10 数据集创建训练数据加载器。
    CIFAR-10 包含 50000 张 32x32 的彩色图像，类别丰富。

    返回：
        train_loader: 训练数据加载器
        sample_images: 少量原始图像用于验证
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                              # [0,255] → [0,1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5],          # [0,1] → [-1,1]
                             std=[0.5, 0.5, 0.5]),
    ])

    # 下载并加载 CIFAR-10 训练集
    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(trainset, batch_size=batch_size,
                              shuffle=True, num_workers=2)

    # 另外取一批不归一化的图像（用于可视化），范围 [0,1]
    vis_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    vis_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=False, transform=vis_transform
    )
    sample_images = torch.stack([vis_set[i][0] for i in range(8)], dim=0)

    print(f"训练样本数：{len(trainset)}，每批大小：{batch_size}")
    return train_loader, sample_images


def create_lr_image(hr_tensor):
    """
    根据高分辨率图像生成低分辨率图像。
    策略：双三次下采样 2 倍，再上采样回原尺寸，得到模糊版本。

    参数：
        hr_tensor: [B, 3, H, W]，值范围 [-1, 1] 或 [0, 1]

    返回：
        lr_tensor: [B, 3, H, W]，值范围与输入相同
    """
    B, C, H, W = hr_tensor.shape
    # 下采样到 1/2 尺寸
    lr_small = F.interpolate(hr_tensor, scale_factor=0.5, mode="bicubic",
                             align_corners=False)
    # 再上采样回原始尺寸（得到模糊版本）
    lr_tensor = F.interpolate(lr_small, size=(H, W), mode="bicubic",
                              align_corners=False)
    return lr_tensor


# ---------------------------
# 3. 训练函数
# ---------------------------
def train(model, train_loader, num_epochs=10, device="cuda",
          save_path="sr_model.pth"):
    """
    训练超分辨率模型。

    参数：
        model:      SRCNN 模型实例
        train_loader: 训练数据加载器
        num_epochs:   训练轮数
        device:       'cuda' 或 'cpu'
        save_path:    模型保存路径
    """
    # 将模型放到指定设备
    model = model.to(device)

    # 损失函数：均方误差（MSE），衡量像素级差异
    criterion = nn.MSELoss()

    # 优化器：Adam，学习率 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 记录训练损失
    loss_history = []

    print("开始训练...")
    print(f"设备：{device}，训练轮数：{num_epochs}")
    print("-" * 50)

    for epoch in range(num_epochs):
        model.train()  # 训练模式
        running_loss = 0.0

        for i, (images, _) in enumerate(train_loader):
            # images: [B, 3, 32, 32]，值范围 [-1, 1]

            # 构建低分辨率图像：下采样 → 上采样
            lr_images = create_lr_image(images)
            lr_images = lr_images.to(device)

            # 高分辨率图像作为目标
            hr_images = images.to(device)

            # 前向传播
            sr_images = model(lr_images)  # 超分辨率重建结果

            # 计算损失
            loss = criterion(sr_images, hr_images)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累积损失
            running_loss += loss.item()

        # 计算平均损失
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]  损失: {avg_loss:.6f}")

    print("-" * 50)
    print("训练完成！")

    # 保存模型
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至：{save_path}")

    return loss_history


# ---------------------------
# 4. 可视化函数
# ---------------------------
def plot_training_curve(loss_history, save_path="training_curve.png"):
    """
    绘制训练损失曲线并保存。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history,
             marker="o", linestyle="-", color="b", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.title("Training Loss Curve", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练损失曲线已保存至：{save_path}")


def show_samples(model, sample_images, device="cuda",
                 save_path="sample_results.png"):
    """
    展示模型在测试图像上的超分辨率效果。
    每行显示：低分辨率输入 | 超分辨率输出 | 高分辨率真实值
    共展示 4 组对比结果。
    """
    model.eval()
    # 将样本图像送入设备（sample_images 范围 [0,1]）
    # 为适配模型输入，先归一化到 [-1, 1]
    sample_norm = sample_images * 2.0 - 1.0  # [0,1] → [-1,1]
    sample_norm = sample_norm.to(device)

    with torch.no_grad():
        # 生成低分辨率图像
        lr_norm = create_lr_image(sample_norm)
        # 超分辨率重建
        sr_norm = model(lr_norm)

    # 将张量移回 CPU 并转换回 [0,1] 范围用于显示
    lr_disp = (lr_norm.cpu() + 1.0) / 2.0      # [-1,1] → [0,1]
    sr_disp = (sr_norm.cpu() + 1.0) / 2.0      # [-1,1] → [0,1]
    hr_disp = sample_images                     # 已经是 [0,1]

    # 截断到有效范围
    lr_disp = torch.clamp(lr_disp, 0.0, 1.0)
    sr_disp = torch.clamp(sr_disp, 0.0, 1.0)
    hr_disp = torch.clamp(hr_disp, 0.0, 1.0)

    # 转换为 NumPy 用于 matplotlib 显示
    lr_np = lr_disp.numpy()
    sr_np = sr_disp.numpy()
    hr_np = hr_disp.numpy()

    # 创建画布：4 行 × 3 列
    n_samples = 4
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))

    titles = ["Low-Resolution (LR Input)", "Super-Resolution (SR Output)",
              "High-Resolution (HR Ground Truth)"]

    for i in range(n_samples):
        for j in range(3):
            ax = axes[i, j]
            # 选择对应的图像
            if j == 0:
                img = lr_np[i].transpose(1, 2, 0)  # CHW → HWC
            elif j == 1:
                img = sr_np[i].transpose(1, 2, 0)
            else:
                img = hr_np[i].transpose(1, 2, 0)

            ax.imshow(img)
            ax.set_title(titles[j], fontsize=10)
            ax.axis("off")

        # 在每行左侧添加标签
        axes[i, 0].set_ylabel(f"Sample {i+1}", fontsize=11)

    plt.suptitle("Image Super-Resolution Results", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"样本对比结果已保存至：{save_path}")


# ---------------------------
# 5. 主函数
# ---------------------------
if __name__ == "__main__":
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")

    # 超参数
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    MODEL_SAVE_PATH = "sr_model.pth"
    CURVE_SAVE_PATH = "training_curve.png"
    RESULT_SAVE_PATH = "sample_results.png"

    # 创建模型
    model = SRCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{total_params:,}")

    # 加载数据
    train_loader, sample_images = get_cifar10_dataloader(
        batch_size=BATCH_SIZE, root="./data"
    )

    # 训练模型
    loss_history = train(
        model, train_loader,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_path=MODEL_SAVE_PATH,
    )

    # 绘制训练曲线
    plot_training_curve(loss_history, save_path=CURVE_SAVE_PATH)

    # 展示样本结果
    show_samples(model, sample_images, device=device,
                 save_path=RESULT_SAVE_PATH)

    print("\n所有任务完成！生成的文件：")
    print(f"  - {MODEL_SAVE_PATH}      (训练好的模型权重)")
    print(f"  - {CURVE_SAVE_PATH}   (训练损失曲线)")
    print(f"  - {RESULT_SAVE_PATH} (样本对比结果)")
