"""
【数据加载模块 - data_loader.py】

【作用】
    加载 MNIST 手写数字数据集，将图像归一化到 [-1, 1] 范围，
    返回 PyTorch DataLoader，供训练和测试使用。

【关键函数】
    get_dataloaders(batch_size, num_workers, data_dir)
        → 返回 (train_loader, test_loader)

【输入输出】
    输入: batch_size=512, num_workers=4, data_dir="./data"
    输出: train_loader (60000 样本), test_loader (10000 样本)
          每个 batch: (images[B,1,28,28], labels[B])，像素值 ∈ [-1, 1]
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(
    batch_size: int = 512,
    num_workers: int = 4,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """
    获取 MNIST 训练集和测试集的 DataLoader。

    Args:
        batch_size: 每批样本数，V100 推荐 512
        num_workers: 数据加载子进程数
        data_dir: MNIST 数据存储目录

    Returns:
        (train_loader, test_loader): 训练和测试 DataLoader
    """
    # 数据预处理流水线：ToTensor 映射到 [0,1]，再归一化到 [-1,1]
    # 公式: (x - 0.5) / 0.5 = 2x - 1，将 [0,1] → [-1,1]
    transform = transforms.Compose([
        transforms.ToTensor(),                        # [0,255] → [0.0, 1.0]
        transforms.Normalize((0.5,), (0.5,)),         # [0,1] → [-1,1]
    ])

    # 下载并加载 MNIST 数据集
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # 构建 DataLoader
    # pin_memory=True 加速 GPU 数据传输（将数据锁页在 CPU 内存中）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,                    # 训练集需要打乱
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,                  # 丢弃不完整的最后一批，保证 batch 尺寸一致
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,                   # 测试集无需打乱
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"📦 MNIST 数据加载完成:")
    print(f"   训练集: {len(train_dataset):,} 样本 → {len(train_loader)} 批次")
    print(f"   测试集: {len(test_dataset):,} 样本 → {len(test_loader)} 批次")
    print(f"   图像尺寸: 28×28 (单通道灰度)")
    print(f"   像素范围: [-1, 1]")

    return train_loader, test_loader


# ============================================================
# 快速测试：直接运行此文件验证数据加载是否正常
# ============================================================
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(batch_size=64, num_workers=0)
    images, labels = next(iter(train_loader))
    print(f"\n🔍 单批数据验证:")
    print(f"   images shape: {images.shape}  (应为 [64, 1, 28, 28])")
    print(f"   labels shape: {labels.shape}  (应为 [64])")
    print(f"   images 值域: [{images.min().item():.3f}, {images.max().item():.3f}]  (应为 [-1, 1])")
    print(f"   labels 示例: {labels[:10].tolist()}")
