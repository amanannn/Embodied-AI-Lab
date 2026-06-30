"""
MNIST 手写数字识别 —— 训练脚本
================================
使用 PyTorch 和 torchvision 加载 MNIST 数据集，
定义一个简单的卷积神经网络（CNN）进行训练，
并保存模型权重及训练/预测可视化结果。

用法：
    python mnist_train.py

输出文件（均在 code/ 目录下）：
    - mnist_model.pth         : 训练好的模型权重
    - training_curve.png      : 训练损失曲线
    - sample_predictions.png  : 测试集上的预测样例（4×4 网格）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，便于结果复现
torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. 定义卷积神经网络
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """一个简单的 CNN 模型，适用于 MNIST 28×28 灰度图像。"""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积：输入通道 1，输出通道 32，卷积核 3×3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # 第二层卷积：输入通道 32，输出通道 64，卷积核 3×3
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # 池化层：2×2 最大池化
        self.pool = nn.MaxPool2d(kernel_size=2)
        # 全连接层
        # 经过两次 "卷积 3×3 + 池化 2×2" 后，特征图尺寸变为 5×5（28→26→13→11→5）
        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        # Dropout 层，防止过拟合
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 卷积 → ReLU → 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积 → ReLU → 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 展平，准备输入全连接层
        x = x.view(-1, 64 * 5 * 5)
        # 全连接 → ReLU → Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 输出层（logits）
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# 2. 训练函数
# ---------------------------------------------------------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个 epoch。
    返回 (平均损失, 准确率) 以及该 epoch 中每个 batch 的损失列表。
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_losses = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        batch_losses.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, batch_losses


# ---------------------------------------------------------------------------
# 3. 可视化辅助函数
# ---------------------------------------------------------------------------
def plot_training_curve(all_batch_losses, save_path="code/training_curve.png"):
    """绘制并保存训练损失曲线。"""
    plt.figure(figsize=(10, 5))
    plt.plot(all_batch_losses, alpha=0.6, linewidth=0.8)
    plt.xlabel("Batch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss Curve", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[可视化] 训练损失曲线已保存至 {save_path}")


def plot_sample_predictions(model, dataloader, device, save_path="code/sample_predictions.png"):
    """从测试集中取 16 张图像，显示模型预测结果（4×4 网格）。"""
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images[:16].to(device), labels[:16].to(device)

    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predicted = torch.max(probabilities, 1)

    # 绘制 4×4 网格
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for idx, ax in enumerate(axes.flat):
        img = images[idx].cpu().squeeze(0).numpy()
        ax.imshow(img, cmap="gray")
        true_label = labels[idx].item()
        pred_label = predicted[idx].item()
        conf = confidences[idx].item()
        color = "green" if pred_label == true_label else "red"
        ax.set_title(
            f"True: {true_label} / Pred: {pred_label} ({conf:.2f})",
            color=color,
            fontsize=9,
        )
        ax.axis("off")
    plt.suptitle("Sample Predictions on Test Set", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[可视化] 样本预测结果已保存至 {save_path}")


# ---------------------------------------------------------------------------
# 4. 主函数
# ---------------------------------------------------------------------------
def main():
    """主训练流程。"""
    # 设备选择（GPU 优先）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用 {device}\n")

    # ---------- 数据预处理 ----------
    # 将像素值归一化到 [0, 1] 再标准化到均值为 0.1307、标准差为 0.3081
    # （这两个数值是 MNIST 数据集的全局统计量）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # 下载训练集与测试集
    print("[数据] 正在下载 / 加载 MNIST 数据集……")
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print(f"[数据] 训练样本数: {len(train_dataset)}")
    print(f"[数据] 测试样本数: {len(test_dataset)}\n")

    # ---------- 模型、损失函数、优化器 ----------
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"[模型] SimpleCNN 参数量: {sum(p.numel() for p in model.parameters()):,}\n")

    # ---------- 训练循环 ----------
    num_epochs = 5
    all_batch_losses = []  # 记录所有 batch 的损失，用于绘图

    for epoch in range(1, num_epochs + 1):
        avg_loss, accuracy, batch_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        all_batch_losses.extend(batch_losses)
        print(
            f"[Epoch {epoch:2d}/{num_epochs}] "
            f"Loss: {avg_loss:.4f}  |  Train Acc: {accuracy:.2f}%"
        )

    # ---------- 保存模型 ----------
    save_path = "mnist_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n[模型] 权重已保存至 {save_path}")

    # ---------- 可视化：训练损失曲线 ----------
    plot_training_curve(all_batch_losses, save_path="training_curve.png")

    # 可视化：测试集预测样例
    plot_sample_predictions(model, test_loader, device, save_path="sample_predictions.png")

    print("\n[完成] 全部任务执行完毕！")


if __name__ == "__main__":
    main()
