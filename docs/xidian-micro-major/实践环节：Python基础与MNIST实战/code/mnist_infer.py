"""
MNIST 手写数字识别 —— 推理脚本
================================
加载训练好的 CNN 模型权重，对测试集中的若干样本
进行推理，并将图像与预测结果（含置信度）一并显示。

用法：
    python mnist_infer.py

前置条件：
    mnist_model.pth 文件须存在于 code/ 目录下
    （可由 mnist_train.py 训练生成）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. 定义与训练时完全相同的 CNN 结构
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """与训练脚本中完全一致的 CNN 模型。"""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# 2. 推理函数：对单个 batch 的图像进行预测并可视化
# ---------------------------------------------------------------------------
def infer_and_visualize(model, dataloader, device, num_samples=16):
    """
    从 dataloader 中取一个 batch，用 model 进行推理，
    并将结果以网格形式显示（包含图像、预测标签和置信度）。
    """
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images[:num_samples].to(device), labels[:num_samples].to(device)

    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)       # 转为概率分布
        confidences, predicted = torch.max(probabilities, dim=1)  # 最高概率及其索引

    # 计算网格尺寸（尽可能接近正方形）
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    # 如果 axes 是一维的，统一处理
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, ax in enumerate(axes):
        if idx < num_samples:
            # 显示灰度图像
            img = images[idx].cpu().squeeze(0).numpy()
            ax.imshow(img, cmap="gray")

            true_label = labels[idx].item()
            pred_label = predicted[idx].item()
            confidence = confidences[idx].item()

            # 预测正确显示绿色，错误显示红色
            color = "green" if pred_label == true_label else "red"
            ax.set_title(
                f"Pred: {pred_label} ({confidence:.2%})",
                color=color,
                fontsize=10,
            )
            ax.set_xlabel(f"True: {true_label}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            # 多余的子图隐藏
            ax.axis("off")

    plt.suptitle("MNIST Inference Results", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    print("[推理] 预测结果展示完毕。")


# ---------------------------------------------------------------------------
# 3. 主函数
# ---------------------------------------------------------------------------
def main():
    """加载模型并对测试集样本进行推理。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用 {device}")

    # ---------- 数据预处理（与训练时一致） ----------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # 加载测试集
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # ---------- 加载模型权重 ----------
    model_path = "mnist_model.pth"
    print(f"[模型] 正在加载权重: {model_path}")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("[模型] 权重加载成功！")

    # ---------- 推理并可视化 ----------
    infer_and_visualize(model, test_loader, device, num_samples=16)


if __name__ == "__main__":
    main()
