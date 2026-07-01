"""
图像超分辨率推理脚本
========================
加载训练好的 SRCNN 模型，对测试图像进行超分辨率重建，
并将低分辨率（LR）、超分辨率（SR）、高分辨率（HR）三列对比图保存到文件。

用法：
    python infer_sr.py
"""

import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------
# 1. 定义与训练时相同的网络结构
# ---------------------------
class SRCNN(torch.nn.Module):
    """
    轻量级超分辨率卷积神经网络。
    必须与训练时的结构完全一致，否则无法加载权重。
    """
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 3, kernel_size=5, padding=2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        out = identity + out  # 残差连接
        return out


# ---------------------------
# 2. 辅助函数
# ---------------------------
def create_lr_image(hr_tensor):
    """
    通过双三次下采样+上采样生成低分辨率图像。
    """
    B, C, H, W = hr_tensor.shape
    lr_small = F.interpolate(hr_tensor, scale_factor=0.5, mode="bicubic",
                             align_corners=False)
    lr_tensor = F.interpolate(lr_small, size=(H, W), mode="bicubic",
                              align_corners=False)
    return lr_tensor


def load_test_images(source="cifar10", num_images=4, root="./data"):
    """
    加载测试图像。

    参数：
        source:     'cifar10' 使用 CIFAR-10 测试集，或提供图片目录路径
        num_images: 需要加载的图像数量
        root:       CIFAR-10 数据存放目录

    返回：
        images:  [N, 3, H, W] 的张量，值范围 [0, 1]
    """
    # 用于显示的变换（仅转为 Tensor，不归一化）
    vis_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if source == "cifar10":
        # 从 CIFAR-10 测试集加载
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=vis_transform
        )
        images = torch.stack([testset[i][0] for i in range(num_images)], dim=0)

    else:
        # 从指定目录加载图片文件
        image_dir = source
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
        files = [f for f in os.listdir(image_dir)
                 if f.lower().endswith(valid_exts)]
        files.sort()
        if len(files) == 0:
            raise FileNotFoundError(f"目录 '{image_dir}' 中没有找到图片文件。")

        # 读取并调整图片为相同尺寸
        img_list = []
        for f in files[:num_images]:
            pil_img = Image.open(os.path.join(image_dir, f)).convert("RGB")
            # 调整到 32x32（与 CIFAR-10 一致）
            pil_img = pil_img.resize((32, 32), Image.BICUBIC)
            img_tensor = vis_transform(pil_img)  # [3, 32, 32]
            img_list.append(img_tensor)

        images = torch.stack(img_list, dim=0)

    return images


# ---------------------------
# 3. 推理与可视化
# ---------------------------
def run_inference(model_path="sr_model.pth", test_source="cifar10",
                  num_images=4, device="cuda",
                  save_path="inference_result.png"):
    """
    运行推理，生成 LR → SR → HR 对比图。

    参数：
        model_path:   训练好的模型权重路径（.pth 文件）
        test_source:  'cifar10' 或图片目录路径
        num_images:   显示的图像数量
        device:       'cuda' 或 'cpu'
        save_path:    结果保存路径
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"模型文件 '{model_path}' 不存在！\n"
            f"请先运行 train_sr.py 训练模型。"
        )

    print(f"使用设备：{device}")
    print(f"加载模型：{model_path}")

    # 创建模型并加载权重
    model = SRCNN()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("模型加载成功！")

    # 加载测试图像（值范围 [0, 1]）
    test_images = load_test_images(source=test_source, num_images=num_images)
    print(f"加载了 {num_images} 张测试图像")

    # 归一化到 [-1, 1] 用于模型推理
    test_norm = test_images * 2.0 - 1.0  # [0,1] → [-1,1]
    test_norm = test_norm.to(device)

    with torch.no_grad():
        # 生成低分辨率版本
        lr_norm = create_lr_image(test_norm)
        # 超分辨率重建
        sr_norm = model(lr_norm)

    print("推理完成！")

    # 转回 [0, 1] 范围用于显示
    lr_disp = torch.clamp((lr_norm.cpu() + 1.0) / 2.0, 0.0, 1.0)
    sr_disp = torch.clamp((sr_norm.cpu() + 1.0) / 2.0, 0.0, 1.0)
    hr_disp = test_images  # 已经是 [0, 1]

    # 转换为 NumPy，形状 (N, H, W, C)
    lr_np = lr_disp.permute(0, 2, 3, 1).numpy()
    sr_np = sr_disp.permute(0, 2, 3, 1).numpy()
    hr_np = hr_disp.permute(0, 2, 3, 1).numpy()

    # ---------- 绘制对比图 ----------
    n = num_images
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))

    titles = [
        "Low-Resolution (Input)",
        "Super-Resolution (Output)",
        "High-Resolution (Ground Truth)",
    ]

    for i in range(n):
        for j, (img, title) in enumerate(zip(
                [lr_np[i], sr_np[i], hr_np[i]], titles)):
            ax = axes[i, j] if n > 1 else axes[j]
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        if n > 1:
            axes[i, 0].set_ylabel(f"Test {i+1}", fontsize=11)

    plt.suptitle("Image Super-Resolution Inference Results",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"推理对比图已保存至：{save_path}")


# ---------------------------
# 4. 主函数
# ---------------------------
if __name__ == "__main__":
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 运行推理
    run_inference(
        model_path="sr_model.pth",     # 训练好的模型权重
        test_source="cifar10",         # 测试数据来源：'cifar10' 或图片目录路径
        num_images=4,                  # 展示的图像数量
        device=device,
        save_path="inference_result.png",
    )
