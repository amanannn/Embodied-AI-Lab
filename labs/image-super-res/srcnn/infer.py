import argparse
from pathlib import Path
import sys

import torch
from PIL import Image

from model import SRCNN
from utils import image_to_tensor, tensor_to_image


@torch.no_grad()
def main(args):
    # 1. 验证输入
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: input file not found: {input_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        sys.exit(f"Error: checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # 2. 加载模型
    model = SRCNN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    trained_scale = checkpoint.get("scale", args.scale)
    if trained_scale != args.scale:
        print(f"Warning: checkpoint trained with scale={trained_scale}, "
              f"inferring with scale={args.scale}")

    # 3. 读取 LR 图像，Bicubic 上采样到目标尺寸
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    target_w, target_h = w * args.scale, h * args.scale
    lr_up = img.resize((target_w, target_h), Image.BICUBIC)

    # 4. 推理
    lr_tensor = image_to_tensor(lr_up).unsqueeze(0).to(device)
    sr_tensor = model(lr_tensor).squeeze(0).cpu()

    # 5. 保存结果
    sr_img = tensor_to_image(sr_tensor)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sr_img.save(output_path)
    print(f"Saved result to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/results/infer_srcnn.png")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pth")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
