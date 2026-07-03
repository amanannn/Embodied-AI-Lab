import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from model_edsr import EDSR
from utils import image_to_tensor, tensor_to_image


@torch.no_grad()
def tile_forward(model, lr_tensor, scale, tile_size, tile_overlap, device, use_fp16):
    _, _, h, w = lr_tensor.shape
    sr_h, sr_w = h * scale, w * scale
    sr = torch.zeros(1, 3, sr_h, sr_w, device=device)
    weight = torch.zeros(1, 1, sr_h, sr_w, device=device)

    step = tile_size - tile_overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = min(y, h - tile_size) if y + tile_size > h else y
            x1 = min(x, w - tile_size) if x + tile_size > w else x
            y2 = y1 + tile_size
            x2 = x1 + tile_size

            patch = lr_tensor[:, :, y1:y2, x1:x2]

            with torch.autocast(device_type=device.type, enabled=use_fp16):
                out = model(patch)

            out = out.float()

            out_y1, out_x1 = y1 * scale, x1 * scale
            out_y2, out_x2 = y2 * scale, x2 * scale

            sr[:, :, out_y1:out_y2, out_x1:out_x2] += out
            weight[:, :, out_y1:out_y2, out_x1:out_x2] += 1.0

    sr = sr / weight
    return sr


@torch.no_grad()
def main(args):
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Error: input file not found: {input_path}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        sys.exit(f"Error: checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    model = EDSR(scale=args.scale).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    trained_scale = checkpoint.get("scale", args.scale)
    if trained_scale != args.scale:
        print(f"Warning: checkpoint trained with scale={trained_scale}, "
              f"inferring with scale={args.scale}")

    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    lr_tensor = image_to_tensor(img).unsqueeze(0).to(device)

    use_fp16 = device.type == "cuda" and args.fp16
    use_tile = w > args.tile_size or h > args.tile_size

    if use_tile:
        print(f"Tiled inference: tile={args.tile_size}, overlap={args.tile_overlap}, fp16={use_fp16}")
        sr_tensor = tile_forward(
            model, lr_tensor, args.scale,
            args.tile_size, args.tile_overlap, device, use_fp16,
        ).squeeze(0).cpu()
        torch.cuda.empty_cache()
    else:
        with torch.autocast(device_type=device.type, enabled=use_fp16):
            sr_tensor = model(lr_tensor).float().squeeze(0).cpu()

    sr_img = tensor_to_image(sr_tensor)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sr_img.save(output_path)

    print(f"Input: {w}x{h}  →  Output: {w * args.scale}x{h * args.scale}")
    print(f"Saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/results_edsr/infer.png")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints_edsr/best.pth")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--tile-overlap", type=int, default=8)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.fp16 = not args.fp32
    main(args)
