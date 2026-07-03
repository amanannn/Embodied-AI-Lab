import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_edsr import EDSRDataset
from model_edsr import EDSR
from utils import calc_psnr, save_compare_image, tensor_to_image


@torch.no_grad()
def evaluate(model, dataset, loader, device):
    model.eval()
    total_bicubic = 0.0
    total_srcnn = 0.0

    for batch in tqdm(loader, desc="Testing"):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        sr = model(lr)

        lr_up = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)

        total_bicubic += calc_psnr(lr_up, hr)
        total_srcnn += calc_psnr(sr, hr)

    n = len(dataset)
    return total_bicubic / n, total_srcnn / n


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    model = EDSR(scale=args.scale).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    for name, hr_dir in [("Set5", args.set5_dir), ("Set14", args.set14_dir)]:
        dataset = EDSRDataset(hr_dir, scale=args.scale, train=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        bicubic_psnr, srcnn_psnr = evaluate(model, dataset, loader, device)

        print(f"{name}: Bicubic={bicubic_psnr:.2f} dB | EDSR={srcnn_psnr:.2f} dB | Δ={srcnn_psnr - bicubic_psnr:+.2f} dB")

        output_dir = Path(args.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        for batch in loader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            name_img = batch["name"][0]

            sr = model(lr)
            lr_up = torch.nn.functional.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)

            save_compare_image(
                tensor_to_image(lr_up[0]),
                tensor_to_image(sr[0]),
                tensor_to_image(hr[0]),
                output_dir / f"{name_img}_compare.png",
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints_edsr/best.pth")
    parser.add_argument("--set5-dir", type=str, default="data/benchmark/Set5/HR")
    parser.add_argument("--set14-dir", type=str, default="data/benchmark/Set14/HR")
    parser.add_argument("--output-dir", type=str, default="outputs/results_edsr")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
