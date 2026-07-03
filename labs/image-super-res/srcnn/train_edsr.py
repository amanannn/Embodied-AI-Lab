import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_edsr import EDSRDataset
from model_edsr import EDSR
from utils import calc_psnr


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_psnr = 0.0
    count = 0
    for batch in data_loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        sr = model(lr)
        total_psnr += calc_psnr(sr, hr)
        count += 1
    return total_psnr / count


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    train_set = EDSRDataset(
        hr_dir=args.train_dir,
        scale=args.scale,
        patch_size=args.patch_size,
        train=True,
        samples_per_epoch=args.samples_per_epoch,
    )

    set5 = EDSRDataset(args.set5_dir, scale=args.scale, patch_size=args.patch_size, train=False)
    set14 = EDSRDataset(args.set14_dir, scale=args.scale, patch_size=args.patch_size, train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    set5_loader = DataLoader(set5, batch_size=1, shuffle=False, num_workers=0)
    set14_loader = DataLoader(set14, batch_size=1, shuffle=False, num_workers=0)

    model = EDSR(
        scale=args.scale,
        n_feats=args.n_feats,
        n_resblocks=args.n_resblocks,
        res_scale=args.res_scale,
    ).to(device)

    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_score = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in progress:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)

            sr = model(lr)
            loss = loss_fn(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.6f}"})

        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        set5_psnr = evaluate(model, set5_loader, device)
        set14_psnr = evaluate(model, set14_loader, device)
        mean_psnr = (set5_psnr + set14_psnr) / 2.0

        print(
            f"Epoch {epoch:03d} | "
            f"loss={avg_loss:.6f} | "
            f"Set5={set5_psnr:.2f} dB | "
            f"Set14={set14_psnr:.2f} dB | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "scale": args.scale,
            "set5_psnr": set5_psnr,
            "set14_psnr": set14_psnr,
        }

        torch.save(checkpoint, save_dir / "last.pth")

        if mean_psnr > best_score:
            best_score = mean_psnr
            torch.save(checkpoint, save_dir / "best.pth")
            print(f"Saved best model with mean PSNR: {best_score:.2f} dB")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", type=str, default="data/DIV2K/DIV2K_train_HR")
    parser.add_argument("--set5-dir", type=str, default="data/benchmark/Set5/HR")
    parser.add_argument("--set14-dir", type=str, default="data/benchmark/Set14/HR")
    parser.add_argument("--save-dir", type=str, default="outputs/checkpoints_edsr")

    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--samples-per-epoch", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step", type=int, default=200)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--n-feats", type=int, default=256)
    parser.add_argument("--n-resblocks", type=int, default=32)
    parser.add_argument("--res-scale", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
