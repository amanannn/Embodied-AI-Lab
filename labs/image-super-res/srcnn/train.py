import argparse 
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SRDataset
from model import SRCNN
from utils import calc_psnr

@torch.no_grad()

def evaluate(model, data_loader, device):
    model.eval()
    total_psnr = 0.0
    image_count = 0
    for batch in data_loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        sr = model(lr)
        psnr = calc_psnr(sr, hr)
        total_psnr += psnr
        image_count += 1

    return total_psnr / image_count 

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    train_set = SRDataset(
        hr_dir=args.train_dir,
        scale=args.scale,
        patch_size=args.patch_size,
        train=True,
        samples_per_epoch=args.samples_per_epoch
    )

    set5 =SRDataset(args.set5_dir, scale=args.scale, train=False)
    set14 = SRDataset(args.set14_dir, scale=args.scale, train=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    set5_loader = DataLoader(set5, batch_size=1, shuffle=False, num_workers=0)
    set14_loader = DataLoader(set14, batch_size=1, shuffle=False, num_workers=0)

    model = SRCNN().to(device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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

        avg_loss = total_loss / len(train_loader)

        set5_psnr = evaluate(model, set5_loader, device)
        set14_psnr = evaluate(model, set14_loader, device)

        mean_psnr = (set5_psnr + set14_psnr) / 2.0

        print(
            f"Epoch {epoch:03d} | "
            f"loss={avg_loss:.6f} | "
            f"Set5={set5_psnr:.2f} dB | "
            f"Set14={set14_psnr:.2f} dB | "
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
    parser.add_argument("--save-dir", type=str, default="outputs/checkpoints")

    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--samples-per-epoch", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)