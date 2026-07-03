import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SRDataset
from model import SRCNN
from utils import calc_psnr, save_compare_image, tensor_to_image

@torch.no_grad()
def test_dataset(model, dataset_name, hr_dir, args, device):
    dataset = SRDataset(hr_dir, scale=args.scale, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    output_dir = Path(args.output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_bicubic_psnr = 0.0
    total_srcnn_psnr = 0.0
    
    model.eval()
    
    for batch in tqdm(loader, desc=f"Testing {dataset_name}"):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        name = batch["name"][0]
        
        sr = model(lr)
        
        bicubic_psnr = calc_psnr(lr, hr)
        srcnn_psnr = calc_psnr(sr, hr)
        
        total_bicubic_psnr += bicubic_psnr
        total_srcnn_psnr += srcnn_psnr
        
        lr_image = tensor_to_image(lr[0])
        sr_image = tensor_to_image(sr[0])
        hr_image = tensor_to_image(hr[0])
        
        save_compare_image(
            lr_image,
            sr_image,
            hr_image,
            output_dir / f"{name}_compare.png",
        )
    
    count = len(dataset)
    avg_bicubic = total_bicubic_psnr / count
    avg_srcnn = total_srcnn_psnr / count
    
    print(f"{dataset_name} Bicubic PSNR: {avg_bicubic:.2f} dB")
    print(f"{dataset_name} SRCNN PSNR: {avg_srcnn:.2f} dB")
    print(f"{dataset_name} Improvement: {avg_srcnn - avg_bicubic:.2f} dB")
    
    return avg_bicubic, avg_srcnn

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    model = SRCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    test_dataset(model, "Set5", args.set5_dir, args, device)
    test_dataset(model, "Set14", args.set14_dir, args, device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pth")
    parser.add_argument("--set5-dir", type=str, default="data/benchmark/Set5/HR")
    parser.add_argument("--set14-dir", type=str, default="data/benchmark/Set14/HR")
    parser.add_argument("--output-dir", type=str, default="outputs/results")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)