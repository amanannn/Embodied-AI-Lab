import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

def list_image_files(folder):
    folder = Path(folder)
    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    file = []
    for path in folder.rglob("*"):
        if path.suffix.lower() in image_exts:
            file.append(path)
    return sorted(file)

def image_to_tensor(image):
    image = image.convert("RGB")
    array = np.array(image).astype(np.float32) 
    array = array / 255.0
    array = array.transpose(2, 0, 1) 
    return torch.from_numpy(array)

def tensor_to_image(tensor):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    array = tensor.numpy().transpose(1, 2, 0) 
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)

def calc_psnr(sr, hr):
    sr = sr.detach().clamp(0.0, 1.0)
    hr = hr.detach().clamp(0.0, 1.0)
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / mse)

def save_compare_image(lr_image, sr_image, hr_image, save_path):
    width, height = hr_image.size
    label_height = 28
    canvas = Image.new("RGB", (width * 3, height + label_height), "white")
    canvas.paste(lr_image, (0, label_height))
    canvas.paste(sr_image, (width, label_height))
    canvas.paste(hr_image, (width * 2, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), "Bicubic", fill=(0, 0, 0))
    draw.text((width + 10, 8), "SRCNN", fill=(0, 0, 0))
    draw.text((width * 2 + 10, 8), "HR", fill=(0, 0, 0))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)