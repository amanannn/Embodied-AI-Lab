#!/bin/bash
# CIFAR-10 数据集下载脚本（从 Hugging Face 镜像）
export HF_ENDPOINT=https://hf-mirror.com

pip install huggingface_hub -q

python3 << 'EOF'
from huggingface_hub import hf_hub_download
import tarfile, os

print("正在从 HF 镜像下载 CIFAR-10 (170MB)...")
path = hf_hub_download(
    repo_id="cifar10",
    repo_type="dataset",
    filename="cifar-10-python.tar.gz",
    local_dir="./data"
)
print(f"下载完成: {path}")

# 解压（torchvision 也能自动解压，这里显式解压确保万无一失）
extract_dir = "./data/cifar-10-batches-py"
if not os.path.exists(extract_dir):
    print("正在解压...")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(path="./data", filter="data")
    print(f"解压完成: {extract_dir}")
else:
    print("数据已就绪")

print("\n可以开始训练了: python train_sr.py")
EOF
