#!/bin/bash
# CIFAR-10 数据集下载脚本
set -e

mkdir -p ./data
FILE="./data/cifar-10-python.tar.gz"
EXTRACT_DIR="./data/cifar-10-batches-py"

if [ -d "$EXTRACT_DIR" ]; then
    echo "数据已就绪: $EXTRACT_DIR"
    echo "可以直接训练: python train_sr.py"
    exit 0
fi

# 清理可能损坏的旧文件
rm -f "$FILE"

echo "正在下载 CIFAR-10 (170MB)..."
# HF 镜像直链，失败回退官网
wget -c -O "$FILE" --timeout=30 --tries=3 \
    https://hf-mirror.com/datasets/cifar10/resolve/main/cifar-10-python.tar.gz \
    || wget -c -O "$FILE" --timeout=60 --tries=5 \
    https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

echo "下载完成，正在解压..."
tar -xzf "$FILE" -C ./data/
echo "解压完成: $EXTRACT_DIR"
echo ""
echo "可以开始训练了: python train_sr.py"
