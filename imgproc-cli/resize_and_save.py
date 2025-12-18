#!/usr/bin/env python3
"""
简单图像处理示例：读取、resize、保存
依赖：opencv-python, numpy
"""
import cv2
import numpy as np
import sys
import os

def main():
    # 创建一张合成图像（避免依赖外部文件）
    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Resize 到 224x224（常见输入尺寸）
    resized = cv2.resize(img, (224, 224))
    
    # 保存
    output_path = "/tmp/resized_output.jpg"
    cv2.imwrite(output_path, resized)
    
    print(f"✅ 图像已处理并保存到: {output_path}")
    print(f"原始尺寸: {img.shape} → 新尺寸: {resized.shape}")
    print(f"NumPy 版本: {np.__version__}")
    print(f"OpenCV 版本: {cv2.__version__}")

if __name__ == "__main__":
    main()
