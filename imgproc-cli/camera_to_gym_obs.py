#!/usr/bin/env python3
"""
从摄像头读取 → 预处理 → 保存为 Gym 观测格式 (84x84 灰度, float32, [0,1])
输出: observation.npy (形状: (84, 84))
兼容 Gymnasium 和 Stable-Baselines3
"""
import cv2
import numpy as np
import os
import sys

def preprocess_frame(frame, output_size=(84, 84)):
    """
    将原始 BGR 帧转为 Gym 兼容观测
    输入: (H, W, 3) uint8 BGR
    输出: (84, 84) float32 [0.0, 1.0]
    """
    # 1. 转灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize 到目标尺寸
    resized = cv2.resize(gray, output_size, interpolation=cv2.INTER_AREA)
    
    # 3. 归一化到 [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized

def main():
    print("🎥 启动摄像头... (按 'q' 退出)")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头！请检查设备权限。")
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 无法读取帧")
                break

            # 预处理
            obs = preprocess_frame(frame)

            # 显示原始 + 预处理结果
            cv2.imshow('Original', frame)
            cv2.imshow('Gym Observation (84x84)', obs)

            # 保存最新观测（覆盖写入）
            np.save('/tmp/gym_observation.npy', obs)
            print(f"\r💾 已保存观测: shape={obs.shape}, dtype={obs.dtype}", end='', flush=True)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ 摄像头已关闭")

    # 最终验证文件
    if os.path.exists('/tmp/gym_observation.npy'):
        loaded = np.load('/tmp/gym_observation.npy')
        print(f"🔍 最终观测验证: {loaded.shape}, min={loaded.min():.2f}, max={loaded.max():.2f}")
        
        # 检查是否符合 Gym 标准
        assert loaded.shape == (84, 84), "观测尺寸错误！"
        assert loaded.dtype == np.float32, "数据类型应为 float32"
        assert 0.0 <= loaded.min() and loaded.max() <= 1.0, "值应在 [0,1] 范围内"
        print("✅ 观测格式完全兼容 Gymnasium！")

if __name__ == "__main__":
    main()
