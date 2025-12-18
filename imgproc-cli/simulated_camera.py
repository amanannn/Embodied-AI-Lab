#!/usr/bin/env python3
"""
模拟摄像头：用动态合成图像代替真实摄像头
适用于 WSL2 / 无摄像头设备 / CI 测试
"""
import cv2
import numpy as np
import time

def generate_synthetic_frame(t):
    """生成带时间变化的合成图像"""
    # 创建 480x640 黑底
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 画一个移动的圆（模拟物体运动）
    x = int(320 + 200 * np.sin(t))
    y = int(240 + 100 * np.cos(t * 0.7))
    cv2.circle(frame, (x, y), 50, (0, 255, 0), -1)
    
    # 添加文字
    cv2.putText(frame, f"Time: {t:.1f}s", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def preprocess_frame(frame, output_size=(84, 84)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, output_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

def main():
    print("🖼️  使用合成图像模拟摄像头（无需真实设备）")
    print("   按 Ctrl+C 退出")
    
    start_time = time.time()
    try:
        while True:
            t = time.time() - start_time
            frame = generate_synthetic_frame(t)
            obs = preprocess_frame(frame)
            
            # 显示
            cv2.imshow('Simulated Camera', frame)
            cv2.imshow('Gym Observation', obs)
            
            # 保存
            np.save('/tmp/gym_observation.npy', obs)
            print(f"\r💾 已保存观测: shape={obs.shape}, time={t:.1f}s", end='', flush=True)
            
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("\n✅ 模拟摄像头已停止")

if __name__ == "__main__":
    main()
