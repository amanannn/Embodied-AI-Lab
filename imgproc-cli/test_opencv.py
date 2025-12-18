import cv2
import numpy as np

# 创建一张测试图
img = np.zeros((100, 100, 3), dtype=np.uint8)
img[:, :] = [255, 0, 0]  # Blue

# 保存
cv2.imwrite("test_output.jpg", img)
print("✅ OpenCV 成功生成 test_output.jpg")

# 读取
loaded = cv2.imread("test_output.jpg")
print(f"✅ 图像形状: {loaded.shape}, 像素值: {loaded[0,0]}")
