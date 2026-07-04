"""YOLO 训练脚本 — VOC 数据集，imgsz=640 和 768 对比实验"""
from ultralytics import YOLO

# ==================== 配置参数 ====================
IMGSZ = 768
NAME = "voc_yolo11n_768"
# =================================================

model = YOLO("yolo11n.pt")

results = model.train(
    data="yolo_voc.yaml",  # 请根据实际数据集路径修改
    epochs=100,
    imgsz=IMGSZ,
    amp=False,        # 关闭 AMP，避免额外下载权重导致中断
    device=0,
    name=NAME,
)
