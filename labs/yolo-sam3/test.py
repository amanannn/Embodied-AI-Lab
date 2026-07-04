"""YOLO 测试/推理脚本 — 在单张图片或视频上运行检测"""
from ultralytics import YOLO

# ==================== 配置参数 ====================
WEIGHTS = "runs/detect/voc_yolo11n_768/weights/best.pt"  # 修改为你的权重路径
IMGSZ = 768          # 768 模型时改为 768
SOURCE = "datasets/VOC/images/test/000025.jpg"  # 修改为你的测试图片路径
CONF = 0.5           # 置信度阈值
# =================================================

model = YOLO(WEIGHTS)

results = model(
    SOURCE,
    save=True,
    imgsz=IMGSZ,
    conf=CONF,
    line_width=2,
    show_labels=True,
    show_conf=True,
)

for r in results:
    print(f"\n检测到 {len(r.boxes)} 个目标")
    for box in r.boxes:
        cls_name = model.names[int(box.cls)]
        conf = box.conf.item()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  - 类别：{cls_name}，置信度：{conf:.2f}，框：({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
