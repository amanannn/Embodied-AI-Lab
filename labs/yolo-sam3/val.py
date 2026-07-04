"""YOLO 验证脚本 — 计算 Precision, Recall, mAP50, mAP50-95, 推理速度"""
from ultralytics import YOLO

# ==================== 配置参数 ====================
WEIGHTS = "runs/detect/voc_yolo11n_768/weights/best.pt"  # 修改为你的权重路径
IMGSZ = 768
DATA_CFG = "yolo_voc.yaml"  # 数据集配置文件
# =================================================

model = YOLO(WEIGHTS)

results = model.val(
    data=DATA_CFG,
    imgsz=IMGSZ,
    device=0,
    batch=16,
)

# ---- 提取指标 ----
map50 = results.box.map50       # mAP@0.5
map_all = results.box.map        # mAP@0.5:0.95
prec = results.box.p.mean()      # Precision
recall = results.box.r.mean()    # Recall

print("\n" + "=" * 50)
print(f"整体 mAP50:      {map50:.4f}")
print(f"整体 mAP50-95:   {map_all:.4f}")
print(f"平均精确率 Precision: {prec:.4f}")
print(f"平均召回率 Recall:    {recall:.4f}")
print(f"推理速度: {results.speed}")
print("=" * 50)
