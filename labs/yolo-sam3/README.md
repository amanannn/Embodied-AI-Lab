# YOLO + SAM3 实验

基于 VOC2007 数据集的 YOLO11 目标检测实验，配合 SAM3 进行分割分析。

## 项目结构

```
yolo-sam3/
├── YOLO与SAM3实验说明及验收要求.docx ← 实验文档
├── yolo_voc.yaml                     ← VOC 数据集配置（20类）
├── train.py                          ← YOLO 训练脚本
├── val.py                            ← YOLO 验证脚本（mAP/Precision/Recall）
├── test.py                           ← YOLO 推理/测试脚本
└── README.md
```

## 依赖

```bash
pip install ultralytics
# SAM3 相关依赖请参考官方文档
```

## 用法

```bash
# 训练（需先准备 VOC2007 数据集）
python train.py

# 验证
python val.py

# 推理
python test.py
```

## 实验内容

- YOLO11 nano 在 VOC2007 上的训练
- imgsz=640 vs 768 对比实验
- SAM3 分割结果分析
