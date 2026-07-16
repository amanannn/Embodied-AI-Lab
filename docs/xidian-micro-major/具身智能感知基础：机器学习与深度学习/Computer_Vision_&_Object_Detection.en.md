# Computer Vision — Object Detection

> Xidian University · Embodied AI Micro-Major
> Course Notes

---

## 1. Object Detection: Answering "Where is the Object"

Classification answers "what is in the image"; detection further requires **finding the location of the object**.

| Task | Output Granularity | Output Form | Representative Method |
|------|---------|---------|---------|
| Image Classification | Image-level | "This is a picture of a cat" | ResNet, ViT |
| **Object Detection** | Region-level | "Cat at (x1,y1)→(x2,y2)" | R-CNN, YOLO, DETR |
| Semantic Segmentation | Pixel-level | "Each pixel belongs to cat" | FCN, DeepLab |
| Instance Segmentation | Pixel-level + Instance | "This is the first cat, this is the second" | Mask R-CNN, SAM |

---

## 2. Core Challenges of Detection

| Challenge | Description | Solution |
|------|------|------|
| **Localization** | Objects can appear anywhere in the image | Dense candidate boxes + regression refinement |
| **Scale Variation** | Same class of objects can vary greatly in size | Multi-scale feature pyramid (FPN) |
| **Occlusion** | Objects may be partially hidden | Data augmentation + context reasoning |
| **Speed** | Real-time detection requires millisecond latency | Single-stage (YOLO) |
| **Small Objects** | Distant small objects carry little information | High resolution + feature fusion |

---

## 3. Evolution of Detection Methods

### 3.1 Two-Stage Detection

```
Input → Region Proposal → Region Classification + Location Refinement → Output Boxes
```

| Model | Year | Core Innovation | Characteristics |
|------|------|---------|------|
| **R-CNN** | 2014 | Selective Search → CNN classification | Slow but pioneering paradigm |
| **Fast R-CNN** | 2015 | One CNN pass per image → RoI Pooling | 25× faster than R-CNN |
| **Faster R-CNN** | 2015 | RPN replaces Selective Search | End-to-end trainable |
| **Mask R-CNN** | 2017 | + Segmentation branch | Detection + Instance segmentation unified |

#### Faster R-CNN Four Components

| Component | Function |
|------|------|
| **Backbone** | CNN extracts image feature maps |
| **RPN** | Sliding anchors → "Is there an object here?" + coarse location refinement |
| **RoI Pooling** | Unifies candidate regions of different sizes to a fixed size |
| **Head** | Classification (which class?) + Regression (exact location?) |

### 3.2 One-Stage Detection

```
Input → Dense Prediction (simultaneous classification + regression) → Output Boxes
```

| Model | Year | Core Innovation | Characteristics |
|------|------|---------|------|
| **YOLOv1** | 2016 | S×S grid + each cell predicts B boxes | 45 FPS |
| **SSD** | 2016 | Multi-scale feature layers detect simultaneously | Balances large and small objects |
| **RetinaNet** | 2017 | Focal Loss addresses positive-negative sample imbalance | Accuracy matches two-stage |
| **YOLOv3→v10** | 2018-2024 | FPN/anchor optimization/data augmentation | De facto industry standard |

> The philosophy of one-stage: abandon "rough screening then refinement," predict end-to-end directly. Trade speed for accuracy — the gap is now very small.

### 3.3 Transformer Detection

| Model | Year | Core Innovation |
|------|------|---------|
| **DETR** | 2020 | CNN → Transformer → Bipartite matching — no NMS needed |
| **Deformable DETR** | 2021 | Deformable attention, 10× faster convergence |
| **Grounding DINO** | 2023 | Text-conditioned detection — directly locate "the red cup" |

> DETR's paradigm shift: from "anchor + NMS post-processing" to "set prediction" — no more hand-crafted anchors and deduplication.

---

## 4. Comparative Summary

| Dimension | Two-Stage (Faster R-CNN) | One-Stage (YOLO) | Transformer (DETR) |
|------|----------------------|---------------|-------------------|
| Accuracy | ⭐⭐⭐ | ⭐⭐→⭐⭐⭐ | ⭐⭐⭐ |
| Speed | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| Small Objects | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Post-processing | NMS | NMS | No NMS needed |
| Deployment Friendly | Low | Very High | Medium |

---

## 5. Relationship with Embodied Intelligence

| Application | Role of Detection |
|------|-----------|
| **Object Grasping** | "Where is the thing I need to pick up?" |
| **Navigation & Obstacle Avoidance** | "There is a chair ahead, move around it" |
| **Scene Understanding** | Detect objects + spatial relationships |
| **Human-Robot Interaction** | Detect gestures, poses, actions |

> Detection is the "eyes" of embodied intelligence — almost every physical task begins with "finding the target."

---

*Related notes: [What are Foundation Models](./What_is_Foundation_Models.md), [From Detection, Segmentation to Segment Anything](./从检测、分割到Segment_Anything.md), [What is a Convolutional Network](./What_is_Convolution_network.md)*
