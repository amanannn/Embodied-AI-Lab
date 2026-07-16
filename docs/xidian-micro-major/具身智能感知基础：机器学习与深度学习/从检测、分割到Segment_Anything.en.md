# From Detection, Segmentation to Segment Anything

> Xidian University · Embodied AI Micro-Major
> Course Notes

---

## 1. The Granularity Progression of Vision Tasks

| Task | Granularity | Question | Output |
|------|------|------|------|
| Classification | Image-level | "What is in the image?" | Class label |
| Detection | Region-level | "Where is the object?" | Bounding box |
| Semantic Segmentation | Pixel-level | "Which class does each pixel belong to?" | Per-pixel class |
| Instance Segmentation | Pixel + ID | "What are the boundaries of each object instance?" | Per-pixel Mask + Instance ID |
| **General Segmentation** | **Any granularity** | **"You point, I segment"** | **Prompt → Mask** |

---

## 2. Semantic Segmentation

### 2.1 Definition

Assign a semantic class to every pixel in the image — "mark sky pixels as blue, road pixels as gray."

### 2.2 Classic Methods

| Method | Year | Core Idea |
|------|------|---------|
| **FCN** | 2015 | Fully convolutional network — replaces fully connected layers with convolutions, outputs pixel-level predictions |
| **U-Net** | 2015 | Encoder-decoder + skip connections — standard for medical imaging |
| **DeepLab** | 2016-2018 | Dilated convolution + ASPP + CRF post-processing |
| **PSPNet** | 2017 | Pyramid pooling captures multi-scale context |

> The historical significance of FCN: first to prove that "fully convolutional" networks can perform end-to-end pixel-level prediction.

### 2.3 Limitations of Semantic Segmentation

| Limitation | Description |
|------|------|
| **No Instance Distinction** | Two cats side by side → same class label, cannot distinguish individuals |
| **Fixed Classes** | Can only segment classes defined in the training set |
| **No Interaction** | Cannot specify the segmentation target according to user intent |

---

## 3. Instance Segmentation

### 3.1 Definition

On top of semantic segmentation, distinguish each instance — "this is the first cat, this is the second cat."

### 3.2 Representative Methods

| Method | Year | Idea |
|------|------|------|
| **Mask R-CNN** | 2017 | Faster R-CNN + segmentation branch — detection and segmentation in one |
| **YOLACT** | 2019 | Real-time instance segmentation (30+ FPS) |
| **SOLO** | 2020 | Distinguishes instances by position — no detection box needed |

> Mask R-CNN three outputs: class + box + Mask. The three heads share the backbone network, serving as a paradigm bridge from detection to segmentation.

---

## 4. SAM — General Segmentation

### 4.1 The Predicament Before SAM

| Problem | Traditional Solution |
|------|---------|
| Need to segment a new category | Annotate thousands of images, train a specialized model |
| Need to segment "the red one on the left" | Cannot describe with a class name |
| Need interactive segmentation | Requires a specially trained interactive tool |

### 4.2 SAM's Core Design

**Promptable Segmentation** — "You provide a prompt, I produce a Mask":

| Prompt Type | Example | How SAM Handles It |
|---------|------|------------|
| **Point** | Click on a location of the object | "Grow" a Mask outward from that point |
| **Box** | Draw a rectangle | Segment the region inside the box |
| **Text** | "the red cup" | Works with CLIP / Grounding DINO |
| **Mask** | Rough scribble | Refine to precise boundaries |

### 4.3 SAM Architecture

```
Input Image
    ↓
Image Encoder (ViT) → Image Embedding (once, reusable)
    ↓
Prompt Encoder → Prompt Embedding (point/box/text/mask)
    ↓
Mask Decoder → Output Mask + Confidence Score
```

| Component | Role | Characteristics |
|------|------|------|
| **Image Encoder** | ViT extracts image features | Heaviest part, only needs one forward pass, can be cached |
| **Prompt Encoder** | Encodes user prompts | Lightweight, real-time response |
| **Mask Decoder** | Fuses image + prompt → Mask | Can output multiple candidate masks |

### 4.4 Why is SAM Important?

| Significance | Description |
|------|------|
| **Data Flywheel** | Use SAM to auto-annotate → train specialized models → more data |
| **Zero-Shot Segmentation** | Can segment any object without training |
| **Composable** | SAM + CLIP = open-vocabulary segmentation; SAM + Grounding DINO = text → Mask |
| **Foundation for Embodied AI** | Robots need precise target masks before grasping |

> SAM is a watershed moment for vision foundation models — it is the first to prove the feasibility of **general segmentation**. It is not "just another segmentation model," but transforms segmentation from a "task" into a "capability."

---

## 5. The Complete Pipeline from Detection to SAM

```
Image
  ↓
[Detection] R-CNN / YOLO / DETR → Find object boxes
  ↓
[Classification] ResNet / ViT → Determine what is inside the box
  ↓
[Segmentation] FCN / Mask R-CNN → Per-pixel labeling
  ↓
[General Segmentation] SAM → Any prompt → Any object Mask
  ↓
[Open Vocabulary] CLIP + SAM → Natural language → Mask of specified object
  ↓
[Embodied Execution] Mask → 3D projection → Grasp planning
```

---

*Related notes: [Object Detection](./Computer_Vision_&_Object_Detection.md), [What are Foundation Models](./What_is_Foundation_Models.md), [Embodied Intelligence in the Era of Large Models](../具身智能认知进阶：多模态大模型与语义理解/大模型时代的具身智能.md)*
