# MAE — Masked Autoencoders

> Xidian University · Embodied AI Micro-Major
> Course Notes
> Paper: Masked Autoencoders Are Scalable Vision Learners (He et al., CVPR 2022)

---

## 1. Motivation: Why Does Visual Pre-training Lag Behind NLP?

| Dimension | NLP (BERT) | CV (Before MAE) |
|------|------------|---------------|
| **Self-Supervised Task** | MLM — masked word prediction, naturally self-supervised | Relies on ImageNet labels or contrastive learning |
| **Data Efficiency** | Very high — 15% tokens predicted per batch | Relatively low |
| **Architecture Unification** | Transformer is a natural fit | Dominated by CNN, ViT just emerged |
| **Scaling** | GPT-3 demonstrated Scaling Law | Vision model scale far behind |

> MAE's core insight: visual signals are far more redundant than text — even with 75% of image patches masked, humans can still understand the content. This means **self-supervised reconstruction with extremely high mask ratios is feasible.**

---

## 2. MAE Core Design

### 2.1 Asymmetric Encoder-Decoder

| Component | Design | Computation |
|------|------|--------|
| **Encoder** | Standard ViT, processes **only visible patches** (25%) | Heavy — but only processes 1/4 of the patches |
| **Decoder** | Lightweight Transformer (fewer layers, narrower width), processes **all patches** (visible + masked) | Lightweight — discarded entirely after pre-training |

### 2.2 75% Extremely High Mask Ratio

| Mask Ratio | Meaning | Used by MAE |
|--------|------|---------|
| 15% (BERT) | Mask 1 out of every 7 tokens | Unsuitable for images — surrounding pixels can easily infer masked content |
| 50% | Half visible | Still too easy |
| **75% (MAE)** | **3/4 masked, only 1/4 visible** | ✅ Forces the model to learn global semantics |

> BERT only needs a 15% mask ratio because text semantics are dense; image pixels are extremely redundant — even with 75% masked, reconstruction is still possible. This is the most essential difference between MAE and BERT.

### 2.3 Training Pipeline

```
Input Image (224×224)
    ↓ Patchify — split into 196 patches (16×16)
    ↓ Random Mask — retain 25% (49) visible patches
    ↓
Encoder (ViT) — encode only visible patches → obtain visible patch features
    ↓
Decoder — visible patch features + mask tokens → reconstruct all 196 patches
    ↓
Loss — compute MSE only on masked patches (no penalty for visible patch reconstruction)
```

---

## 3. Key Technical Details

### 3.1 Why Does the Encoder Process Only Visible Patches?

| Approach | Computation | Effect |
|------|--------|------|
| Encoder processes all patches (visible + mask tokens) | $O(N^2)$, N=196 | Wasteful — significant computation spent on mask tokens |
| **Encoder processes only visible patches** | $O((N/4)^2)$ | Reduces computation by 16× |

> This is the key to MAE's efficiency — the Encoder runs on only 25% of the tokens, and the lightweight Decoder handles the full reconstruction.

### 3.2 Loss Function

$$L = \frac{1}{N_{mask}}\sum_{i \in masked} \| \hat{p}_i - p_i \|^2$$

- Computes MSE only for masked patches
- $p_i$ is the original pixel value (after normalization), $\hat{p}_i$ is the reconstructed pixel value
- Feature-space losses (e.g., CLIP features) can also be used, with better results but more complexity

### 3.3 Visual Reconstruction Results

A trained MAE can reconstruct a plausible full image from only 25% of visible patches — demonstrating that the model has learned global semantic understanding.

---

## 4. Core Differences Between MAE and BERT

| Dimension | BERT | MAE |
|------|------|-----|
| **Modality** | Natural language | Image |
| **Information Density** | High (each word carries rich semantics) | Low (adjacent pixels are nearly identical) |
| **Mask Ratio** | 15% | 75% |
| **Encoder Processes** | All tokens (including [MASK]) | Only visible patches |
| **Decoder** | None (MLM is a classification task, outputs directly) | Yes (pixel reconstruction is a regression task) |
| **Pre-training Task** | Word-level classification | Pixel-level regression |

---

## 5. Impact of MAE

| Area | Impact |
|------|------|
| **Self-Supervised Learning** | Proved the feasibility of high mask ratio + asymmetric architecture |
| **ViT Adoption** | MAE is the best pre-training companion for ViT |
| **Multimodal** | MAE ideas extend to video (VideoMAE), point clouds |
| **Computational Efficiency** | Only 1/4 tokens enter the Encoder → significantly reduces pre-training cost |

---

## 6. Limitations and Subsequent Work

| Limitation | Subsequent Exploration |
|------|---------|
| Pixel reconstruction loss does not directly equate to downstream task performance | Feature-space reconstruction (e.g., CLIP features replacing pixels) |
| Fixed mask ratio | Adaptive masking strategies |
| Images only | VideoMAE, MultiMAE (multimodal) |

---

*Related notes: [What are Foundation Models](./What_is_Foundation_Models.md), [From Detection, Segmentation to Segment Anything](./从检测、分割到Segment_Anything.md), [Multimodal Large Model Pre-training Topic](../具身智能认知进阶：多模态大模型与语义理解/多模态大模型预训练—微调—后训练专题研究.md)*
