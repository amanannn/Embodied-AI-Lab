# What is VLA — Vision-Language-Action Model

> Xidian University · Embodied AI Micro-Major
> Course Notes

---

## 1. What is VLA?

VLA (Vision-Language-Action) is the core model paradigm of embodied intelligence:

| Component | Role | Form |
|------|------|------|
| **V (Vision)** | Understand the environment | RGB image / Depth map / Point cloud |
| **L (Language)** | Understand the task | Natural language instruction |
| **A (Action)** | Generate actions | Joint angles / End-effector pose / Control commands |

$$A = f_\theta(V, L)$$

> VLA = enabling robots to "see" (Vision), "understand" (Language), and "do" (Action).

---

## 2. Why is VLA Needed?

### 2.1 The Predicament of Traditional Robot Programming

| Traditional Approach | Problem |
|---------|------|
| Hand-code every action | Change task = rewrite code |
| Fixed skill library | Scenarios too complex to enumerate |
| Teach programming | Requires human step-by-step demonstration |

### 2.2 The Breakthrough of VLA

| Capability | Traditional | VLA |
|------|------|-----|
| Task Instruction | Program / API | **Natural language** |
| Scene Adaptation | Fixed environment | **Generalizes to new scenes** |
| Novel Objects | Requires re-annotation | **Zero-shot understanding** |
| Action Generation | Pre-programmed | **End-to-end learning** |

---

## 3. VLA Architecture

### 3.1 General Architecture Pattern

```
Visual Input (Camera)                   Language Instruction (Text)
      ↓                                      ↓
Vision Encoder                       Language Encoder
(ViT / SigLIP / DINO)                (LLaMA / Gemma / Qwen)
      ↓                                      ↓
      └────────────┬─────────────────────────┘
                   ↓
           Multimodal Fusion
         (Cross-Attention / Concat / Q-Former)
                   ↓
            Action Decoder
         (Diffusion / MLP / Autoregressive)
                   ↓
             Action Output
      (Joint Angles / End-effector Pose / Trajectory)
```

### 3.2 Three Main Paradigms

| Paradigm | Action Representation | Representative Model | Characteristics |
|------|---------|---------|------|
| **Discretization** | Discretize action space into tokens | RT-2, Octo | Reuses LLM's next-token prediction |
| **Diffusion** | Use diffusion model to denoise continuous actions | Diffusion Policy, Pi0 | Handles multimodal action distributions |
| **Regression** | Directly predict continuous action values | ACT, BC | Simple and direct, suitable for unimodal |

---

## 4. Representative VLA Models

| Model | Year | Institution | Core Idea |
|------|------|------|---------|
| **RT-2** | 2023 | Google DeepMind | Vision-language model + action tokens — "treating actions as words" |
| **Octo** | 2024 | UC Berkeley | Open-source general robot policy — multi-embodiment, multi-task |
| **Pi0 ($\pi_0$)** | 2024 | Physical Intelligence | Diffusion actions + flow matching — "robot GPT moment" |
| **OpenVLA** | 2024 | Stanford | Open-source 7B VLA — based on Llama + DINOv2 + SigLIP |
| **GR00T** | 2024 | NVIDIA | Multimodal humanoid robot foundation model |

### RT-2 in Detail

```
"Pick up the red cup on the left"
        ↓
PaLI-X / PaLM-E (Vision-Language Backbone)
        ↓
Output token: "<grip> 128 256 45 <release> 320 180 90"
        ↓
Token → Joint angles / End-effector pose → Robot executes
```

> RT-2's core trick: **discretize the continuous action space into tokens**, reusing the LLM architecture and pre-training — no specialized "action network" needed.

### Pi0 in Detail

| Component | Role |
|------|------|
| **PaliGemma / Vision Backbone** | Image → Features |
| **Gemma / Language Backbone** | Instruction → Features |
| **Diffusion Action Head** | Denoise to generate continuous action sequences |
| **Flow Matching** | More efficient diffusion sampling than DDPM |

---

## 5. VLA Training Paradigm

| Stage | Objective | Data | Method |
|------|------|------|------|
| **Pre-training** | General vision-language alignment | Internet image-text pairs | CLIP-style contrastive learning |
| **Action Fine-tuning** | Learn "see + listen → act" | Robot expert demonstrations | Behavior Cloning / Diffusion Policy |
| **Alignment** | Action safety + task success rate | Human preferences | RLHF / DPO (early exploration) |

---

## 6. Current Challenges for VLA

| Challenge | Severity | Description |
|------|---------|------|
| **Data Scarcity** | 🔴 Most critical | Robot data is 4-5 orders of magnitude less than text data |
| **Cross-Embodiment Generalization** | 🟡 | Different robots have different degrees of freedom and action spaces |
| **Real-Time Performance** | 🟡 | Perception → reasoning → action must complete within 100ms |
| **Safety** | 🟡 | VLA hallucinations can be disastrous in the physical world |
| **Evaluation Standards** | 🟡 | Lack of unified benchmark |

---

## 7. VLA Ecosystem

| Project | Role |
|------|------|
| **Open X-Embodiment** | Largest robot dataset — 1M+ trajectories |
| **LIBERO** | Standard VLA benchmark |
| **robosuite / MuJoCo** | Simulation environments |
| **HuggingFace LeRobot** | Open-source VLA training framework |
| **FluxVLA** (LimX) | Industrial-grade VLA engineering platform |

---

## 8. Relationship with Your GraspAnything Project

GraspAnything is essentially a **lightweight VLA system**:

```
Vision: Grounding DINO + SAM → Open-vocabulary object detection and segmentation
Language: Natural language describing the target object
Action: robosuite grasp execution
```

> It is the best starting point for understanding the full VLA stack — starting from a simple system and gradually understanding the design motivations of large models like RT-2 and Pi0.

---

*Related notes: [Embodied Intelligence in the Era of Large Models](./大模型时代的具身智能.md), [Multimodal Large Model Pre-training Topic](./多模态大模型预训练—微调—后训练专题研究.md), [What is Transformer](./What_is_Transformer.md)*
