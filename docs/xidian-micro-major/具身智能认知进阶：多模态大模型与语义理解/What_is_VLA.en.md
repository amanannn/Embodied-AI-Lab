# What is VLA — Vision-Language-Action Model

> Xidian University · Embodied AI Micro-Major
> Course Notes

---

## 1. VLA Technology Origins

### 1.1 What is VLA?

VLA (Vision-Language-Action) is the core model paradigm of embodied intelligence — enabling robots to simultaneously "see" the environment, "understand" instructions, and "execute" actions:

| Component | Role | Form |
|------|------|------|
| **V (Vision)** | Understand the environment | RGB image / Depth map / Point cloud |
| **L (Language)** | Understand the task | Natural language instruction |
| **A (Action)** | Generate actions | Joint angles / End-effector pose / Control commands |

$$A = f_\theta(V, L)$$

> VLA = enabling robots to "see" (Vision), "understand" (Language), and "do" (Action).

### 1.2 Why VLA? — The End of the Traditional Paradigm

#### The Predicament of Traditional Robot Programming

| Traditional Approach | Problem |
|---------|------|
| Hand-code every action | Change task = rewrite code |
| Fixed skill library | Scenarios too complex to enumerate |
| Teach programming | Requires human step-by-step demonstration |
| Modular pipeline | Perception → Planning → Control — errors amplify at each stage |

The fundamental contradiction of traditional robotics: **structured programming vs. the open world**. Real environments contain infinitely many combinations of objects, scenes, and tasks — code can never cover them all.

#### The Breakthrough of VLA

| Capability | Traditional | VLA |
|------|------|-----|
| Task Instruction | Program / API | **Natural language** |
| Scene Adaptation | Fixed environment | **Generalizes to new scenes** |
| Novel Objects | Requires re-annotation | **Zero-shot understanding** |
| Action Generation | Pre-programmed | **End-to-end learning** |
| Knowledge Transfer | None | **Internet-scale pre-training knowledge** |

### 1.3 Evolution Path: From Modular to End-to-End

VLA did not emerge from nowhere — it is the natural extension of Vision-Language Models (VLMs) into embodied intelligence:

```
Traditional Robotics (Modular)
    Perception → Planning → Control
         ↓
Vision-Language Models (VLM)
    CLIP (2021) → BLIP-2 (2023) → LLaVA (2023)
         ↓
VLA Pioneering Work
    SayCan (2022)     —— LLM for high-level planning, skill library for execution
    PaLM-E (2023)     —— Multimodal LLM directly fusing vision and robot state
    RT-1 (2022)       —— Transformer for end-to-end vision-action mapping
    Gato (2022)       —— Single Transformer for multiple tasks (including robot control)
         ↓
VLA Paradigm Established
    RT-2 (2023)       —— "Actions as tokens" — VLM + action discretization
    Octo / OpenVLA (2024) —— Open-source VLA foundation models
    Pi0 (2024)        —— Diffusion actions + flow matching
```

**Key turning points:**
- **SayCan** proved LLMs can plan robot tasks (but actions still require a pre-programmed skill library)
- **RT-1** proved Transformers can directly output actions (but relies on robot-specific data)
- **RT-2** unified both — a single Transformer that both understands language and outputs actions, achieving true "end-to-end VLA"

### 1.4 Three Foundational Action Generation Paradigms

| Paradigm | Action Representation | Representative Model | Core Idea |
|------|---------|---------|---------|
| **Discretization** | Discretize action space into tokens | RT-2, Octo | Reuses LLM's next-token prediction — "treating actions as words" |
| **Diffusion** | Denoise to generate continuous actions | Diffusion Policy, Pi0 | Handles multimodal action distributions — ideal for dexterous manipulation |
| **Regression** | Directly predict continuous action values | ACT, BC | Simple and direct, training-efficient |

> These three paradigms form the technical foundation for all subsequent advanced VLA models.

---

## 2. VLA Advanced Foundation Technology

### 2.1 Universal Architecture: Four-Layer Stack

Modern VLA foundation models generally follow a four-layer architecture:

```
Visual Input (Camera)                   Language Instruction (Text)
      ↓                                      ↓
┌─────────────────────┐    ┌─────────────────────┐
│  Vision Encoder     │    │  Language Encoder   │
│  ViT / SigLIP /     │    │  LLaMA / Gemma /    │
│  DINOv2 / CLIP      │    │  Qwen / InternLM    │
└─────────┬───────────┘    └─────────┬───────────┘
          ↓                          ↓
          └────────────┬─────────────┘
                       ↓
          ┌─────────────────────────┐
          │   Multimodal Fusion     │
          │  Cross-Attention /      │
          │  Concat / Q-Former /    │
          │  MLP Projector          │
          └─────────────┬───────────┘
                        ↓
          ┌─────────────────────────┐
          │    Action Decoder       │
          │  Diffusion Head /       │
          │  Autoregressive /       │
          │  Flow Matching          │
          └─────────────┬───────────┘
                        ↓
                   Action Output
          (Joint Angles / End-effector Pose / Trajectory)
```

### 2.2 Representative Models — Deep Dive

#### Overview

| Model | Year | Institution | Core Idea | Parameters | Open-Source |
|------|------|------|---------|---------|------|
| **RT-2** | 2023 | Google DeepMind | VLM + action tokenization | 55B | ✗ |
| **Octo** | 2024 | UC Berkeley | Open-source general policy, multi-embodiment, multi-task | 27M–93M | ✓ |
| **OpenVLA** | 2024 | Stanford | 7B open-source VLA, Llama + DINOv2 + SigLIP | 7B | ✓ |
| **Pi0 ($\pi_0$)** | 2024 | Physical Intelligence | Diffusion actions + flow matching | 3B | ✗ |
| **GR00T N1** | 2024–25 | NVIDIA | Humanoid robot foundation model | ~2B | ✓ |
| **RDT-1B** | 2024 | Tsinghua | 1B-parameter diffusion VLA, bimanual manipulation | 1B | ✓ |

---

#### RT-2 (Google DeepMind, 2023)

**Core idea: Treat actions as "language"**

```
"Pick up the red cup on the left"
        ↓
PaLI-X / PaLM-E (Vision-Language Backbone, 55B)
        ↓
Output token sequence:
    "<grip> 128 256 45 <release> 320 180 90"
        ↓
Token → Joint angles / End-effector pose → Robot executes
```

**Technical highlights:**
- **Action tokenization**: Discretize continuous 6-DoF actions into 256 bins, each bin corresponding to a token
- **Co-fine-tuning**: Jointly fine-tune on internet image-text data + robot data
- **Key finding**: Large-scale VLM pre-training provides semantic understanding that directly transfers to robot control tasks

> RT-2's core insight: **no specialized "action network" is needed** — if a model can understand the semantics of "grasp the red cup," it can also learn the corresponding action token sequence.

---

#### Octo (UC Berkeley, 2024)

**Core idea: One model to rule multiple robots**

| Design Goal | Implementation |
|---------|---------|
| **Multi-embodiment support** | Input includes embodiment description (DoF, camera params); model learns embodiment-invariant representations |
| **Multi-task unification** | Supports joint training on 25+ datasets |
| **Lightweight & open-source** | As small as 27M params, fine-tunable on consumer GPUs |
| **Plug-and-play** | HuggingFace models + standard inference interface |

**Architecture:**
- Vision encoder: Lightweight ResNet / ViT
- Language encoder: T5-base
- Action head: Transformer decoder + diffusion head
- Training data: Full Open X-Embodiment (1M+ trajectories)

> Octo's significance: proved that **open-source VLA foundation models are viable** — no longer just DeepMind can do it.

---

#### OpenVLA (Stanford, 2024)

**Core idea: Build the strongest open-source VLA with the strongest open-source components**

```
Visual Input (multi-view RGB)        Language Instruction
      ↓                                    ↓
┌──────────────┐                 ┌──────────────┐
│ SigLIP (ViT) │                 │   LLaMA-7B   │
│  + DINOv2    │                 │  (language)   │
│  (vision)    │                 │               │
└──────┬───────┘                 └──────┬────────┘
       ↓                                ↓
       └──────────┬─────────────────────┘
                  ↓
         ┌───────────────┐
         │  MLP Projector│
         │  (fusion)     │
         └───────┬───────┘
                 ↓
         ┌───────────────┐
         │  Action Head  │
         │  (discretized  │
         │   tokens)      │
         └───────────────┘
                 ↓
            Action Output
```

**Key innovations:**
- **Dual vision encoder fusion**: SigLIP (semantic understanding) + DINOv2 (spatial understanding) — complementary
- **Efficient fine-tuning**: LoRA on Open X-Embodiment, updating only ~1% of parameters
- **Performance**: Approaches RT-2-X (55B) on LIBERO benchmark with only 7B parameters

---

#### Pi0 / $\pi_0$ (Physical Intelligence, 2024)

**Core idea: Diffusion models are the best choice for robot action generation**

| Component | Selection | Role |
|------|------|------|
| **Vision Backbone** | PaliGemma | Image → multimodal features |
| **Language Backbone** | Gemma | Instruction → text features |
| **Diffusion Action Head** | Flow Matching | Iteratively denoise to generate continuous action sequences |
| **Output Frequency** | 50Hz | Meets real-time dexterous manipulation requirements |

**Why diffusion?**
- Robot action distributions are often **multimodal** (the same task can have multiple valid grasp strategies)
- Diffusion models naturally model multimodal distributions — regression models can only learn the "average action," leading to execution failures
- **Flow Matching** samples 10× faster than traditional DDPM, making real-time control feasible

> "The GPT moment for robotics" — Pi0 is widely regarded as the GPT-3 moment for robot foundation models.

---

#### GR00T N1 (NVIDIA, 2024–2025)

**Core idea: A general-purpose foundation model for humanoid robots**

| Feature | Description |
|------|------|
| **Target hardware** | Humanoid robots (bipedal / dual-arm) |
| **Training data** | Synthetic data + teleoperation data + internet video |
| **Simulation platform** | Isaac Sim / Isaac Lab |
| **Open-source** | Model weights and training framework open-sourced in 2025 |
| **Key capabilities** | Full-body coordinated control, bimanual manipulation, long-horizon tasks |

**Tech stack:**
- Vision-Language backbone: Based on NVIDIA VILA architecture
- Action generation: Diffusion Transformer (DiT)
- Training strategy: Large-scale sim pre-training → real-robot fine-tuning (Sim-to-Real)

---

#### RDT-1B (Tsinghua, 2024)

**Core idea: Diffusion Transformer + large-scale parameters = dexterous bimanual hands**

- 1B-parameter diffusion model, specifically designed for **bimanual coordinated manipulation**
- Jointly trained on 46 datasets
- Demonstrated that scaling laws hold for robot diffusion policies: larger models → stronger generalization

### 2.3 Technical Route Divergence — Summary

```
VLA Foundation Technology Tree
├── Discretization Route (token-based)
│   ├── RT-2 (2023)     —— Foundation stone, closed-source
│   ├── Octo (2024)     —— Lightweight open-source
│   └── OpenVLA (2024)  —— 7B open-source benchmark
│
├── Diffusion Route (diffusion-based)
│   ├── Diffusion Policy (2023) —— Academic foundation
│   ├── Pi0 (2024)      —— "GPT moment"
│   ├── RDT-1B (2024)   —— Bimanual manipulation scaling
│   └── GR00T N1 (2025) —— Humanoid robots
│
└── Hybrid Route
    └── Future trend: discrete tokens for high-level planning + diffusion for low-level execution
```

---

## 3. VLA Training Data and Benchmarks

### 3.1 Training Paradigm

| Stage | Objective | Data | Method |
|------|------|------|------|
| **Pre-training** | General vision-language alignment | Internet image-text pairs (billions) | CLIP-style contrastive learning |
| **Action Pre-training** | Cross-embodiment action representations | Multi-robot dataset mixture | Behavior Cloning / Joint training |
| **Action Fine-tuning** | Task-specific "see + listen → act" | Target robot expert demonstrations | BC / Diffusion Policy / LoRA |
| **Alignment** | Safety + success rate | Human preferences | RLHF / DPO (early exploration) |

### 3.2 Training Datasets — Panoramic View

#### Flagship Datasets

| Dataset | Scale | Robot Count | Task Types | Year |
|--------|------|-----------|---------|------|
| **Open X-Embodiment (OXE)** | 1M+ trajectories | 22 robot types | Grasping, pushing, folding, etc. | 2023 |
| **DROID** | 76K trajectories / 350 hours | 1 (Franka) | Diverse tabletop manipulation | 2024 |
| **BridgeData V2** | 60K+ trajectories | 1 (WidowX) | Kitchen/tabletop manipulation | 2023 |
| **RH20T** | 110K+ trajectories | 4 robot types | Multi-skill manipulation | 2024 |
| **AgiBot World** | 1M+ trajectories | 1 | Full-task real-world scenarios | 2025 |

#### Data Collection Methods

| Method | Cost | Scale | Quality | Representative |
|------|------|------|------|------|
| **Teleoperation** | High (labor-intensive) | Small | High | DROID, RoboTurk |
| **Human demonstration videos** | Medium | Medium | Medium | Ego4D, YouTube |
| **Simulation generation** | Low (GPU) | Very large | Medium (Sim-to-Real gap) | Isaac Sim, MuJoCo |
| **Self-supervised / Auto-collection** | Low | Large | Low–Medium | SayCan auto-retry |

> 🔴 **Data scarcity remains the most critical bottleneck for VLA**: robot trajectory data is 4–5 orders of magnitude less than internet text data. This is a "big model" challenge in a "small data" domain.

### 3.3 Mainstream Benchmarks

#### Simulation Benchmarks

| Benchmark | Environment | Task Count | Evaluation Dimensions | Characteristics |
|-----------|------|--------|---------|------|
| **LIBERO** | robosuite | 130 | Spatial, object, goal, and compositional generalization | **Standard VLA benchmark** |
| **CALVIN** | PyBullet | 34 | Long-horizon task execution | Progressive sequence length evaluation |
| **RLBench** | CoppeliaSim | 100+ | Per-task success rate and sample efficiency | Richest task variety |
| **SIMPLER** | Real→sim reconstruction | — | **Sim proxy for real robot performance** | Highly correlated with real results |
| **ManiSkill** | SAPIEN | 100+ | Dexterous manipulation | Physically high-fidelity |

#### Real-Robot Benchmarks

| Benchmark | Robot | Core Metrics |
|-----------|--------|---------|
| **Google Robot Setups** | Everyday Robots / ALOHA | Task success rate, generalization to novel objects |
| **FMB (FurnitureBench)** | Franka | Complex assembly, long-horizon |
| **TidyBot++** | Mobile manipulator | Household item generalization |

### 3.4 Evaluation Dimension System

```
VLA Evaluation Dimensions
├── Generalization
│   ├── Novel objects
│   ├── Novel scenes
│   ├── Novel tasks
│   └── Novel embodiments
│
├── Execution Quality
│   ├── Success rate
│   ├── Completion time
│   └── Trajectory smoothness
│
├── Robustness
│   ├── Disturbance recovery
│   ├── Visual occlusion
│   └── Initial state variation
│
└── Efficiency
    ├── Inference latency (target <100ms)
    ├── Data efficiency (few-shot adaptation)
    └── Parameter efficiency
```

### 3.5 Core Challenges

| Challenge | Severity | Description | Current Progress |
|------|---------|------|---------|
| **Data Scarcity** | 🔴 Most critical | Robot data is 4–5 orders of magnitude less than text data | OXE aggregation, sim data augmentation, video pre-training |
| **Cross-Embodiment Generalization** | 🟡 | Different robots have different DoF, action spaces, cameras | Multi-embodiment training paradigms from Octo/OpenVLA |
| **Real-Time Performance** | 🟡 | Perception → reasoning → action must complete within 100ms | Flow matching, model distillation, edge deployment |
| **Safety** | 🟡 | VLA hallucinations can be disastrous in the physical world | Constrained decoding, runtime safety layer, RLHF |
| **Evaluation Standards** | 🟡 | Lack of unified real-world benchmark | SIMPLER, LIBERO continuous iteration |
| **Long-Horizon Tasks** | 🟡 | Multi-step tasks prone to error accumulation | Hierarchical VLA (high-level planning + low-level execution) |

### 3.6 Ecosystem Toolchain

| Project | Positioning | Function |
|------|------|------|
| **Open X-Embodiment** | Data | Largest robot dataset — 1M+ trajectories, 22 robot types |
| **LIBERO** | Benchmark | Standard VLA benchmark — 130 tasks, 4 generalization dimensions |
| **robosuite / MuJoCo** | Simulation | Classic robot simulation environments |
| **Isaac Sim / Isaac Lab** | Simulation | NVIDIA high-fidelity GPU simulation + reinforcement learning |
| **HuggingFace LeRobot** | Training | Open-source VLA training framework — datasets, models, deployment integrated |
| **FluxVLA** (LimX) | Engineering | Industrial-grade VLA engineering platform — data collection → training → deployment full pipeline |
| **ManiSkill** | Simulation+Eval | High-fidelity dexterous manipulation simulation benchmark |

---

## 4. VLA Advanced Improvement Techniques

### 4.1 Reasoning-Augmented VLA: From "Reflex" to "Thinking"

The problem with traditional VLA is that "perception → action" is nearly reflexive — lacking an intermediate reasoning step. When tasks require multi-step planning, counting, or spatial relationship reasoning, they often fail.

#### Technical Route Comparison

| Method | Core Idea | Representative Work | Effect |
|------|---------|---------|------|
| **Embodied CoT** | Generate reasoning chains before actions, explicitly outputting "thought processes" | ECoT (2024), ReVLA (2025) | Multi-step task success +25% |
| **Inner Monologue** | Inject environmental feedback into the reasoning loop, enabling self-correction | Inner Monologue (2022), Code-as-Policies (2023) | Dynamic replanning |
| **ReAct-style VLA** | Alternate "reasoning → observation → action," step-by-step decision making | RT-2-Reason (2024), Helix (Figure, 2025) | Complex instruction following |

#### ECoT (Embodied Chain-of-Thought) in Detail

```
Traditional VLA:
    "Place the blue block to the right of the red block"
        → [action sequence]

ECoT VLA:
    "Place the blue block to the right of the red block"
        → Reasoning: red block at (x=0.3, y=0.1), blue block at (x=0.5, y=0.4)
        → Sub-task 1: grasp blue block
        → Sub-task 2: move to target position (x=0.4, y=0.1)
        → Sub-task 3: place
        → [action sequence]
```

**Key design:**
- **Interleaved reasoning + action tokens**: Model first outputs natural language reasoning, then action tokens
- **Multimodal CoT data synthesis**: Automatically annotate reasoning chains for robot data using GPT-4V
- **Reasoning compression**: Reasoning incurs no inference latency overhead — reasoning and action are parallel-encoded

> Core insight: **Let the VLA "say what it's thinking"** — reasoning chains not only improve success rates but also let humans understand why the model made a particular action.

#### Helix (Figure, 2025)

Figure's commercial VLA system for humanoid robots, using a **dual-model architecture**:

```
System 1 (high-frequency ~200Hz)    System 2 (low-frequency ~7Hz)
  "Cerebellum"                        "Cerebrum"
  │                                    │
  Fast reactive control                Semantic understanding + reasoning + planning
  (80M param ViT + MLP)               (7B param VLM)
  │                                    │
  └──────────┬─────────────────────────┘
             ↓
       Latent space fusion
             ↓
        Action output (full-body coordination)
```

**Key innovations:**
- High-low frequency decoupling: Dexterous manipulation needs 200Hz, large model reasoning only needs 7Hz
- Latent conditioning: System 2's semantic intent is injected into System 1 as a latent vector, not token-by-token
- **Full-body coordination**: Unified control of dual arms + dual hands + torso + head

---

### 4.2 Spatial & 3D VLA: From 2D to the World

#### Why 3D?

The fundamental limitation of 2D VLA: grasping an object 10cm above a table surface is a **3D spatial problem requiring depth information**, and 2D images lose the z-axis.

| Method | Input | Spatial Representation | Representative |
|------|------|---------|------|
| **2D+Depth** | RGB-D | 2.5D feature map (pixel + depth channel) | CLIPort, PerAct |
| **Point Cloud VLA** | 3D point cloud | PointNet++ / 3D Transformer | 3D-ViTAC, Act3D |
| **3D Scene Graph** | Multi-view RGB → NeRF/3DGS | Explicit 3D feature field | SpatialVLA, RoboPoint |
| **Multi-view Fusion** | 2–4 RGB cameras | Cross-view attention | RVT, OpenVLA (multi-view) |

#### SpatialVLA (2024–2025)

```
RGB multi-view images
      ↓
┌──────────────────────┐
│  NeRF / 3D Gaussian  │
│  Scene reconstruction │
│  + feature distillation│
└──────────┬───────────┘
           ↓
    3D feature field (explicit spatial coordinates)
           ↓
┌──────────────────────┐
│  Spatial Grounding   │
│  "Red block is 5cm   │
│   left of blue block" │
└──────────┬───────────┘
           ↓
      Action (6-DoF end-effector pose)
```

**Key capabilities:**
- **Spatial relation reasoning**: Understanding spatial prepositions like "left," "above," "between"
- **Metric manipulation**: Precisely executing quantitative instructions like "move 5cm," "rotate 30°"
- **Occlusion reasoning**: Even when targets are partially occluded, 3D representations infer complete poses

---

### 4.3 Action Generation Frontiers: Faster, More Accurate, More Robust

#### From DDPM to Consistency Models

| Method | Sampling Steps | Inference Latency | Action Quality |
|------|---------|---------|---------|
| DDPM (classic diffusion) | 100–1000 | ~1s | Highest |
| DDIM | 10–100 | ~100ms | High |
| Flow Matching (Pi0) | 5–10 | ~50ms | High |
| **Consistency Model** | **1–2** | **<10ms** | High (approaching DDPM) |
| Autoregressive (RT-2) | 1 (per token) | ~20ms | Medium |

#### Diffusion Forcing (2024–2025)

The pain point of traditional diffusion policies: they generate a **fixed-length future action sequence**, unable to flexibly handle tasks whose duration is uncertain.

**Diffusion Forcing** solution:
- Denoise only one action token per timestep, based on previously determined action history
- Supports **variable-length** action sequences — "stop when done"
- Combines diffusion quality + autoregressive flexibility

```
Diffusion Policy (traditional):    [a₁ a₂ ... aₙ]  ← denoise N steps at once
Diffusion Forcing:                   a₁ → a₂ → a₃ → ... ← stepwise denoising, terminable anytime
```

#### Hierarchical Action Generation

```
Hierarchical VLA Action Architecture
├── High-level: Task Planning
│    Output: sub-task sequence
│    "Open drawer → take object → close drawer"
│    Frequency: ~1Hz, LLM-based
│
├── Mid-level: Skill Execution
│    Output: end-effector trajectory keypoints
│    Frequency: ~10Hz, diffusion-policy-based
│
└── Low-level: Motion Control
     Output: joint angles / torques
     Frequency: ~100–1000Hz, classical controller or learned policy
```

> Current trend: **Not using one model for all frequencies, but letting each level use the most suitable technique**.

---

### 4.4 Efficient Adaptation: Small Data Leveraging Large Models

#### VLA Fine-Tuning Tech Stack

| Technique | Trainable Params | Typical Performance | Use Case |
|------|---------|---------|---------|
| **Full Fine-tuning** | 100% | Highest (overfitting risk) | Large-scale robot data |
| **LoRA** | 0.1%–1% | Near full fine-tuning | **Most common** — rapid new task adaptation |
| **QLoRA** | 0.1%–1% | Slightly below LoRA | Consumer GPU (<24GB VRAM) |
| **Prompt Tuning** | <0.01% | Acceptable for basic tasks | Ultra-low resource, multi-task switching |
| **Frozen + Action Head** | 1%–5% | Task-dependent | Maximal pre-trained knowledge retention |

#### OpenVLA LoRA Practice

```
OpenVLA-7B full parameters: 7B
LoRA fine-tuned parameters: ~8M (0.1%)
├── Vision encoder: frozen (well pre-trained)
├── LLM backbone: LoRA injected into attention layers
└── Action head: fully trained
```

**Key lessons:**
- LoRA rank = 32–64 suffices
- Inject into both attention query/value and MLP down-projection
- 50–100 high-quality demonstrations enable new task adaptation

#### Test-Time Adaptation

Adapting to environmental changes in real time at deployment:

| Method | Principle | Typical Application |
|------|------|---------|
| **Visual Domain Adaptation** | Online adjustment of vision encoder BN statistics | Lighting/background changes |
| **Action Residual Learning** | Learn "base policy + environment-dependent offset" | Different friction/payload |
| **In-Context Learning** | Inject successful examples as context into prompt | Rapid novel object adaptation |

---

### 4.5 Safety & Alignment: VLA Cannot Have "Hallucinations"

LLM hallucinations are text mistakes — VLA hallucinations are **physical disasters**. This has spawned specialized VLA safety techniques:

#### Safety Technology Panorama

| Level | Method | Mechanism |
|------|------|------|
| **Data Level** | Safety data filtering + adversarial example training | Inject safety constraints during training |
| **Inference Level** | Constrained Decoding | Force actions within safety bounds (joint limits, torque caps) |
| **Execution Level** | Runtime safety monitor | Independent process detecting abnormal actions and emergency-stopping |
| **Alignment Level** | RLHF / Constitutional VLA | Train "safe" behavioral policies using human preferences |

```
Constrained Decoding example:
    Model outputs joint angle [-180° ~ 180°]
        ↓ constraint layer
    Angle ∈ [0°, 90°]  ← physical limit mapping
    Torque ∈ [0, 5 N·m] ← safety torque cap
    Velocity ∈ [0, 2 rad/s] ← smooth motion constraint
```

#### Constitutional VLA (2025 exploratory direction)

Encoding "Three Laws of Robotics" as training constraints for VLA:
- Before each action output, automatically check for safety rule violations
- Use violation cases as negative examples for preference alignment during training
- Similar to Anthropic's Constitutional AI, but constraining the **action space**

---

### 4.6 Inference Acceleration: Large Models onto Robots

VLA inference latency is the core bottleneck for deployment. Typical performance targets:

| Control Level | Target Frequency | Target Latency | Current VLA Bottleneck |
|---------|---------|---------|-------------|
| High-level Planning | 1–5 Hz | 200–1000ms | ✅ VLM inference already sufficient |
| Skill Execution | 10–50 Hz | 20–100ms | ⚠️ Diffusion sampling steps limit |
| Motion Control | 100–1000 Hz | 1–10ms | ❌ Large models cannot reach |

#### Acceleration Technology Matrix

| Technique | Speedup | Accuracy Loss | Principle |
|------|---------|---------|------|
| **INT8/INT4 Quantization** | 2×–4× | <1% | Low-precision weights/activations |
| **Knowledge Distillation** | 2×–10× | 1%–3% | Large model teaches small model |
| **Speculative Decoding** | 1.5×–3× | 0% | Small model predicts + large model verifies |
| **TensorRT/ONNX Compilation** | 1.5×–3× | <0.5% | Computation graph optimization + operator fusion |
| **Flow Matching Step Reduction** | 5×–10× | 1%–2% | Fewer steps, equivalent sampling |
| **KV-Cache Reuse** | 1.2×–1.5× | 0% | Historical frame cache sharing |

#### Production Deployment: Edge VLA

```
Cloud (offline training)
    Large model pre-training + LoRA fine-tuning
        ↓ distillation
Edge Device (online inference)
    Student model (<1B)
    + INT8 quantization
    + TensorRT optimization
    → Inference latency ~15ms (Jetson AGX Orin)
```

> Typical pipeline: OpenVLA-7B teacher → **TinyVLA-300M** student → INT4 quantization → Deploy on Jetson Orin, inference latency drops from 500ms to 25ms.

---

### 4.7 Cutting-Edge Trend Overview

| Direction | Current Status | 2025–2026 Outlook |
|------|---------|---------------|
| **End-to-end full-body control** | Initial implementation by Pi0, GR00T | Unified humanoid running + jumping + manipulation |
| **Video generation → robot policy** | Early exploration (UniPi, SuSIE) | Video prediction as universal "world model" |
| **Language → reward function** | Practical systems exist (Eureka) | Natural language auto-design of training objectives |
| **Sim-to-Real 2.0** | Domain randomization dominant | 3D Gaussian splatting + real data closed loop |
| **Multi-robot collaborative VLA** | Academic exploration | 2+ robots sharing VLA policy |
| **Lifelong learning VLA** | Early exploration | Continuous learning from interaction post-deployment |

---

## 5. Relationship with Your GraspAnything Project

GraspAnything is essentially a **lightweight VLA system**:

```
Vision:  Grounding DINO + SAM → Open-vocabulary object detection and segmentation
Language: Natural language describing the target object
Action:  robosuite grasp execution
```

Its position in the VLA landscape:

```
Full VLA (RT-2 / Pi0):     End-to-end perception → reasoning → action
        ↑
GraspAnything:             Modular VLA (detection → segmentation → grasp planning → execution)
        ↑
Traditional Methods:       Pre-programmed skills + fixed object set
```

> It is the best starting point for understanding the full VLA stack — beginning with a simple system and gradually understanding the design motivations behind large models like RT-2 and Pi0: why end-to-end? Why massive data? Why does diffusion outperform regression?

---

*Related notes: [Embodied Intelligence in the Era of Large Models](./大模型时代的具身智能.md), [Multimodal Large Model Pre-training Topic](./多模态大模型预训练—微调—后训练专题研究.md), [What is Transformer](./What_is_Transformer.md)*
