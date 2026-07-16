# What are Foundation Models

> Xidian University · Embodied AI Micro-Major
> Course Notes

---

## 1. Definition and Core Characteristics

Foundation Model is a concept proposed by Stanford HAI in 2021: **a general-purpose model pre-trained on large-scale, diverse data that can be adapted to a wide range of downstream tasks**.

| Feature | Description |
|------|------|
| **Large-Scale Pre-training** | Self-supervised / semi-supervised learning on massive unlabeled data |
| **Emergent Abilities** | Capabilities that small models lack, appearing when scale crosses a threshold |
| **Multi-Task Transfer** | The same model can be adapted to classification, generation, reasoning, translation, etc. |
| **Multimodal Fusion** | Unified modeling of text, image, speech, and video |

> The paradigm shift of foundation models: from "training a specialized model for each task" to "training one general-purpose backbone and adapting it to all tasks."

---

## 2. Evolution of Foundation Models

| Period | Stage | Representative | Key Change |
|------|------|------|---------|
| 2018-2019 | Pre-trained Language Models | BERT, GPT-1/2 | Demonstrated the effectiveness of the "pre-training + fine-tuning" paradigm |
| 2020-2021 | Large Language Models | GPT-3 (175B) | In-context learning replaced fine-tuning; emergent abilities observed |
| 2021-2022 | Multimodal Foundation Models | CLIP, DALL-E, Stable Diffusion | Vision and language entered the same semantic space |
| 2023 | Unified Multimodal | GPT-4V, Gemini | Visual understanding + language reasoning + code generation integrated |
| 2024+ | Embodied Foundation Models | RT-2, Octo, Pi0 | Foundation models move from screens to the physical world |

---

## 3. Three Levels of Foundation Models

### 3.1 Language Foundation Models

| Model | Parameter Scale | Core Capability | Paradigm |
|------|---------|---------|------|
| BERT | 110M / 340M | Bidirectional context understanding | Pre-training + Fine-tuning |
| GPT-3 | 175B | In-context learning, natural language generation | Prompt + Few-shot |
| GPT-4 | ~1.8T | Multimodal understanding, complex reasoning, code generation | Instruction following + Agent |
| LLaMA Series | 7B-405B | Open ecosystem, community-driven innovation | Open weights |

### 3.2 Vision Foundation Models

| Model | Core Capability | Key Innovation |
|------|---------|---------|
| ViT | Image classification backbone | Patchification — images can also be processed with Transformer |
| MAE | Self-supervised visual pre-training | 75% extremely high mask ratio + asymmetric encoder-decoder |
| SAM | Universal image segmentation | Promptable — points/boxes/text to segment any object |
| DINO v2 | Strong visual representations | Self-supervised learning produces features rivaling supervised learning |

### 3.3 Multimodal Foundation Models

| Model | Capability | Architecture |
|------|------|------|
| CLIP | Image-text matching + Zero-shot classification | Dual tower (ViT + Transformer) → Contrastive learning |
| BLIP-2 | Image-text understanding + generation | Q-Former bridging vision encoder and LLM |
| LLaVA | Multimodal dialogue | ViT + LLM + projection layer |
| GPT-4V | Visual reasoning + dialogue | End-to-end multimodal Transformer |

---

## 4. Why Are Foundation Models Important?

| Significance | Description |
|------|------|
| **Unified Paradigm** | NLP + CV + Speech → same set of architectures and methodologies |
| **Lowered Barrier** | One person + one pre-trained model = output of an entire team before |
| **Accelerated Research** | No longer training from scratch; fine-tune or prompt on foundation models |
| **Catalyzed Applications** | ChatGPT, Midjourney, Copilot — all built on this |

> Foundation models are not "just another new model," but a fundamental shift in AI R&D — from "building wheels" to "using wheels."

---

## 5. Key Technologies of Foundation Models

| Technology | Role | Representative |
|------|------|------|
| **Transformer** | Unified architecture of foundation models | Self-Attention + FFN |
| **Self-Supervised Learning** | Escape annotation dependency | MLM, Contrastive Learning, MAE |
| **Scaling Law** | Guides the scaling of model and data | Kaplan et al., 2020 |
| **Instruction Fine-Tuning** | Enables models to follow human intent | InstructGPT, FLAN |
| **RLHF / DPO** | Aligns with human values | ChatGPT, Claude |
| **MoE** | Sparse activation increases capacity without increasing inference cost | Mixtral, DeepSeek-V2 |

---

## 6. Foundation Models and Embodied Intelligence

Foundation models are moving from the pure software world to the physical world:

| Direction | Representative Work | Role of Foundation Models |
|------|---------|--------------------------|
| Visual Perception | SAM, Grounding DINO | Universal object localization and segmentation |
| Task Understanding | GPT-4V, LLaVA | Natural language instruction → Task planning |
| Action Generation | RT-2, Octo, Pi0 | Vision + Language → Robot actions |
| General Policy | GATO, RT-X | Multi-task, multi-embodiment unified model |

> The ultimate form of embodied intelligence = Vision Foundation Model (understand the world) + Language Foundation Model (understand tasks) + Action Foundation Model (execute actions).

---

*Related notes: [What is a Neural Network](./What_is_neural_network.md), [What is Transformer](../具身智能认知进阶：多模态大模型与语义理解/What_is_Transformer.md), [Multimodal Large Model Pre-training Topic](../具身智能认知进阶：多模态大模型与语义理解/多模态大模型预训练—微调—后训练专题研究.md)*
