# Agentic RL — Reinforcement Learning for Agents

> Xidian University · Embodied AI Micro-Major
> Course Notes

---

## I. What is Agentic RL

Agentic RL applies the **reinforcement learning paradigm to the training of large language models / multimodal models**, endowing the model with autonomous decision-making, tool use, multi-step reasoning, and other Agent capabilities.

| Concept | Description |
|---------|-------------|
| **Agent** | An autonomous entity that can perceive the environment, make plans, use tools, and execute actions |
| **RL** | Learning optimal policies through trial and error and reward signals |
| **Agentic RL** | Using RL to train models to become better Agents |

> Traditional RL trains Atari games, AlphaGo, and robot control; Agentic RL trains large models to learn tool invocation, code generation, multi-step reasoning, self-reflection, and self-correction.

### 1.1 Fundamental Paradigm Shift

| Paradigm | Training Objective | Weakness |
|----------|--------------------|----------|
| **Next-token prediction** | Maximize $\log P(y \mid x)$ | Can only continue text, cannot reason |
| **SFT (Supervised Fine-Tuning)** | Imitate human-annotated correct answers | Ceiling is the annotator's level |
| **Agentic RL** | Maximize long-term cumulative reward | Can surpass human annotations |

---

## II. Why Agentic RL is Needed

### 2.1 The Ceiling of SFT

| Problem | Specific Manifestation | How Agentic RL Solves It |
|---------|----------------------|--------------------------|
| **Distribution shift** | Training "correct trajectories" cannot cover all real scenarios | RL can explore strategies beyond training data |
| **Sparse rewards** | In 10-step reasoning, only the final answer is right or wrong | RL naturally handles delayed rewards |
| **Imitation ceiling** | SFT can only reach the annotator's level, cannot surpass it | RL can surpass humans through self-play |
| **Error accumulation** | One wrong step leads to all subsequent steps being wrong; SFT cannot "self-correct" | RL can learn recovery strategies from error trajectories |

### 2.2 Core Advantages of RL

| RL Capability | Significance for Agents | Example |
|---------------|------------------------|---------|
| **Trial-and-error learning** | Extract signals from failure trajectories | Grasp failed → move down 2cm → success |
| **Long-term planning** | Optimize cumulative reward, not per-step accuracy | Sacrifice one piece to win the whole game |
| **Exploration** | Discover strategies not present in training data | AlphaGo's move 37 |
| **Credit assignment** | Know which step in a long sequence was right/wrong | The search query in step 3 was critical |

---

## III. Core Technical Approaches

### 3.1 RLHF — Reinforcement Learning from Human Feedback

#### Three-Step Process in Detail

```
Step 1: SFT (Supervised Fine-Tuning)
  Input: Pretrained model + human high-quality demonstrations (prompt, ideal_answer)
  Output: SFT model that follows instruction format
  Cost: $$$$ — requires large amounts of manual demonstrations

Step 2: Train Reward Model (RM)
  Input: Same prompt → SFT model generates K responses → Human ranking (A > B > C > D)
  Output: A reward model r_θ(x, y) that predicts "which response humans would prefer"
  Cost: $$ — ranking is 10x faster than writing demonstrations

Step 3: PPO Reinforcement Learning
  Input: SFT model + Reward Model
  Process: Model generates → RM scores → PPO updates parameters
  Output: Aligned final model
  Cost: $ — no human involvement needed (only GPU compute)
```

#### PPO Objective Function

$$L_{PPO} = \mathbb{E}\left[r_\theta(x, y) - \beta \cdot D_{KL}\left(\pi_\theta(y|x) \| \pi_{ref}(y|x)\right)\right]$$

| Term | Meaning | Role |
|------|---------|------|
| $r_\theta(x, y)$ | RM score for response $y$ | Encourages "good" responses |
| $\beta \cdot D_{KL}$ | KL divergence penalty | Prevents the model from drifting too far (avoids forgetting what SFT learned) |
| $\beta$ | KL penalty coefficient | 0.01~0.1 — larger values make the model more conservative |

```python
# Simplified PPO training
for batch in dataloader:
    # 1. Current policy generates responses
    responses = policy_model.generate(prompts)

    # 2. RM scores + Reference Model computes KL
    rewards = reward_model(prompts, responses)
    with torch.no_grad():
        ref_logps = reference_model(prompts, responses)
    kl_penalty = (policy_logps - ref_logps).sum(-1)

    # 3. PPO loss
    score = rewards - beta * kl_penalty
    advantages = (score - score.mean()) / (score.std() + 1e-8)
    ratio = torch.exp(policy_logps - old_policy_logps)
    loss = -torch.min(
        ratio * advantages,                      # Normal update
        torch.clamp(ratio, 1-eps, 1+eps) * advantages  # Clip to prevent large steps
    ).mean()

    loss.backward(); optimizer.step()
```

### 3.2 GRPO — Group Relative Policy Optimization

The innovative algorithm used by DeepSeek-R1, completely eliminating the reward model:

| Comparison | PPO (RLHF) | GRPO |
|------------|------------|------|
| Needs reward model? | Yes — requires separate training and maintenance | No — rules or group comparison |
| Needs Reference Model? | Yes — needed for KL computation | No — group normalization replaces it |
| Memory requirement | Loads 4 models simultaneously | Only 1 model |
| Training stability | PPO is notoriously hard to tune | More stable — no adversarial training |

#### GRPO Algorithm Flow

```python
# GRPO complete pseudocode
for prompt in batch:
    # 1. Generate K candidate responses
    responses = [model.generate(prompt) for _ in range(K)]

    # 2. Rule-based scoring (math → check answer, code → run test cases)
    rewards = [rule_based_score(prompt, r) for r in responses]

    # 3. Group normalization — good answers get relatively higher scores within the group
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 4. Compute GRPO loss
    for i, (response, advantage) in enumerate(zip(responses, advantages)):
        log_probs = model.log_prob(response)
        ratio = torch.exp(log_probs - old_log_probs[i])
        loss += -torch.min(
            ratio * advantage,
            torch.clamp(ratio, 0.8, 1.2) * advantage
        )

    # 5. Optional: KL regularization (added directly to loss, no reference model needed)
    loss += kl_coef * estimate_kl(original_model, model, prompt)
```

> DeepSeek-R1's key finding: For verifiable tasks (math, code), rule-based rewards + group comparison are sufficient — no expensive human annotation or reward model training is needed.

### 3.3 Process Reward vs Outcome Reward

| Reward Type | Granularity | When Available | Representative |
|-------------|-------------|----------------|----------------|
| **ORM (Outcome Reward Model)** | Correctness of the entire response | Math/code with clear answers | DeepSeek-R1, GPT-4 Math |
| **PRM (Process Reward Model)** | Correctness of each reasoning step | Requires human-annotated intermediate steps | OpenAI o1, Let's Verify Step by Step |

```text
Question: "Xiao Ming has 5 apples, eats 2, then buys 3 more. How many are left?"

ORM score: Final answer = 6 → ✅ reward = +1
PRM score:
  Step 1: 5 - 2 = 3 → ✅ reward = +0.3
  Step 2: 3 + 3 = 6 → ✅ reward = +0.3
  Format compliance: → ✅ reward = +0.1
  Final answer correct: → ✅ reward = +0.3
  Total reward = +1.0
```

### 3.4 More Frontier Methods

| Method | Core Idea | Applicable Scenarios | Representative Work |
|--------|-----------|---------------------|---------------------|
| **STaR** | Model reasons → filters correct reasoning → trains itself on correct reasoning | Reasoning tasks | Zelikman et al., 2022 |
| **ReST** | Generate → filter → SFT → regenerate → RL iterative loop | General | DeepMind, 2023 |
| **SPIN** | Current version vs. previous version self-play | General | Chen et al., 2024 |
| **RLVR** | Relies solely on verifiable rewards, no human annotation | Math/code | DeepSeek-R1 |
| **Iterative DPO** | Each round uses the updated model to generate new preference pairs | General | Llama 3 |
| **Online RLHF** | Real-time human feedback collection during RL training | Pursuing highest quality | ChatGPT production environment |

---

## IV. Reward Signal Design

| Reward Type | Provider | Applicable Scenario | Latency | Noise |
|-------------|----------|---------------------|---------|-------|
| **Human preference** | Annotator ranking | Open-ended Q&A, creative writing | Low | Medium |
| **Rule-based verification** | Automated program | Math, code, logic | Low | Low |
| **Environment feedback** | Simulated/real environment | Embodied AI, games | High | Medium |
| **Tool invocation result** | API/function return value | Agent tasks | Medium | Low |
| **Self-evaluation** | The model itself | General reasoning | Low | High |
| **LLM-as-Judge** | Stronger model scoring | General | Low | Medium |

> Reward design is the hardest part of Agentic RL — the reward function determines what the model will learn. Poorly designed rewards lead to reward hacking.

---

## V. Agentic RL and Embodied Intelligence

| Scenario | RL Approach | Current Status |
|----------|-------------|----------------|
| **VLA policy optimization** | Grasp success +1, collision -0.1, drop -1 | Works well in simulation; Sim2Real gap remains |
| **Task planning** | +0.2 per sub-task completed, +1 for full completion | Under exploration |
| **Exploration behavior** | Curiosity reward — bonus for high-surprise states | Research frontier |
| **Sim-to-Real Transfer** | RL in simulation → Domain Randomization → Transfer | Partially successful |

---

*Related notes: [Agent Development and Classic Frameworks](./Agent%E5%8F%91%E5%B1%95%E4%B8%8E%E7%BB%8F%E5%85%B8%E6%A1%86%E6%9E%B6.en.md), [Post-Training and Alignment for Multimodal Large Models](./%E5%A4%9A%E6%A8%A1%E6%80%81%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%90%8E%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%AF%B9%E9%BD%90.en.md), [What is VLA](../%E7%AB%AF%E5%88%B0%E7%AB%AFVLA%E5%AE%9E%E6%88%98/What_is_VLA.en.md)*
