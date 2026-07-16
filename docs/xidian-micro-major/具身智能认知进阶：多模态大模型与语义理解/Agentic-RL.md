# Agentic RL——智能体的强化学习

> 西安电子科技大学 · 具身智能微专业
> 课程笔记

---

## 一、什么是 Agentic RL

Agentic RL 是将**强化学习范式应用于大语言模型/多模态模型的训练**，使模型具备自主决策、工具使用、多步推理等 Agent 能力。

| 概念 | 说明 |
|------|------|
| **Agent** | 能感知环境、制定计划、使用工具、执行行动的自主智能体 |
| **RL** | 通过试错和奖励信号学习最优策略 |
| **Agentic RL** | 用 RL 训练模型成为更好的 Agent |

> 传统 RL 训练 Atari 游戏、AlphaGo、机器人控制；Agentic RL 训练大模型学会工具调用、代码生成、多步推理、自我反思和自我纠正。

### 1.1 训练范式的根本变化

| 范式 | 训练目标 | 弱点 |
|------|---------|------|
| **Next-token prediction** | 最大化 $\log P(y \mid x)$ | 只会续写，不会推理 |
| **SFT（监督微调）** | 模仿人类标注的正确回答 | 天花板是标注者水平 |
| **Agentic RL** | 最大化长期累积奖励 | 可以超越人类标注 |

---

## 二、为什么需要 Agentic RL

### 2.1 SFT 的天花板

| 问题 | 具体表现 | Agentic RL 如何解决 |
|------|---------|-------------------|
| **分布偏移** | 训练中的"正确轨迹"无法覆盖所有真实场景 | RL 可以探索训练数据之外的策略 |
| **稀疏奖励** | 10 步推理中只有最终答案有对错 | RL 天然处理延迟奖励 |
| **模仿上限** | SFT 只能达到标注者水平，无法超越 | RL 可以通过自我对弈超越人类 |
| **错误累积** | Agent 一步错→步步错，SFT 不会"自我纠正" | RL 可以从错误轨迹中学到恢复策略 |

### 2.2 RL 的核心优势

| RL 能力 | 对 Agent 的意义 | 实例 |
|---------|---------------|------|
| **试错学习** | 从失败轨迹中提取信号 | 抓取失败→下移 2cm→成功 |
| **长期规划** | 优化累积奖励，不是单步正确率 | 牺牲一个棋子换整盘胜利 |
| **探索** | 发现训练数据中没有的策略 | AlphaGo 的第 37 手 |
| **信用分配** | 知道长序列中哪一步做对了/错了 | 第 3 步的搜索查询是关键 |

---

## 三、核心技术路线

### 3.1 RLHF——从人类反馈中学习

#### 三步流程详解

```
Step 1: SFT（监督微调）
  输入: 预训练模型 + 人类高质量示范（prompt, ideal_answer）
  输出: 会遵循指令格式的 SFT 模型
  成本: $$$$——需要大量人工写示范

Step 2: 训练奖励模型（RM）
  输入: 同一 prompt → SFT 模型生成 K 个回答 → 人类排序（A > B > C > D）
  输出: 一个能预测"人类更喜欢哪个"的奖励模型 r_θ(x, y)
  成本: $$——排序比写示范快 10 倍

Step 3: PPO 强化学习
  输入: SFT 模型 + 奖励模型
  过程: 模型生成→RM 打分→PPO 更新参数
  输出: 对齐后的最终模型
  成本: $——不需要人类参与（仅 GPU 算力）
```

#### PPO 目标函数

$$L_{PPO} = \mathbb{E}\left[r_\theta(x, y) - \beta \cdot D_{KL}\left(\pi_\theta(y|x) \| \pi_{ref}(y|x)\right)\right]$$

| 项 | 含义 | 作用 |
|----|------|------|
| $r_\theta(x, y)$ | RM 对回答 $y$ 的评分 | 鼓励回答"好" |
| $\beta \cdot D_{KL}$ | KL 散度惩罚 | 防止模型偏离太远（避免忘记 SFT 学到的东西） |
| $\beta$ | KL 惩罚系数 | 0.01~0.1——越大模型越保守 |

```python
# PPO 训练简化版
for batch in dataloader:
    # 1. 当前 policy 生成回答
    responses = policy_model.generate(prompts)

    # 2. RM 打分 + Reference Model 计算 KL
    rewards = reward_model(prompts, responses)
    with torch.no_grad():
        ref_logps = reference_model(prompts, responses)
    kl_penalty = (policy_logps - ref_logps).sum(-1)

    # 3. PPO 损失
    score = rewards - beta * kl_penalty
    advantages = (score - score.mean()) / (score.std() + 1e-8)
    ratio = torch.exp(policy_logps - old_policy_logps)
    loss = -torch.min(
        ratio * advantages,                      # 正常更新
        torch.clamp(ratio, 1-eps, 1+eps) * advantages  # clip 防止太大步
    ).mean()

    loss.backward(); optimizer.step()
```

### 3.2 GRPO——群组相对策略优化

DeepSeek-R1 使用的革新算法，彻底抛弃奖励模型：

| 对比 | PPO（RLHF） | GRPO |
|------|------------|------|
| 需要奖励模型？ | 是——需单独训练维护 | 否——规则或群组对比 |
| 需要 Reference Model？ | 是——计算 KL 需要 | 否——群组归一化替代 |
| 显存需求 | 同时加载 4 个模型 | 仅 1 个模型 |
| 训练稳定性 | PPO 出名难调 | 更稳定——无对抗训练 |

#### GRPO 算法流程

```python
# GRPO 完整伪代码
for prompt in batch:
    # 1. 生成 K 个候选回答
    responses = [model.generate(prompt) for _ in range(K)]

    # 2. 基于规则评分（数学→核对答案，代码→跑测试用例）
    rewards = [rule_based_score(prompt, r) for r in responses]

    # 3. 群组归一化——好答案在组内相对得分高
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 4. 计算 GRPO loss
    for i, (response, advantage) in enumerate(zip(responses, advantages)):
        log_probs = model.log_prob(response)
        ratio = torch.exp(log_probs - old_log_probs[i])
        loss += -torch.min(
            ratio * advantage,
            torch.clamp(ratio, 0.8, 1.2) * advantage
        )

    # 5. 可选：KL 正则化（直接加在 loss 上，不需要 reference model）
    loss += kl_coef * estimate_kl(original_model, model, prompt)
```

> DeepSeek-R1 的核心发现：对于可验证任务（数学、代码），规则奖励 + 群组对比就足够了——不需要昂贵的人类标注和奖励模型训练。

### 3.3 过程奖励 vs 结果奖励

| 奖励类型 | 粒度 | 何时可用 | 代表 |
|---------|------|---------|------|
| **ORM（Outcome Reward Model）** | 整个回答的最终对错 | 数学/代码有明确答案 | DeepSeek-R1, GPT-4 数学 |
| **PRM（Process Reward Model）** | 每一步推理是否正确 | 需要人工标注中间步骤 | OpenAI o1, Let's Verify Step by Step |

```text
问题: "小明有 5 个苹果，吃了 2 个，又买了 3 个，还剩几个？"

ORM 评分: 最终答案 = 6 → ✅ reward = +1
PRM 评分:
  Step 1: 5 - 2 = 3 → ✅ reward = +0.3
  Step 2: 3 + 3 = 6 → ✅ reward = +0.3
  格式规范: → ✅ reward = +0.1
  最终答案正确: → ✅ reward = +0.3
  总 reward = +1.0
```

### 3.4 更多前沿方法

| 方法 | 核心思路 | 适用场景 | 代表工作 |
|------|---------|---------|---------|
| **STaR** | 模型自己推理→筛出正确推理→用正确推理训练自己 | 推理任务 | Zelikman et al., 2022 |
| **ReST** | 生成→过滤→SFT→再生成→RL 迭代循环 | 通用 | DeepMind, 2023 |
| **SPIN** | 当前版本 vs 上一版本自我对弈 | 通用 | Chen et al., 2024 |
| **RLVR** | 只靠可验证奖励，不做人类标注 | 数学/代码 | DeepSeek-R1 |
| **Iterative DPO** | 每轮用更新后的模型生成新的偏好对 | 通用 | Llama 3 |
| **Online RLHF** | 在 RL 训练过程中实时采集人类反馈 | 追求最高质量 | ChatGPT 生产环境 |

---

## 四、奖励信号设计

| 奖励类型 | 谁提供 | 适用场景 | 延迟 | 噪声 |
|---------|--------|---------|------|------|
| **人类偏好** | 标注员排序 | 开放问答、创意写作 | 低 | 中 |
| **规则验证** | 程序自动 | 数学、代码、逻辑 | 低 | 低 |
| **环境反馈** | 仿真/真实环境 | 具身智能、游戏 | 高 | 中 |
| **工具调用结果** | API/函数返回值 | Agent 任务 | 中 | 低 |
| **自我评估** | 模型自身 | 通用推理 | 低 | 高 |
| **LLM-as-Judge** | 更强的模型评分 | 通用 | 低 | 中 |

> 奖励设计是 Agentic RL 中最难的部分——奖励函数决定了模型会学到什么。错误设计的奖励会导致 reward hacking。

---

## 五、Agentic RL 与具身智能

| 场景 | RL 方案 | 当前状态 |
|------|--------|---------|
| **VLA 策略优化** | 抓取成功 +1，碰撞 -0.1，掉落 -1 | 仿真中效果好，Sim2Real 仍有 gap |
| **任务规划** | 每完成一个子任务 +0.2，全部完成 +1 | 正在探索 |
| **探索行为** | 好奇心奖励——对 surprise 高的状态加分 | 研究前沿 |
| **Sim-to-Real Transfer** | 仿真中 RL→Domain Randomization→迁移 | 部分成功 |

---

*相关笔记：[Agent发展与经典框架](./Agent发展与经典框架.md)、[多模态大模型后训练与对齐](./多模态大模型后训练与对齐.md)、[什么是VLA](../端到端VLA实战/What_is_VLA.md)*
