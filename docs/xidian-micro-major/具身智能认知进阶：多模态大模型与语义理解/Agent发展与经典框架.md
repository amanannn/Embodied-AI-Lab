# Agent 发展与经典框架

> 西安电子科技大学 · 具身智能微专业
> 课程笔记

---

## 一、什么是 AI Agent

AI Agent 是能够**自主感知、规划、使用工具、执行、记忆、反思**的智能体。

| 能力 | 说明 | 实现方式 |
|------|------|---------|
| **感知** | 理解环境状态和用户意图 | 多模态模型读图/读文档/看视频 |
| **规划** | 分解任务为可执行步骤 | CoT / ReAct / Tree-of-Thought |
| **工具使用** | 调用外部 API/函数/代码 | Function Calling / MCP 协议 |
| **记忆** | 短期（对话上下文）+ 长期（向量数据库） | 上下文窗口 + RAG |
| **执行** | 实际完成操作 | 代码执行 / 浏览器操作 / API 调用 |
| **反思** | 评估输出并自我修正 | Self-Reflection / Self-Critique |

> Agent ≠ Chatbot。Chatbot 一问一答；Agent 是自主闭环——设定目标、拆解、执行、验证、修正。

### 1.1 Agent 的认知架构

```
        ┌─────────────┐
        │   用户输入    │
        └──────┬──────┘
               ↓
┌──────────────────────────────┐
│         规划 (Plan)          │
│  "我需要做 A→B→C 三步"       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       执行 (Act)             │──→ 调用工具 / API / 代码
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       观察 (Observe)         │←── 工具返回结果
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       反思 (Reflect)         │
│  "结果对吗？需要调整吗？"     │
└──────────────┬───────────────┘
               ↓
       ┌───────┴───────┐
       │ 完成 → 输出     │  不完成 → 回到 Plan
       └───────────────┘
```

---

## 二、Agent 发展四阶段

| 阶段 | 时期 | 关键词 | 代表 | 里程碑 |
|------|------|--------|------|--------|
| **萌芽期** | 2018-2022 | 概念验证 | ReAct, SayCan, Toolformer | 证明 LLM 能调用工具 |
| **爆炸期** | 2023 H1 | 开源狂热 | AutoGPT, BabyAGI, LangChain | GitHub 上 Star 过万 |
| **成熟期** | 2023 H2-2024 | 平台化 | GPT-4 Functions, Claude Computer Use | Agent 成为 LLM 标配 |
| **隐形化** | 2025+ | Agent 即应用 | MCP 协议, OpenAI Agents SDK, 各厂 Agent 平台 | Agent 退到后台——用户只看到功能 |

---

## 三、经典框架详解

### 3.1 ReAct——推理与行动交织

ReAct（Yao et al., 2022）的核心洞察：**思考（Thought）和行动（Action）交替进行，相互增强**。

```
用户: "北京和上海的时差是多少？"

Thought: 时差取决于经度，北京东八区，上海也是东八区
Action: verify_timezone("北京")
Observation: 北京 = UTC+8

Thought: 北京确认了，查上海
Action: verify_timezone("上海")
Observation: 上海 = UTC+8

Thought: 两个城市都在东八区
Action: FINISH("北京和上海没有时差，都是 UTC+8")
```

#### ReAct 为什么有效？

| 机制 | 说明 |
|------|------|
| **推理指导行动** | Thought 决定了下一步该做什么（而非随机试） |
| **行动反馈推理** | Observation 提供事实依据（减少幻觉） |
| **可解释性** | 每一步做了什么、为什么——完全透明 |
| **错误恢复** | 看到错误 Observation 后可以调整方向 |

### 3.2 更多推理范式

| 范式 | 流程 | 适用场景 |
|------|------|---------|
| **CoT** | 直接输出推理链，不调用工具 | 数学、逻辑 |
| **CoT-SC** | 多次 CoT → 投票选最佳 | 需要高准确率 |
| **Tree-of-Thought** | 每步生成多个候选→ BFS/DFS 搜索 | 复杂规划 |
| **ReAct** | Thought→Action→Obs 循环 | 需要外部信息 |

### 3.3 Function Calling——工业标准

不依赖 Prompt Engineering，而是**在模型架构层面定义可用工具**。

```python
# OpenAI Function Calling 定义
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
}]
# 模型输出: tool_calls=[{name:"get_weather", arguments:{city:"北京",unit:"celsius"}}]
# 而不是自然语言 "我需要查一下北京的天气"
```

| 优势 | 说明 |
|------|------|
| **结构化** | JSON Schema 定义参数——强类型，不会传错参数 |
| **并行调用** | 无关的 tool call 可以同时执行 |
| **流式** | tool call 的 arguments 可以流式输出 |
| **多模态** | GPT-4V 可以 "看到" 图像后调用工具 |

### 3.4 MCP 协议——Agent 的"USB 接口"

Model Context Protocol（Anthropic, 2024）：统一 Agent 与外部工具/数据的交互标准。

| MCP 角色 | 说明 | 类比 |
|---------|------|------|
| **MCP Host** | Claude Desktop / ChatGPT App | 电脑 |
| **MCP Client** | 模型端的协议实现 | USB 控制器 |
| **MCP Server** | 工具/数据提供方（文件系统、数据库、API） | USB 设备 |

> MCP 解决了 Agent 最大的碎片化问题——过去每个 Agent 框架都有自己的工具接入方式，现在一个协议统一所有。

---

## 四、Multi-Agent 协作模式

### 4.1 四种协作拓扑

| 拓扑 | 结构 | 适用场景 |
|------|------|---------|
| **流水线** | A→B→C 串行 | 固定流程（文档→翻译→审校） |
| **星型** | 一个 Orchestrator 调度 N 个 Worker | 任务分解（调研→分析→写作→审校） |
| **辩论型** | N 个 Agent 互相辩论 → 裁判裁决 | 需多角度思考（方案评审） |
| **层级型** | 树状结构——上级拆解、下级执行 | 复杂项目（软件开发团队） |

### 4.2 代表框架对比

| 框架 | 特色 | 学习曲线 | 生产就绪 |
|------|------|---------|---------|
| **AutoGen** (Microsoft) | 多 Agent 对话 + 代码沙箱执行 | 中 | ⭐⭐⭐ |
| **CrewAI** | 角色驱动——定义"研究员""分析师""作家" | 低 | ⭐⭐ |
| **LangGraph** | 有状态图——每个节点是 Agent，边是交互 | 高 | ⭐⭐⭐ |
| **OpenAI Swarm** | 极简——handoff 切换 Agent | 低 | ⭐（实验性） |
| **OpenAI Agents SDK** | 生产级——Guardrails + Tracing + Handoffs | 中 | ⭐⭐⭐ |

---

## 五、Agent 工程框架

| 框架 | 核心理念 | 适合谁 |
|------|---------|--------|
| **LangChain** | Prompt→LLM→Tool 链式调用 | 想快速搭原型的开发者 |
| **LangGraph** | 有向图限状态机——可控、可调试 | 复杂多步 Agent |
| **LlamaIndex** | 以数据索引为核心 | 知识密集型 Agent（RAG） |
| **Semantic Kernel** | 微软出品——企业级插件架构 | .NET/企业开发者 |
| **CrewAI** | 角色扮演——"你是研究员" | 非技术用户也能理解 |
| **Dify / Coze** | 拖拽搭建 + 模板市场 | 完全不会写代码的人 |

---

## 六、Agent 评估

| Benchmark | 测什么 | 代表任务 |
|-----------|--------|---------|
| **WebArena** | 网页操作 Agent | 购物、订票、发邮件 |
| **SWE-bench** | 代码 Agent | 修 GitHub Issue |
| **AgentBench** | 综合 Agent 能力 | 8 种环境——OS/DB/Web/游戏 |
| **GAIA** | 通用 AI 助手 | 多步推理 + 多模态输入 |
| **τ-bench** | 工具使用 | 真实 API 调用 |

---

## 七、Agent 核心挑战与解决方案

| 挑战 | 表现 | 当前最佳实践 |
|------|------|------------|
| **幻觉工具** | 调用不存在的 API | Function Calling JSON Schema 约束 |
| **发散** | 10 步后忘记原始目标 | 每 N 步回查目标；Orchestrator 检查 |
| **速度** | 串行 10+ 步 = 30 秒+ | 并行 tool call + 流式输出 |
| **成本** | 每步都调 GPT-4 → 一次任务 $1+ | 小模型做简单步骤，复杂步骤才用大模型 |
| **安全** | 删除文件、发送错误邮件 | 敏感操作需人类确认；权限分级 |

---

*相关笔记：[Agentic RL](./Agentic-RL.md)、[智能体经典框架实现](./智能体经典框架实现.md)、[大模型时代的具身智能](./大模型时代的具身智能.md)*
