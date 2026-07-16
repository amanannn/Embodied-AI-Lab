# Agent Development and Classic Frameworks

> Xidian University · Embodied AI Micro-Major
> Course Notes

---

## I. What is an AI Agent

An AI Agent is an intelligent entity capable of **autonomous perception, planning, tool use, execution, memory, and reflection**.

| Capability | Description | Implementation |
|------------|-------------|----------------|
| **Perception** | Understanding environmental state and user intent | Multimodal models read images/documents/videos |
| **Planning** | Decomposing tasks into executable steps | CoT / ReAct / Tree-of-Thought |
| **Tool Use** | Calling external APIs/functions/code | Function Calling / MCP Protocol |
| **Memory** | Short-term (conversation context) + Long-term (vector database) | Context window + RAG |
| **Execution** | Actually completing operations | Code execution / Browser operations / API calls |
| **Reflection** | Evaluating output and self-correcting | Self-Reflection / Self-Critique |

> Agent ≠ Chatbot. Chatbot is question-and-answer; Agent is an autonomous closed loop — set goals, decompose, execute, verify, correct.

### 1.1 Agent Cognitive Architecture

```
           ┌─────────────┐
           │  User Input  │
           └──────┬──────┘
                  ↓
┌──────────────────────────────┐
│         Plan                 │
│  "I need to do A→B→C in 3 steps"
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│         Act                  │──→ Invoke tool / API / code
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       Observe                │←── Tool returns result
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       Reflect                │
│  "Is the result correct? Any adjustments needed?"
└──────────────┬───────────────┘
               ↓
       ┌───────┴───────┐
       │ Done → Output  │   Not done → Back to Plan
       └───────────────┘
```

---

## II. Four Stages of Agent Development

| Stage | Period | Keywords | Representative | Milestone |
|-------|--------|----------|----------------|-----------|
| **Germination** | 2018-2022 | Concept validation | ReAct, SayCan, Toolformer | Proved LLMs can call tools |
| **Explosion** | 2023 H1 | Open-source frenzy | AutoGPT, BabyAGI, LangChain | Thousands of GitHub stars |
| **Maturation** | 2023 H2-2024 | Platformization | GPT-4 Functions, Claude Computer Use | Agents become LLM standard |
| **Invisibilization** | 2025+ | Agent as Application | MCP Protocol, OpenAI Agents SDK, Various Agent Platforms | Agents fade into the background — users only see functionality |

---

## III. Detailed Classic Frameworks

### 3.1 ReAct — Interleaving Reasoning and Action

The core insight of ReAct (Yao et al., 2022): **Thought and Action alternate, mutually reinforcing each other**.

```
User: "What is the time difference between Beijing and Shanghai?"

Thought: Time difference depends on longitude. Beijing is UTC+8, Shanghai is also UTC+8
Action: verify_timezone("Beijing")
Observation: Beijing = UTC+8

Thought: Beijing confirmed, now check Shanghai
Action: verify_timezone("Shanghai")
Observation: Shanghai = UTC+8

Thought: Both cities are in UTC+8
Action: FINISH("There is no time difference between Beijing and Shanghai; both are UTC+8")
```

#### Why ReAct Works

| Mechanism | Description |
|-----------|-------------|
| **Reasoning guides action** | Thought determines what to do next (rather than random trial) |
| **Action feeds reasoning** | Observation provides factual evidence (reduces hallucination) |
| **Interpretability** | Every step shows what was done and why — fully transparent |
| **Error recovery** | Can adjust course after seeing erroneous Observations |

### 3.2 More Reasoning Paradigms

| Paradigm | Process | Applicable Scenario |
|----------|---------|---------------------|
| **CoT** | Directly outputs reasoning chain without tool invocation | Math, logic |
| **CoT-SC** | Multiple CoT → vote for the best | Requires high accuracy |
| **Tree-of-Thought** | Multiple candidates per step → BFS/DFS search | Complex planning |
| **ReAct** | Thought→Action→Obs loop | Requires external information |

### 3.3 Function Calling — Industry Standard

Rather than relying on Prompt Engineering, it **defines available tools at the model architecture level**.

```python
# OpenAI Function Calling definition
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a specified city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
}]
# Model output: tool_calls=[{name:"get_weather", arguments:{city:"Beijing",unit:"celsius"}}]
# Rather than natural language "I need to check the weather in Beijing"
```

| Advantage | Description |
|-----------|-------------|
| **Structured** | JSON Schema defines parameters — strong typing, cannot pass wrong parameters |
| **Parallel calls** | Unrelated tool calls can execute simultaneously |
| **Streaming** | tool call arguments can be streamed |
| **Multimodal** | GPT-4V can "see" images before calling tools |

### 3.4 MCP Protocol — The "USB Interface" for Agents

Model Context Protocol (Anthropic, 2024): A unified standard for Agent interaction with external tools/data.

| MCP Role | Description | Analogy |
|----------|-------------|---------|
| **MCP Host** | Claude Desktop / ChatGPT App | Computer |
| **MCP Client** | Protocol implementation on the model side | USB Controller |
| **MCP Server** | Tool/Data provider (filesystem, database, API) | USB Device |

> MCP solves the biggest fragmentation problem for Agents — previously every Agent framework had its own tool integration method; now a single protocol unifies everything.

---

## IV. Multi-Agent Collaboration Patterns

### 4.1 Four Collaboration Topologies

| Topology | Structure | Applicable Scenario |
|----------|-----------|---------------------|
| **Pipeline** | A→B→C serial | Fixed workflows (document → translate → proofread) |
| **Star** | One Orchestrator dispatches N Workers | Task decomposition (research → analyze → write → review) |
| **Debate** | N Agents debate each other → Judge decides | Needs multi-perspective thinking (proposal review) |
| **Hierarchical** | Tree structure — superiors decompose, subordinates execute | Complex projects (software development team) |

### 4.2 Representative Framework Comparison

| Framework | Highlight | Learning Curve | Production Ready |
|-----------|-----------|----------------|------------------|
| **AutoGen** (Microsoft) | Multi-Agent conversation + code sandbox execution | Medium | ⭐⭐⭐ |
| **CrewAI** | Role-driven — define "Researcher" "Analyst" "Writer" | Low | ⭐⭐ |
| **LangGraph** | Stateful graph — each node is an Agent, edges are interactions | High | ⭐⭐⭐ |
| **OpenAI Swarm** | Minimalist — handoff to switch Agents | Low | ⭐ (Experimental) |
| **OpenAI Agents SDK** | Production-grade — Guardrails + Tracing + Handoffs | Medium | ⭐⭐⭐ |

---

## V. Agent Engineering Frameworks

| Framework | Core Philosophy | Best For |
|-----------|-----------------|----------|
| **LangChain** | Prompt→LLM→Tool chaining | Developers wanting rapid prototyping |
| **LangGraph** | Directed graph finite state machine — controllable, debuggable | Complex multi-step Agents |
| **LlamaIndex** | Data indexing at its core | Knowledge-intensive Agents (RAG) |
| **Semantic Kernel** | Microsoft — enterprise-grade plugin architecture | .NET / enterprise developers |
| **CrewAI** | Role-playing — "You are a researcher" | Non-technical users can understand |
| **Dify / Coze** | Drag-and-drop building + template marketplace | People who cannot code at all |

---

## VI. Agent Evaluation

| Benchmark | What It Tests | Representative Tasks |
|-----------|---------------|----------------------|
| **WebArena** | Web operation Agent | Shopping, booking, sending emails |
| **SWE-bench** | Code Agent | Fixing GitHub Issues |
| **AgentBench** | Comprehensive Agent capabilities | 8 environments — OS/DB/Web/Games |
| **GAIA** | General AI assistant | Multi-step reasoning + multimodal input |
| **τ-bench** | Tool use | Real API calls |

---

## VII. Core Agent Challenges and Solutions

| Challenge | Manifestation | Current Best Practice |
|-----------|---------------|-----------------------|
| **Hallucinated tools** | Calling non-existent APIs | Function Calling JSON Schema constraints |
| **Divergence** | Forgetting original goal after 10 steps | Re-check goal every N steps; Orchestrator verification |
| **Speed** | Serial 10+ steps = 30+ seconds | Parallel tool calls + streaming output |
| **Cost** | Every step calls GPT-4 → $1+ per task | Small model for simple steps, large model only for complex steps |
| **Safety** | Deleting files, sending wrong emails | Sensitive operations require human confirmation; permission分级 |

---

*Related notes: [Agentic RL](./Agentic-RL.en.md), [Classic Agent Framework Implementation](./%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E6%A1%86%E6%9E%B6%E5%AE%9E%E7%8E%B0.en.md), [Embodied Intelligence in the Era of Large Models](./%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%97%B6%E4%BB%A3%E7%9A%84%E5%85%B7%E8%BA%AB%E6%99%BA%E8%83%BD.en.md)*
