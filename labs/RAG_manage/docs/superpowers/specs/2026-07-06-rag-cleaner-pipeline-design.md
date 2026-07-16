# RAG 数据清洗管道 — 架构设计文档

**日期**: 2026-07-06
**状态**: 已确认
**作者**: 架构师 + 工程搭档

---

## 1. 项目概述

构建一个基于小模型（Qwen-32B via Ollama）的本地部署 RAG 数据清洗工具，将脏文本转换为标准化 Markdown 文件。提供 Gradio 前端 UI 供零基础用户使用。

### 1.1 核心哲学（不可违背）

1. **改造优于重构**：所有 Prompt 基于已验证的旧 Prompt 进行"协议不变，载荷替换"的改造
2. **小模型做"体力活"**：小模型只负责遍历 → 提取 → 填充固定 JSON Schema
3. **协议不变，载荷替换**：保留低温度（0.15）+ 强校验 + JSON 外壳
4. **大模型/代码做决策，小模型做执行**：路由、入库、重试、并发控制全部由 Python 胶水代码完成

### 1.2 用户流程

```
用户上传/粘贴文本 → Gradio UI → 后端管道（分片→调用Ollama→校验→重试→写入）→ 预览+下载MD
```

### 1.3 用户使用流程

**启动（一条命令）**：`python app.py`，自动打开浏览器访问 `http://localhost:7860`。

**Tab 1 - 📝 文本清洗（单条/少量）**：
- 选择来源类型（网页/笔记/聊天/PDF）→ 粘贴脏文本 → 点击清洗
- 结果区展示：标题、标签、质量标识、清洗后的 Markdown 预览
- 支持单独下载 `.md` 文件或一键复制

**Tab 2 - 📁 批量上传（多文件）**：
- 选择来源类型 → 拖拽或选择多个 `.txt`/`.md` 文件 → 点击开始
- 显示实时进度条（X/N 完成）
- 完成后展示统计：valid/partial/invalid 数量 + 逐条结果
- 一键下载 ZIP 包，内部按 quality_flag 分 `valid/`、`partial/`、`invalid/` 三个子目录

**Tab 3 - 📊 处理历史**：
- 表格展示历次清洗记录（时间、文件数、各类质量数量）
- 点击某次记录可展开查看详情

---

## 2. 技术选型

| 层次 | 技术 | 理由 |
|------|------|------|
| UI | Gradio 4.x | 纯 Python，async 原生支持，`gr.Interface` 一键启动 |
| 推理引擎 | Ollama + Qwen3:32B | 本地部署，OpenAI 兼容 API |
| 异步框架 | asyncio + httpx | 避免阻塞 UI 线程 |
| 数据校验 | json.loads + 字段强校验 + 正则闲聊检测 | 确定性校验，不依赖 LLM |
| 输出格式 | Standard Markdown (.md) + 按 quality_flag 三分流 | 直接可用，无需二次转换 |

---

## 3. 项目结构

```
exobrain-air/
├── app.py                    # Gradio 入口，一键启动
├── pyproject.toml            # 依赖管理
├── prompts/                  # Prompt 资产（不动）
│   ├── ingest-single.md      # rag-single-cleaner
│   └── ingest-batch.md       # rag-batch-cleaner
├── src/
│   ├── __init__.py
│   ├── config.py             # 配置中心（dataclass + 环境变量 + 默认值）
│   ├── client.py             # Ollama API 异步客户端 + 速率控制
│   ├── chunker.py            # 数据分片器（≤5条/批）
│   ├── validator.py          # JSON 校验 + RetryableError + 闲聊检测
│   ├── retry.py              # 指数退避重试状态机
│   ├── cleaner.py            # 清洗管道编排器
│   └── exporter.py           # Markdown 文件输出器
├── output/                   # 清洗结果输出目录
│   ├── valid/                # quality_flag=valid
│   ├── partial/              # quality_flag=partial
│   └── invalid/              # quality_flag=invalid
└── tests/
    └── test_pipeline.py      # 集成测试
```

### 设计原则
- `prompts/` 完全不动，由 `client.py` 在运行时读取并跳过 YAML frontmatter
- `output/` 按 `quality_flag` 三分流
- 每个 `src/` 模块职责单一，可独立测试
- `app.py` 仅做 UI 相关逻辑，不包含业务逻辑

---

## 4. 数据流

```
用户上传/粘贴
    │
    ▼
chunker.py       ← 输入拆分为 ≤5条/批的 chunks
    │
    ▼
cleaner.py       ← 管道编排器，遍历每个 chunk
    │
    ├─ chunk_size == 1 → 使用 rag-single-cleaner
    ├─ chunk_size >= 2 → 使用 rag-batch-cleaner
    │
    ├─ client.py     ← 异步 POST Ollama API（串行处理多 chunk）
    ├─ validator.py  ← JSON 解析 + 字段校验 + 闲聊检测
    ├─ retry.py      ← 校验失败 → 指数退避 → 重新调用（最多3次）
    │
    ▼
exporter.py      ← 按 quality_flag 分目录写入
    │
    ├─ valid/     ← {title}.md
    ├─ partial/   ← {title}.md + {title}.meta.json
    └─ invalid/   ← {title}.md（原文 + [内容无效] 前缀）
    │
    ▼
Gradio UI        ← 展示统计摘要 + 下载链接
```

### 重试路径

```
validator 抛出 RetryableError
    │
    ▼
等待 min(base_delay * 2^n + jitter, max_delay)
  n=0: ~1.0s, n=1: ~2.0s, n=2: ~4.0s
    │
    ▼
重新调用 Ollama API（同 chunk）
    │
    ▼
再次校验 → 仍失败 → FatalError → 标记 "manual_review" → output/invalid/
```

### 并发模型

- 多个 chunk 之间**串行处理**（Ollama 单卡推理，并发无意义且可能 OOM）
- **异步 I/O** 仅用于 API 调用本身（避免阻塞 Gradio UI 线程）

---

## 5. 核心模块接口契约

### 5.1 `config.py`

```python
@dataclass
class PipelineConfig:
    model_name: str = "qwen3:32b"
    temperature: float = 0.15
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    chunk_size: int = 5
    ollama_base_url: str = "http://localhost:11434/v1"
    request_timeout: float = 120.0
    prompt_dir: Path = Path("prompts")
    output_dir: Path = Path("output")
```

从环境变量 + YAML 配置文件加载，有合理默认值。

### 5.2 `chunker.py`

- `chunk_items(items, max_size=5) -> list[list[dict]]` — 分片
- `chunk_text(raw_texts, source_type) -> list[dict]` — 原始文本包装为 item

约定：`chunk_size=1` → `rag-single-cleaner`，`chunk_size>=2` → `rag-batch-cleaner`

### 5.3 `validator.py`

- `RetryableError` — JSON 解析失败 / 字段缺失 / 闲聊前置废话 → 触发重试
- `FatalError` — 超出最大重试 / 模型完全崩溃 → 不再重试
- `validate_single_result(raw_text, expected_id=None) -> dict`
- `validate_batch_result(raw_text, input_ids) -> list[dict]`

校验步骤：`json.loads` → 闲聊检测 → 字段完整性 → id 映射一致性

### 5.4 `retry.py`

- `with_retry(fn, validator, max_retries=3, base_delay=1.0, max_delay=10.0) -> dict`
- 延迟公式：`min(base_delay * 2^n + jitter, max_delay)`
- jitter：随机 ±0~10%

### 5.5 `client.py`

- `OllamaClient(config)` — 异步客户端
- `load_prompt(name) -> str` — 加载 Prompt（跳过 YAML frontmatter）
- `clean_single(content, metadata) -> str` — 调用 rag-single-cleaner
- `clean_batch(items) -> str` — 调用 rag-batch-cleaner
- `_chat(system_prompt, user_input) -> str` — 底层 POST，异步 + 超时控制

### 5.6 `exporter.py`

- `export_results(results, output_dir) -> dict` — 按 quality_flag 分目录写入，返回统计摘要

输出格式：
- `valid/` → `{title}.md`（cleaned_markdown 内容）
- `partial/` → `{title}.md` + `{title}.meta.json`（保留元数据）
- `invalid/` → `{title}.md`（原文 + [内容无效] 标注）

### 5.7 `cleaner.py`

- `RAGCleaner(config)` — 管道编排器
- `clean(items, progress_cb=None) -> dict` — 主入口：分片 → 调用 → 校验 → 重试 → 导出

### 5.8 `app.py`

Gradio 三标签页界面：
- Tab "📝 文本清洗"：粘贴 → 清洗 → 预览 + 下载
- Tab "📁 批量上传"：拖拽/上传 → 清洗 → ZIP 打包下载
- Tab "📊 处理历史"：历次统计

---

## 6. 错误处理策略

| 错误类型 | 处理方式 |
|----------|----------|
| JSON 解析失败 | RetryableError，最多重试 3 次 |
| 字段缺失 | RetryableError，最多重试 3 次 |
| 闲聊前置废话 | RetryableError，最多重试 3 次 |
| ID 映射不一致 | RetryableError，最多重试 3 次 |
| API 超时 | RetryableError，延迟指数退避 |
| API 返回非 JSON | RetryableError，最多重试 3 次 |
| 3次重试后仍失败 | FatalError，标记 manual_review，写入 invalid/ |
| Ollama 服务不可达 | 立即终止，通过 Gradio 提示用户检查服务 |

---

## 7. 不在范围内的（明确排除）

1. 不修改 `prompts/` 下的任何文件
2. 不引入向量数据库集成（纯文件系统输出）
3. 不引入 YAML/Markdown 混合输出格式
4. 不提高 Temperature（保持 0.15）
5. 不需要 CUDA/GPU 特定的配置（Ollama 已封装）
6. 不需要用户认证/多租户

---

## 8. 待实现清单（按优先级）

1. `src/config.py` — 配置中心
2. `src/validator.py` — 统一校验器
3. `src/retry.py` — 指数退避重试
4. `src/client.py` — Ollama 异步客户端
5. `src/chunker.py` — 数据分片器
6. `src/exporter.py` — Markdown 输出器
7. `src/cleaner.py` — 管道编排器
8. `app.py` — Gradio UI
9. `pyproject.toml` — 项目配置与依赖
10. `tests/test_pipeline.py` — 集成测试
