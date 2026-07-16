# RAG 数据清洗管道 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个基于 Gradio + Ollama + Qwen3:32B 的本地 RAG 数据清洗工具，将脏文本转换为标准化 Markdown 文件

**Architecture:** 三层结构 — Gradio UI（表现层）→ Python 管道（调度/容错层）→ Ollama + Prompt（清洗层）。模块按 config → validator → retry → client → chunker → exporter → cleaner → app 顺序构建，每个模块独立可测

**Tech Stack:** Python 3.11+, Gradio 4.x, httpx (async), openai SDK, asyncio, pytest

---

## File Structure

| 文件 | 职责 | 创建/修改 |
|------|------|-----------|
| `pyproject.toml` | 项目元数据 + 依赖声明 | Create |
| `src/__init__.py` | 包初始化 | Create |
| `src/config.py` | PipelineConfig dataclass | Create |
| `src/validator.py` | JSON 校验 + RetryableError + 闲聊检测 | Create |
| `src/retry.py` | 指数退避重试状态机 | Create |
| `src/client.py` | Ollama 异步客户端 + Prompt 加载 | Create |
| `src/chunker.py` | 数据分片器 | Create |
| `src/exporter.py` | Markdown 文件输出器 | Create |
| `src/cleaner.py` | 管道编排器（组装所有模块） | Create |
| `app.py` | Gradio UI 入口 | Create |
| `tests/test_pipeline.py` | 集成测试 | Create |

---

### Task 1: 项目骨架与依赖

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "exobrain-air"
version = "0.1.0"
description = "RAG数据清洗管道 - 基于小模型的本地文本清洗工具"
requires-python = ">=3.11"
dependencies = [
    "gradio>=4.0.0",
    "httpx>=0.27.0",
    "openai>=1.30.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
include = ["src*"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: 创建 src/__init__.py**

```python
"""RAG 数据清洗管道 - 基于小模型的本地文本清洗工具."""
```

- [ ] **Step 3: 安装依赖**

```bash
cd /home/amanannn/Projects/exobrain-air && pip install -e ".[dev]"
```

---

### Task 2: 配置中心 config.py

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: 创建 src/config.py**

```python
"""配置中心 - 所有可调参数集中管理."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """RAG 清洗管道全局配置.

    优先级: 构造函数参数 > 环境变量 > 默认值
    """

    # --- Ollama 连接 ---
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv(
            "EXOBRAIN_OLLAMA_URL", "http://localhost:11434/v1"
        )
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("EXOBRAIN_MODEL", "qwen3:32b")
    )

    # --- 推理参数 ---
    temperature: float = 0.15
    request_timeout: float = 120.0  # 单次 API 超时（秒）

    # --- 重试参数 ---
    max_retries: int = 3
    base_delay: float = 1.0   # 重试基础延迟（秒）
    max_delay: float = 10.0   # 重试最大延迟（秒）

    # --- 分片参数 ---
    chunk_size: int = 5        # 批量清洗最大条目数（≤5）

    # --- 路径 ---
    prompt_dir: Path = field(default_factory=lambda: Path("prompts"))
    output_dir: Path = field(default_factory=lambda: Path("output"))

    def __post_init__(self) -> None:
        """确保路径为 Path 对象."""
        if isinstance(self.prompt_dir, str):
            self.prompt_dir = Path(self.prompt_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
```

- [ ] **Step 2: 验证可导入**

```bash
cd /home/amanannn/Projects/exobrain-air && python -c "from src.config import PipelineConfig; c = PipelineConfig(); print(f'model={c.model_name}, chunk_size={c.chunk_size}')"
```

Expected: `model=qwen3:32b, chunk_size=5`

---

### Task 3: 校验器 validator.py

**Files:**
- Create: `src/validator.py`

- [ ] **Step 1: 创建 src/validator.py**

```python
"""统一校验器 - JSON 解析 + 字段完整性 + 闲聊检测."""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# 异常定义
# ---------------------------------------------------------------------------


class RetryableError(Exception):
    """可重试错误：JSON 解析失败 / 字段缺失 / 闲聊前置废话."""

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


class FatalError(Exception):
    """致命错误：超出最大重试次数或模型完全崩溃，不再重试."""

    def __init__(self, message: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output = raw_output


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"title", "tags", "summary", "cleaned_markdown", "quality_flag"}

VALID_QUALITY_FLAGS = {"valid", "invalid", "partial"}

# 闲聊前置废话模式：中文解释性前缀 + 非 JSON 文本
CHATTY_PATTERNS = [
    re.compile(r"^(好的|没问题|以下是|根据您|让我|我来|下面|这是|输出如下|处理结果)"),
    re.compile(r"^(OK|Sure|Here|Let me|Below|The following)", re.IGNORECASE),
    re.compile(r"^```(?:json)?\s*"),  # markdown 代码块包装
]


# ---------------------------------------------------------------------------
# 核心校验函数
# ---------------------------------------------------------------------------


def _strip_chatty_prefix(text: str) -> str:
    """检测并剥离闲聊/解释性前置文本，仅保留 JSON 部分.

    返回剥离后的文本。若检测到闲聊但无法定位 JSON 起始位置，抛出 RetryableError.
    """
    stripped = text.strip()

    # 检测 markdown 代码块包装（保留内部 JSON）
    md_match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", stripped, re.DOTALL)
    if md_match:
        return md_match.group(1).strip()

    for pattern in CHATTY_PATTERNS:
        if pattern.match(stripped):
            # 尝试找到第一个 '{' 或 '[' 作为 JSON 起点
            json_start = re.search(r"[\[\{]", stripped)
            if json_start:
                return stripped[json_start.start() :]
            raise RetryableError(
                "检测到闲聊前置文本，但无法定位 JSON 起始位置",
                raw_output=text,
            )

    return stripped


def _validate_fields(result: dict, expected_id: str | None = None) -> None:
    """校验 JSON 对象包含所有必需字段且值合法."""
    missing = REQUIRED_FIELDS - set(result.keys())
    if missing:
        raise RetryableError(f"缺少必需字段: {sorted(missing)}")

    if not isinstance(result["title"], str) or not result["title"].strip():
        raise RetryableError("title 字段为空或非字符串")

    if not isinstance(result["tags"], list) or len(result["tags"]) == 0:
        raise RetryableError("tags 字段为空或非数组")

    if not isinstance(result["summary"], str) or not result["summary"].strip():
        raise RetryableError("summary 字段为空或非字符串")

    if not isinstance(result["cleaned_markdown"], str) or not result["cleaned_markdown"].strip():
        raise RetryableError("cleaned_markdown 字段为空或非字符串")

    if result["quality_flag"] not in VALID_QUALITY_FLAGS:
        raise RetryableError(
            f"quality_flag 值非法: '{result['quality_flag']}'，允许值: {VALID_QUALITY_FLAGS}"
        )

    if expected_id is not None:
        if "id" not in result:
            raise RetryableError("batch 模式缺少 id 字段")
        if result["id"] != expected_id:
            raise RetryableError(
                f"ID 不匹配: 期望 '{expected_id}'，实际 '{result['id']}'"
            )


def validate_single_result(raw_text: str, expected_id: str | None = None) -> dict:
    """校验单条清洗结果.

    Args:
        raw_text: 模型原始输出文本.
        expected_id: batch 模式下的期望 ID，None 表示 single 模式.

    Returns:
        清洗后的有效结果 dict.

    Raises:
        RetryableError: 可重试的校验失败.
        FatalError: 不可恢复的格式错误.
    """
    # 1. 剥离闲聊前缀
    cleaned = _strip_chatty_prefix(raw_text)

    # 2. JSON 解析
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RetryableError(f"JSON 解析失败: {e}", raw_output=raw_text) from e

    if not isinstance(result, dict):
        raise FatalError(f"期望 JSON 对象，实际类型: {type(result).__name__}", raw_output=raw_text)

    # 3. 字段校验
    _validate_fields(result, expected_id)

    # 4. invalid 文档的特殊校验
    if result["quality_flag"] == "invalid":
        if result["title"] != "无效文档":
            raise RetryableError("quality_flag=invalid 时 title 必须为 '无效文档'")
        if result["tags"] != ["待人工审核"]:
            raise RetryableError("quality_flag=invalid 时 tags 必须为 ['待人工审核']")

    return result


def validate_batch_result(raw_text: str, input_ids: list[str]) -> list[dict]:
    """校验批量清洗结果.

    Args:
        raw_text: 模型原始输出文本.
        input_ids: 输入的 id 列表，用于数量+顺序校验.

    Returns:
        清洗后的有效结果列表.

    Raises:
        RetryableError: 可重试的校验失败.
        FatalError: 不可恢复的格式错误.
    """
    # 1. 剥离闲聊前缀
    cleaned = _strip_chatty_prefix(raw_text)

    # 2. JSON 解析
    try:
        wrapper = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RetryableError(f"JSON 解析失败: {e}", raw_output=raw_text) from e

    if not isinstance(wrapper, dict):
        raise FatalError(f"期望 JSON 对象，实际类型: {type(wrapper).__name__}", raw_output=raw_text)

    if "results" not in wrapper:
        raise RetryableError("batch 输出缺少 'results' 字段")

    results = wrapper["results"]
    if not isinstance(results, list):
        raise FatalError(f"'results' 字段应为数组，实际类型: {type(results).__name__}")

    # 3. 数量校验
    if len(results) != len(input_ids):
        raise RetryableError(
            f"results 数量不匹配: 期望 {len(input_ids)}，实际 {len(results)}"
        )

    # 4. 逐条校验
    for i, (item, expected_id) in enumerate(zip(results, input_ids)):
        if not isinstance(item, dict):
            raise RetryableError(f"results[{i}] 不是 JSON 对象")
        _validate_fields(item, expected_id)

        if item["quality_flag"] == "invalid":
            if item["title"] != "无效文档":
                raise RetryableError(f"results[{i}] quality_flag=invalid 时 title 必须为 '无效文档'")
            if item["tags"] != ["待人工审核"]:
                raise RetryableError(f"results[{i}] quality_flag=invalid 时 tags 必须为 ['待人工审核']")

    return results
```

- [ ] **Step 2: 验证校验器可导入并手动测试**

```bash
cd /home/amanannn/Projects/exobrain-air && python -c "
from src.validator import validate_single_result, RetryableError

# 正例
r = validate_single_result('{\"title\":\"测试\",\"tags\":[\"a\"],\"summary\":\"s\",\"cleaned_markdown\":\"c\",\"quality_flag\":\"valid\"}')
print('正例通过:', r['title'])

# 反例 - 缺少字段
try:
    validate_single_result('{\"title\":\"x\"}')
except RetryableError as e:
    print('反例正确触发:', str(e)[:60])
"
```

Expected: 两条输出均正确显示

---

### Task 4: 重试状态机 retry.py

**Files:**
- Create: `src/retry.py`

- [ ] **Step 1: 创建 src/retry.py**

```python
"""指数退避重试状态机."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

from src.validator import FatalError, RetryableError

logger = logging.getLogger(__name__)


async def with_retry(
    fn: Callable[[], Awaitable[str]],
    validator: Callable[[str], object],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
) -> object:
    """带指数退避的重试执行器.

    Args:
        fn: 异步可调用对象，返回模型原始输出字符串.
        validator: 校验函数，接收原始文本，成功返回 dict/list，失败抛出 RetryableError.
        max_retries: 最大重试次数（不含首次调用）.
        base_delay: 重试基础延迟（秒）.
        max_delay: 重试最大延迟上限（秒）.

    Returns:
        校验通过的结果.

    Raises:
        FatalError: 所有重试耗尽后仍失败.
    """
    last_error: RetryableError | None = None
    total_attempts = 1 + max_retries  # 首次 + 重试

    for attempt in range(total_attempts):
        try:
            raw = await fn()
            result = validator(raw)
            if attempt > 0:
                logger.info("第 %d 次重试成功", attempt)
            return result
        except RetryableError as e:
            last_error = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = delay * random.uniform(0, 0.1)  # 0~10% 抖动
                actual_delay = delay + jitter
                logger.warning(
                    "校验失败 (attempt %d/%d): %s — %.1fs 后重试",
                    attempt + 1, total_attempts, str(e)[:100], actual_delay,
                )
                await asyncio.sleep(actual_delay)
            else:
                logger.error("所有 %d 次重试已耗尽", total_attempts)

    raise FatalError(
        f"重试 {max_retries} 次后仍失败: {last_error}",
        raw_output=last_error.raw_output if last_error else "",
    )
```

- [ ] **Step 2: 验证重试逻辑**

```bash
cd /home/amanannn/Projects/exobrain-air && python -c "
import asyncio
from src.validator import RetryableError
from src.retry import with_retry

async def flaky_fn():
    flaky_fn.calls += 1
    if flaky_fn.calls < 3:
        return 'bad json'
    return '{\"title\":\"ok\",\"tags\":[\"a\"],\"summary\":\"s\",\"cleaned_markdown\":\"c\",\"quality_flag\":\"valid\"}'
flaky_fn.calls = 0

async def main():
    result = await with_retry(flaky_fn, lambda r: __import__('src.validator', fromlist=['validate_single_result']).validate_single_result(r), max_retries=2, base_delay=0.01)
    print(f'重试 {flaky_fn.calls-1} 次后成功:', result['title'])

asyncio.run(main())
"
```

Expected: `重试 2 次后成功: ok`

---

### Task 5: Ollama 异步客户端 client.py

**Files:**
- Create: `src/client.py`

- [ ] **Step 1: 创建 src/client.py**

```python
"""Ollama 异步客户端 - Prompt 加载 + API 调用."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
from openai import AsyncOpenAI

from src.config import PipelineConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """封装 Ollama OpenAI 兼容 API 的异步客户端."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            base_url=config.ollama_base_url,
            api_key="ollama",  # Ollama 不需要真实 key
            timeout=httpx.Timeout(config.request_timeout),
        )
        # 缓存已加载的 prompt
        self._prompt_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Prompt 加载
    # ------------------------------------------------------------------

    def load_prompt(self, name: str) -> str:
        """从 prompts/{name}.md 加载 system prompt（跳过 YAML frontmatter）.

        Args:
            name: prompt 文件名（不含 .md 后缀），如 'ingest-single'.

        Returns:
            system prompt 文本内容.
        """
        if name in self._prompt_cache:
            return self._prompt_cache[name]

        file_path = self.config.prompt_dir / f"{name}.md"
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {file_path}")

        content = file_path.read_text(encoding="utf-8")
        # 跳过 YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2].strip()
            else:
                content = content.strip()
        else:
            content = content.strip()

        self._prompt_cache[name] = content
        logger.debug("已加载 prompt: %s (%d 字符)", name, len(content))
        return content

    # ------------------------------------------------------------------
    # 清洗接口
    # ------------------------------------------------------------------

    async def clean_single(self, content: str, metadata: dict | None = None) -> str:
        """调用 rag-single-cleaner 清洗单条文本.

        Args:
            content: 待清洗的原始文本.
            metadata: 元数据 dict，包含 source_type 和 timestamp.

        Returns:
            模型原始输出字符串.
        """
        if metadata is None:
            metadata = {"source_type": "web", "timestamp": ""}

        system_prompt = self.load_prompt("ingest-single")
        user_input = json.dumps(
            {"content": content, "metadata": metadata},
            ensure_ascii=False,
        )
        return await self._chat(system_prompt, user_input)

    async def clean_batch(self, items: list[dict]) -> str:
        """调用 rag-batch-cleaner 批量清洗.

        Args:
            items: [{"id": str, "source_type": str, "raw_text": str}, ...].

        Returns:
            模型原始输出字符串.
        """
        system_prompt = self.load_prompt("ingest-batch")
        user_input = json.dumps({"items": items}, ensure_ascii=False)
        return await self._chat(system_prompt, user_input)

    # ------------------------------------------------------------------
    # 底层 API
    # ------------------------------------------------------------------

    async def _chat(self, system_prompt: str, user_input: str) -> str:
        """底层异步 POST，带超时控制.

        Args:
            system_prompt: system role 消息内容.
            user_input: user role 消息内容（JSON 字符串）.

        Returns:
            模型响应文本.

        Raises:
            httpx.TimeoutException: 请求超时.
        """
        response = await self._client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        logger.debug("API 响应 (%d 字符)", len(raw))
        return raw
```

- [ ] **Step 2: 验证 client 可导入并测试 prompt 加载**

```bash
cd /home/amanannn/Projects/exobrain-air && python -c "
from src.config import PipelineConfig
from src.client import OllamaClient
c = OllamaClient(PipelineConfig())
p = c.load_prompt('ingest-single')
print('Prompt 加载成功:', p[:80], '...')
assert 'rag-single-cleaner' in p or 'RAG数据清洗' in p or '清洗' in p, 'Prompt 内容异常'
print('✅ client.py 就绪')
"
```

Expected: `Prompt 加载成功: ... ✅ client.py 就绪`

---

### Task 6: 数据分片器 chunker.py

**Files:**
- Create: `src/chunker.py`

- [ ] **Step 1: 创建 src/chunker.py**

```python
"""数据分片器 - 将输入拆分为 ≤5条/批的 chunks."""

from __future__ import annotations

import uuid


def chunk_items(items: list[dict], max_size: int = 5) -> list[list[dict]]:
    """将输入条目按 max_size 切分为批次.

    约定:
      - chunk 大小为 1 → 调用 rag-single-cleaner
      - chunk 大小为 2~5 → 调用 rag-batch-cleaner

    Args:
        items: [{"id": str, "source_type": str, "raw_text": str}, ...].
        max_size: 每批次最大条目数（默认 5）.

    Returns:
        批次列表，每个批次是一个 item dict 的 list.
    """
    if not items:
        return []
    return [items[i : i + max_size] for i in range(0, len(items), max_size)]


def make_items(
    raw_texts: list[str],
    source_type: str = "web",
) -> list[dict]:
    """将原始文本列表包装为带 ID 的 item 结构.

    Args:
        raw_texts: 原始文本字符串列表.
        source_type: 来源类型标识.

    Returns:
        [{"id": str, "source_type": str, "raw_text": str}, ...].
    """
    items = []
    for text in raw_texts:
        if not text or not text.strip():
            continue  # 跳过空白文本
        item_id = f"doc_{uuid.uuid4().hex[:8]}"
        items.append({
            "id": item_id,
            "source_type": source_type,
            "raw_text": text.strip(),
        })
    return items


def classify_chunk(chunk: list[dict]) -> str:
    """判断某个 chunk 应使用的策略.

    Returns:
        'single' 或 'batch'.
    """
    return "single" if len(chunk) == 1 else "batch"
```

- [ ] **Step 2: 验证分片逻辑**

```bash
cd /home/amanannn/Projects/exobrain-air && python -c "
from src.chunker import chunk_items, make_items, classify_chunk

# 7条 → 应为2个批次（5+2）
items = make_items(['text1', 'text2', 'text3', 'text4', 'text5', 'text6', 'text7'])
chunks = chunk_items(items, 5)
print(f'7条 → {len(chunks)} 个批次: {[len(c) for c in chunks]}')
assert len(chunks) == 2
assert len(chunks[0]) == 5
assert len(chunks[1]) == 2

# 分类
assert classify_chunk(chunks[0]) == 'batch'
single_chunk = chunk_items(make_items(['only one']), 5)
assert classify_chunk(single_chunk[0]) == 'single'

# 空输入
assert chunk_items([], 5) == []

# 跳过空白
empty_items = make_items(['', '  ', 'valid'])
assert len(empty_items) == 1
print('✅ chunker.py 就绪')
"
```

Expected: `✅ chunker.py 就绪`

---

### Task 7: Markdown 输出器 exporter.py

**Files:**
- Create: `src/exporter.py`

- [ ] **Step 1: 创建 src/exporter.py**

```python
"""Markdown 文件输出器 - 按 quality_flag 分目录写入."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str, max_len: int = 60) -> str:
    """将标题转换为安全文件名.

    Args:
        name: 原始标题.
        max_len: 最大文件名长度（不含扩展名）.

    Returns:
        安全文件名（保留中文，移除非法字符）.
    """
    # 移除文件系统非法字符
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    # 去除首尾空白和点
    safe = safe.strip(" .")
    if len(safe) > max_len:
        safe = safe[:max_len]
    return safe or "untitled"


def export_results(results: list[dict], output_dir: Path) -> dict:
    """按 quality_flag 分目录写入清洗结果.

    Args:
        results: 清洗结果列表，每个 dict 包含:
            title, tags, summary, cleaned_markdown, quality_flag, id(可选).
        output_dir: 输出根目录（内部自动创建 valid/ partial/ invalid/ 子目录）.

    Returns:
        统计摘要: {"total": int, "valid": int, "partial": int, "invalid": int, "export_dir": str}.
    """
    valid_dir = output_dir / "valid"
    partial_dir = output_dir / "partial"
    invalid_dir = output_dir / "invalid"

    for d in (valid_dir, partial_dir, invalid_dir):
        d.mkdir(parents=True, exist_ok=True)

    stats = {"total": len(results), "valid": 0, "partial": 0, "invalid": 0}

    for i, r in enumerate(results):
        quality = r.get("quality_flag", "invalid")
        title = r.get("title", f"untitled_{i}")
        safe_name = _sanitize_filename(title)
        # 避免重名
        doc_id = r.get("id", f"{i:04d}")
        base_name = f"{doc_id}_{safe_name}"

        if quality == "valid":
            _write_md(valid_dir / f"{base_name}.md", r)
            stats["valid"] += 1

        elif quality == "partial":
            md_path = partial_dir / f"{base_name}.md"
            meta_path = partial_dir / f"{base_name}.meta.json"
            _write_md(md_path, r)
            _write_meta(meta_path, r)
            stats["partial"] += 1

        else:  # invalid
            _write_md(invalid_dir / f"{base_name}.md", r)
            stats["invalid"] += 1

    logger.info(
        "导出完成: total=%d valid=%d partial=%d invalid=%d → %s",
        stats["total"], stats["valid"], stats["partial"], stats["invalid"], str(output_dir),
    )
    stats["export_dir"] = str(output_dir.resolve())
    return stats


def _write_md(filepath: Path, result: dict) -> None:
    """写入 Markdown 文件，添加元数据头."""
    title = result.get("title", "未命名")
    tags = result.get("tags", [])
    summary = result.get("summary", "")
    quality = result.get("quality_flag", "invalid")
    body = result.get("cleaned_markdown", "")

    tags_str = " ".join(f"#{t}" for t in tags)

    content = (
        f"---\n"
        f"title: {title}\n"
        f"tags: [{', '.join(tags)}]\n"
        f"summary: {summary}\n"
        f"quality: {quality}\n"
        f"---\n\n"
        f"{body}\n"
    )
    filepath.write_text(content, encoding="utf-8")


def _write_meta(filepath: Path, result: dict) -> None:
    """写入元数据 JSON 文件（仅 partial）."""
    meta = {
        "title": result.get("title"),
        "tags": result.get("tags"),
        "summary": result.get("summary"),
        "quality_flag": result.get("quality_flag"),
    }
    filepath.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def create_zip(output_dir: Path) -> Path:
    """将输出目录打包为 ZIP 文件.

    Args:
        output_dir: 输出根目录.

    Returns:
        ZIP 文件路径.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_path = output_dir / f"cleaned_{timestamp}.zip"
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
        for sub in ("valid", "partial", "invalid"):
            sub_dir = output_dir / sub
            if sub_dir.exists():
                for f in sub_dir.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(output_dir))
    logger.info("ZIP 已创建: %s", zip_path)
    return zip_path
```

- [ ] **Step 2: 测试导出功能**

```bash
cd /home/amanannn/Projects/exobrain-air && python -c "
from pathlib import Path
from src.exporter import export_results, create_zip

results = [
    {'id': 'doc_01', 'title': 'Q3预算会议', 'tags': ['预算', 'Q3'], 'summary': '削减费用20%', 'cleaned_markdown': '# 会议纪要', 'quality_flag': 'valid'},
    {'id': 'doc_02', 'title': '', 'tags': ['待人工审核'], 'summary': '内容不足', 'cleaned_markdown': '[内容无效] ...', 'quality_flag': 'invalid'},
    {'id': 'doc_03', 'title': 'Python教程', 'tags': ['Python'], 'summary': '部分有效', 'cleaned_markdown': '# 教程\n> 推广已移除', 'quality_flag': 'partial'},
]
out = Path('/tmp/test_exporter_output')
# 清理旧数据
import shutil
shutil.rmtree(out, ignore_errors=True)

stats = export_results(results, out)
print('统计:', {k: v for k, v in stats.items() if k != 'export_dir'})
assert stats['valid'] == 1
assert stats['partial'] == 1
assert stats['invalid'] == 1

# 验证文件存在
assert (out / 'valid').exists()
assert (out / 'partial').exists()
assert (out / 'invalid').exists()
valid_files = list((out / 'valid').glob('*.md'))
print('valid 目录文件:', [f.name for f in valid_files])

# 测试 ZIP
zip_path = create_zip(out)
print('ZIP:', zip_path.name, f'({zip_path.stat().st_size} bytes)')
assert zip_path.exists()

shutil.rmtree(out, ignore_errors=True)
print('✅ exporter.py 就绪')
"
```

Expected: `✅ exporter.py 就绪`

---

### Task 8: 管道编排器 cleaner.py

**Files:**
- Create: `src/cleaner.py`

- [ ] **Step 1: 创建 src/cleaner.py**

```python
"""管道编排器 - 组装分片/客户端/校验/重试/导出全流程."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from src.chunker import chunk_items, classify_chunk
from src.client import OllamaClient
from src.config import PipelineConfig
from src.exporter import create_zip, export_results
from src.retry import with_retry
from src.validator import (
    FatalError,
    RetryableError,
    validate_batch_result,
    validate_single_result,
)

logger = logging.getLogger(__name__)


class RAGCleaner:
    """RAG 数据清洗管道编排器.

    用法:
        config = PipelineConfig()
        cleaner = RAGCleaner(config)
        stats = await cleaner.clean(items, progress_cb=my_callback)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.client = OllamaClient(self.config)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    async def clean(
        self,
        items: list[dict],
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """清洗主流程.

        Args:
            items: [{"id": str, "source_type": str, "raw_text": str}, ...].
            progress_cb: 进度回调 (current, total, message).

        Returns:
            统计摘要 dict，含 results 列表和文件路径.
        """
        if not items:
            return {"total": 0, "valid": 0, "partial": 0, "invalid": 0, "results": []}

        chunks = chunk_items(items, self.config.chunk_size)
        all_results: list[dict] = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            strategy = classify_chunk(chunk)
            msg = f"处理第 {i+1}/{total_chunks} 批 ({len(chunk)} 条, {strategy})"
            logger.info(msg)
            if progress_cb:
                progress_cb(i + 1, total_chunks, msg)

            try:
                if strategy == "single":
                    result = await self._clean_single_with_retry(chunk[0])
                    all_results.append(result)
                else:
                    batch_results = await self._clean_batch_with_retry(chunk)
                    all_results.extend(batch_results)
            except FatalError as e:
                # 单条失败 → 生成 invalid 结果继续
                logger.error("批次 %d 清洗失败: %s", i + 1, str(e)[:100])
                for item in chunk:
                    all_results.append(self._make_invalid_result(item))
                if progress_cb:
                    progress_cb(i + 1, total_chunks, f"第 {i+1} 批失败，已标记为 invalid")

        # 导出
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        stats = export_results(all_results, self.config.output_dir)
        zip_path = create_zip(self.config.output_dir)
        stats["zip_path"] = str(zip_path.resolve())
        stats["results"] = all_results

        if progress_cb:
            progress_cb(total_chunks, total_chunks, "完成")
        return stats

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _clean_single_with_retry(self, item: dict) -> dict:
        """带重试的单条清洗."""
        validator = validate_single_result

        async def call_fn() -> str:
            return await self.client.clean_single(
                content=item["raw_text"],
                metadata={"source_type": item.get("source_type", "web"), "timestamp": ""},
            )

        try:
            return await with_retry(
                call_fn,
                lambda raw: validator(raw, expected_id=item.get("id")),
                max_retries=self.config.max_retries,
                base_delay=self.config.base_delay,
                max_delay=self.config.max_delay,
            )
        except FatalError:
            raise

    async def _clean_batch_with_retry(self, chunk: list[dict]) -> list[dict]:
        """带重试的批量清洗."""
        input_ids = [item["id"] for item in chunk]

        async def call_fn() -> str:
            return await self.client.clean_batch(chunk)

        try:
            return await with_retry(
                call_fn,
                lambda raw: validate_batch_result(raw, input_ids),
                max_retries=self.config.max_retries,
                base_delay=self.config.base_delay,
                max_delay=self.config.max_delay,
            )
        except FatalError:
            raise

    def _make_invalid_result(self, item: dict) -> dict:
        """当清洗完全失败时，构造一个 invalid 结果."""
        raw = item.get("raw_text", "")
        return {
            "id": item.get("id", "unknown"),
            "title": "无效文档",
            "tags": ["待人工审核"],
            "summary": "内容不足",
            "cleaned_markdown": f"[内容无效] 清洗失败: {raw[:200]}",
            "quality_flag": "invalid",
        }
```

- [ ] **Step 2: 验证 cleaner 可导入**

```bash
cd /home/amanannn/Projects/exobrain-air && python -c "
from src.config import PipelineConfig
from src.cleaner import RAGCleaner
c = RAGCleaner(PipelineConfig())
print('✅ cleaner.py 就绪')
"
```

Expected: `✅ cleaner.py 就绪`

---

### Task 9: Gradio UI app.py

**Files:**
- Create: `app.py`

- [ ] **Step 1: 创建 app.py**

```python
#!/usr/bin/env python3
"""RAG 数据清洗工具 - Gradio 前端入口.

用法:
    python app.py
    # 浏览器自动打开 http://localhost:7860
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

import gradio as gr

from src.chunker import make_items
from src.cleaner import RAGCleaner
from src.config import PipelineConfig

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# ---------------------------------------------------------------------------
# 全局
# ---------------------------------------------------------------------------
config = PipelineConfig()
cleaner = RAGCleaner(config)

# ---------------------------------------------------------------------------
# Tab 1: 单条文本清洗
# ---------------------------------------------------------------------------


def clean_single_text(source_type: str, raw_text: str) -> tuple[str, str, str]:
    """清洗单条文本，返回 (markdown_preview, title, stats_str)."""
    if not raw_text or not raw_text.strip():
        return "", "", "❌ 请输入文本"

    items = make_items([raw_text], source_type=source_type)
    if not items:
        return "", "", "❌ 输入为空"

    async def _run():
        return await cleaner.clean(items)

    try:
        stats = asyncio.run(_run())
    except Exception as e:
        logger.exception("清洗失败")
        return "", "", f"❌ 清洗失败: {e}"

    results = stats.get("results", [])
    if not results:
        return "", "", "⚠️ 无结果"

    r = results[0]
    title = r.get("title", "")
    quality = r.get("quality_flag", "")
    tags = " ".join(f"#{t}" for t in r.get("tags", []))
    body = r.get("cleaned_markdown", "")

    preview = f"## {title}\n\n🏷️ {tags}\n\n📊 质量: {quality}\n\n---\n\n{body}"

    flag_emoji = {"valid": "✅", "partial": "⚠️", "invalid": "❌"}
    status_line = f"{flag_emoji.get(quality, '❓')} {quality} | {title}"

    return preview, title, status_line


# ---------------------------------------------------------------------------
# Tab 2: 批量上传
# ---------------------------------------------------------------------------


def clean_batch_files(source_type: str, files: list[str]) -> tuple[str, str]:
    """批量清洗上传的文件，返回 (summary_md, zip_path)."""
    if not files:
        return "❌ 请上传至少一个文件", ""

    # 读取所有文件内容
    raw_texts: list[str] = []
    for fp in files:
        try:
            text = Path(fp).read_text(encoding="utf-8")
            if text.strip():
                raw_texts.append(text)
        except Exception as e:
            logger.warning("读取文件失败 %s: %s", fp, e)

    if not raw_texts:
        return "❌ 所有文件为空或无法读取", ""

    items = make_items(raw_texts, source_type=source_type)
    total = len(items)
    progress = gr.Progress()

    async def _run():
        return await cleaner.clean(
            items,
            progress_cb=lambda cur, tot, msg: progress((cur, tot), desc=msg),
        )

    try:
        stats = loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stats = loop.run_until_complete(_run())
        loop.close()
    except Exception as e:
        logger.exception("批量清洗失败")
        return f"❌ 清洗失败: {e}", ""

    # 构建摘要
    results = stats.get("results", [])
    zip_path = stats.get("zip_path", "")

    lines = [
        f"## 📊 处理完成",
        f"",
        f"| 指标 | 数量 |",
        f"|------|------|",
        f"| 总计 | {stats.get('total', 0)} |",
        f"| ✅ valid | {stats.get('valid', 0)} |",
        f"| ⚠️ partial | {stats.get('partial', 0)} |",
        f"| ❌ invalid | {stats.get('invalid', 0)} |",
        f"",
        f"---",
        f"",
        f"### 详细结果",
        f"",
    ]

    flag_map = {"valid": "✅", "partial": "⚠️", "invalid": "❌"}
    for r in results:
        q = r.get("quality_flag", "invalid")
        title = r.get("title", "未命名")
        summary = r.get("summary", "")
        lines.append(f"- {flag_map.get(q, '❓')} **{title}**: {summary}")

    return "\n".join(lines), zip_path


# ---------------------------------------------------------------------------
# Tab 3: 处理历史 (简化版)
# ---------------------------------------------------------------------------


def list_history() -> str:
    """列出 output 目录下的历史处理记录."""
    out = config.output_dir
    if not out.exists():
        return "📭 暂无处理记录"

    lines = ["## 📊 处理历史", "", "| 时间 | 文件 |", "|------|------|"]

    # 列出 ZIP 文件作为历史记录
    zips = sorted(out.glob("cleaned_*.zip"), reverse=True)
    for z in zips[:20]:  # 最近 20 条
        ts = z.stem.replace("cleaned_", "")
        size_kb = z.stat().st_size // 1024
        lines.append(f"| {ts} | {size_kb} KB |")

    if not zips:
        lines.append("| (空) | (空) |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI 构建
# ---------------------------------------------------------------------------

CSS = """
.gradio-container { max-width: 900px; margin: 0 auto; }
footer { display: none !important; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG 数据清洗工具", css=CSS, theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """# 🧹 RAG 数据清洗工具
            基于 Qwen3:32B 的本地文本清洗与 Markdown 结构化引擎。
            """
        )

        with gr.Tab("📝 文本清洗"):
            with gr.Row():
                source_type = gr.Dropdown(
                    choices=["web", "note", "chat", "pdf"],
                    value="web",
                    label="来源类型",
                    scale=1,
                )
            raw_input = gr.Textbox(
                label="粘贴待清洗的原始文本",
                placeholder="在此粘贴脏文本...支持中文、Markdown、纯文本",
                lines=8,
            )
            clean_btn = gr.Button("🔍 清洗", variant="primary")
            with gr.Group(visible=True) as result_group:
                status_line = gr.Textbox(label="状态", interactive=False, visible=True)
                preview = gr.Markdown("等待输入...", label="清洗结果")

        with gr.Tab("📁 批量上传"):
            with gr.Row():
                batch_source = gr.Dropdown(
                    choices=["web", "note", "chat", "pdf"],
                    value="web",
                    label="来源类型",
                    scale=1,
                )
            file_input = gr.File(
                label="上传文件（支持 .txt / .md）",
                file_count="multiple",
                file_types=[".txt", ".md", "text"],
            )
            batch_btn = gr.Button("🚀 开始批量清洗", variant="primary")
            batch_summary = gr.Markdown("等待上传...")
            batch_download = gr.File(label="📦 下载结果 (ZIP)")

        with gr.Tab("📊 处理历史"):
            refresh_btn = gr.Button("🔄 刷新")
            history_md = gr.Markdown("点击刷新查看历史记录")

        # --- 事件绑定 ---
        clean_btn.click(
            fn=clean_single_text,
            inputs=[source_type, raw_input],
            outputs=[preview, gr.Textbox(), status_line],
        )

        batch_btn.click(
            fn=clean_batch_files,
            inputs=[batch_source, file_input],
            outputs=[batch_summary, batch_download],
        )

        refresh_btn.click(
            fn=list_history,
            inputs=[],
            outputs=[history_md],
        )

    return app


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"🚀 RAG 数据清洗工具启动中...")
    print(f"📌 模型: {config.model_name}")
    print(f"📌 Ollama: {config.ollama_base_url}")
    print(f"🌐 浏览器打开: http://localhost:7860")
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
```

- [ ] **Step 2: 验证 app 可导入且 UI 结构正确**

```bash
cd /home/amanannn/Projects/exobrain-air && python -c "
from app import build_ui
blocks = build_ui()
print(f'UI blocks 构建成功 (type={type(blocks).__name__})')
# 验证三个 Tab
tab_names = [t.label for t in blocks.children if hasattr(t, 'label')]
print(f'Tabs: {tab_names}')
print('✅ app.py 就绪')
"
```

Expected: UI blocks 构建成功 + 三个 Tab 名称

---

### Task 10: 集成测试

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: 创建 tests/__init__.py**

```python
"""RAG 清洗管道测试套件."""
```

- [ ] **Step 2: 创建 tests/test_pipeline.py**

```python
"""管道集成测试 - 使用 mock 避免依赖真实 Ollama."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.chunker import chunk_items, classify_chunk, make_items
from src.config import PipelineConfig
from src.exporter import _sanitize_filename, export_results
from src.validator import (
    FatalError,
    RetryableError,
    validate_batch_result,
    validate_single_result,
)


class TestChunker:
    def test_chunk_items_normal(self):
        items = [{"id": f"d{i}"} for i in range(7)]
        chunks = chunk_items(items, max_size=5)
        assert len(chunks) == 2
        assert len(chunks[0]) == 5
        assert len(chunks[1]) == 2

    def test_chunk_items_empty(self):
        assert chunk_items([], 5) == []

    def test_chunk_items_single(self):
        items = [{"id": "d0"}]
        chunks = chunk_items(items, 5)
        assert len(chunks) == 1
        assert classify_chunk(chunks[0]) == "single"

    def test_make_items_skip_blank(self):
        items = make_items(["", "  ", "hello"])
        assert len(items) == 1
        assert items[0]["raw_text"] == "hello"

    def test_make_items_has_id_and_source(self):
        items = make_items(["text"], source_type="pdf")
        assert items[0]["source_type"] == "pdf"
        assert items[0]["id"].startswith("doc_")


class TestValidator:
    def test_validate_single_valid(self):
        raw = json.dumps({
            "title": "测试", "tags": ["a"], "summary": "s",
            "cleaned_markdown": "c", "quality_flag": "valid",
        })
        r = validate_single_result(raw)
        assert r["title"] == "测试"

    def test_validate_single_invalid_doc(self):
        raw = json.dumps({
            "title": "无效文档", "tags": ["待人工审核"], "summary": "内容不足",
            "cleaned_markdown": "[内容无效] ...", "quality_flag": "invalid",
        })
        r = validate_single_result(raw)
        assert r["quality_flag"] == "invalid"

    def test_validate_single_invalid_doc_wrong_title_raises(self):
        raw = json.dumps({
            "title": "普通文档", "tags": ["待人工审核"], "summary": "内容不足",
            "cleaned_markdown": "[内容无效] ...", "quality_flag": "invalid",
        })
        with pytest.raises(RetryableError):
            validate_single_result(raw)

    def test_validate_missing_field_raises(self):
        raw = json.dumps({"title": "x"})
        with pytest.raises(RetryableError):
            validate_single_result(raw)

    def test_validate_chatty_prefix(self):
        raw = '好的，以下是处理结果：\n{"title":"测试","tags":["a"],"summary":"s","cleaned_markdown":"c","quality_flag":"valid"}'
        r = validate_single_result(raw)
        assert r["title"] == "测试"

    def test_validate_batch_success(self):
        raw = json.dumps({
            "results": [
                {"id": "d1", "title": "a", "tags": ["x"], "summary": "s", "cleaned_markdown": "c", "quality_flag": "valid"},
                {"id": "d2", "title": "b", "tags": ["y"], "summary": "t", "cleaned_markdown": "d", "quality_flag": "valid"},
            ],
            "batch_summary": "2条",
        })
        results = validate_batch_result(raw, ["d1", "d2"])
        assert len(results) == 2

    def test_validate_batch_count_mismatch(self):
        raw = json.dumps({"results": [{"id": "d1", "title": "a", "tags": ["x"], "summary": "s", "cleaned_markdown": "c", "quality_flag": "valid"}]})
        with pytest.raises(RetryableError):
            validate_batch_result(raw, ["d1", "d2"])


class TestExporter:
    def test_sanitize_filename(self):
        assert _sanitize_filename("hello") == "hello"
        assert _sanitize_filename("file:name?") == "file_name_"
        assert _sanitize_filename("a" * 100) == "a" * 60

    def test_export_results(self, tmp_path):
        results = [
            {"id": "1", "title": "Valid Doc", "tags": ["a"], "summary": "s", "cleaned_markdown": "# ok", "quality_flag": "valid"},
            {"id": "2", "title": "Partial Doc", "tags": ["b"], "summary": "p", "cleaned_markdown": "# part", "quality_flag": "partial"},
            {"id": "3", "title": "无效文档", "tags": ["待人工审核"], "summary": "bad", "cleaned_markdown": "[内容无效]", "quality_flag": "invalid"},
        ]
        stats = export_results(results, tmp_path)
        assert stats["valid"] == 1
        assert stats["partial"] == 1
        assert stats["invalid"] == 1
        assert (tmp_path / "valid").exists()
        assert (tmp_path / "partial").exists()
        assert (tmp_path / "invalid").exists()
        # partial 应有 .meta.json
        metas = list((tmp_path / "partial").glob("*.meta.json"))
        assert len(metas) == 1


class TestRetry:
    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self):
        from src.retry import with_retry

        call_count = [0]

        async def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                return "bad json"
            return json.dumps({"title": "ok", "tags": ["t"], "summary": "s", "cleaned_markdown": "c", "quality_flag": "valid"})

        result = await with_retry(fn, validate_single_result, max_retries=3, base_delay=0.01)
        assert result["title"] == "ok"
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_fatal(self):
        from src.retry import with_retry

        call_count = [0]

        async def fn():
            call_count[0] += 1
            return "always bad json"

        with pytest.raises(FatalError):
            await with_retry(fn, validate_single_result, max_retries=2, base_delay=0.01)
        assert call_count[0] == 3  # 1 initial + 2 retries


class TestConfig:
    def test_defaults(self):
        c = PipelineConfig()
        assert c.model_name == "qwen3:32b"
        assert c.temperature == 0.15
        assert c.max_retries == 3
        assert c.chunk_size == 5

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("EXOBRAIN_MODEL", "qwen3:14b")
        c = PipelineConfig()
        assert c.model_name == "qwen3:14b"
```

- [ ] **Step 3: 运行测试**

```bash
cd /home/amanannn/Projects/exobrain-air && python -m pytest tests/ -v
```

Expected: 16-17 tests all PASS

- [ ] **Step 4: 清理临时 exporter 测试产生的目录**

```bash
rm -rf /tmp/test_exporter_output
```

---

### Task 11: 端到端验证

- [ ] **Step 1: 确认所有模块可正常导入**

```bash
cd /home/amanannn/Projects/exobrain-air && python -c "
from src.config import PipelineConfig
from src.validator import RetryableError, FatalError, validate_single_result, validate_batch_result
from src.retry import with_retry
from src.client import OllamaClient
from src.chunker import chunk_items, make_items, classify_chunk
from src.exporter import export_results, create_zip
from src.cleaner import RAGCleaner
print('✅ 所有模块导入成功')
"
```

- [ ] **Step 2: 使用 mock client 测试完整管道**

```bash
cd /home/amanannn/Projects/exobrain-air && python -c "
import asyncio, json
from unittest.mock import AsyncMock, patch
from src.config import PipelineConfig
from src.chunker import make_items
from src.cleaner import RAGCleaner
from pathlib import Path

async def test():
    config = PipelineConfig()
    config.output_dir = Path('/tmp/e2e_test_output')
    import shutil
    shutil.rmtree(config.output_dir, ignore_errors=True)

    cleaner = RAGCleaner(config)

    # Mock OllamaClient 的 clean_single 和 clean_batch
    async def mock_single(content, metadata=None):
        return json.dumps({
            'title': '测试文档', 'tags': ['测试'], 'summary': 'mock结果',
            'cleaned_markdown': '# 测试', 'quality_flag': 'valid'
        }, ensure_ascii=False)

    async def mock_batch(items):
        results = []
        for it in items:
            results.append({
                'id': it['id'], 'title': f'文档_{it[\"id\"]}', 'tags': ['t'],
                'summary': 's', 'cleaned_markdown': '# doc', 'quality_flag': 'valid'
            })
        return json.dumps({'results': results, 'batch_summary': f'{len(results)}条'})

    cleaner.client.clean_single = mock_single
    cleaner.client.clean_batch = mock_batch

    # 9条数据 → 应拆为 5+4 两批
    texts = [f'text_{i}' for i in range(9)]
    items = make_items(texts)
    stats = await cleaner.clean(items)

    print(f'统计: total={stats[\"total\"]}, valid={stats[\"valid\"]}, partial={stats[\"partial\"]}, invalid={stats[\"invalid\"]}')
    assert stats['total'] == 9
    assert stats['valid'] == 9
    assert Path(stats['zip_path']).exists()
    print('ZIP:', stats['zip_path'])
    print('✅ 端到端测试通过')

    shutil.rmtree(config.output_dir, ignore_errors=True)

asyncio.run(test())
"
```

Expected: `✅ 端到端测试通过`

---

### Task 12: 最终检查清单

- [ ] **检查 1: 运行完整测试套件**

```bash
cd /home/amanannn/Projects/exobrain-air && python -m pytest tests/ -v
```

- [ ] **检查 2: 验证 prompts/ 目录未被修改**

```bash
cd /home/amanannn/Projects/exobrain-air && git status prompts/ 2>/dev/null || echo "(非 git 仓库，人工确认 prompts/ 文件未被编辑工具修改即可)"
```

- [ ] **检查 3: 确认项目结构完整**

```bash
cd /home/amanannn/Projects/exobrain-air && find . -type f -not -path './.git/*' -not -path './__pycache__/*' -not -path '*.pyc' -not -path './output/*' -not -path './docs/*' | sort
```

Expected: 所有 `src/` 模块、`app.py`、`pyproject.toml`、`tests/` 均已创建
