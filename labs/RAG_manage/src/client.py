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
