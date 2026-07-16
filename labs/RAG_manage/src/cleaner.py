"""管道编排器 - 组装分片/客户端/校验/重试/导出全流程."""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.chunker import chunk_items, classify_chunk
from src.client import OllamaClient
from src.config import PipelineConfig
from src.exporter import create_zip, export_results
from src.retry import with_retry
from src.validator import (
    FatalError,
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
            except FatalError:
                # 单条失败 → 生成 invalid 结果继续
                logger.error("批次 %d 清洗失败", i + 1)
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

        async def call_fn() -> str:
            return await self.client.clean_single(
                content=item["raw_text"],
                metadata={"source_type": item.get("source_type", "web"), "timestamp": ""},
            )

        def validate_fn(raw: str) -> dict:
            return validate_single_result(raw, expected_id=item.get("id"))  # type: ignore[return-value]

        return await with_retry(  # type: ignore[return-value]
            call_fn,
            validate_fn,
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
        )

    async def _clean_batch_with_retry(self, chunk: list[dict]) -> list[dict]:
        """带重试的批量清洗."""
        input_ids = [item["id"] for item in chunk]

        async def call_fn() -> str:
            return await self.client.clean_batch(chunk)

        def validate_fn(raw: str) -> list[dict]:
            return validate_batch_result(raw, input_ids)  # type: ignore[return-value]

        return await with_retry(  # type: ignore[return-value]
            call_fn,
            validate_fn,
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
        )

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
