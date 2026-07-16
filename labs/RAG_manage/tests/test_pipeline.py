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
