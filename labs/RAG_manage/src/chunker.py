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
