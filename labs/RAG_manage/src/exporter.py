"""Markdown 文件输出器 - 按 quality_flag 分目录写入."""

from __future__ import annotations

import json
import logging
import re
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
