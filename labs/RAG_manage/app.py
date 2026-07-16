#!/usr/bin/env python3
"""RAG 数据清洗工具 - Gradio 前端入口.

用法:
    python app.py
    # 浏览器自动打开 http://localhost:7860
"""

from __future__ import annotations

import asyncio
import logging
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

    async def _run():
        return await cleaner.clean(items)

    try:
        stats = asyncio.run(_run())
    except Exception as e:
        logger.exception("批量清洗失败")
        return f"❌ 清洗失败: {e}", ""

    # 构建摘要
    results = stats.get("results", [])
    zip_path = stats.get("zip_path", "")

    lines = [
        "## 📊 处理完成",
        "",
        "| 指标 | 数量 |",
        "|------|------|",
        f"| 总计 | {stats.get('total', 0)} |",
        f"| ✅ valid | {stats.get('valid', 0)} |",
        f"| ⚠️ partial | {stats.get('partial', 0)} |",
        f"| ❌ invalid | {stats.get('invalid', 0)} |",
        "",
        "---",
        "",
        "### 详细结果",
        "",
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

    lines = ["## 📊 处理历史", "", "| 时间 | 大小 |", "|------|------|"]

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
    with gr.Blocks(title="RAG 数据清洗工具") as app:
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
            status_line = gr.Textbox(label="状态", interactive=False)
            preview = gr.Markdown("等待输入...")

            clean_btn.click(
                fn=clean_single_text,
                inputs=[source_type, raw_input],
                outputs=[preview, gr.Textbox(), status_line],
            )

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

            batch_btn.click(
                fn=clean_batch_files,
                inputs=[batch_source, file_input],
                outputs=[batch_summary, batch_download],
            )

        with gr.Tab("📊 处理历史"):
            refresh_btn = gr.Button("🔄 刷新")
            history_md = gr.Markdown("点击刷新查看历史记录")

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
    print("🚀 RAG 数据清洗工具启动中...")
    print(f"📌 模型: {config.model_name}")
    print(f"📌 Ollama: {config.ollama_base_url}")
    print("🌐 浏览器打开: http://localhost:7860")
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True, css=CSS, theme=gr.themes.Soft())
