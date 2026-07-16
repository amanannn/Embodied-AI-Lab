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
