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
