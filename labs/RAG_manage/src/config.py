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
