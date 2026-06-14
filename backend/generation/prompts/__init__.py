"""
backend.generation.prompts — Prompt 模板加载器
================================================================================
技术决策 (P2 阶段):
- Prompt 全部抽到本目录的 v*.yaml 文件，git-tracked，方便 review/diff/rollback。
- PromptBuilder 在启动时 load_prompts()，version 由 PROMPT_VERSION 控制。
- 每次 LLM 调用时把 prompt_hash + template_version 写入 trace，
  CI 中 eval diff gate 用 prompt_hash 区分"prompt 改 vs 数据改"对 NDCG 的影响。
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent
_DEFAULT_VERSION = "v1.0.0"
_cache: dict[str, dict[str, Any]] = {}


def get_prompt_version() -> str:
    """从环境变量 PROMPT_VERSION 读取，默认 v1.0.0"""
    return os.getenv("PROMPT_VERSION", _DEFAULT_VERSION)


def load_prompts(version: str | None = None) -> dict[str, Any]:
    """
    加载指定版本的 prompt 模板

    Returns:
        dict 包含 keys: system, requirements, structured_template, version, ...
    """
    version = version or get_prompt_version()
    if version in _cache:
        return _cache[version]

    path = _PROMPTS_DIR / f"{version}.yaml"
    if not path.exists():
        # Fallback: 用 default
        logger.warning(
            f"Prompt version '{version}' not found at {path}; falling back to {_DEFAULT_VERSION}"
        )
        path = _PROMPTS_DIR / f"{_DEFAULT_VERSION}.yaml"
        version = _DEFAULT_VERSION

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    _cache[version] = data
    return data


def get_prompt_hash(prompts: dict[str, Any]) -> str:
    """计算 prompt 内容的 SHA256，用于 trace / eval diff gate"""
    canonical = f"{prompts.get('version', '')}\n{prompts.get('system', '')}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def get_prompts_with_hash() -> tuple[dict[str, Any], str]:
    """convenience: load + hash"""
    prompts = load_prompts()
    return prompts, get_prompt_hash(prompts)

