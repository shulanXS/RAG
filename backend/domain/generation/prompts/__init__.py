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
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent
_DEFAULT_VERSION = "v1.0.0"


def get_prompt_version() -> str:
    """从环境变量 PROMPT_VERSION 读取，默认 v1.0.0"""
    return os.getenv("PROMPT_VERSION", _DEFAULT_VERSION)


@lru_cache(maxsize=8)
def load_prompts(version: str | None = None) -> dict[str, Any]:
    """
    加载指定版本的 prompt 模板（按 version 缓存，P1-7: 替换原模块级 _cache dict）。

    P1-7: 原实现用模块级 `_cache: dict` 缓存，多 worker / reload 代码时
    _cache 不共享，反成一致性 bug 源。lru_cache 装饰函数按 version
    隔离，且 Python 多进程模式下每个 worker 独立 lru_cache 实例，行为一致。

    Returns:
        dict 包含 keys: system, requirements, structured_template, version, ...
    """
    version = version or get_prompt_version()
    path = _PROMPTS_DIR / f"{version}.yaml"
    if not path.exists():
        logger.warning(
            f"Prompt version '{version}' not found at {path}; falling back to {_DEFAULT_VERSION}"
        )
        path = _PROMPTS_DIR / f"{_DEFAULT_VERSION}.yaml"

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_prompt_hash(prompts: dict[str, Any]) -> str:
    """
    计算 prompt 内容的 SHA256，用于 trace / eval diff gate。

    P0-2: 原实现只用 `version + system` 拼接，改 `requirements / user_template` 等
    字段时 hash 不变 → CI eval diff gate 误判 "prompt 未变"。
    修复：对整个 dict 做稳定序列化（sort_keys + ensure_ascii=False），
    保证所有字段改动都能反映在 hash 上。
    """
    canonical = json.dumps(prompts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def get_prompts_with_hash() -> tuple[dict[str, Any], str]:
    """convenience: load + hash"""
    prompts = load_prompts()
    return prompts, get_prompt_hash(prompts)
