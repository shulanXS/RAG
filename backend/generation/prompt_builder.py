"""
prompt_builder.py — Prompt 模板与上下文组装
================================================================================
技术决策记录:
- prompt 内容从代码搬到 backend/generation/prompts/v{version}.yaml，git-tracked。
  PromptBuilder 在构造时 load_prompts()，version 由 PROMPT_VERSION 控制。
  prompt_hash 暴露给 trace，CI 跑 eval 时记录"prompt 改了 → NDCG 涨/跌"。
- 引用标注格式: 在 Prompt 中明确要求每个关键陈述附带引用标记。
- 置信度评估: 让 LLM 对答案的置信度做自评（high/medium/low/insufficient）。
- 这些设计在 RAGAS 评估中对应 Context Precision 和 Faithfulness 指标。
"""

from __future__ import annotations

from typing import Any, Literal

from backend.generation.prompts import load_prompts, get_prompt_hash


class PromptBuilder:
    """
    Prompt 构建器

    技术要点:
    - build_context(): 将检索到的 chunks 组装为带引用标注的上下文
    - build_prompt(): 将用户查询和上下文组装为完整的 LLM Prompt
    - 引用格式: [来源: doc_id / section_path]
    - prompt 文本从 backend/generation/prompts/v{version}.yaml 加载，便于 review/diff/rollback

    提示词设计原则:
    1. 角色定义: 明确 AI 的角色和职责
    2. 上下文注入: 检索结果作为「证据」而非「参考」
    3. 约束条件: 不要编造、不要超出上下文范围
    """

    def __init__(self):
        prompts, self._prompt_hash = load_prompts(), get_prompt_hash(load_prompts())
        self._system_prompt = prompts.get("system", "").strip()
        self._requirements = prompts.get("requirements", {}) or {}
        self._prompt_version = prompts.get("version", "unknown")

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    @property
    def prompt_hash(self) -> str:
        return self._prompt_hash

    def build_context(self, chunks: list[dict]) -> str:
        """
        将检索到的 chunks 组装为带引用标注的上下文文本

        技术要点:
        - 每个 chunk 都标注来源（doc_id + section_path）
        - chunks 按相关性得分降序排列（由 Reranker 保证）
        - 文本被截断到合理长度（避免超出 LLM 上下文）
        """
        if not chunks:
            return "（未检索到相关文档）"

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            doc_id = chunk.get("doc_id", "unknown")
            section = chunk.get("section_path", "")
            source_label = f"{doc_id}"
            if section:
                source_label += f" / {section}"

            text = chunk.get("text", "")
            # 截断到 500 字符，避免上下文过长
            if len(text) > 500:
                text = text[:500] + "..."

            context_parts.append(
                f"[{i}] 来源: {source_label}\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def build_prompt(
        self,
        query: str,
        context: str,
        *,
        require_citations: bool = True,
        require_confidence: bool = True,
    ) -> str:
        """
        构建完整的 LLM Prompt

        Args:
            query: 用户查询
            context: 检索到的上下文（已格式化为 build_context 输出）
            require_citations: 是否要求引用标注
            require_confidence: 是否要求置信度自评
        """
        prompt_parts = [self._system_prompt]

        prompt_parts.append(f"\n\n[检索到的上下文]\n{context}")

        prompt_parts.append(f"\n\n[用户问题]\n{query}")

        # 构建回答要求
        requirements = []
        if require_citations:
            requirements.append(self._requirements.get("with_citations", "").strip())
        if require_confidence:
            requirements.append(self._requirements.get("with_confidence", "").strip())

        prompt_parts.append("\n\n" + "\n".join(requirements))

        return "\n".join(prompt_parts)
