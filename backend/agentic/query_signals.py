"""
query_signals.py — 查询信号的纯规则分析器 (P2-B6)

目标: 给 query 提取一组"廉价信号" (无 LLM, < 1ms),
让下游 (B7 路由, B1 动态 RRF, B6 OTel attributes) 都能用。

设计原则 (YAGNI):
- 只做"启发式就能判"的事; 复杂的 NLU / 命名实体识别属于 P3+
- 所有信号都是 optional, 信号缺失不应让上层崩溃
- signals 是 dataclass 而非 dict: 类型安全, OTel attribute 时再 to_dict()
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict


# 中文代词 / 英文代词 (粗略, 覆盖大部分对话场景)
_PRONOUN_RE = re.compile(
    r"\b(我|你|他|她|它|们|这|那|这些|那些|此|本|该|其)"
    r"|\b(I|you|he|she|it|they|we|this|that|these|those|such|said)\b",
    re.IGNORECASE,
)

# multi-hop 关键词 (中英文)
_MULTIHOP_RE = re.compile(
    r"(之后|然后|并且|以及|对比|相比|比较|影响|关系|交叉|结合|综合|"
    r"and then|after|afterward|combine|compare|contrast|impact|"
    r"relationship|cross|merge|synthesize)",
    re.IGNORECASE,
)

# 英文大写词 / 中文专名 (粗略启发: 连续 2-3 个大写字母开头的词)
_UPPER_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
# 中文专名 (粗略: 2-4 个中文字符后跟专名标志: 公司/集团/先生/女士/教授/博士 等)
_CN_ENTITY_SUFFIXES = (
    "公司", "集团", "大学", "学院", "银行", "保险", "证券",
    "先生", "女士", "教授", "博士", "总裁", "经理", "主任",
)


@dataclass
class QuerySignals:
    """
    查询信号集合 — 路由 + 检索 + 编排决策的"廉价 hint"

    所有字段都是 cheap heuristic, O(N) 时间, < 1ms
    """
    has_pronoun: bool = False
    entity_count: int = 0
    is_multi_hop: bool = False
    query_length: int = 0
    has_quote: bool = False

    def to_dict(self) -> dict:
        """OTel attribute / JSON 序列化友好版本"""
        return asdict(self)

    def complexity_hint(self) -> str:
        """
        基于 signals 给一个"无 LLM 启发式"复杂度 hint, 供 routing 兜底
        - 长查询 + multi-hop + 多实体 → complex
        - 有代词 (依赖上下文) → moderate (默认走 ReAct)
        - 短且无 multi-hop → simple
        """
        if self.is_multi_hop and (self.entity_count >= 2 or self.query_length >= 30):
            return "complex"
        if self.has_pronoun and self.query_length < 30:
            return "moderate"  # 代词 → 需查 history, 走 ReAct 更稳
        if self.query_length <= 15 and self.entity_count <= 1:
            return "simple"
        return "moderate"


class QueryAnalyzer:
    """
    P2-B6: 纯规则查询分析器 (无 LLM, 无外部依赖)

    用法:
        analyzer = QueryAnalyzer()
        signals = analyzer.analyze("他和她的合同")
        # signals.has_pronoun == True
        # signals.complexity_hint() == "moderate"
    """

    def analyze(self, query: str) -> QuerySignals:
        """
        分析查询文本, 返回 QuerySignals

        性能: 4 个正则 + 字符串计数, 在 1000 字符 query 上 < 0.5ms
        """
        if not query:
            return QuerySignals()

        has_pronoun = bool(_PRONOUN_RE.search(query))
        is_multi_hop = bool(_MULTIHOP_RE.search(query))
        has_quote = ('"' in query) or ("'" in query) or ('"' in query) or ('"' in query)

        # 实体数: 英文大写实体 + 中文专名 (粗略)
        en_entities = _UPPER_ENTITY_RE.findall(query)
        cn_entities = []
        for suffix in _CN_ENTITY_SUFFIXES:
            cn_entities.extend(re.findall(rf"[\u4e00-\u9fff]{{2,4}}{suffix}", query))
        entity_count = len(set(en_entities)) + len(set(cn_entities))

        return QuerySignals(
            has_pronoun=has_pronoun,
            entity_count=entity_count,
            is_multi_hop=is_multi_hop,
            query_length=len(query),
            has_quote=has_quote,
        )
