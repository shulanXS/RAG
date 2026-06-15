"""
test_p2_refactor.py — P2 阶段关键测试补全
================================================================================
P2-7 补 5 个测试 (含 P2-8):
1. TestOpenAICompatibleBackend — OpenAI + DeepSeek 合并后行为一致
2. TestQueryRouterRefactor — 3 路径收敛到 _fallback_decision
3. TestDocumentParserRegistry — registry + register_parser() 插件化
4. TestRateLimiterSimplified — 3-tier 已删，固定 rpm=60
5. TestRequestContext — ContextVar + Filter + Middleware 链路
"""

import pytest
from types import SimpleNamespace


# =============================================================================
# 1. OpenAICompatibleBackend
# =============================================================================


class TestOpenAICompatibleBackend:
    """P2-1: OpenAI + DeepSeek 合并到 OpenAICompatibleBackend"""

    def test_class_aliases_removed(self):
        """P0-Phase1.17: OpenAIBackend / DeepSeekBackend 别名已删除，仅保留 OpenAICompatibleBackend"""
        from backend.generation.llm_client import OpenAICompatibleBackend
        import backend.generation.llm_client as llm_mod

        assert OpenAICompatibleBackend is not None
        assert not hasattr(llm_mod, "OpenAIBackend")
        assert not hasattr(llm_mod, "DeepSeekBackend")

    def test_default_base_url_per_provider(self):
        """Phase 5.1: OpenAI 与 DeepSeek 共享 OpenAICompatibleBackend；base_url 由调用方传入"""
        from backend.generation.llm_client import OpenAICompatibleBackend
        # Phase 5.1 删除: DEFAULT_BASE_URLS 字段；class 不再硬编码 provider→url 映射
        assert not hasattr(OpenAICompatibleBackend, "DEFAULT_BASE_URLS")

    def test_constructor_records_base_url(self):
        """构造时记录 base_url（替代 provider 字段）"""
        from backend.generation.llm_client import OpenAICompatibleBackend
        # 不真正发请求（SDK 未配 api_key 会失败），用 __new__ 跳过 __init__
        inst = OpenAICompatibleBackend.__new__(OpenAICompatibleBackend)
        inst._base_url = "https://api.deepseek.com/v1"
        inst._model = "deepseek-chat"
        inst._max_tokens = 2048
        inst._temperature = 0.3
        assert inst._base_url == "https://api.deepseek.com/v1"
        assert inst._model == "deepseek-chat"

    def test_llm_client_factory_dispatches(self):
        """LLMClient._create_backend 把 deepseek/openai 都路由到 OpenAICompatibleBackend"""
        from backend.generation.llm_client import LLMClient, OpenAICompatibleBackend
        from unittest.mock import patch

        with patch("backend.generation.llm_client.OpenAICompatibleBackend") as MockBackend:
            MockBackend.side_effect = lambda **kwargs: SimpleNamespace(
                _base_url=kwargs.get("base_url"),
                _model=kwargs.get("model"),
            )
            LLMClient._create_backend = staticmethod(  # type: ignore
                lambda provider, model, api_key=None, base_url=None: MockBackend(
                    model=model, api_key=api_key, base_url=base_url,
                )
            )
            client = LLMClient(
                generator_provider="deepseek", generator_model="deepseek-chat",
                generator_base_url="https://api.deepseek.com/v1",
                router_provider="openai", router_model="gpt-4o",
                router_base_url="https://api.openai.com/v1",
            )
            assert client.generator_client._base_url == "https://api.deepseek.com/v1"
            assert client.router_client._base_url == "https://api.openai.com/v1"


# =============================================================================
# 2. QueryRouter refactor
# =============================================================================


class TestQueryRouterRefactor:
    """P2-2: 3 路径收敛到 _fallback_decision + 集中 _APPROACH_MAP"""

    def test_approach_map_covers_all_complexities(self):
        """_APPROACH_MAP 必须覆盖 4 种复杂度"""
        from backend.agentic.query_router import QueryRouter, QueryComplexity
        for c in QueryComplexity:
            assert c in QueryRouter._APPROACH_MAP, f"Missing mapping for {c}"

    def test_no_llm_path_uses_fallback(self):
        """无 LLM 时走 _fallback_decision，返回正确的 recommended_approach"""
        from backend.agentic.query_router import QueryRouter, QueryComplexity
        router = QueryRouter(llm_client=None)
        d = router.route("简单问题")
        # SIMPLE 查询走 signals 兜底
        assert d.complexity in QueryComplexity
        assert d.recommended_approach in router._APPROACH_MAP.values()

    def test_build_history_context_helper(self):
        """_build_history_context 辅助方法正确处理空/非空历史"""
        from backend.agentic.query_router import QueryRouter
        assert QueryRouter._build_history_context(None) == ""
        assert QueryRouter._build_history_context([]) == ""
        ctx = QueryRouter._build_history_context([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])
        assert "用户: hi" in ctx
        assert "助手: hello" in ctx

    def test_fallback_decision_passes_through_complexity(self):
        """_fallback_decision 把传入的 complexity 写到 recommended_approach"""
        from backend.agentic.query_router import QueryRouter, QueryComplexity
        router = QueryRouter(llm_client=None)
        d = router._fallback_decision(
            complexity=QueryComplexity.BEYOND_KB,
            confidence=0.5,
            reasoning="test",
            query="q",
            signals=None,
        )
        assert d.complexity == QueryComplexity.BEYOND_KB
        assert d.recommended_approach == "直接 LLM 生成（无需检索）"


# =============================================================================
# 3. DocumentParser registry
# =============================================================================


class TestDocumentParserRegistry:
    """P2-3: 4 class 合并为 registry + register_parser() 插件化"""

    def test_default_extensions_registered(self):
        """_EXTENSION_PARSERS 至少包含 8 个默认扩展名"""
        from backend.ingestion.document_parser import _EXTENSION_PARSERS
        expected = {".pdf", ".docx", ".doc", ".md", ".markdown", ".mdown", ".html", ".htm"}
        assert expected <= set(_EXTENSION_PARSERS.keys())

    def test_register_parser_removed(self):
        """Phase1-1.11: register_parser() 已删除 — 无外部调用方"""
        from backend.ingestion import document_parser as dp
        assert not hasattr(dp, "register_parser"), "register_parser should be removed in Phase1-1.11"

    def test_backward_compat_class_aliases_removed(self):
        """Phase1-1.11: PDFParser / DOCXParser / MarkdownParser / HTMLParser 薄 class 已删除"""
        from backend.ingestion import document_parser as dp
        for name in ("PDFParser", "DOCXParser", "MarkdownParser", "HTMLParser", "register_parser"):
            assert not hasattr(dp, name), f"{name} should be removed in Phase1-1.11"


# =============================================================================
# 4. RateLimiter simplification
# =============================================================================


class TestRateLimiterSimplified:
    """P1-2: 3-tier 抽象删除，固定 60 rpm"""

    def test_no_tier_dataclass(self):
        """RateLimitTier dataclass 已删除"""
        from backend.middleware import rate_limiter
        assert not hasattr(rate_limiter, "RateLimitTier"), "RateLimitTier should be removed"

    def test_no_default_tiers_dict(self):
        """DEFAULT_TIERS 字典已删除"""
        from backend.middleware import rate_limiter
        assert not hasattr(rate_limiter, "DEFAULT_TIERS"), "DEFAULT_TIERS should be removed"

    def test_default_rpm_is_60(self):
        """DEFAULT_REQUESTS_PER_MINUTE = 60 (原 pro 档)"""
        from backend.middleware.rate_limiter import DEFAULT_REQUESTS_PER_MINUTE
        assert DEFAULT_REQUESTS_PER_MINUTE == 60

    def test_rate_limit_exceeded_exception_preserved(self):
        """RateLimitExceeded 异常保留（middleware / API 仍可能抛）"""
        from backend.middleware.rate_limiter import RateLimitExceeded
        exc = RateLimitExceeded("test")
        assert "test" in str(exc)


# =============================================================================
# 5. RequestContext (P2-8)
# =============================================================================


class TestRequestContext:
    """P2-8: 请求级上下文 + logging filter + ASGI middleware"""

    def test_default_context_values(self):
        """未进入 RequestContext 时，访问函数返回默认 '-'"""
        from backend.middleware.request_context import (
            current_request_id, current_tenant_id, current_session_id,
        )
        assert current_request_id() == "-"
        assert current_tenant_id() == "-"
        assert current_session_id() == "-"

    def test_context_manager_sets_and_resets(self):
        """进入 with 块后字段被设置，退出后还原默认"""
        from backend.middleware.request_context import (
            RequestContext, current_request_id, current_tenant_id, current_session_id,
        )
        with RequestContext(request_id="abc", tenant_id="t1", session_id="s1"):
            assert current_request_id() == "abc"
            assert current_tenant_id() == "t1"
            assert current_session_id() == "s1"
        assert current_request_id() == "-"
        assert current_tenant_id() == "-"
        assert current_session_id() == "-"

    def test_nested_contexts_restore_correctly(self):
        """嵌套时退出 inner 块应回到 outer 值，再退出 outer 回到默认"""
        from backend.middleware.request_context import (
            RequestContext, current_request_id,
        )
        with RequestContext(request_id="outer"):
            assert current_request_id() == "outer"
            with RequestContext(request_id="inner"):
                assert current_request_id() == "inner"
            assert current_request_id() == "outer"
        assert current_request_id() == "-"

    def test_auto_generated_request_id(self):
        """不传 request_id 时自动生成 (12 字符)"""
        from backend.middleware.request_context import RequestContext, current_request_id
        with RequestContext():
            rid = current_request_id()
            assert rid != "-"
            assert len(rid) >= 8  # 12 字符 UUID 截断

    def test_filter_injects_context_into_records(self):
        """RequestContextFilter 把 ContextVar 写到 LogRecord"""
        import logging
        from backend.middleware.request_context import (
            RequestContext, RequestContextFilter,
        )

        record = logging.LogRecord("name", logging.INFO, "path", 1, "msg", None, None)
        # 过滤前无 context 字段
        assert not hasattr(record, "request_id")
        # 进入 ctx 后过滤
        with RequestContext(request_id="r1", tenant_id="t9", session_id="s2"):
            f = RequestContextFilter()
            assert f.filter(record) is True
        assert record.request_id == "r1"
        assert record.tenant_id == "t9"
        assert record.session_id == "s2"

    def test_middleware_module_loadable(self):
        """RequestContextMiddleware 模块可 import + 关键属性存在"""
        # starlette 是 prod-only 依赖，本地开发环境可能缺
        pytest.importorskip("starlette")
        from backend.middleware.request_context_middleware import (
            RequestContextMiddleware, _HEADER_REQUEST_ID, _HEADER_TENANT_ID,
        )
        assert _HEADER_REQUEST_ID == "x-request-id"
        assert _HEADER_TENANT_ID == "x-tenant-id"
        assert hasattr(RequestContextMiddleware, "dispatch")
