"""
web_search.py — Web Search Tool (DuckDuckGo lite) (P2.4)
================================================================================
技术决策:
- 作品集 demo 用 DuckDuckGo lite HTML 端点（不需要 API key）
- 真实生产应接 SerpAPI / Bing / Tavily，并设置成本/速率控制
- 带 fail-soft: 联网失败时返回 1 条 stub 答案，Agent 不会卡死
"""

from __future__ import annotations

import logging
from urllib.parse import quote_plus

from backend.agentic.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Web search tool wrapping DuckDuckGo's HTML lite endpoint."""

    def __init__(self, max_results: int = 5, timeout: float = 5.0):
        super().__init__(
            name="web_search",
            description="在公网检索最新信息（股价 / 新闻 / 文档），返回 top-K 摘要列表。",
        )
        self._max_results = max_results
        self._timeout = timeout

    def _get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索 query，例如 'AAPL stock price' 或 '今日上证指数'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回的最大结果数（默认 5）",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, query: str, max_results: int | None = None) -> ToolResult:
        n = max_results or self._max_results
        if not query or not query.strip():
            return ToolResult(success=False, error="query is empty")

        try:
            import httpx
        except ImportError:
            return ToolResult(
                success=False,
                error="httpx not installed; pip install httpx",
            )

        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Bot/1.0)"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                html = resp.text
        except Exception as e:
            logger.warning(f"WebSearch 网络失败: {e}; 返回 stub")
            return ToolResult(
                success=True,
                result=[{
                    "title": "(web unavailable)",
                    "snippet": f"无法访问公网以验证 '{query}'；Agent 应在答案中说明此限制。",
                    "url": "",
                }],
                metadata={"query": query, "error": str(e), "stub": True},
            )

        # 简易 HTML 解析：抓 result__a 链接 + result__snippet 摘要
        results: list[dict] = []
        try:
            from html.parser import HTMLParser

            class ResultParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._in_title = False
                    self._in_snippet = False
                    self._buffer = ""
                    self._title: str = ""
                    self._snippet: str = ""
                    self._href: str = ""
                    self._stage = "idle"

                def handle_starttag(self, tag, attrs):
                    a = dict(attrs)
                    if tag == "a" and "result__a" in a.get("class", ""):
                        self._stage = "title"
                        self._buffer = ""
                        self._in_title = True
                        self._href = a.get("href", "")
                    elif tag == "a" and "result__url" in a.get("class", ""):
                        self._href = a.get("href", "")
                    elif tag == "td" and "result__snippet" in a.get("class", ""):
                        self._stage = "snippet"
                        self._buffer = ""
                        self._in_snippet = True

                def handle_data(self, data):
                    if self._in_title or self._in_snippet:
                        self._buffer += data

                def handle_endtag(self, tag):
                    if tag == "a" and self._in_title:
                        self._title = self._buffer.strip()
                        self._in_title = False
                    elif tag == "td" and self._in_snippet:
                        self._snippet = self._buffer.strip()
                        self._in_snippet = False
                        if self._title and self._snippet:
                            results.append({
                                "title": self._title,
                                "snippet": self._snippet,
                                "url": self._href,
                            })
                            self._title = ""
                            self._snippet = ""
                            self._href = ""
                            self._stage = "idle"

            parser = ResultParser()
            parser.feed(html)
        except Exception as e:
            logger.warning(f"WebSearch 解析失败: {e}")
            return ToolResult(success=False, error=f"HTML parse error: {e}", result=[])

        return ToolResult(
            success=True,
            result=results[:n],
            metadata={"query": query, "count": len(results)},
        )
