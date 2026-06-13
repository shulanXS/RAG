"""
streaming.py — LLM 流式输出
================================================================================
技术决策记录:
- 流式输出可将首 token 响应时间从 2-3s 降低到 <500ms
- Anthropic: 使用 messages.stream() API
- OpenAI/DeepSeek: 使用 chat.completions.create(stream=True)
- 适用于 RAG 生成、对话等长文本场景
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class LLMStreamer:
    """
    LLM 流式输出生成器

    支持 Anthropic / OpenAI / DeepSeek 流式 API
    """

    def __init__(self):
        pass

    async def stream_anthropic(
        self,
        prompt: str,
        model: str = "claude-3-7-sonnet-20250620",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """Anthropic 流式输出"""
        try:
            import anthropic
        except ImportError:
            raise ImportError("需要安装 anthropic: pip install anthropic")

        client = anthropic.Anthropic()
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    async def stream_openai(
        self,
        prompt: str,
        model: str = "gpt-4o",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """OpenAI 流式输出"""
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def stream_deepseek(
        self,
        prompt: str,
        model: str = "deepseek-chat",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
    ) -> AsyncIterator[str]:
        """DeepSeek 流式输出（OpenAI 兼容 API）"""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def stream(
        self,
        prompt: str,
        provider: str = "anthropic",
        model: str = "claude-3-7-sonnet-20250620",
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        统一流式输出接口

        Args:
            prompt: 用户提示
            provider: 提供商 (anthropic | openai | deepseek)
            model: 模型名称
        """
        if provider == "anthropic":
            async for token in self.stream_anthropic(prompt, model=model, **kwargs):
                yield token
        elif provider == "openai":
            async for token in self.stream_openai(prompt, model=model, **kwargs):
                yield token
        elif provider == "deepseek":
            async for token in self.stream_deepseek(prompt, model=model, **kwargs):
                yield token
        else:
            raise ValueError(f"不支持的流式 provider: {provider}")
