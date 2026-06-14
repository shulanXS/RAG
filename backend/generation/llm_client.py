"""
llm_client.py — 统一 LLM 接口
================================================================================
技术决策记录:
- 统一接口: 不管底层是 Anthropic/OpenAI/Google/DeepSeek，对外暴露一致的 generate() 接口。
- 分层使用: Router 用轻量模型，Generator 用主力模型。分层策略节省 60-70% LLM 成本。
- DeepSeek: 兼容 OpenAI API 格式，成本约为 Claude/GPT-4 的 1/10-1/20。
- Structured Output: 使用各 SDK 的原生功能（Pydantic + json_schema），
  而不是提示词工程。消除 json.loads() 解析的脆弱性。
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Literal, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMBackend(ABC):
    """LLM 后端抽象基类"""

    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> str:
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...

    async def generate_stream_async(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        流式生成，yield 每个 token 片段。
        默认实现：调用 generate_async 返回完整结果后 yield。
        子类可覆盖以提供真正的流式。
        """
        result = await self.generate_async(prompt, **kwargs)
        yield result


class AnthropicBackend(LLMBackend):
    """Anthropic (Claude) 后端"""

    def __init__(
        self,
        model: str = "claude-3-7-sonnet-20250620",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _record_usage(self, usage: Any, mc: Any) -> None:
        """记录 token 使用量到 metrics"""
        try:
            mc.record_llm_tokens(
                self._model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
        except Exception:
            pass

    async def generate_async(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        system = kwargs.get("system", None)
        structured_schema = kwargs.get("structured_schema", None)
        metrics_collector = kwargs.get("_metrics_collector", None)

        def _call():
            extra = {}
            if structured_schema:
                extra["output"] = {"type": "json_object", "schema": structured_schema}
            return self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                **extra,
            )

        message = await asyncio.to_thread(_call)
        if metrics_collector:
            self._record_usage(message.usage, metrics_collector)
        return message.content[0].text

    async def generate_stream_async(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        P2.1: Anthropic 真流式 — 使用 messages.stream() API。
        之前 LLMClient.generate_stream_async 调用基类默认实现，
        实际是一句话一次性 yield（不是 token 级流）。
        """
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        system = kwargs.get("system", None)

        from anthropic import AsyncAnthropic

        if not hasattr(self, "_async_client") or self._async_client is None:
            self._async_client = AsyncAnthropic()

        try:
            # 关键 API：messages.stream 才是真正的 SSE 流
            async with self._async_client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
        except AttributeError:
            # SDK 版本太老，messages.stream 不可用 → 退到非流式
            result = await self.generate_async(prompt, **kwargs)
            if result:
                yield result

    def generate(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        system = kwargs.get("system", None)

        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


class OpenAIBackend(LLMBackend):
    """OpenAI 后端"""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        from openai import AsyncOpenAI, OpenAI
        self._client = OpenAI()
        self._async_client = AsyncOpenAI()
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _record_usage(self, usage: Any, mc: Any) -> None:
        try:
            mc.record_llm_tokens(
                self._model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            )
        except Exception:
            pass

    async def generate_async(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        structured_schema = kwargs.get("structured_schema", None)
        metrics_collector = kwargs.get("_metrics_collector", None)

        create_kwargs = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if structured_schema:
            create_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": structured_schema},
            }

        response = await self._async_client.chat.completions.create(**create_kwargs)
        if metrics_collector and response.usage:
            self._record_usage(response.usage, metrics_collector)
        return response.choices[0].message.content or ""

    def generate(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def generate_stream_async(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """OpenAI 流式生成"""
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)

        stream = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class GoogleGenAIBackend(LLMBackend):
    """Google Gemini 后端"""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        from google import genai
        self._client = genai.Client()
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _record_usage(self, response: Any, mc: Any) -> None:
        try:
            usage = response.usage_metadata
            mc.record_llm_tokens(
                self._model,
                input_tokens=usage.prompt_token_count,
                output_tokens=usage.candidates_token_count,
            )
        except Exception:
            pass

    async def generate_async(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        structured_schema = kwargs.get("structured_schema", None)
        metrics_collector = kwargs.get("_metrics_collector", None)

        config = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if structured_schema:
            config["response_schema"] = structured_schema
            config["response_mime_type"] = "application/json"

        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=prompt,
            config=config,
        )
        if metrics_collector:
            self._record_usage(response, metrics_collector)
        return response.text

    def generate(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return response.text


class DeepSeekBackend(LLMBackend):
    """
    DeepSeek 后端

    技术决策:
    - DeepSeek API 兼容 OpenAI 格式，使用 openai 包 + 自定义 base_url
    - 支持 deepseek-chat (主力模型) 和 deepseek-coder (代码场景)
    - 成本仅为 Claude/GPT-4 的 1/10-1/20，适合大规模调用
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
    ):
        from openai import AsyncOpenAI, OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _record_usage(self, usage: Any, mc: Any) -> None:
        try:
            mc.record_llm_tokens(
                self._model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
            )
        except Exception:
            pass

    async def generate_async(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        structured_schema = kwargs.get("structured_schema", None)
        metrics_collector = kwargs.get("_metrics_collector", None)

        create_kwargs = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if structured_schema:
            create_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"Name": "output", "schema": structured_schema},
            }

        response = await self._async_client.chat.completions.create(**create_kwargs)
        if metrics_collector and response.usage:
            self._record_usage(response.usage, metrics_collector)
        return response.choices[0].message.content or ""

    def generate(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def generate_stream_async(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """DeepSeek 流式生成（OpenAI 兼容 API）"""
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)

        stream = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LLMClient:
    """
    LLM 统一客户端

    设计模式: 门面模式 + 工厂模式
    - 对外提供统一的 generate() 和 generate_async() 接口
    - 内部根据配置选择具体后端

    技术决策:
    - 为什么分层（Router/Generator）: Router 只做简单分类，不需要生成模型的推理能力。
      Haiku 成本仅为 Sonnet 的 1/20，分层使用可节省 60-70% LLM 成本。
    - Circuit Breaker: 每个 provider 独立熔断，防止单点故障雪崩。
    """

    def __init__(
        self,
        generator_provider: Literal["anthropic", "openai", "google", "deepseek"] = "anthropic",
        generator_model: str = "claude-3-7-sonnet-20250620",
        router_provider: Literal["anthropic", "openai", "google", "deepseek"] = "anthropic",
        router_model: str = "claude-3-5-haiku-20250620",
    ):
        from backend.middleware.circuit_breaker import get_breaker, CircuitBreakerConfig
        from backend.observability.metrics import create_metrics_collector

        self.generator_client = self._create_backend(
            generator_provider, generator_model
        )
        self._generator_model = generator_model
        self._generator_provider = generator_provider
        self._generator_breaker = get_breaker(
            f"llm:{generator_provider}:{generator_model}",
            CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0),
        )

        self.router_client = self._create_backend(router_provider, router_model)
        self._router_model = router_model
        self._router_provider = router_provider
        self._router_breaker = get_breaker(
            f"llm:{router_provider}:{router_model}",
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0),
        )

        self._metrics = create_metrics_collector()

    @property
    def generator_model(self) -> str:
        return self._generator_model

    def _create_backend(
        self,
        provider: str,
        model: str,
    ) -> LLMBackend:
        if provider == "anthropic":
            return AnthropicBackend(model=model)
        elif provider == "openai":
            return OpenAIBackend(model=model)
        elif provider == "google":
            return GoogleGenAIBackend(model=model)
        elif provider == "deepseek":
            return DeepSeekBackend(model=model)
        else:
            raise ValueError(f"不支持的 LLM provider: {provider}")

    async def generate_async(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system: str | None = None,
        structured_schema: dict | None = None,
        use_breaker: bool = True,
        use_retry: bool = True,
    ) -> str:
        """
        异步生成

        Args:
            prompt: 用户提示
            model: 可选，指定模型（默认使用 generator）
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            system: 系统提示
            structured_schema: 可选，JSON Schema 用于 Structured Output
            use_breaker: 是否启用熔断器
            use_retry: 是否启用指数退避重试（默认 True）
        """
        client = self.generator_client

        if model and model != self._generator_model:
            client = self._create_backend(self._generator_provider, model)

        call_kwargs = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            structured_schema=structured_schema,
            _metrics_collector=self._metrics,
        )

        async def _invoke() -> str:
            return await client.generate_async(prompt, **call_kwargs)

        try:
            if use_breaker and use_retry:
                return await self._generator_breaker.call_async(
                    self._invoke_with_retry, _invoke
                )
            if use_breaker:
                return await self._generator_breaker.call(
                    client.generate_async, prompt, **call_kwargs
                )
            if use_retry:
                return await self._invoke_with_retry(_invoke)
            return await client.generate_async(prompt, **call_kwargs)
        except Exception as e:
            logger.error(f"LLM generate failed, all retries exhausted or breaker open: {e}")
            return ""

    @staticmethod
    async def _invoke_with_retry(invoke: Callable[[], Awaitable[str]]) -> str:
        """带指数退避重试的调用包装"""
        from backend.generation.retry import with_retry, RetryConfig

        return await with_retry(invoke, RetryConfig(max_attempts=3))

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        structured_schema: dict | None = None,
    ) -> str:
        """同步生成"""
        client = self.generator_client
        if model and model != self._generator_model:
            client = self._create_backend(self._generator_provider, model)

        return client.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            structured_schema=structured_schema,
            _metrics_collector=self._metrics,
        )

    async def generate_stream_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        system: str | None = None,
    ) -> AsyncIterator[str]:
        """
        流式生成，yield 每个 token 片段。

        Args:
            prompt: 用户提示
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            system: 系统提示（流式时忽略 structured_schema）

        Yields:
            每个 token 片段字符串
        """
        client = self.generator_client

        call_kwargs = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
        )

        # 流式端点的熔断接入：检查 OPEN 状态，CLOSED/HALF_OPEN 走原始流；
        # 不在 try 内 yield 整个流，否则熔断器统计会因流中断而误判。
        # 失败 / 成功统计在 _generator_breaker.call 内自动完成。
        if self._generator_breaker.state.value == "open":
            logger.warning("Circuit breaker open, returning empty stream")
            return

        try:
            async for token in client.generate_stream_async(prompt, **call_kwargs):
                yield token
            # 流式成功完成：显式记录成功（call() 会自动记录）
            self._generator_breaker._on_success_sync()
        except Exception as e:
            logger.error(f"LLM stream failed: {e}")
            self._generator_breaker._on_failure_sync()
            return
