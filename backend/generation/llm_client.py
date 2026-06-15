"""
llm_client.py — 统一 LLM 接口
================================================================================
技术决策记录:
- 统一接口: 不管底层是 OpenAI 还是 DeepSeek，对外暴露一致的 generate() 接口。
  DeepSeek API 兼容 OpenAI ChatCompletion 协议，无独立 class 必要。
- 分层使用: Router 用轻量模型，Generator 用主力模型。分层策略节省 60-70% LLM 成本。
- DeepSeek: 兼容 OpenAI API 格式，成本约为 Claude/GPT-4 的 1/10-1/20。
- Structured Output: 使用各 SDK 的原生功能（Pydantic + json_schema），
  而不是提示词工程。消除 json.loads() 解析的脆弱性。
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Awaitable, Callable, Literal, TypeVar

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

    @abstractmethod
    async def generate_stream_async(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """流式生成，yield 每个 token 片段（子类必须实现真正的 token 级流）。"""
        if False:  # pragma: no cover - 强制子类实现
            yield ""


class OpenAICompatibleBackend(LLMBackend):
    """
    OpenAI 兼容协议后端（OpenAI / DeepSeek / 其他 OpenAI-like 服务）

    Phase5-5.1 简化: 移除 ``provider`` 字段，OpenAI 与 DeepSeek 共享一个 class。
    provider 之间通过 base_url 区分（默认 base_url 由调用方传入）。
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        from openai import AsyncOpenAI, OpenAI
        self._base_url = base_url
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)
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
        """流式生成（OpenAI 兼容协议）"""
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
    - 内部根据配置选择具体后端（仅 OpenAI / DeepSeek 两种）

    技术决策:
    - 为什么分层（Router/Generator）: Router 只做简单分类，不需要生成模型的推理能力。
      轻量模型成本仅为大模型的 1/10-1/20，分层使用可节省 60-70% LLM 成本。
    - Circuit Breaker: 每个 provider 独立熔断，防止单点故障雪崩。
    """

    def __init__(
        self,
        generator_provider: Literal["openai", "deepseek"] = "deepseek",
        generator_model: str = "deepseek-chat",
        router_provider: Literal["openai", "deepseek"] = "deepseek",
        router_model: str = "deepseek-chat",
        generator_api_key: str | None = None,
        generator_base_url: str | None = None,
        router_api_key: str | None = None,
        router_base_url: str | None = None,
    ):
        from backend.middleware.circuit_breaker import get_breaker, CircuitBreakerConfig
        from backend.observability.metrics import create_metrics_collector

        self.generator_client = self._create_backend(
            provider=generator_provider,
            model=generator_model,
            api_key=generator_api_key,
            base_url=generator_base_url,
        )
        self._generator_model = generator_model
        self._generator_provider = generator_provider
        self._generator_breaker = get_breaker(
            f"llm:{generator_provider}:{generator_model}",
            CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0),
        )

        self.router_client = self._create_backend(
            provider=router_provider,
            model=router_model,
            api_key=router_api_key,
            base_url=router_base_url,
        )
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
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> LLMBackend:
        if provider in ("openai", "deepseek"):
            return OpenAICompatibleBackend(
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
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
            client = self._create_backend(
                self._generator_provider, model,
            )

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
        if self._generator_breaker.state.value == "open":
            logger.warning("Circuit breaker open, returning empty stream")
            return

        try:
            async for token in client.generate_stream_async(prompt, **call_kwargs):
                yield token
            self._generator_breaker._on_success_sync()
        except Exception as e:
            logger.error(f"LLM stream failed: {e}")
            self._generator_breaker._on_failure_sync()
            return
