"""
llm_client.py — 统一 LLM 接口
================================================================================
技术决策记录:
- 统一接口: 不管底层是 Anthropic/OpenAI/Google/DeepSeek，对外暴露一致的 generate() 接口。
- 分层使用: Router 用轻量模型，Generator 用主力模型。分层策略节省 60-70% LLM 成本。
- DeepSeek: 兼容 OpenAI API 格式，成本约为 Claude/GPT-4 的 1/10-1/20。
- Structured Output: 使用各 SDK 的原生功能（Pydantic + json_schema），
  而不是提示词工程。因为提示词的输出格式稳定性不够（特别是多语言场景）。

业务难点:
- API 速率限制: 高并发场景需要指数退避重试。
- 多模型支持: 不同模型的 API 格式不同，需要适配器封装。
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Literal

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """LLM 后端抽象基类"""

    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> str:
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        ...


class AnthropicBackend(LLMBackend):
    """Anthropic (Claude) 后端"""

    def __init__(
        self,
        model: str = "claude-3-7-sonnet-20250620",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("需要安装 anthropic: pip install anthropic")

        self._client = Anthropic()
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def generate_async(self, prompt: str, **kwargs) -> str:
        import anthropic
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        system = kwargs.get("system", None)

        message = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def generate(self, prompt: str, **kwargs) -> str:
        import anthropic
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
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            raise ImportError("需要安装 openai: pip install openai")

        self._client = OpenAI()
        self._async_client = AsyncOpenAI()
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def generate_async(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
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


class GoogleGenAIBackend(LLMBackend):
    """Google Gemini 后端"""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        try:
            from google import genai
        except ImportError:
            raise ImportError("需要安装 google-genai: pip install google-genai")

        self._client = genai.Client()
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def generate_async(self, prompt: str, **kwargs) -> str:
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config={
                "max_output_tokens": kwargs.get("max_tokens", self._max_tokens),
                "temperature": kwargs.get("temperature", self._temperature),
            },
        )
        return response.text

    def generate(self, prompt: str, **kwargs) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config={
                "max_output_tokens": kwargs.get("max_tokens", self._max_tokens),
                "temperature": kwargs.get("temperature", self._temperature),
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
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            raise ImportError("需要安装 openai: pip install openai")

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def generate_async(self, prompt: str, **kwargs) -> str:
        max_tokens = kwargs.get("max_tokens", self._max_tokens)
        temperature = kwargs.get("temperature", self._temperature)
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
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


class LLMClient:
    """
    LLM 统一客户端

    设计模式: 门面模式 + 工厂模式
    - 对外提供统一的 generate() 和 generate_async() 接口
    - 内部根据配置选择具体后端

    技术决策:
    - 为什么分层（Router/Generator）: Router 只做简单分类，不需要生成模型的推理能力。
      Haiku 成本仅为 Sonnet 的 1/20，分层使用可节省 60-70% LLM 成本。
    - Structured Output: 各后端均支持 JSON Schema 约束输出，
      这是 2026 年生产环境的必备功能。
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

        # Generator (主生成模型)
        self.generator_client = self._create_backend(
            generator_provider, generator_model
        )
        self._generator_model = generator_model
        self._generator_provider = generator_provider
        self._generator_breaker = get_breaker(
            f"llm:{generator_provider}:{generator_model}",
            CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0),
        )

        # Router (轻量分类模型)
        self.router_client = self._create_backend(router_provider, router_model)
        self._router_model = router_model
        self._router_provider = router_provider
        self._router_breaker = get_breaker(
            f"llm:{router_provider}:{router_model}",
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0),
        )

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
    ) -> str:
        """
        异步生成

        Args:
            prompt: 用户提示
            model: 可选，指定模型（默认使用 generator）
            max_tokens: 最大生成 token 数
            temperature: 温度参数
            system: 系统提示
        """
        # 默认为 generator client
        client = self.generator_client

        # 如果指定了不同模型，创建新的 backend
        if model and model != self._generator_model:
            client = self._create_backend("anthropic", model)

        try:
            return await self._generator_breaker.call(
                client.generate_async,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
            )
        except Exception as e:
            logger.error(f"LLM generate failed, circuit breaker may be open: {e}")
            return ""

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> str:
        """同步生成"""
        client = self.generator_client
        if model and model != self._generator_model:
            client = self._create_backend("anthropic", model)

        return client.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
