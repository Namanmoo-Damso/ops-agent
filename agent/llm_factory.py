"""
LLM Plugin Factory - Dynamically load LLM providers based on environment variables.

Supported providers:
- aws: AWS Bedrock (Claude)
- ollama: Self-hosted Ollama (via OpenAI-compatible API)
- openai: OpenAI-compatible APIs (vLLM, etc.)
"""

import logging
import os
from typing import Any

from livekit.plugins import aws, openai

logger = logging.getLogger(__name__)


def create_llm(**kwargs) -> Any:
    """
    Create LLM instance based on environment variables.

    Environment variables:
        LLM_PROVIDER: Provider type (aws | ollama | openai)
        LLM_MODEL: Model name
        LLM_BASE_URL: Base URL for OpenAI-compatible APIs (ollama, vLLM)
        LLM_API_KEY: API key for OpenAI (optional for local servers)
        LLM_TEMPERATURE: Temperature (default: 0.7)

    Returns:
        LLM instance from livekit.plugins

    Raises:
        ValueError: If LLM_PROVIDER is invalid or required env vars are missing
    """
    provider = os.getenv("LLM_PROVIDER", "aws").lower()
    model = os.getenv("LLM_MODEL")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # Allow kwargs to override env vars
    provider = kwargs.get("provider", provider)
    model = kwargs.get("model", model)
    temperature = kwargs.get("temperature", temperature)

    logger.info(
        f"Creating LLM: provider={provider}, model={model}, temperature={temperature}"
    )

    if provider == "aws":
        if not model:
            raise ValueError("LLM_MODEL is required for AWS provider")
        return aws.LLM(
            model=model,
            temperature=temperature,
        )

    elif provider == "ollama":
        if not model:
            raise ValueError("LLM_MODEL is required for Ollama provider")

        base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        base_url = kwargs.get("base_url", base_url)

        logger.info(f"Using Ollama at {base_url}")
        return openai.LLM(
            model=model,
            base_url=base_url,
            api_key="ollama",  # Ollama doesn't require real API key
            temperature=temperature,
        )

    elif provider == "openai":
        if not model:
            raise ValueError("LLM_MODEL is required for OpenAI provider")

        base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
        base_url = kwargs.get("base_url", base_url)
        api_key = os.getenv("LLM_API_KEY", "EMPTY")  # vLLM doesn't require API key

        logger.info(f"Using OpenAI-compatible API at {base_url}")

        # Check if Qwen3 model (needs thinking mode disabled)
        if "qwen3" in model.lower() or "qwen-3" in model.lower():
            logger.info("Qwen3 detected: disabling thinking mode via extra_body")
            return openai.LLM(
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

        llm = openai.LLM(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        logger.info(f"Created OpenAI LLM: model={model}, base_url={base_url}")
        return llm

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider}. Supported providers: aws, ollama, openai"
        )
