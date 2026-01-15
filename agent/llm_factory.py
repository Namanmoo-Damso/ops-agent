"""
LLM Plugin Factory - Dynamically load LLM providers based on environment variables.

Supported providers:
- aws: AWS Bedrock (Claude)
- ollama: Self-hosted Ollama (via OpenAI-compatible API)
- openai: OpenAI API
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
        LLM_BASE_URL: Base URL for OpenAI-compatible APIs (ollama/openai)
        LLM_API_KEY: API key for OpenAI
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

        base_url = os.getenv("LLM_BASE_URL", "http://ollama:11434/v1")
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

        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY is required for OpenAI provider")

        base_url = os.getenv("LLM_BASE_URL")  # Optional for OpenAI
        base_url = kwargs.get("base_url", base_url)

        logger.info(f"Using OpenAI with model {model}")
        return openai.LLM(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider}. "
            f"Supported providers: aws, ollama, openai"
        )
