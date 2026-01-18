"""Configuration and environment validation for AI agents"""

import os
import sys
from typing import Optional


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid"""
    pass


def validate_env_vars() -> dict[str, str]:
    """
    Validate required environment variables for AI agent.

    Returns:
        dict: Validated environment variables

    Raises:
        ConfigError: If any required environment variable is missing
    """
    required_vars = [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "REDIS_URL",
    ]

    missing_vars = []
    config = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            config[var] = value

    if missing_vars:
        error_msg = (
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please set these variables in your .env file or environment."
        )
        raise ConfigError(error_msg)

    # Validate URL format
    livekit_url = config["LIVEKIT_URL"]
    if not (livekit_url.startswith("ws://") or livekit_url.startswith("wss://")):
        raise ConfigError(
            f"LIVEKIT_URL must start with ws:// or wss://, got: {livekit_url}"
        )

    # Validate Redis URL format
    redis_url = config["REDIS_URL"]
    if not redis_url.startswith("redis://"):
        raise ConfigError(
            f"REDIS_URL must start with redis://, got: {redis_url}"
        )

    # Validate AWS credentials format (basic check)
    aws_key = config["AWS_ACCESS_KEY_ID"]
    if not aws_key.startswith("AKIA"):
        raise ConfigError(
            "AWS_ACCESS_KEY_ID appears invalid (should start with 'AKIA')"
        )

    # Validate KMA MCP URL format (optional)
    kma_mcp_url = os.getenv("KMA_MCP_URL")
    if kma_mcp_url:
        if not (kma_mcp_url.startswith("http://") or kma_mcp_url.startswith("https://")):
            raise ConfigError(
                f"KMA_MCP_URL must start with http:// or https://, got: {kma_mcp_url}"
            )
        config["KMA_MCP_URL"] = kma_mcp_url

    return config


def get_optional_config() -> dict[str, Optional[str]]:
    """Get optional configuration variables with defaults"""
    return {
        "OPS_API_URL": os.getenv("OPS_API_URL", "http://backend:8080"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    }
