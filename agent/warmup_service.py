"""
vLLM Prefix Cache Warmup Service.

vLLM이 ready된 후 시스템 프롬프트를 미리 캐싱하여
첫 번째 요청부터 prefix cache hit을 보장합니다.

Usage:
    python -m agent.warmup_service
"""

import logging
import os
import sys
import time

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def warmup_vllm_prefix_cache() -> bool:
    """vLLM prefix cache를 시스템 프롬프트로 워밍업."""
    from .prompts.builder import PromptBuilder

    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    llm_model = os.getenv("LLM_MODEL", "Qwen/Qwen3-8B-AWQ")

    # 정적 시스템 프롬프트 생성
    builder = PromptBuilder("sodam")
    system_prompt = builder.build()

    logger.info(f"System prompt length: {len(system_prompt)} chars")

    # Warmup 요청 (max_tokens=1로 최소 생성)
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{llm_base_url}/chat/completions",
                json={
                    "model": llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "안녕"},
                    ],
                    "max_tokens": 1,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response.raise_for_status()
            logger.info("vLLM prefix cache warmup complete!")
            return True
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        return False


def main() -> None:
    """메인 진입점 - vLLM health 확인 후 warmup 실행."""
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    health_url = llm_base_url.replace("/v1", "/health")

    logger.info(f"Waiting for vLLM at {health_url}...")

    # vLLM ready 대기 (최대 5분)
    max_wait = 300
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(health_url)
                if resp.status_code == 200:
                    logger.info("vLLM is ready!")
                    break
        except Exception:
            pass
        time.sleep(1)
    else:
        logger.error("vLLM health check timeout")
        sys.exit(1)

    # Warmup 실행
    if warmup_vllm_prefix_cache():
        logger.info("Warmup service completed successfully")
        sys.exit(0)
    else:
        logger.error("Warmup service failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
