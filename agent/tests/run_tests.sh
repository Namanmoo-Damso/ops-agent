#!/bin/bash
# 테스트 실행 스크립트 (Docker 환경)
# 사용법: ./run_tests.sh [test_name]
# 예시:
#   ./run_tests.sh                              # 전체 테스트 (타이밍 리포트)
#   ./run_tests.sh test_very_short_answers      # 짧은 답변 테스트만
#   ./run_tests.sh test_short_answers           # Short 카테고리만
#   ./run_tests.sh test_medium_answers          # Medium 카테고리만
#   ./run_tests.sh test_long_answers            # Long 카테고리만
#   ./run_tests.sh test_very_long_answers       # Very Long 카테고리만

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "🐳 Running tests in Docker..."
echo "=================================================="
echo ""

# pytest asyncio 옵션 포함
PYTEST_OPTS="-p pytest_asyncio --asyncio-mode=auto -s -v"

# pytest 설치 후 테스트 실행
if [ -z "$1" ]; then
    # 기본: 타이밍 리포트 포함 전체 테스트
    docker compose -f docker-compose.dev.yml run --rm agent \
        bash -c "pip install pytest pytest-asyncio -q && cd /app/agent && LIVEKIT_EVALS_VERBOSE=1 python -m pytest tests/test_agent.py::test_all_with_timing_report $PYTEST_OPTS"
else
    # 특정 테스트
    docker compose -f docker-compose.dev.yml run --rm agent \
        bash -c "pip install pytest pytest-asyncio -q && cd /app/agent && LIVEKIT_EVALS_VERBOSE=1 python -m pytest tests/test_agent.py::$1 $PYTEST_OPTS"
fi
