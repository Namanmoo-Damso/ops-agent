"""
어르신 돌봄 AI 에이전트 테스트

이 테스트는 LiveKit Agents 테스트 프레임워크를 사용하여
텍스트 기반으로 에이전트 동작을 검증합니다.

실행 방법:
    cd /home/ubuntu/wip/ops-agent/agent
    LIVEKIT_EVALS_VERBOSE=1 pytest -s tests/test_agent.py

필요 패키지:
    pip install pytest pytest-asyncio
"""

import time
from dataclasses import dataclass

import pytest
from livekit.agents import AgentSession
from livekit.plugins import aws
from voice_agent import CallDirection, VoiceAgent


@dataclass
class TestCase:
    """테스트 케이스 정의"""

    question: str
    expected_intent: str
    category: str  # very_short, short, medium, long, very_long
    max_response_time: float = 10.0  # seconds


# ==============================================================================
# 테스트 케이스 정의 (20개)
# ==============================================================================

TEST_CASES = [
    # 📊 Very Short Answers (1-5 words)
    TestCase(
        question="안녕?",
        expected_intent="친절하게 인사하고 안부를 묻는다",
        category="very_short",
    ),
    TestCase(
        question="오늘 날씨 어때?",
        expected_intent="날씨에 대해 대답한다 (직접 모르면 추측하거나 질문한다)",
        category="very_short",
    ),
    TestCase(
        question="몇 시야?",
        expected_intent="현재 시간을 알려준다",
        category="very_short",
    ),
    TestCase(
        question="배고파?",
        expected_intent="식사에 대해 물어보거나 공감한다",
        category="very_short",
    ),
    TestCase(
        question="잘 지냈어?",
        expected_intent="안부에 대답하고 어르신의 안부도 묻는다",
        category="very_short",
    ),
    # 💬 Short Answers (1-2 sentences)
    TestCase(
        question="오늘 점심 뭐 먹을까?",
        expected_intent="점심 메뉴를 제안하거나 어르신 취향을 묻는다",
        category="short",
    ),
    TestCase(
        question="산책 나가도 될까?",
        expected_intent="산책에 대해 조언하고 날씨나 건강을 고려한다",
        category="short",
    ),
    TestCase(
        question="물 좀 마셔야겠다",
        expected_intent="물 마시는 것을 긍정하고 격려한다",
        category="short",
    ),
    TestCase(
        question="약 먹었나?",
        expected_intent="약 복용에 대해 물어보거나 확인해준다",
        category="short",
    ),
    TestCase(
        question="오늘 무슨 요일이지?",
        expected_intent="오늘 요일을 알려준다",
        category="short",
    ),
    # 📝 Medium Answers (3-5 sentences)
    TestCase(
        question="무릎이 아픈데 어떡하지?",
        expected_intent="무릎 통증에 공감하고 온찜질, 스트레칭, 병원 방문 등을 조언한다",
        category="medium",
    ),
    TestCase(
        question="심심한데 뭐 할까?",
        expected_intent="여러 가지 활동을 제안한다 (라디오, TV, 취미 등)",
        category="medium",
    ),
    TestCase(
        question="오늘 누가 온다고 했는데 누구였지?",
        expected_intent="방문 예정에 대해 물어보거나 기억을 도와주려 한다",
        category="medium",
    ),
    TestCase(
        question="잠을 잘 못 잤어",
        expected_intent="공감하고 수면에 도움이 되는 조언을 한다",
        category="medium",
    ),
    TestCase(
        question="병원 언제 가야 하지?",
        expected_intent="병원 일정에 대해 물어보거나 확인을 도와주려 한다",
        category="medium",
    ),
    # 📖 Long Answers (6-10 sentences)
    TestCase(
        question="요즘 외로운데 어떡하지?",
        expected_intent="외로움에 깊이 공감하고, 함께 있다고 위로하며, 가족 방문이나 복지관 등 사회적 연결을 제안한다",
        category="long",
    ),
    TestCase(
        question="건강 관리 어떻게 해야 하나?",
        expected_intent="약 복용, 물 마시기, 산책, 스트레칭 등 종합적인 건강 관리 조언을 한다",
        category="long",
    ),
    TestCase(
        question="오늘 하루 어떻게 보내면 좋을까?",
        expected_intent="하루 일과를 시간대별로 제안한다 (아침 산책, 점심, 휴식, 저녁 등)",
        category="long",
    ),
    # 🎵 Very Long Answers (특수 케이스)
    TestCase(
        question="애국가 1절 불러줘",
        expected_intent="노래를 직접 부를 수 없다고 정중히 거절하고, 대안을 제시한다",
        category="very_long",
        max_response_time=15.0,
    ),
    TestCase(
        question="요즘 기억력이 자꾸 나빠지는데 걱정돼",
        expected_intent="기억력 걱정에 공감하고 위로하며, 기억력 유지 방법과 의사 상담을 권유한다",
        category="very_long",
        max_response_time=15.0,
    ),
]


# ==============================================================================
# 테스트 픽스처
# ==============================================================================


@pytest.fixture
def llm():
    """LLM 인스턴스 (평가용)"""
    return aws.LLM(
        model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        temperature=0.3,
    )


# ==============================================================================
# 개별 테스트 함수
# ==============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case", TEST_CASES, ids=[tc.question[:20] for tc in TEST_CASES]
)
async def test_agent_response(test_case: TestCase, llm):
    """에이전트 응답 테스트"""

    async with (
        llm,
        AgentSession(llm=llm) as session,
    ):
        # 에이전트 시작
        agent = VoiceAgent(
            ward_context="",
            call_direction=CallDirection.INBOUND,
        )
        await session.start(agent)

        # 타이밍 측정 시작
        start_time = time.time()

        # 사용자 입력 실행
        result = await session.run(user_input=test_case.question)

        # 타이밍 측정 종료
        elapsed_time = time.time() - start_time

        # 결과 출력
        print(f"\n{'=' * 60}")
        print(f"📝 질문: {test_case.question}")
        print(f"📂 카테고리: {test_case.category}")
        print(f"⏱️  응답 시간: {elapsed_time:.3f}s")

        # 응답 내용 확인
        response_event = result.expect.next_event()
        response_event.is_message(role="assistant")

        # 응답 내용 출력
        event_item = response_event.event().item
        if hasattr(event_item, "content"):
            content = event_item.content
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            print(f"💬 응답: {content}")

        # LLM 판정
        print(f"🎯 기대 의도: {test_case.expected_intent}")
        await response_event.judge(
            llm,
            intent=test_case.expected_intent,
        )
        print("✅ 판정: PASS")

        # 응답 시간 체크
        assert elapsed_time < test_case.max_response_time, (
            f"응답 시간 {elapsed_time:.2f}s가 최대 {test_case.max_response_time}s를 초과했습니다"
        )

        # 더 이상 이벤트가 없는지 확인
        result.expect.no_more_events()


# ==============================================================================
# 카테고리별 테스트 (그룹 실행용)
# ==============================================================================


@pytest.mark.asyncio
async def test_very_short_answers(llm):
    """Very Short 카테고리 전체 테스트"""
    cases = [tc for tc in TEST_CASES if tc.category == "very_short"]
    await _run_category_tests(cases, llm)


@pytest.mark.asyncio
async def test_short_answers(llm):
    """Short 카테고리 전체 테스트"""
    cases = [tc for tc in TEST_CASES if tc.category == "short"]
    await _run_category_tests(cases, llm)


@pytest.mark.asyncio
async def test_medium_answers(llm):
    """Medium 카테고리 전체 테스트"""
    cases = [tc for tc in TEST_CASES if tc.category == "medium"]
    await _run_category_tests(cases, llm)


@pytest.mark.asyncio
async def test_long_answers(llm):
    """Long 카테고리 전체 테스트"""
    cases = [tc for tc in TEST_CASES if tc.category == "long"]
    await _run_category_tests(cases, llm)


@pytest.mark.asyncio
async def test_very_long_answers(llm):
    """Very Long 카테고리 전체 테스트"""
    cases = [tc for tc in TEST_CASES if tc.category == "very_long"]
    await _run_category_tests(cases, llm)


async def _run_category_tests(cases: list[TestCase], llm):
    """카테고리 테스트 실행 헬퍼"""
    results = []

    async with (
        llm,
        AgentSession(llm=llm) as session,
    ):
        agent = VoiceAgent(
            ward_context="",
            call_direction=CallDirection.INBOUND,
        )
        await session.start(agent)

        for test_case in cases:
            start_time = time.time()
            result = await session.run(user_input=test_case.question)
            elapsed_time = time.time() - start_time

            response_event = result.expect.next_event()
            response_event.is_message(role="assistant")

            # 응답 내용 추출
            event_item = response_event.event().item
            content = ""
            if hasattr(event_item, "content"):
                content = event_item.content
                if isinstance(content, list):
                    content = " ".join(str(c) for c in content)

            results.append(
                {
                    "question": test_case.question,
                    "response": content,
                    "time": elapsed_time,
                }
            )

            print(f"\n질문: {test_case.question}")
            print(f"응답: {content[:100]}...")
            print(f"시간: {elapsed_time:.3f}s")

    # 결과 요약
    print(f"\n{'=' * 60}")
    print("📊 카테고리 테스트 결과 요약")
    print(f"{'=' * 60}")
    total_time = sum(r["time"] for r in results)
    avg_time = total_time / len(results) if results else 0
    print(f"총 테스트: {len(results)}개")
    print(f"평균 응답 시간: {avg_time:.3f}s")
    print(f"총 소요 시간: {total_time:.3f}s")


# ==============================================================================
# 전체 테스트 + 타이밍 리포트
# ==============================================================================


@pytest.mark.asyncio
async def test_all_with_timing_report(llm):
    """전체 테스트 + 타이밍 리포트 생성"""

    results = []

    async with (
        llm,
        AgentSession(llm=llm) as session,
    ):
        agent = VoiceAgent(
            ward_context="",
            call_direction=CallDirection.INBOUND,
        )
        await session.start(agent)

        for test_case in TEST_CASES:
            start_time = time.time()

            try:
                result = await session.run(user_input=test_case.question)
                elapsed_time = time.time() - start_time

                response_event = result.expect.next_event()
                response_event.is_message(role="assistant")

                event_item = response_event.event().item
                content = ""
                if hasattr(event_item, "content"):
                    content = event_item.content
                    if isinstance(content, list):
                        content = " ".join(str(c) for c in content)

                results.append(
                    {
                        "category": test_case.category,
                        "question": test_case.question,
                        "response": content,
                        "time": elapsed_time,
                        "success": True,
                        "error": None,
                    }
                )

            except Exception as e:
                elapsed_time = time.time() - start_time
                results.append(
                    {
                        "category": test_case.category,
                        "question": test_case.question,
                        "response": "",
                        "time": elapsed_time,
                        "success": False,
                        "error": str(e),
                    }
                )

    # 리포트 출력
    print("\n" + "=" * 80)
    print("📊 PIPELINE TIMING REPORT (Text-based, LLM only)")
    print("=" * 80)

    categories = ["very_short", "short", "medium", "long", "very_long"]

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue

        avg_time = sum(r["time"] for r in cat_results) / len(cat_results)
        success_count = sum(1 for r in cat_results if r["success"])

        print(f"\n📂 {cat.upper()}")
        print("-" * 40)

        for r in cat_results:
            status = "✅" if r["success"] else "❌"
            print(f"  {status} [{r['time']:.2f}s] {r['question'][:30]}")
            if r["response"]:
                print(f"     → {r['response'][:60]}...")

        print(f"\n  평균: {avg_time:.3f}s | 성공: {success_count}/{len(cat_results)}")

    # 전체 요약
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    total_time = sum(r["time"] for r in results)
    avg_time = total_time / len(results)
    success_count = sum(1 for r in results if r["success"])

    print(f"총 테스트: {len(results)}개")
    print(f"성공: {success_count}/{len(results)}")
    print(f"평균 LLM 응답 시간: {avg_time:.3f}s")
    print(f"총 소요 시간: {total_time:.3f}s")

    # 카테고리별 평균
    print("\n카테고리별 평균 LLM 응답 시간:")
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            avg = sum(r["time"] for r in cat_results) / len(cat_results)
            print(f"  {cat:12}: {avg:.3f}s")
