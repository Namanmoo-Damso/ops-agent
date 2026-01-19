"""
Prompt Builder - YAML + Jinja2 기반 프롬프트 빌더

역할: sodam.yaml을 읽어서 Jinja2 템플릿으로 시스템 프롬프트 생성
"""

import logging
from datetime import datetime
from pathlib import Path

import yaml
from jinja2 import Template

from ..constants import AGENT_TZINFO

logger = logging.getLogger(__name__)


class PromptBuilder:
    """YAML + Jinja2 기반 프롬프트 빌더."""

    def __init__(self, template_name: str = "sodam"):
        """
        Initialize prompt builder.

        Args:
            template_name: YAML 템플릿 파일명 (확장자 제외)
        """
        yaml_path = Path(__file__).parent / f"{template_name}.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Loaded prompt template: {yaml_path}")

        # Jinja2 메인 템플릿
        self.template_str = """
#############################################
# 절대 금지 규칙 (ABSOLUTE PROHIBITIONS)
#############################################
아래 규칙을 어기면 대화가 실패합니다. 반드시 지키세요:

1. 이모지 절대 금지: 😊 😄 ❤️ 등 모든 이모지 사용 금지
2. 대화 중 "안녕하세요" 금지: 첫 인사 이후에는 절대 "안녕하세요"로 시작하지 마세요
3. 조기 작별 인사 금지: 어르신이 먼저 끊겠다고 하기 전에는 "좋은 하루 보내세요" 등 작별 인사 금지
4. 영어 금지: 모든 응답은 한국어로만
5. 엉뚱한 답변 금지: 어르신이 말한 내용에만 응답하세요

#############################################

You are a {{ persona.role }}.
Your name is '{{ persona.name }}' (Sodam).

# 페르소나 성격
{% for trait in persona.personality -%}
- {{ trait }}
{% endfor %}

# 현재 시각 (한국 시간)
- 지금은 {{ current_time }} (KST) 입니다
- 오늘 날짜: {{ current_date }}

# CRITICAL RULE: Language
{% for rule in language.critical_rules -%}
- {{ rule }}
{% endfor %}

{% if location_info -%}
# 어르신 위치 정보
- 위도(latitude): {{ location_info.latitude }}
- 경도(longitude): {{ location_info.longitude }}
- 날씨 조회 시 이 좌표를 사용하세요

{% endif -%}
{% if retrieved_memories -%}
# {{ memory_context.title }}
{% for rule in memory_context.instructions -%}
- {{ rule }}
{% endfor %}

{{ retrieved_memories }}

{% endif -%}
{% if ward_context -%}

# 어르신 정보 (참고용)
{{ ward_context }}

{% endif -%}
# 말하기 스타일
{% for rule in output_rules.speech_style -%}
- {{ rule }}
{% endfor %}

# 맞춤법 및 문법
{% for rule in output_rules.grammar_and_spelling -%}
- {{ rule }}
{% endfor %}

# 포맷팅
{% for rule in output_rules.formatting -%}
- {{ rule }}
{% endfor %}

# CRITICAL: 대화 연속성 규칙
{% for rule in output_rules.conversation_continuity -%}
- {{ rule }}
{% endfor %}

# CRITICAL: 문맥 일치 규칙
{% for rule in output_rules.context_matching -%}
- {{ rule }}
{% endfor %}

# 감정 지능 - 공감 규칙
{% for rule in emotional_intelligence.empathy_rules -%}
- {{ rule }}
{% endfor %}

# 감정 지능 - 톤 매칭
{% for rule in emotional_intelligence.tone_matching -%}
- {{ rule }}
{% endfor %}

# 대화 흐름 원칙
{% for principle in conversational_flow.principles -%}
- {{ principle }}
{% endfor %}

# 나쁜 예시 (하지 말 것)
{% for example in conversational_flow.bad_examples -%}
- BAD: "{{ example.wrong }}" → 이유: {{ example.reason }}
{% endfor %}

# 좋은 예시 (참고)
{% for example in conversational_flow.good_examples -%}
- 상황: {{ example.context }} → 응답: "{{ example.response }}"
{% endfor %}

# {{ tools.weather.title }}
- 날씨 관련 질문 트리거: {% for trigger in tools.weather.triggers %}'{{ trigger }}'{% if not loop.last %}, {% endif %}{% endfor %}
{% for api in tools.weather.apis -%}
- {{ api.name }}: {{ api.description }}
{% endfor -%}
{% for instruction in tools.weather.instructions -%}
- {{ instruction }}
{% endfor %}

# {{ interruptions.title }}
{% for rule in interruptions.rules -%}
- {{ rule }}
{% endfor %}

# {{ guardrails.title }}
{% for rule in guardrails.rules -%}
- {{ rule.detail }}
{% endfor %}

# {{ topics.title }}
{% for topic in topics.list -%}
- {{ topic }}
{% endfor -%}
""".strip()

    def build(
        self,
        ward_context: str = "",
        latitude: float | None = None,
        longitude: float | None = None,
        retrieved_memories: str = "",
    ) -> str:
        """
        Build full agent instructions.

        Args:
            ward_context: Pre-fetched context about the ward
            latitude: Ward's location latitude
            longitude: Ward's location longitude
            retrieved_memories: Formatted RAG memory segments

        Returns:
            Complete instruction string
        """
        template = Template(self.template_str)

        # 현재 시각 (KST)
        tz = AGENT_TZINFO
        local_now = datetime.now(tz)
        current_time = local_now.strftime("%Y년 %m월 %d일 %H시 %M분")
        current_date = local_now.strftime("%Y년 %m월 %d일")

        # 위치 정보
        location_info = None
        if latitude is not None and longitude is not None:
            location_info = {"latitude": latitude, "longitude": longitude}

        # RAG 메모리 지침

        # 템플릿 렌더링
        rendered_prompt = template.render(
            persona=self.config["persona"],
            language=self.config["language"],
            output_rules=self.config["output_rules"],
            emotional_intelligence=self.config["emotional_intelligence"],
            conversational_flow=self.config["conversational_flow"],
            tools=self.config["tools"],
            interruptions=self.config["interruptions"],
            guardrails=self.config["guardrails"],
            memory_context=self.config.get("guardrails", {}).get("memory_context", {}),
            topics=self.config["topics"],
            current_time=current_time,
            current_date=current_date,
            location_info=location_info,
            retrieved_memories=retrieved_memories,
            ward_context=ward_context,
        )

        return rendered_prompt
