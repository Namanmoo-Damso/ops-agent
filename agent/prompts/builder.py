"""
Prompt Builder - YAML + Jinja2 기반 프롬프트 빌더

역할: sodam.yaml을 읽어서 Jinja2 템플릿으로 시스템 프롬프트 생성
"""

import logging
from datetime import datetime
from pathlib import Path

import yaml
from ..constants import AGENT_TZINFO
from jinja2 import Template

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
You are a {{ persona.role }}.
Your name is '{{ persona.name }}' (Sodam).

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
{{ memory_instructions }}

{% if ward_context -%}
# 어르신 정보 (참고용)
{{ ward_context }}

{% endif -%}
# Output rules
{% for rule in output_rules.speech_style -%}
- {{ rule }}
{% endfor -%}
{% for rule in output_rules.formatting -%}
- {{ rule }}
{% endfor %}

# Conversational flow
{% for flow in conversational_flow -%}
- {{ flow }}
{% endfor %}

# {{ tools.search_memory.title }}
Format: {{ tools.search_memory.format }}
{% for instruction in tools.search_memory.instructions -%}
- {{ instruction }}
{% endfor %}
- Trigger patterns: {% for trigger in tools.search_memory.trigger_patterns %}'{{ trigger }}'{% if not loop.last %}, {% endif %}{% endfor %}
- Keywords: {% for keyword in tools.search_memory.keywords %}{{ keyword }}{% if not loop.last %}, {% endif %}{% endfor %}

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
        memory_instructions: str = "",
    ) -> str:
        """
        Build full agent instructions.

        Args:
            ward_context: Pre-fetched context about the ward
            latitude: Ward's location latitude
            longitude: Ward's location longitude

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
            conversational_flow=self.config["conversational_flow"],
            tools=self.config["tools"],
            interruptions=self.config["interruptions"],
            guardrails=self.config["guardrails"],
            topics=self.config["topics"],
            current_time=current_time,
            current_date=current_date,
            location_info=location_info,
            memory_instructions=memory_instructions,
            ward_context=ward_context,
        )

        return rendered_prompt
