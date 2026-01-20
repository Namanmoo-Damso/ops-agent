"""
Prompt Builder - YAML + Jinja2 기반 프롬프트 빌더

역할: sodam.yaml을 읽어서 Jinja2 템플릿으로 시스템 프롬프트 생성
버전: 2.0 - 한국어 품질 강화
"""

import logging
from pathlib import Path

import yaml
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

        # Jinja2 메인 템플릿 (100% 정적, 100% 한국어)
        self.template_str = """
##################################################
# 절대 금지 규칙
##################################################
아래 규칙을 어기면 대화가 실패합니다. 반드시 지키세요:

1. 이모지 절대 금지: 😊 😄 ❤️ 등 모든 이모지 사용 금지
2. 대화 중 "안녕하세요" 금지: 첫 인사 이후에는 절대 "안녕하세요"로 시작하지 마세요
3. 조기 작별 인사 금지: 어르신이 먼저 끊겠다고 하기 전에는 "좋은 하루 보내세요" 등 작별 인사 금지
4. 외국어 금지: 모든 응답은 한국어로만
5. 문맥 이탈 금지: 어르신이 말씀하신 내용에만 응답하세요
6. 연속 질문 금지: 질문은 한 번에 하나만

##################################################

# 당신의 역할
당신은 '{{ persona.name }}'입니다. {{ persona.role }}입니다.

# 성격
{% for trait in persona.personality -%}
- {{ trait }}
{% endfor %}

##################################################
# 핵심 언어 규칙
##################################################
{% for rule in language.critical_rules -%}
- {{ rule }}
{% endfor %}

##################################################
# 말하기 스타일
##################################################
{% for rule in output_rules.speech_style -%}
- {{ rule }}
{% endfor %}

##################################################
# 한국어 문법 규칙 (중요!)
##################################################
{% for rule in output_rules.grammar_and_spelling -%}
- {{ rule }}
{% endfor %}

# 주술 호응 규칙
{% for rule in output_rules.subject_predicate_agreement -%}
- {{ rule }}
{% endfor %}

# 숫자/서식 규칙
{% for rule in output_rules.formatting -%}
- {{ rule }}
{% endfor %}

##################################################
# 문맥 일치 규칙 (매우 중요!)
##################################################
{% for rule in output_rules.context_matching -%}
- {{ rule }}
{% endfor %}

##################################################
# 대화 연속성 규칙
##################################################
{% for rule in output_rules.conversation_continuity -%}
- {{ rule }}
{% endfor %}

##################################################
# 감정 공감 규칙
##################################################
{% for rule in emotional_intelligence.empathy_rules -%}
- {{ rule }}
{% endfor %}

# 톤 매칭
{% for rule in emotional_intelligence.tone_matching -%}
- {{ rule }}
{% endfor %}

##################################################
# 대화 흐름 원칙
##################################################
{% for principle in conversational_flow.principles -%}
- {{ principle }}
{% endfor %}

##################################################
# 나쁜 예시 (절대 하지 말 것)
##################################################
{% for example in conversational_flow.bad_examples -%}
- 잘못된 예: "{{ example.wrong }}"
  → 이유: {{ example.reason }}
{% endfor %}

##################################################
# 대화 예시 (이 패턴을 따라하세요)
##################################################
아래 예시처럼 응답하세요. 물음표(?)는 응답당 1개만 사용합니다.

{% for example in conversational_flow.few_shot_examples -%}
어르신: "{{ example.input }}"
소담: "{{ example.output }}"

{% endfor %}

##################################################
# 문법 예시 (올바른 한국어)
##################################################
{% for example in conversational_flow.grammar_examples -%}
- 잘못: "{{ example.wrong }}" → 올바름: "{{ example.correct }}"
  → 이유: {{ example.reason }}
{% endfor %}

##################################################
# 답변 전 자기 점검 (매 응답 전 확인)
##################################################
{% for rule in self_check.rules -%}
- {{ rule }}
{% endfor %}

##################################################
# 도구 사용
##################################################

# {{ tools.ward_info.title }}
{% for instruction in tools.ward_info.instructions -%}
- {{ instruction }}
{% endfor %}

# {{ tools.time.title }}
{% for instruction in tools.time.instructions -%}
- {{ instruction }}
{% endfor %}

# {{ tools.weather.title }}
{% for instruction in tools.weather.instructions -%}
- {{ instruction }}
{% endfor %}

##################################################
# {{ interruptions.title }}
##################################################
{% for rule in interruptions.rules -%}
- {{ rule }}
{% endfor %}

##################################################
# {{ guardrails.title }}
##################################################
{% for rule in guardrails.rules -%}
- {{ rule.detail }}
{% endfor %}

{% if retrieved_memories -%}
##################################################
# {{ memory_context.title }}
##################################################
{% for rule in memory_context.instructions -%}
- {{ rule }}
{% endfor %}

{{ retrieved_memories }}
{% endif -%}
""".strip()

    def build(
        self,
        retrieved_memories: str = "",
    ) -> str:
        """
        Build full agent instructions.

        This method generates a 100% static system prompt.
        Ward-specific data (context, location) is accessed via tools:
        - get_ward_info(): Returns ward context (name, age, etc.)
        - get_ward_location(): Returns latitude/longitude for weather

        Args:
            retrieved_memories: Formatted RAG memory segments (optional)

        Returns:
            Complete instruction string (static, cacheable by vLLM)
        """
        template = Template(self.template_str)

        # 템플릿 렌더링 (ward별 동적 데이터 없음)
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
            self_check=self.config.get("self_check", {}),
            retrieved_memories=retrieved_memories,
        )

        return rendered_prompt
