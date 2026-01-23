"""
Negative Keyword Detector - 부정적 키워드 감지

사용자 발화에서 통증, 비관적 표현, 긴급 상황 등의 부정적 키워드를 감지하여
emotion caution/critical 상태를 트리거합니다.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """위험 수준"""
    NORMAL = "normal"
    CAUTION = "caution"
    CRITICAL = "critical"


class KeywordCategory(str, Enum):
    """키워드 카테고리"""
    # 통증 관련
    PAIN_HEAD = "pain_head"           # 두통
    PAIN_CHEST = "pain_chest"         # 흉통, 가슴 통증
    PAIN_ABDOMINAL = "pain_abdominal" # 복통
    PAIN_BACK = "pain_back"           # 허리, 등 통증
    PAIN_JOINT = "pain_joint"         # 관절 통증
    PAIN_GENERAL = "pain_general"     # 일반 통증
    
    # 증상 관련
    SYMPTOM_DIZZINESS = "symptom_dizziness"     # 어지럼증
    SYMPTOM_BREATHING = "symptom_breathing"     # 호흡 곤란
    SYMPTOM_NAUSEA = "symptom_nausea"           # 메스꺼움, 구토
    SYMPTOM_WEAKNESS = "symptom_weakness"       # 무기력, 기운 없음
    SYMPTOM_NUMBNESS = "symptom_numbness"       # 저림, 마비
    SYMPTOM_FEVER = "symptom_fever"             # 열, 체온
    SYMPTOM_SLEEP = "symptom_sleep"             # 수면 문제
    
    # 정서적 고통
    EMOTIONAL_DEPRESSION = "emotional_depression"   # 우울
    EMOTIONAL_ANXIETY = "emotional_anxiety"         # 불안
    EMOTIONAL_LONELINESS = "emotional_loneliness"   # 외로움
    EMOTIONAL_FEAR = "emotional_fear"               # 두려움
    EMOTIONAL_ANGER = "emotional_anger"             # 분노
    
    # 자해/자살 관련 (최고 위험)
    SUICIDE_IDEATION = "suicide_ideation"       # 자살 사고
    SELF_HARM = "self_harm"                     # 자해 언급
    
    # 긴급 상황
    EMERGENCY_FALL = "emergency_fall"           # 낙상
    EMERGENCY_HELP = "emergency_help"           # 도움 요청
    EMERGENCY_ACCIDENT = "emergency_accident"   # 사고
    
    # 건강 걱정
    HEALTH_CONCERN = "health_concern"           # 건강 불안


# 카테고리별 키워드 리스트 (형태소 변형 포함)
KEYWORD_PATTERNS: dict[KeywordCategory, list[str]] = {
    # ==================== 통증 관련 ====================
    KeywordCategory.PAIN_HEAD: [
        "머리", "두통", "머리가 아", "머리 아", "머리통",
        "편두통", "골이 지끈", "지끈지끈", "머리 깨질",
        "관자놀이", "뒷머리", "앞머리가 아",
    ],
    KeywordCategory.PAIN_CHEST: [
        "가슴", "흉통", "가슴이 아", "가슴 아", "가슴이 답답",
        "심장", "심장이 아", "가슴을 쥐어", "가슴이 조이",
        "명치", "명치가 아", "가슴이 뻐근", "가슴이 쿵쿵",
        "가슴이 쓰리", "흉부", "갈비뼈",
    ],
    KeywordCategory.PAIN_ABDOMINAL: [
        "배", "복통", "배가 아", "배 아", "뱃속",
        "위", "위가 아", "위장", "위경련", "속이 쓰리",
        "장", "장이 아", "설사", "변비", "소화",
        "속이 안 좋", "속이 불편", "명치", "속이 더부룩",
        "배탈", "체", "체했", "소화불량", "구역질",
    ],
    KeywordCategory.PAIN_BACK: [
        "허리", "허리가 아", "허리 아", "요통", "등",
        "등이 아", "등 아", "척추", "디스크", "목",
        "목이 아", "목 아", "뒷목", "어깨", "어깨가 아",
        "허리가 끊어질", "허리 삐끗", "등짝", "견갑골",
    ],
    KeywordCategory.PAIN_JOINT: [
        "무릎", "무릎이 아", "무릎 아", "관절", "관절이 아",
        "손목", "손목이 아", "발목", "발목이 아",
        "팔꿈치", "손가락이 아", "어깨 관절", "고관절",
        "다리가 아", "다리 아", "팔이 아", "팔 아",
        "뼈마디", "뼈가 아", "쑤시", "삐었", "삐끗",
    ],
    KeywordCategory.PAIN_GENERAL: [
        "아파", "아프다", "아프네", "아픕니다", "아파요",
        "통증", "고통", "신음", "끙끙", "아이고",
        "아야", "악", "앗", "욱신", "욱신거리", "쿡쿡",
        "찌릿", "찌릿거리", "시리", "화끈", "뻣뻣",
        "뻐근", "결리", "저리", "수술", "병원",
    ],
    
    # ==================== 증상 관련 ====================
    KeywordCategory.SYMPTOM_DIZZINESS: [
        "어지러", "어지럽", "현기증", "빙빙", "휘청",
        "어질어질", "아찔", "정신이 몽롱", "기절할 것 같",
        "눈앞이 깜깜", "핑", "핑 돌", "쓰러질 것 같",
    ],
    KeywordCategory.SYMPTOM_BREATHING: [
        "숨", "숨이 차", "숨이 안", "호흡", "호흡이 안",
        "숨쉬기", "숨쉬기 힘들", "숨이 막히", "가빠",
        "헐떡", "헐떡거리", "숨차", "숨 가쁘", "질식",
        "산소", "기침", "가래", "기관지", "천식",
    ],
    KeywordCategory.SYMPTOM_NAUSEA: [
        "메스꺼", "메슥", "구역", "구역질", "토할",
        "토하", "구토", "울렁", "울렁거리", "속이 안 좋",
        "속 메스", "속이 뒤집", "게우", "헛구역",
    ],
    KeywordCategory.SYMPTOM_WEAKNESS: [
        "기운", "기운이 없", "기력", "기력이 없",
        "힘이 없", "힘이 안", "무기력", "나른",
        "축 처", "늘어", "녹초", "지치", "피곤",
        "탈진", "힘들어 죽", "몸이 안", "몸이 말을 안",
    ],
    KeywordCategory.SYMPTOM_NUMBNESS: [
        "저리", "저림", "저려", "저립", "마비",
        "감각이 없", "감각이 안", "뻣뻣", "굳어",
        "팔이 안 올라", "다리가 안", "손이 안",
        "움직이기 힘", "뻣뻣해", "경직",
    ],
    KeywordCategory.SYMPTOM_FEVER: [
        "열", "열이 나", "열나", "체온", "뜨거워",
        "한기", "오한", "으슬으슬", "싸늘", "식은 땀",
        "땀이 비 오듯", "땀이 뻘뻘", "미열", "고열",
    ],
    KeywordCategory.SYMPTOM_SLEEP: [
        "잠", "잠이 안", "못 자", "불면", "불면증",
        "잠을 못", "뜬눈", "밤새", "꿈", "악몽",
        "가위", "가위눌리", "수면", "수면제",
        "졸리", "졸음", "깨어", "새벽에 깨",
    ],
    
    # ==================== 정서적 고통 ====================
    KeywordCategory.EMOTIONAL_DEPRESSION: [
        "우울", "우울해", "우울하", "울적", "침울",
        "슬퍼", "슬프", "눈물", "울고 싶", "눈물이 나",
        "마음이 아", "가슴이 먹먹", "마음이 무거",
        "의욕이 없", "의욕", "재미가 없", "재미없",
        "삶이 무의미", "공허", "허무", "허탈",
        "비참", "암담", "절망", "희망이 없",
    ],
    KeywordCategory.EMOTIONAL_ANXIETY: [
        "불안", "불안해", "불안하", "걱정", "걱정되",
        "초조", "조마조마", "안절부절", "가슴이 두근",
        "두근두근", "긴장", "긴장되", "두려", "두렵",
        "무서", "무섭", "겁", "겁나", "공포", "패닉",
        "심장이 빨리", "손에 땀", "식은땀",
    ],
    KeywordCategory.EMOTIONAL_LONELINESS: [
        "외로", "외롭", "혼자", "혼자서", "홀로",
        "버림받", "버려진", "소외", "고독", "쓸쓸",
        "적적", "허전", "아무도 없", "나 혼자만",
        "친구가 없", "가족이 없", "연락도 없",
    ],
    KeywordCategory.EMOTIONAL_FEAR: [
        "무서워", "무섭다", "두려워", "두렵다", "겁나",
        "겁이 나", "공포", "떨려", "떨리다", "소름",
        "오싹", "등골이 서늘", "식겁", "혼비백산",
    ],
    KeywordCategory.EMOTIONAL_ANGER: [
        # "화" 단독 제거 - "대화", "회화", "전화" 등 오탐지 방지
        "화가 나", "화나", "화났", "화내", "짜증", "빡치",
        "열받", "분노", "억울", "원망", "증오",
        "답답", "답답해", "미치겠", "돌아버리",
        "폭발할 것 같", "참을 수 없", "화가 치밀",
    ],
    
    # ==================== 자해/자살 관련 (최고 위험) ====================
    KeywordCategory.SUICIDE_IDEATION: [
        "죽고 싶", "죽을래", "죽어버리", "죽었으면",
        "살기 싫", "살고 싶지 않", "사라지고 싶",
        "없어지고 싶", "다 끝내", "끝내버리",
        "세상을 떠나", "이 세상에 없", "죽음",
        "자살", "투신", "목을 매", "떨어지면",
        "이렇게 살 바에", "사는 게 힘들",
        "삶의 의미가 없", "태어나지 말았", "존재 의미",
    ],
    KeywordCategory.SELF_HARM: [
        "자해", "자상", "손목을 그", "팔을 긋",
        "피가 나", "상처", "상처를 내", "아프게 해",
        "스스로 다치", "나를 해치", "때리고 싶",
    ],
    
    # ==================== 긴급 상황 ====================
    KeywordCategory.EMERGENCY_FALL: [
        "넘어", "넘어졌", "쓰러", "쓰러졌", "굴러",
        "떨어", "떨어졌", "미끄러", "미끄러졌",
        "주저앉", "엎어", "자빠", "꽈당", "쿵",
        "일어나기 힘", "못 일어나", "바닥에",
    ],
    KeywordCategory.EMERGENCY_HELP: [
        "도와줘", "도와주세요", "살려줘", "살려주세요",
        "구해줘", "119", "응급", "응급실", "병원",
        "구급차", "앰뷸런스", "빨리 와", "어서 와",
        "큰일", "큰일났", "위급", "긴급", "급해",
    ],
    KeywordCategory.EMERGENCY_ACCIDENT: [
        "사고", "사고가 나", "다쳤", "다치", "부딪혔",
        "부딪", "충돌", "화상", "데였", "베였",
        "찔렸", "깨졌", "골절", "부러", "부러졌",
        "피나", "피가 나", "출혈", "멍", "타박상",
    ],
    
    # ==================== 건강 걱정 ====================
    KeywordCategory.HEALTH_CONCERN: [
        "병원", "병원에 가", "검사", "진찰", "진료",
        "약 먹", "약을 안 먹", "약이 없", "처방",
        "어디 아프", "몸이 안 좋", "건강이 안 좋",
        "컨디션", "컨디션이 안", "최근에 아",
        "자꾸 아파", "계속 아파", "안 낫", "안 나아",
    ],
}

# 카테고리별 위험 수준 매핑
CATEGORY_RISK_LEVELS: dict[KeywordCategory, RiskLevel] = {
    # 통증 - caution (주의)
    KeywordCategory.PAIN_HEAD: RiskLevel.CAUTION,
    KeywordCategory.PAIN_CHEST: RiskLevel.CRITICAL,  # 흉통은 critical
    KeywordCategory.PAIN_ABDOMINAL: RiskLevel.CAUTION,
    KeywordCategory.PAIN_BACK: RiskLevel.CAUTION,
    KeywordCategory.PAIN_JOINT: RiskLevel.CAUTION,
    KeywordCategory.PAIN_GENERAL: RiskLevel.CAUTION,
    
    # 증상 - 상황에 따라 다름
    KeywordCategory.SYMPTOM_DIZZINESS: RiskLevel.CAUTION,
    KeywordCategory.SYMPTOM_BREATHING: RiskLevel.CRITICAL,  # 호흡곤란은 critical
    KeywordCategory.SYMPTOM_NAUSEA: RiskLevel.CAUTION,
    KeywordCategory.SYMPTOM_WEAKNESS: RiskLevel.CAUTION,
    KeywordCategory.SYMPTOM_NUMBNESS: RiskLevel.CRITICAL,   # 마비는 critical (뇌졸중 의심)
    KeywordCategory.SYMPTOM_FEVER: RiskLevel.CAUTION,
    KeywordCategory.SYMPTOM_SLEEP: RiskLevel.CAUTION,
    
    # 정서적 고통 - caution
    KeywordCategory.EMOTIONAL_DEPRESSION: RiskLevel.CAUTION,
    KeywordCategory.EMOTIONAL_ANXIETY: RiskLevel.CAUTION,
    KeywordCategory.EMOTIONAL_LONELINESS: RiskLevel.CAUTION,
    KeywordCategory.EMOTIONAL_FEAR: RiskLevel.CAUTION,
    KeywordCategory.EMOTIONAL_ANGER: RiskLevel.CAUTION,
    
    # 자해/자살 - critical (최고 위험)
    KeywordCategory.SUICIDE_IDEATION: RiskLevel.CRITICAL,
    KeywordCategory.SELF_HARM: RiskLevel.CRITICAL,
    
    # 긴급 상황 - critical
    KeywordCategory.EMERGENCY_FALL: RiskLevel.CRITICAL,
    KeywordCategory.EMERGENCY_HELP: RiskLevel.CRITICAL,
    KeywordCategory.EMERGENCY_ACCIDENT: RiskLevel.CRITICAL,
    
    # 건강 걱정 - caution
    KeywordCategory.HEALTH_CONCERN: RiskLevel.CAUTION,
}

# 카테고리별 한글 라벨
CATEGORY_LABELS: dict[KeywordCategory, str] = {
    KeywordCategory.PAIN_HEAD: "두통",
    KeywordCategory.PAIN_CHEST: "흉통",
    KeywordCategory.PAIN_ABDOMINAL: "복통",
    KeywordCategory.PAIN_BACK: "요통/등통증",
    KeywordCategory.PAIN_JOINT: "관절통",
    KeywordCategory.PAIN_GENERAL: "통증",
    
    KeywordCategory.SYMPTOM_DIZZINESS: "어지럼증",
    KeywordCategory.SYMPTOM_BREATHING: "호흡곤란",
    KeywordCategory.SYMPTOM_NAUSEA: "구역질/메스꺼움",
    KeywordCategory.SYMPTOM_WEAKNESS: "무기력/피로",
    KeywordCategory.SYMPTOM_NUMBNESS: "저림/마비",
    KeywordCategory.SYMPTOM_FEVER: "발열",
    KeywordCategory.SYMPTOM_SLEEP: "수면장애",
    
    KeywordCategory.EMOTIONAL_DEPRESSION: "우울감",
    KeywordCategory.EMOTIONAL_ANXIETY: "불안감",
    KeywordCategory.EMOTIONAL_LONELINESS: "외로움",
    KeywordCategory.EMOTIONAL_FEAR: "두려움",
    KeywordCategory.EMOTIONAL_ANGER: "분노",
    
    KeywordCategory.SUICIDE_IDEATION: "자살사고",
    KeywordCategory.SELF_HARM: "자해언급",
    
    KeywordCategory.EMERGENCY_FALL: "낙상",
    KeywordCategory.EMERGENCY_HELP: "긴급도움요청",
    KeywordCategory.EMERGENCY_ACCIDENT: "사고/부상",
    
    KeywordCategory.HEALTH_CONCERN: "건강걱정",
}


@dataclass
class KeywordDetectionResult:
    """키워드 감지 결과"""
    detected: bool
    category: Optional[KeywordCategory] = None
    category_label: Optional[str] = None
    matched_keyword: Optional[str] = None
    risk_level: RiskLevel = RiskLevel.NORMAL
    original_text: Optional[str] = None
    
    # 다중 매칭된 경우
    all_matches: list[tuple[KeywordCategory, str]] = None
    
    def __post_init__(self):
        if self.all_matches is None:
            self.all_matches = []


class NegativeKeywordDetector:
    """부정적 키워드 감지기"""
    
    def __init__(self):
        # 컴파일된 정규식 패턴 (성능 최적화)
        self._compiled_patterns: dict[KeywordCategory, list[re.Pattern]] = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """키워드 패턴을 정규식으로 컴파일"""
        for category, keywords in KEYWORD_PATTERNS.items():
            self._compiled_patterns[category] = []
            for keyword in keywords:
                # 부분 일치 패턴 (형태소 변형 허용)
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                self._compiled_patterns[category].append((keyword, pattern))
    
    def detect(self, text: str) -> KeywordDetectionResult:
        """텍스트에서 부정적 키워드 감지
        
        Args:
            text: 분석할 텍스트 (사용자 발화)
            
        Returns:
            KeywordDetectionResult: 감지 결과
        """
        if not text or not text.strip():
            return KeywordDetectionResult(detected=False)
        
        text_lower = text.lower()
        all_matches: list[tuple[KeywordCategory, str]] = []
        highest_risk = RiskLevel.NORMAL
        highest_risk_match: tuple[KeywordCategory, str] = None
        
        # 모든 카테고리에서 매칭 검색
        for category, patterns in self._compiled_patterns.items():
            for keyword, pattern in patterns:
                if pattern.search(text_lower):
                    all_matches.append((category, keyword))
                    
                    category_risk = CATEGORY_RISK_LEVELS.get(category, RiskLevel.CAUTION)
                    
                    # 가장 높은 위험 수준 추적
                    if category_risk == RiskLevel.CRITICAL:
                        highest_risk = RiskLevel.CRITICAL
                        if highest_risk_match is None or highest_risk != RiskLevel.CRITICAL:
                            highest_risk_match = (category, keyword)
                    elif category_risk == RiskLevel.CAUTION and highest_risk == RiskLevel.NORMAL:
                        highest_risk = RiskLevel.CAUTION
                        highest_risk_match = (category, keyword)
                    
                    # 첫 번째 critical 발견 시 바로 결과 반환 (최적화)
                    if category_risk == RiskLevel.CRITICAL:
                        break
            
            if highest_risk == RiskLevel.CRITICAL:
                break
        
        if not all_matches:
            return KeywordDetectionResult(detected=False)
        
        # 가장 중요한 매칭 결과 반환
        primary_category, primary_keyword = highest_risk_match or all_matches[0]
        
        result = KeywordDetectionResult(
            detected=True,
            category=primary_category,
            category_label=CATEGORY_LABELS.get(primary_category),
            matched_keyword=primary_keyword,
            risk_level=highest_risk,
            original_text=text,
            all_matches=all_matches,
        )
        
        logger.info(
            f"[KeywordDetector] Detected: category={primary_category.value}, "
            f"keyword='{primary_keyword}', risk={highest_risk.value}, "
            f"matches={len(all_matches)}, text='{text[:50]}...'"
        )
        
        return result


# 싱글톤 인스턴스 (전역 사용)
_detector_instance: Optional[NegativeKeywordDetector] = None


def get_detector() -> NegativeKeywordDetector:
    """싱글톤 감지기 인스턴스 반환"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = NegativeKeywordDetector()
    return _detector_instance


def detect_keywords(text: str) -> KeywordDetectionResult:
    """텍스트에서 부정적 키워드 감지 (편의 함수)"""
    return get_detector().detect(text)
