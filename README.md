# OPS Agent

**LiveKit 기반 실시간 AI 음성 에이전트** - 어르신 돌봄 서비스 '소담'의 핵심 음성 대화 엔진

## 프로젝트 개요

OPS Agent는 독거노인 및 요양 시설 어르신을 위한 **AI 말벗 서비스**의 음성 에이전트입니다. WebRTC 기반 실시간 음성 통화를 통해 자연스러운 대화를 제공하고, **위험 상황 감지** 및 **케어 알림** 기능으로 보호자와 요양보호사에게 즉각적인 알림을 전송합니다.

### 핵심 가치

| 기능 | 설명 |
|------|------|
| **실시간 음성 대화** | 200ms 이하 응답 지연으로 자연스러운 대화 경험 |
| **위험 감지 시스템** | 낙상, 화재, 부정 키워드 등 위급 상황 자동 감지 |
| **RAG 기반 컨텍스트** | 과거 대화 기억을 활용한 개인화된 응답 |
| **다중 센서 연동** | IoT 센서 데이터 기반 실시간 위험도 평가 |

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OPS Agent System                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                │
│   │   Mobile    │      │  LiveKit    │      │   ops-api   │                │
│   │    App      │◄────►│   Server    │◄────►│  (NestJS)   │                │
│   └─────────────┘      └──────┬──────┘      └──────┬──────┘                │
│                               │                    │                        │
│                               ▼                    ▼                        │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                        Voice Agent                                 │    │
│   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐              │    │
│   │  │   STT   │  │   LLM   │  │   TTS   │  │  Tools  │              │    │
│   │  │(Whisper)│─►│ (Qwen3) │─►│ (Polly) │  │(RAG,MCP)│              │    │
│   │  └─────────┘  └─────────┘  └─────────┘  └─────────┘              │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                               │                                             │
│          ┌────────────────────┼────────────────────┐                       │
│          ▼                    ▼                    ▼                        │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                │
│   │  Care Alert │      │  Transcript │      │    Redis    │                │
│   │   Handler   │      │   Storage   │      │   PubSub    │                │
│   └─────────────┘      └─────────────┘      └─────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 서비스 구성

| 서비스 | 포트 | 기술 스택 | 역할 |
|--------|------|-----------|------|
| **vLLM** | 8000 | Qwen3-8B-AWQ, CUDA | LLM 추론 서버 (Tool Calling 지원) |
| **AI Server** | 8001 | Whisper Large-v3-Turbo, CUDA | STT/TTS/VAD 처리 |
| **KMA MCP** | 8002 | FastAPI, MCP Protocol | 기상청 날씨 API |
| **Agent** | - | LiveKit Agents SDK | 음성 에이전트 메인 프로세스 |
| **Transcript Storage** | - | Redis PubSub | 대화 기록 비동기 저장 |

---

## 기술 스택

### Core

| 분류 | 기술 | 버전 | 용도 |
|------|------|------|------|
| **Runtime** | Python | 3.12 | 메인 런타임 |
| **Framework** | LiveKit Agents | 1.0+ | 실시간 음성 에이전트 프레임워크 |
| **LLM** | vLLM + Qwen3-8B-AWQ | - | 한국어 특화 LLM (4-bit 양자화) |
| **STT** | Faster-Whisper Large-v3-Turbo | - | GPU 가속 음성 인식 |
| **TTS** | AWS Polly (Seoyeon) | - | 한국어 음성 합성 |

### Infrastructure

| 분류 | 기술 | 용도 |
|------|------|------|
| **Message Broker** | Redis PubSub | 실시간 이벤트 전파 |
| **Container** | Docker Compose | 개발/배포 환경 |
| **GPU** | NVIDIA CUDA | LLM/STT 가속 |
| **Protocol** | WebRTC (LiveKit) | 실시간 음성 스트리밍 |
| **API** | MCP (Model Context Protocol) | 외부 도구 연동 |

---

## 프로젝트 구조

```
ops-agent/
├── agent/                          # 메인 에이전트 코드
│   ├── main.py                     # 진입점 (WorkerOptions, AgentSession)
│   ├── voice_agent.py              # VoiceAgent 클래스 (Agent 상속)
│   ├── config.py                   # 환경변수 설정
│   ├── constants.py                # 상수 정의
│   ├── userdata.py                 # 세션별 사용자 데이터
│   │
│   ├── handlers/                   # 이벤트 핸들러
│   │   ├── care_alert.py           # 케어 알림 (riskLevel, riskScore)
│   │   ├── negative_keyword_detector.py  # 부정 키워드 감지
│   │   ├── sensor_detector.py      # IoT 센서 기반 위험 감지
│   │   ├── transcript.py           # 대화 기록 처리
│   │   ├── session.py              # 세션 생명주기 관리
│   │   └── takeover.py             # 관리자 개입 처리
│   │
│   ├── llm/                        # LLM 관련 모듈
│   │   └── token_budget.py         # 동적 토큰 예산 관리
│   │
│   ├── prompts/                    # 프롬프트 엔지니어링
│   │   ├── builder.py              # Jinja2 템플릿 빌더
│   │   ├── sodam.yaml              # 페르소나 및 규칙 정의
│   │   └── greeting.py             # 인사말 생성
│   │
│   ├── rag/                        # RAG (Retrieval-Augmented Generation)
│   │   ├── orchestrator.py         # RAG 파이프라인 오케스트레이션
│   │   ├── context_formatter.py    # 검색 결과 포맷팅
│   │   ├── temporal_parser.py      # 시간 표현 파싱 ("어제", "지난주" 등)
│   │   ├── threshold_filter.py     # 유사도 임계값 필터링
│   │   └── prompt_template.py      # RAG 프롬프트 템플릿
│   │
│   ├── tools/                      # Agent Tools (Function Calling)
│   │   ├── auto_rag.py             # 자동 RAG 검색 도구
│   │   ├── ward_info.py            # 어르신 정보 조회 도구
│   │   ├── time.py                 # 시간/날짜 조회 도구
│   │   └── memory.py               # 메모리 관리 도구
│   │
│   ├── services/                   # 외부 서비스 클라이언트
│   │   ├── api_client.py           # ops-api REST 클라이언트
│   │   └── redis_pubsub.py         # Redis PubSub 클라이언트
│   │
│   ├── external/                   # External Provider 클라이언트
│   │   ├── external_stt.py         # AI Server STT 클라이언트
│   │   ├── external_tts.py         # AI Server TTS 클라이언트
│   │   └── external_turn_detector.py  # 턴 감지 클라이언트
│   │
│   ├── stt/                        # STT 구현체
│   │   └── faster_whisper_stt.py   # Faster-Whisper 래퍼
│   │
│   ├── storage/                    # 데이터 저장
│   │   └── transcript_listener.py  # 대화 기록 Redis 리스너
│   │
│   ├── llm_factory.py              # LLM Provider Factory
│   ├── pipeline_timer.py           # 파이프라인 성능 측정
│   ├── rag_client.py               # RAG API 클라이언트
│   ├── embedding_client.py         # 임베딩 API 클라이언트
│   └── warmup_service.py           # vLLM 웜업 서비스
│
├── ai_server/                      # GPU 모델 서버
│   ├── server.py                   # FastAPI 서버 (STT/TTS/VAD)
│   ├── Dockerfile                  # GPU 컨테이너 빌드
│   ├── requirements.txt            # Python 의존성
│   ├── scripts/                    # 유틸리티 스크립트
│   └── services/                   # 모델 서비스 구현체
│
├── mcp/                            # MCP 서버 (기상청 API)
│   ├── kma_server.py               # 기상청 API MCP 서버
│   ├── Dockerfile                  # 컨테이너 빌드
│   └── requirements.txt            # Python 의존성
│
├── docker-compose.dev.yml          # 개발 환경 Docker Compose
├── Dockerfile                      # Agent 컨테이너 빌드
├── requirements.txt                # Agent Python 의존성
├── agent.env.example               # 환경변수 템플릿
└── README.md                       # 프로젝트 문서
```

---

## 핵심 기능 상세

### 1. 실시간 음성 대화 파이프라인

```
Audio Input → VAD → STT → LLM (+ Tool Calling) → TTS → Audio Output
     │                          │
     └──────── ~200ms ──────────┘
```

- **VAD (Voice Activity Detection)**: Silero VAD로 음성 구간 감지
- **STT**: Whisper Large-v3-Turbo (GPU 가속, ~50ms)
- **LLM**: Qwen3-8B-AWQ + vLLM (Prefix Caching, ~100ms)
- **TTS**: AWS Polly Neural (Seoyeon, ~50ms)

### 2. 케어 알림 시스템 (Care Alert)

위험 상황 감지 시 보호자/요양보호사에게 실시간 알림 전송:

```python
# 위험도 레벨
riskLevel: 'normal' | 'caution' | 'critical'

# 트리거 조건
- 낙상 감지 (IoT 센서)
- 화재 감지 (IoT 센서)
- 부정 키워드 ("아파", "쓰러졌어", "도와줘" 등)
- 비정상 바이탈 사인
```

### 3. 동적 토큰 예산 시스템

질문 유형에 따라 응답 길이를 자동 조절:

| 질문 유형 | 패턴 예시 | max_tokens |
|-----------|-----------|------------|
| 단순 질문 | "몇 시야?", "이름이 뭐야?" | 40~50 |
| 일반 대화 | "오늘 뭐 했어?", "밥 먹었어?" | 80~100 |
| 상세 설명 | "건강이 안 좋아", "걱정이 돼" | 120~150 |

### 4. RAG (Retrieval-Augmented Generation)

과거 대화를 검색하여 개인화된 응답 생성:

- **시간 표현 파싱**: "어제", "지난주", "3일 전" 등 자연어 시간 표현 처리
- **유사도 필터링**: 임계값 기반 관련성 높은 대화만 선별
- **컨텍스트 포맷팅**: LLM 프롬프트에 최적화된 형식으로 변환

### 5. Tool Calling (Function Calling)

| 도구 | 기능 | 트리거 예시 |
|------|------|-------------|
| `get_current_time` | 현재 시간/날짜 조회 | "지금 몇 시야?" |
| `get_current_weather` | 현재 날씨 조회 | "오늘 날씨 어때?" |
| `get_ward_info` | 어르신 정보 조회 | "내 이름이 뭐야?" |
| `search_memory` | 과거 대화 검색 | (자동 RAG) |

---

## 환경 설정

### 1. 환경변수 파일 생성

```bash
cp agent.env.example agent.env
```

### 2. 필수 설정

```bash
# ===================================================================
# [필수] LiveKit Connection
# ===================================================================
LIVEKIT_URL=wss://your-livekit-server.com
LIVEKIT_API_KEY=<YOUR_LIVEKIT_API_KEY>
LIVEKIT_API_SECRET=<YOUR_LIVEKIT_API_SECRET>

# ===================================================================
# [필수] API Server
# ===================================================================
API_BASE_URL=https://your-api-server.com
API_INTERNAL_TOKEN=<YOUR_API_INTERNAL_TOKEN>

# ===================================================================
# [필수] AWS Credentials (Polly TTS)
# ===================================================================
AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY>
AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_KEY>
AWS_DEFAULT_REGION=ap-northeast-2

# ===================================================================
# [필수] Redis
# ===================================================================
REDIS_URL=redis://localhost:6379
```

### 3. Provider 설정

#### LLM Provider

```bash
# vLLM + Qwen3-8B-AWQ (권장)
LLM_PROVIDER=openai
LLM_MODEL=Qwen/Qwen3-8B-AWQ
LLM_BASE_URL=http://localhost:8000/v1

# AWS Bedrock Claude
LLM_PROVIDER=aws

# Local Ollama
LLM_PROVIDER=ollama
```

#### STT Provider

```bash
# AI Server + Whisper GPU (권장)
STT_PROVIDER=external
AI_SERVER_URL=http://localhost:8001

# AWS Transcribe
STT_PROVIDER=aws
```

#### TTS Provider

```bash
# AWS Polly (권장)
TTS_PROVIDER=aws
TTS_VOICE=Seoyeon

# AI Server (Supertonic)
TTS_PROVIDER=external
```

---

## 실행 방법

### Docker Compose (권장)

```bash
# 전체 서비스 실행
docker compose -f docker-compose.dev.yml up -d

# 로그 확인
docker compose -f docker-compose.dev.yml logs -f agent

# 상태 확인
docker compose -f docker-compose.dev.yml ps

# 서비스 중지
docker compose -f docker-compose.dev.yml down
```

### 서비스 시작 순서

1. **vLLM** (LLM 서버) - 120초 대기 후 health check
2. **AI Server** (STT/TTS) - 60초 대기 후 health check
3. **vLLM Warmup** (캐시 워밍업) - 완료 후 종료
4. **Agent** (음성 에이전트) - vLLM, AI Server, Warmup 완료 후 시작
5. **Transcript Storage** (대화 저장) - Agent 시작 후 시작
6. **KMA MCP** (날씨 API) - 독립 실행

### 로컬 실행 (개발용)

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 에이전트 시작
python -m agent.main start

# 대화 저장 리스너 시작 (별도 터미널)
python -m agent.storage.transcript_listener
```

---

## vLLM 설정

### GPU 메모리 최적화

```yaml
command:
  - --model
  - Qwen/Qwen3-8B-AWQ
  - --quantization
  - awq_marlin              # 4-bit 양자화
  - --max-model-len
  - "32768"                 # 최대 컨텍스트 길이
  - --gpu-memory-utilization
  - "0.65"                  # GPU 메모리 사용률
  - --dtype
  - half                    # FP16
  - --enable-auto-tool-choice
  - --tool-call-parser
  - hermes                  # Tool Calling 파서
  - --enable-prefix-caching # 프롬프트 캐싱
```

### 토큰 예산

| 항목 | 토큰 수 |
|------|---------|
| 시스템 프롬프트 | ~10,400 |
| 사용자 컨텍스트 | ~5,000 |
| RAG 컨텍스트 | ~5,000 |
| 응답 | ~150 |
| **총합** | ~20,550 |
| **max-model-len** | 32,768 |

---

## 성능 지표

| 메트릭 | 목표 | 현재 |
|--------|------|------|
| 응답 지연 (E2E) | < 500ms | ~200ms |
| STT 지연 | < 100ms | ~50ms |
| LLM 지연 | < 200ms | ~100ms |
| TTS 지연 | < 100ms | ~50ms |
| 동시 세션 | 10+ | 10 |

---

## 관련 저장소

| 저장소 | 설명 |
|--------|------|
| [ops-api](../ops-api) | NestJS 백엔드 API 서버 |
| [ops-web](../ops-web) | Next.js 관리자 웹 대시보드 |

---

## 라이선스

Private - All Rights Reserved
