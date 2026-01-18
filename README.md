# OPS Agent

LiveKit 기반 어르신 돌봄 AI 음성 에이전트

## 구조

```
ops-agent/
├── agent/                    # 메인 에이전트 코드
│   ├── main.py              # 진입점 (AgentServer)
│   ├── voice_agent.py       # VoiceAgent 클래스
│   ├── llm_factory.py       # LLM Provider (aws/openai/ollama)
│   ├── pipeline_timer.py    # 파이프라인 타이밍 측정
│   ├── external/            # External AI Server 클라이언트
│   │   ├── external_stt.py  # HTTP STT 클라이언트
│   │   └── external_tts.py  # HTTP TTS 클라이언트
│   ├── prompts/             # 프롬프트 템플릿 (YAML + Jinja2)
│   ├── handlers/            # 이벤트 핸들러
│   ├── tools/               # Agent 도구 (AutoRAG, Time 등)
│   └── services/            # API, Redis 클라이언트
├── ai_server/               # GPU 모델 서버 (STT)
│   ├── server.py           # FastAPI 서버
│   └── services/           # Whisper STT, TTS, Embedding
├── mcp/                     # MCP 서버 (날씨 API)
└── docker-compose.dev.yml   # 개발 환경
```

## 환경 설정

```bash
cp agent.env.example agent.env
# agent.env 편집
```

### 필수 설정

| 변수 | 설명 |
|------|------|
| `LIVEKIT_URL` | LiveKit 서버 URL |
| `LIVEKIT_API_KEY` | LiveKit API Key |
| `LIVEKIT_API_SECRET` | LiveKit API Secret |
| `API_BASE_URL` | ops-api 서버 URL |
| `API_INTERNAL_TOKEN` | API 인증 토큰 |

### 권장 구성 (현재 사용 중)

```bash
# LLM: vLLM + Qwen3-8B-AWQ (GPU)
LLM_PROVIDER=openai
LLM_MODEL=Qwen/Qwen3-8B-AWQ
LLM_BASE_URL=http://localhost:8000/v1

# STT: AI Server + Whisper (GPU)
STT_PROVIDER=external
AI_SERVER_URL=http://localhost:8001

# TTS: AWS Polly
TTS_PROVIDER=aws
TTS_VOICE=Seoyeon

# MCP (날씨)
MCP_ENABLED=true
KMA_MCP_URL=http://localhost:8002/sse
```

### Provider 설정

#### STT Provider
```bash
STT_PROVIDER=aws        # AWS Transcribe
STT_PROVIDER=external   # ai_server (Whisper GPU) - 권장
```

#### LLM Provider
```bash
LLM_PROVIDER=aws        # AWS Bedrock Claude
LLM_PROVIDER=openai     # vLLM (Qwen3-8B-AWQ) - 권장
LLM_PROVIDER=ollama     # Local Ollama
```

#### TTS Provider
```bash
TTS_PROVIDER=aws        # AWS Polly (Seoyeon) - 권장
TTS_PROVIDER=external   # ai_server (Supertonic)
```

## 실행

### Docker Compose (개발)

```bash
# 전체 실행 (vllm + ai-server + agent + kma-mcp)
docker compose -f docker-compose.dev.yml up -d

# 로그 확인
docker compose -f docker-compose.dev.yml logs -f agent

# 상태 확인
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### 서비스 구성

| 서비스 | 포트 | 역할 |
|--------|------|------|
| vllm | 8000 | LLM (Qwen3-8B-AWQ, GPU) |
| ai-server | 8001 | STT (Whisper, GPU) |
| kma-mcp | 8002 | 날씨 API (MCP) |
| agent | - | 음성 에이전트 |
| transcript-storage | - | 대화 저장 |

### 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 에이전트 시작
python -m agent.main start
```

## 주요 기능

- **음성 대화**: LiveKit 기반 실시간 음성 통화
- **AutoRAG**: 자동 과거 대화 검색 및 컨텍스트 주입
- **Tool Calling**: 시간, 날씨 조회 (MCP)
- **위급 감지**: 낙상, 화재 등 알림 처리
- **Token 관리**: 자동 컨텍스트 truncation (MAX_CONTEXT_ITEMS)
