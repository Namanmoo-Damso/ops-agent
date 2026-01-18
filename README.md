# OPS Agent

LiveKit 기반 어르신 돌봄 AI 음성 에이전트

## 구조

```
ops-agent/
├── agent/                    # 메인 에이전트 코드
│   ├── main.py              # 진입점 (AgentServer)
│   ├── voice_agent.py       # VoiceAgent 클래스
│   ├── llm_factory.py       # LLM Provider (aws/openai/ollama)
│   ├── external/            # External AI Server 클라이언트
│   │   ├── external_stt.py  # HTTP STT 클라이언트
│   │   ├── external_tts.py  # HTTP TTS 클라이언트
│   │   └── external_turn_detector.py
│   ├── prompts/             # 프롬프트 템플릿
│   ├── handlers/            # 이벤트 핸들러
│   ├── tools/               # Agent 도구 (RAG 등)
│   └── services/            # API, Redis 클라이언트
├── stt/                     # Custom STT 서버 클라이언트
├── ai_server/               # GPU 모델 서버 (선택)
│   ├── server.py           # FastAPI 서버
│   └── services/           # STT, TTS, Embedding 서비스
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

### Provider 설정

#### STT Provider
```bash
STT_PROVIDER=aws        # AWS Transcribe (기본)
STT_PROVIDER=custom     # stt/ 서버 (FasterWhisper)
STT_PROVIDER=external   # ai_server (ExternalSTT)
```

#### LLM Provider
```bash
LLM_PROVIDER=aws        # AWS Bedrock Claude (기본)
LLM_PROVIDER=openai     # OpenAI-compatible (vLLM 등)
LLM_PROVIDER=ollama     # Local Ollama
```

#### TTS Provider
```bash
TTS_PROVIDER=aws        # AWS Polly (기본)
TTS_PROVIDER=external   # ai_server (Supertonic)
```

## 실행

### Docker Compose (개발)

```bash
# 전체 실행 (stt + ollama + agent)
docker compose -f docker-compose.dev.yml up -d

# 로그 확인
docker compose -f docker-compose.dev.yml logs -f agent
```

### 로컬 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 에이전트 시작
python -m agent.main start
```

## AI Server (선택)

로컬 GPU로 STT/TTS/Embedding을 처리하려면:

```bash
cd ai_server
docker build -t ai-server .
docker run --gpus all -p 8001:8001 ai-server
```

설정:
```bash
STT_PROVIDER=external
TTS_PROVIDER=external
AI_SERVER_URL=http://localhost:8001
```

## 주요 기능

- **음성 대화**: LiveKit 기반 실시간 음성 통화
- **RAG 기억**: 과거 대화 검색 및 컨텍스트 주입
- **날씨 조회**: KMA API 연동 (MCP)
- **위급 감지**: 낙상, 화재 등 알림 처리
