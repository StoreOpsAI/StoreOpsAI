# StoreOps AI · 3.3 질문 파트 — Windows 설치와 실행 (처음부터 끝까지)

Qwen3.6-35B-A3B를 내 PC에서 `llama-server`로 띄우고, 그 위에서 질문 Agent(도구 3개 + 매뉴얼 RAG)를 돌리는 순서입니다.
모든 명령은 **PowerShell**에서 실행합니다.

```
점주 질문 ─▶ Agent(Python) ─▶ llama-server(Qwen3.6-35B-A3B)   ← 도구 이름·입력값만 정함
                │
                ├─ query_records  ─▶ SQLite (사건·발주 초안)
                ├─ search_manual  ─▶ bge-m3 임베딩 + M001 조각 카드 ─▶ Qwen이 문장별 근거 답변
                └─ get_event_video ─▶ SQLite
```

---

## 0. 내 PC가 되는지 확인

Qwen3.6-35B-A3B는 전체 350억 파라미터지만 토큰당 30억만 쓰는 MoE 모델입니다. 4비트 양자화(UD-Q4_K_XL) 기준으로 **VRAM + 시스템 RAM 합계가 약 23GB 이상** 필요합니다(Unsloth 문서 기준). 디스크는 모델 파일용으로 **25GB 이상** 비워 두세요.

| 내 PC                              | 4단계 이후 `start_llm.ps1` 실행법                           |
| ---------------------------------- | ----------------------------------------------------------- |
| NVIDIA GPU VRAM 24GB 이상          | `.\scripts\start_llm.ps1`                                   |
| NVIDIA GPU VRAM 12~16GB + RAM 32GB | `.\scripts\start_llm.ps1 -NCpuMoe 24` (숫자는 아래 팁 참고) |
| GPU 없음 / 약한 GPU, RAM 32GB      | `.\scripts\start_llm.ps1 -NCpuMoe 99 -Ngl 0` (느립니다)     |

> 팁: `-NCpuMoe` 숫자가 클수록 GPU 메모리를 덜 쓰고 느려집니다. 메모리 부족 오류가 나면 숫자를 올리고, 여유가 남으면 내려서 속도를 맞춥니다.

NVIDIA GPU라면 드라이버를 최신으로 올리고 확인합니다.

```powershell
nvidia-smi
```

---

## 1. Python 설치 (3.11 이상)

```powershell
winget install -e --id Python.Python.3.12
```

설치 후 **PowerShell을 닫았다 다시 열고** 확인합니다.

```powershell
py --version
```

---

## 2. 프로젝트 풀기와 가상환경

받은 `storeops_qna.zip`을 원하는 곳(예: `C:\work\storeops_qna`)에 풉니다.

```powershell
cd C:\work\storeops_qna
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

`Activate.ps1`이 막히면(실행 정책 오류) 이 창에서만 허용하고 다시 실행합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

패키지를 설치하고 **먼저 모의 LLM 테스트**를 돌려 코드가 정상인지 확인합니다(Qwen 없이도 됩니다).

```powershell
pip install -r requirements.txt
python -m unittest discover -s tests
```

`OK`와 함께 `Ran 42 tests`가 나오면 정상입니다.

---

## 3. llama.cpp 설치 (llama-server)

1. 브라우저에서 <https://github.com/ggml-org/llama.cpp/releases> 의 최신 릴리스를 엽니다.
2. Assets에서 Windows용 압축 파일을 받습니다. 파일 이름은 빌드마다 조금씩 다를 수 있습니다.
   - **NVIDIA GPU**: `...bin-win-cuda-...-x64.zip` 과 같은 CUDA 버전의 `cudart-...win-cuda-...-x64.zip` 두 개
   - **AMD/Intel GPU**: `...bin-win-vulkan-x64.zip`
   - **GPU 없음**: `...bin-win-cpu-x64.zip` 계열
3. 이 저장소에는 `../llama.cpp/llama.cpp/llama-server.exe`가 이미 있습니다. CUDA 런타임 DLL도 `../llama.cpp/llama.cpp/cudart-llama-bin-win-cuda-12.4-x64/`에 들어 있으며 실행 스크립트가 해당 경로를 자동으로 추가합니다. 다른 곳에 설치했다면 실행 시 `-LlamaDir`로 `llama-server.exe`가 있는 폴더를 지정하세요.

> ⚠️ **CUDA 13.2 빌드는 피하세요.** Unsloth 문서에 CUDA 13.2에서 깨진 글자(gibberish)가 나올 수 있다는 경고가 있습니다. 헷갈리면 12.x 빌드를 받으면 됩니다.

확인:

```powershell
..\llama.cpp\llama.cpp\llama-server.exe --version
```

---

## 4. 모델 다운로드 (Qwen3.6-35B-A3B, 약 22GB)

Unsloth가 올려 둔 GGUF(양자화) 파일을 받습니다. 아래 명령은 이 프로젝트 가상환경이 켜진 상태에서 실행합니다.

```powershell
pip install -U huggingface_hub
hf download unsloth/Qwen3.6-35B-A3B-GGUF --local-dir C:\models\qwen3.6-35b-a3b --include "*UD-Q4_K_XL*"
```

- 사진·영상 입력용 `mmproj` 파일은 필요 없으니 받지 않습니다. 이 프로젝트는 텍스트만 씁니다.
- 중간에 끊기면 같은 명령을 다시 실행하면 이어서 받습니다.
- 메모리가 모자라 더 작은 모델이 필요하면 `UD-Q4_K_XL` 대신 `UD-Q2_K_XL`(Unsloth가 권하는 최소 2비트)을 쓸 수 있지만, 품질은 떨어집니다.

---

## 5. Qwen 서버 실행

**새 PowerShell 창**을 하나 더 열어 프로젝트 폴더에서 실행합니다(이 창은 계속 켜 둡니다).

```powershell
cd C:\work\storeops_qna
.\scripts\start_llm.ps1          # 0단계 표에서 내 PC에 맞는 옵션을 붙이세요
```

모델을 불러오는 데 1~몇 분 걸립니다. 로그에 `server is listening` 같은 문구가 보이면 준비된 것입니다.

원래 창으로 돌아와 확인합니다.

```powershell
Invoke-RestMethod http://127.0.0.1:8003/v1/models
```

`qwen3.6-35b-a3b`가 목록에 나오면 `config.yaml`의 `llm.model`과 일치하는 것입니다.

간단한 대화 시험:

```powershell
$body = @{ model="qwen3.6-35b-a3b"; messages=@(@{role="user"; content="한 문장으로 자기소개 해 줘"}); max_tokens=100 } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8003/v1/chat/completions -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

---

## 6. 질문 기능 준비 (DB와 규정 문서)

프로젝트 창(가상환경 켜진 상태)에서:

```powershell
python -m storeops_qna.cli init-db     # 기획서 예시 값(E014, E015, 합성 P001)을 DB에 넣기
python -m storeops_qna.cli ingest      # M001 규정을 조각 카드로 저장 (임베딩 생성)
```

`ingest`는 처음에 임베딩 모델 **BAAI/bge-m3**(약 2GB+)를 내려받아서 몇 분 걸립니다. 한 번만 받으면 됩니다. CPU만으로도 충분합니다.

---

## 7. 질문해 보기

```powershell
python -m storeops_qna.cli ask "어제 사건이 몇 건이었나요?"
python -m storeops_qna.cli ask "어제 쓰러짐 사건 영상 보여 줘"
python -m storeops_qna.cli ask "카메라 연결 끊김 알림을 확인한 뒤 어떤 내용을 기록해야 하나요?"
python -m storeops_qna.cli ask "E15 영상 보여 줘"
python -m storeops_qna.cli ask "발주 승인해 줘"
```

> `config.yaml`의 `agent.demo_now`는 `null`이므로 "어제"는 현재 한국 날짜로 해석됩니다. `scripts/smoke_live.py`만 예시 데이터 검증을 위해 2026-03-15를 사용합니다. 위의 예시 사건(E014/E015)은 2026-03-14의 시연 데이터입니다.

실행 로그(단계마다 부른 도구, 입력값, 결과 상태, 고른 이유, 반려 사유)는 이렇게 봅니다.

```powershell
python -m storeops_qna.cli log 1       # 1은 질문 끝에 출력된 session_id
```

**명세서 8절 질문 검수 항목을 한 번에 돌리기** (Qwen 서버가 켜져 있어야 함):

```powershell
python scripts\smoke_live.py
```

LLM 응답은 매번 조금씩 달라서 한두 항목이 실패할 수 있습니다. 실패하면 한두 번 더 돌려 보고, 계속 실패하면 `log`로 어떤 도구를 왜 골랐는지 확인하세요.

---

## 8. API 서버로 띄우기 (화면 3 연결용)

```powershell
python -m storeops_qna.cli serve       # http://127.0.0.1:8000
```

기존 StoreOpsAI 백엔드가 8000 포트를 사용 중이면 `python -m storeops_qna.cli serve --port 8002`로 실행하고 아래 API 주소도 8002로 바꿉니다. 이 명령은 토큰 없이 실행할 때만 시연용 `X-Owner-Id` API를 엽니다. 운영 화면의 로그인 연동은 아래 절차를 사용합니다.

```powershell
$body = @{ question="어제 사건이 몇 건이었나요?" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/ask -Method Post -ContentType "application/json; charset=utf-8" `
  -Headers @{ "X-Owner-Id"="owner-01" } -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

응답에는 `answer`, `conditions`(조회 조건), `sources`(출처), `notices`(0건·실패·미수신 문장), `grounded_sentences`(규정 답변의 문장별 근거)가 들어 있습니다. 브라우저에서 `http://127.0.0.1:8000/docs`를 열면 시험용 화면도 있습니다. 점주 식별은 시연용 `X-Owner-Id` 헤더이며 실제 로그인으로 바꿔야 합니다.

## 9. 기존 화면·운영 DB와 연결

로그인한 점주의 질문은 프런트엔드 `/api/ask` → 기존 백엔드(로그인 세션 검증, 매장 사건·발주 조회) → 질문 서버 `/internal/ask` → Qwen 순으로 처리합니다. 모델은 기존 DB나 승인 API를 직접 호출하지 않습니다. 질문 이력·규정 조각은 질문 서비스의 SQLite에 남으며, 사건과 발주 기록은 기존 사건 저장소와 수요 서비스에서 질문마다 읽습니다. 시연 DB 사건을 운영 질문에 섞지 않습니다.

1. `python -m storeops_qna.cli ingest`로 현재 `M001_StoreOps_AI_매장점검규정_v0.1.md`를 bge-m3로 색인합니다. 기존 같은 문서 ID의 검색 조각만 갱신합니다. 이 규정은 **검토용 초안·미시행**이며 답변과 문장별 근거에도 표시됩니다.
2. 루트 `.env`에 backend와 질문 Agent가 공유할 `STOREOPS_QNA_TOKEN`을 설정합니다. 비밀값은 저장소에 커밋하지 마세요. 질문 GGUF 파일이 기본 위치와 다르면 `QNA_LLM_MODEL_PATH`도 지정합니다.
3. Docker Desktop에서 WSL2 NVIDIA GPU 지원을 확인하고 저장소 루트에서 `docker compose up -d --build`를 실행합니다. Compose가 vLLM, llama.cpp, 질문 Agent를 포함한 전체 서비스를 함께 시작합니다. 첫 vLLM 실행은 모델을 Docker 볼륨에 내려받아 시간이 걸립니다. Docker에 GPU 접근 문제가 있으면 VLM은 WSL의 `storeops_ai/tools/start_qwen_vllm.sh`로 별도 실행하고, Compose의 `qwen-vllm` 서비스 대신 호스트 연결 설정이 필요합니다.
4. `docker compose ps`에서 `qwen-vllm`, `qna-llm`, `qna-agent`가 정상 상태인지 확인합니다. 질문 LLM은 `8003`, 질문 Agent는 `8002`, VLM은 `8001`에서 로컬 점검할 수 있습니다. 기존 수동 서버가 이 포트를 사용 중이면 먼저 해당 프로세스를 종료하세요.
5. `http://localhost:5173`에 로그인해 질문 탭에서 기록·규정을 질문합니다. 조회 조건, 출처 원문, 영상, 도구 로그는 같은 화면에서 확인할 수 있습니다.

Compose의 backend와 Agent는 내부 DNS 주소로 통신하며, 같은 토큰은 Compose 환경에서 주입됩니다. 운영 DB의 사건이 수신되지 않았거나 수요 서비스에 연결할 수 없으면 해당 도구는 0건 대신 조회 실패로 응답합니다.

---

## 문제가 생겼을 때

| 증상                                     | 해결                                                                                                                                                                                         |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 답이 깨진 글자·무의미한 글자             | CUDA 13.2 빌드인지 확인(12.x나 13.3으로 교체). `-Ctx`가 너무 작아도 생기니 16384 이상 유지. 계속되면 `start_llm.ps1`의 `serverArgs`에 `"--cache-type-k","bf16","--cache-type-v","bf16"` 추가 |
| `out of memory` / 서버가 바로 꺼짐       | `-NCpuMoe` 숫자를 올리거나 `-Ctx 8192`로 낮춤. 다른 GPU 프로그램 종료                                                                                                                        |
| 도구 호출이 안 되고 글로만 답함          | 서버를 `--jinja`로 띄웠는지 확인(`start_llm.ps1`은 포함). llama.cpp를 최신 빌드로 교체                                                                                                       |
| `unknown argument: --n-cpu-moe` 등       | llama.cpp 빌드가 오래됨 → 최신 릴리스로 교체                                                                                                                                                 |
| `llm_error` (모델에 연결하지 못했습니다) | `qna-llm` 컨테이너 상태와 포트 8003 확인. 단독 실행이면 `config.yaml`의 `llm.base_url` 확인                                                                                                  |
| 첫 답변이 매우 느림                      | 첫 요청은 모델이 메모리에 올라가는 시간이 포함됩니다. 이후 빨라집니다                                                                                                                        |
| `hf` 명령을 찾을 수 없음                 | 가상환경이 켜져 있는지 확인하고 `pip install -U huggingface_hub` 다시 실행                                                                                                                   |
| 한글이 깨져 보임                         | PowerShell에서 `chcp 65001` 실행 (CLI는 UTF-8 출력으로 설정돼 있음)                                                                                                                          |
| 사건 번호나 날짜를 모델이 틀리게 넣음    | 로그(`cli log`)에서 입력값 확인. 입력 검사가 틀린 값을 반려하고 점주에게 되묻는지 확인. 필요하면 `config.yaml`에서 `enable_thinking: true` 시도                                              |

---

## 구현한 것 ↔ 명세서 3.3 (FR-QNA)

| ID                                     | 어디에                                                          | 확인 방법                                         |
| -------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------- |
| 01 LLM은 도구와 입력값만 정함          | `agent.py` 제어 루프 (실행·검사·종료는 코드)                    | `test_agent`                                      |
| 02 도구 3개, 조회만                    | `tools.py` `TOOL_DEFS`, 허용 목록                               | `test_tool_outside_allowlist_is_rejected`         |
| 03 앞 결과를 보고 다음 도구 선택       | 루프가 도구 결과를 messages에 넣고 다시 LLM 호출                | `test_two_tools_in_sequence`                      |
| 04 한국 시간 날짜 범위, 조건 표시      | `timeutil.date_hints`를 프롬프트에 주입, `AskResult.conditions` | `test_yesterday_count_*`                          |
| 05 0건/실패/미수신 구분                | 도구가 세 상태와 고정 문장을 돌려주고 `notices`로 별도 표시     | `test_three_outcomes_*`                           |
| 06 끊김 사건 영상 요청                 | `get_event_video`가 `no_video` + 이유                           | `test_video_for_disconnect_*`                     |
| 07 조각 카드, 버전 변경 시 함께 갱신   | `manual_index.ingest_manual` (한 트랜잭션)                      | `test_new_version_replaces_*`                     |
| 08 문장마다 근거                       | `search_manual`이 JSON으로 문장별 chunk_id를 받아 검증          | `test_manual_*`                                   |
| 09 가까운 조각 없으면 근거 부족        | 유사도 기준값 + `answerable:false`                              | `test_manual_no_close_chunk_*`                    |
| 10 호출 전 입력 검사                   | `validate_input` — 고치지 않고 반려, "E015가 맞나요?" 되묻기    | `test_bad_event_id_*`                             |
| 11 연속 2회 실패 / 6번째 호출에서 멈춤 | `agent.py`                                                      | `test_two_consecutive_*`, `test_more_than_five_*` |
| 12 자기 매장만                         | 매장 번호는 로그인 점주에게서 결정, 다르면 거부                 | `test_other_store_*`                              |
| 13 State는 질문 동안만, 개인정보 없음  | 질문마다 messages를 새로 만듦                                   | `test_no_state_carries_over_*`                    |
| 14 단계별 실행 로그                    | `tool_calls` 테이블                                             | `cli log`                                         |

## 제가 정한 부분 (팀 확정 필요)

명세서에 값이 없거나 열려 있어서 임시로 정한 것들입니다. 모두 `config.yaml`이나 해당 모듈에서 바꿀 수 있습니다.

- **유사도 기준값 0.45** — 명세서가 4주차 미정으로 둔 값입니다. 질문 평가셋(D16)으로 반드시 보정하세요. 실제 bge-m3 점수는 아직 측정하지 못했습니다.
- **호출 횟수는 반려된 시도도 센다**, "E15"처럼 형식이 틀리면 **즉시 멈추고 점주에게 확인**합니다(되살려서 재시도하지 않음).
- **query_records 출력에 `records`(사건 종류·시각) 추가** — 7.2절 출력에는 없지만, "어제 쓰러짐 사건"의 번호를 찾으려면 필요합니다.
- **미수신 판단용 `data_coverage` 테이블 추가** — 어느 기간까지 받았는지 알아야 `not_received`를 구분할 수 있습니다.
- **규정 질문은 한 번의 `search_manual`로 끝나면 도구의 문장별 근거 답변을 그대로** 돌려줍니다. LLM이 다시 쓰면서 근거 연결이 끊기지 않게 하려는 선택입니다.
- **저장소는 SQLite + numpy 검색** — 명세서의 PostgreSQL + pgvector는 개발용 대안(6절 표)에 맞춰 단순화했습니다. 조각이 수십 개 수준이라 성능 차이는 없습니다.
- **LLM 샘플링은 temperature 0.3 / presence_penalty 0** — Qwen 공식 권장(비사고: 0.7, 0.8, presence 1.5)과 다릅니다. 도구 인자가 반복되는 용도라 낮췄으며, 이상하면 공식 값으로 돌려 보세요.
- **독립 시연 API의 `X-Owner-Id` 헤더** — 통합 실행에서는 비활성화되고 기존 백엔드가 로그인 세션을 검증합니다.

## 확인한 것과 남은 제약

- 이 저장소의 llama-server Windows 빌드(11312)에서 CUDA 12.4 DLL 경로를 추가하면 RTX A4000을 인식하고, `-NCpuMoe 24`로 Qwen3.6-35B-A3B 모델을 실행해 `/v1/models` 응답을 확인했습니다.
- Python 모의 LLM 테스트 42개와 기존 예시 규정의 실제 Qwen 검수 7개 시나리오를 확인했습니다. 새 점검 규정 초안도 실제 bge-m3로 조각 7개를 저장하고 규정 질문의 문장별 근거와 초안 경고를 확인했습니다. 다양한 질문에 대한 검색 품질과 유사도 기준 0.45는 아직 평가셋으로 보정하지 않았습니다.
- 기존 백엔드의 세션으로 매장 S01 운영 사건 E001을 조회하고, 실제 영상과 점검 규정 원문이 질문 탭에 표시되는 것을 확인했습니다. 이 질문 서비스는 호스트의 별도 Python 프로세스이므로 서버·토큰을 재시작할 때 9절의 환경변수를 다시 설정해야 합니다.
