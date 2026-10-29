# 세 PC Docker 연결 설정 기록

이 문서는 PC별 실행 설정과 남은 현장 확인 사항을 기록한다. 기존 `docker-compose.yml`은 한 PC 배치를 유지한다. 분리 배치에서는 백엔드 PC가 기존 Compose의 필요한 서비스만 실행하고, 나머지 두 PC가 전용 Compose 파일을 사용한다.

## 목표 배치

| 컴퓨터 | 실행할 서비스 | 다른 PC와 주고받는 데이터 |
| --- | --- | --- |
| 백엔드 PC | `backend`, `postgres`, `demand`, 필요한 경우 `qwen-vllm` | 탐지 PC에서 사건 정보와 영상·이미지를 받고, 프런트 PC에 사건 API와 미디어를 제공 |
| 웹캠·탐지 PC | `storeops-ai` 컨테이너와 Windows 호스트의 웹캠 분석 스크립트 | 웹캠 영상을 분석하고 사건 정보·완성된 미디어를 백엔드로 전송 |
| 프런트 PC | `frontend`와 브라우저 | 백엔드의 API·사건 미디어를 조회 |

프런트엔드는 React와 Nginx이며 FastAPI 서버가 아니다. 프런트 PC의 `/api` 프록시가 백엔드 FastAPI와 통신한다. 프런트 PC 자체에도 FastAPI 프로세스가 꼭 필요하다면 별도 게이트웨이가 필요하며, 현재 구현에는 포함되지 않는다.

## 방화벽 규칙을 바꾸지 않는 연결

세 PC에 같은 Tailscale 네트워크를 구성하고, 각 PC에서 필요한 로컬 서비스만 [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve)로 노출한다. 먼저 세 PC의 Tailscale 연결과 접속 정책을 확인한다. Tailscale은 보통 방화벽 규칙 변경 없이 동작하지만, 외부 연결이 제한된 네트워크에서는 연결 또는 속도를 실제 환경에서 확인해야 한다. 주소는 예시 IP를 코드에 넣지 말고 배포 환경의 Tailscale 이름이나 주소로 설정한다.

- 탐지 PC → 백엔드 PC: 사건 JSON, 이후 영상·이미지 업로드. `localhost`와 Compose 서비스명 `backend`는 다른 PC를 가리키지 않는다.
- 프런트 PC → 백엔드 PC: `/api` 요청. 영상도 백엔드의 인증된 미디어 API로 받는다.
- Docker 컨테이너에서도 상대 PC의 Tailscale 주소를 해석하고 접속할 수 있는지 확인한다. 호스트에서의 연결 성공만으로 컨테이너 연결을 보장할 수 없다.
- 탐지·프런트 전용 Compose는 포트를 `127.0.0.1`에 바인딩한다. 기존 백엔드 Compose의 `8000:8000`은 그대로이므로 백엔드 PC의 LAN 노출 범위를 현장에서 확인한다. Tailscale Serve는 호스트의 8000 포트로 연결한다.

## PC별 설정값

실제 토큰과 주소는 각 PC의 로컬 환경 파일에만 보관하고 저장소에 넣지 않는다.

| 위치 | 설정 | 값의 의미 |
| --- | --- | --- |
| 백엔드 | `STOREOPS_AI_INGEST_TOKENS_JSON` | 탐지 PC의 `EVENT_INGEST_TOKEN`을 실제 매장 ID에 대응시킨 JSON |
| 백엔드 | `STOREOPS_OUTPUT_DIR` | 업로드된 사건 MP4·이미지를 백엔드가 저장·제공할 영속 폴더. 기존 Compose는 `/app/output`과 `./storeops_ai/output`을 연결함 |
| 백엔드 | `FRONTEND_ORIGIN` | 브라우저가 백엔드 API를 직접 호출할 때 프런트 사이트의 정확한 출처. 같은 출처의 `/api` 프록시만 쓰면 CORS 설정이 필요하지 않음 |
| 탐지 PC | `ALERT_WEBHOOK_URL` | 백엔드 PC의 접근 가능한 `/api/internal/events` 주소. 현재 `http://backend:8000/...`과 `localhost`는 사용 불가 |
| 탐지 PC | `VLM_RESULT_WEBHOOK_URL` | 백엔드 PC의 `/api/internal/events/{event_id}/vlm` 주소 |
| 탐지 PC | `EVENT_INGEST_TOKEN` | 백엔드 토큰 매핑의 키와 동일한 값 |
| 탐지 PC | `STOREOPS_MEDIA_UPLOAD_ENABLED=1` | 사건 JSON에서 로컬 미디어 경로를 제외하고 영상 완성 후 파일을 업로드 |
| 프런트 PC | `BACKEND_UPSTREAM` | 백엔드 PC의 Tailscale Serve HTTPS 기본 주소. 전용 Nginx 템플릿의 `/api` 대상 |

`DATABASE_URL`과 `STOREOPS_DEMAND_URL`은 백엔드 PC의 같은 Compose 네트워크에 둘 때 기존 서비스명을 쓸 수 있다. Q&A Agent를 사용한다면 별도 PC의 `STOREOPS_QNA_URL`과 공유 토큰도 유지한다. Qwen VLM을 다른 PC로 옮긴다면 해당 서비스의 주소도 따로 바꿔야 한다.

## 실행 순서

1. 백엔드 PC의 저장소 루트 `.env`에 기존 README의 수요·Q&A 설정과 `STOREOPS_AI_INGEST_TOKENS_JSON={"<탐지-토큰>":"<매장-ID>"}`를 설정한다. 같은 PC에서 `docker compose up -d --build postgres demand backend`를 실행한다. Qwen VLM이 필요하고 GPU가 준비되면 `qwen-vllm`도 실행한다.
2. 백엔드 PC에서 `http://localhost:8000/api/health`를 확인한 뒤 Tailscale Serve로 로컬 8000 포트를 공유한다. 예: `tailscale serve 8000`. 출력된 HTTPS URL의 **호스트 이름만** 아래 `<백엔드-주소>`에 넣는다.
3. 웹캠·탐지 PC의 저장소 루트 `.env`에 아래 값을 설정하고 `docker compose -f docker-compose.detector.yml up -d --build`를 실행한다. 토큰은 백엔드 매핑의 키와 같아야 한다.

   ```dotenv
   BACKEND_EVENTS_URL=https://<백엔드-주소>/api/internal/events
   BACKEND_VLM_URL=https://<백엔드-주소>/api/internal/events/{event_id}/vlm
   STOREOPS_AI_INGEST_TOKEN=<탐지-토큰>
   ```

4. USB 웹캠을 쓰는 Windows 탐지 PC에서는 `storeops_ai/pipelines/path1_webcam.py`를 **호스트에서** 실행한다. 최신 main은 `storeops_ai/.env`를 자동으로 읽으며 이미 지정한 프로세스 환경변수가 우선한다. 저장소 루트 Compose `.env`와는 다른 파일이므로 같은 주소와 토큰을 설정한다. 기존 `storeops_ai/output`을 탐지 컨테이너와 공유한다.

   ```powershell
   $env:ALERT_WEBHOOK_URL = 'https://<백엔드-주소>/api/internal/events'
   $env:VLM_RESULT_WEBHOOK_URL = 'https://<백엔드-주소>/api/internal/events/{event_id}/vlm'
   $env:EVENT_INGEST_TOKEN = '<탐지-토큰>'
   $env:STOREOPS_MEDIA_UPLOAD_ENABLED = '1'
   Push-Location .\storeops_ai
   python -m pipelines.path1_webcam --camera 0 --camera-id CAM001
   Pop-Location
   ```

5. 프런트 PC의 저장소 루트 `.env`에 `BACKEND_UPSTREAM=https://<백엔드-주소>`를 설정하고 `docker compose -f docker-compose.frontend.yml up -d --build`를 실행한다. 해당 PC 브라우저에서 `http://localhost:5173`을 연다. Nginx가 `/api`를 백엔드 FastAPI로 전달한다.

탐지 PC의 Docker 컨테이너에서 백엔드 Tailscale 주소로 접속되는지 확인한다. 파일 입력 분석은 `http://localhost:8100/path1/analyze`에서 기존처럼 실행할 수 있다. Windows 웹캠 스크립트는 호스트에서 실행하며, Docker 안의 탐지 API는 같은 `output`을 볼 수 있다.

## 연결된 코드

탐지기는 사건 생성 시 `event_uid` UUID를 붙이고 JSON에 저장한다. 재전송은 같은 UUID를 유지하며 백엔드는 매장·카메라·UUID 조합으로 기존 사건을 반환한다. 화면의 `E001` 번호와 자릿수는 유지한다. 사건 시각은 UTC 시간대를 포함해 전송하고 백엔드는 한국 시간으로 변환한다.

기존 PostgreSQL 볼륨에는 백엔드의 새 코드를 실행하기 전에 아래 마이그레이션을 적용한다. 새 볼륨은 초기 스키마에 포함되어 별도 적용이 필요 없다.

```powershell
Get-Content -Raw .\backend\migrations\003_event_ingest_identity.sql | docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U storeops -d storeops
```

구형 탐지기의 UUID 없는 요청은 원본 사건 번호와 시각을 조합해 재전송을 판별한다. 이미 저장된 UUID 없는 DB 행에는 소급해서 중복 식별자를 붙이지 않으므로 업그레이드 이전 사건의 중복은 자동 병합하지 않는다.

1. 탐지기는 사건 JSON을 백엔드에 먼저 등록한다. 분리 배치에서는 로컬 파일 경로를 사건 본문에서 제외한다.
2. 영상이 완성되면 `PUT /api/internal/events/{event_id}/media/{media_type}`으로 MP4와 이미지 3장을 각각 보낸다. 백엔드는 토큰·매장·파일 크기·형식을 확인해 저장하고 DB 미디어 경로를 갱신한다. 전송은 일시적 실패 시 최대 3번 시도한다.
3. 프런트는 `/detector` 대신 백엔드의 세션 인증 미디어 API에서 영상과 이미지를 받는다.

## Compose 분리 시 확인할 항목

- PC별 전용 Compose에는 다른 PC의 서비스에 대한 `depends_on`을 두지 않았다. 백엔드 PC에서는 기존 Compose의 서비스 중 필요한 것만 실행한다.
- `frontend/nginx.remote.conf.template`은 원격 백엔드 주소를 환경변수로 받는다. 기존 `frontend/nginx.conf`와 `docker-compose.yml`의 서비스명은 단일 PC 배치에만 사용한다.
- 백엔드 DB·미디어 저장소는 컨테이너 재시작 후에도 유지한다.
- 호스트와 컨테이너 각각에서 상대 서비스의 상태 API에 접속하고, 사건 등록 → 미디어 업로드 → 프런트 재생을 순서대로 확인한다.

## 남은 제약

- MVP에서는 사건 번호를 3자리로 유지한다. 1,000건 경계에서 번호 충돌 또는 탐지기 검증 오류가 발생하므로 해당 건수 전에 번호 정책을 수정해야 한다.

- `path1_webcam.py`는 Windows DirectShow와 화면 창을 사용한다. 탐지 Docker 이미지는 Linux와 `opencv-python-headless`를 사용하므로 USB 웹캠 분석 스크립트까지 컨테이너 안에서 직접 실행하는 방식은 아직 지원하지 않는다. 기존 `/path2/streams`는 연결 감시만 하며 행동 분석을 하지 않는다.
- 네트워크 장애 중 전송을 세 번 모두 실패하면 사건 영상은 탐지 PC에 남고 자동으로 나중에 다시 전송되지는 않는다. 장시간 오프라인 운영에는 영속 재전송 대기열이 추가로 필요하다.
- Compose 구성 검사는 통과했으나 검증 환경의 Docker 소켓 권한이 없어 이미지 빌드와 세 PC 실연결은 아직 검증하지 못했다.


## 최신 main 동기화 후 추가 설정 (2026-10-29)

- 프런트 전용 `docker-compose.frontend.yml`을 추가했다. main의 원격 Nginx 템플릿을 마운트하며 `BACKEND_UPSTREAM`은 경로 없는 기본 주소로 지정한다. `depends_on` 없이 독립 실행한다.
- 실시간 화면은 **백엔드 → 탐지 PC 호스트** 방향도 필요하다. 탐지 호스트의 8101 포트를 사설 네트워크로 제공하고 백엔드 PC 루트 `.env`의 `STOREOPS_AI_LIVE_URL`을 그 주소로 설정한다. 기존 단일 PC 기본값 `host.docker.internal:8101`은 다른 PC를 가리키지 않는다. 탐지 컨테이너의 8100 API와 호스트의 8101 화면 서버는 별개다.
- 탐지 PC의 `storeops_ai/.env`에 원격 `QWEN_VLM_BASE_URL=https://<VLM-주소>/v1`, 필요한 `QWEN_VLM_API_KEY`, 정확한 모델 이름을 지정한다. 컨테이너의 `127.0.0.1`은 VLM PC가 아니다. VLM 포트를 외부에 공개하기보다 사설 네트워크/인증된 프록시를 사용한다.
- 기본 `VLM_GATE_ENABLED=1`, `VLM_GATE_FAIL_OPEN=1`이다. VLM 오류 시 경보가 전송될 수 있으므로 연결 성공을 별도 확인한다. VLM 없는 실행은 `VLM_ENABLED=0`, `VLM_GATE_ENABLED=0`을 탐지 프로세스에도 지정한다. 별도로 실행하는 호스트 Python에도 위 환경변수를 직접 지정한다.
- C1P 기본 실행에는 `storeops_ai/training/runs/ours_stack/`의 `s3d_c1_6.pt`, `pose_head6_rel.pt`, `fusion_c1pr.json`, `object_det.pt`와 YOLO 가중치가 필요하다. 가중치는 이미지 빌드 전에 배치하거나 동일 경로에 읽기 전용 마운트한다. 데이터/가중치는 Git에 추가하지 않는다.
- `output` 볼륨은 미디어와 `output/calibration/<camera_id>.json` 및 색 모델을 유지한다. 카메라 ID와 해상도를 바꿀 때 기존 보정값을 재사용할 수 있는지 확인한다.
- 호스트 Python 일반 실행과 단일 PC Compose는 유지했다. 세 PC 실장비·GPU·카메라·모델이 없는 환경의 config 검사는 실제 스트리밍/VLM/미디어 통합 검증을 대체하지 않는다.
