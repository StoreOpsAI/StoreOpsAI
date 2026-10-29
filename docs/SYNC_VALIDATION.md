# 서비스 동기화 검증 기록

2026-10-29, 시작 연구 커밋 `842e60c`, 기준 main `611c953`.
현재 실행 환경에는 기존 체크아웃이 없어 원격 연구 브랜치를 새로 복제했다. 최초 상태는 clean이었다. 사용자의 다른 PC에만 존재하는 미커밋 변경은 이 검사의 대상이 아니다.

## 실행 결과

| 검증 | 결과 |
| --- | --- |
| Python compileall: 서비스 5개 및 video_research | PASS |
| 백엔드 pytest | PASS: 39 tests + 4 subtests |
| 탐지 pytest (`STACK_MODE=a_c1`) | PASS: 35 tests + 4 subtests, 65.74초 |
| 수요 예측 pytest | PASS: 27 tests |
| Q&A pytest | PASS: 7 tests |
| 연구–서비스 합성 정합성 pytest | PASS: 5 tests |
| 자동 보정, view_calib, lying_gate 자체 검사 | PASS |
| 프런트 npm ci / lint / build | PASS, lint 경고 3건, npm audit high 2건 보고(의존성 변경은 범위 밖) |
| uvicorn `/api/health` | PASS: HTTP 200, status ok, 메모리 저장소 |
| Vite `/` 및 `/api/health` 프록시 | PASS: HTTP 200, status ok |
| Compose main / detector / frontend / novlm 병합 | PASS: 예시 환경값 사용, novlm 게이트 0 확인 |
| Docker 이미지 빌드 | NOT_RUN: 시도했으나 Docker 소켓 연결 권한 거부 |
| C1P 기본 모드 실제 모델 로드·6클래스 추론 | NOT_RUN: 학습 가중치·fusion 설정 없음. 기본 pytest 수집도 이 이유로 실패 |
| 실제 YOLO 추론 / CCTV / 다중 PC / 원격 VLM / DB 통합 | NOT_RUN: 모델·카메라·실장비·서비스 미제공 |
| 영상 Event F1 / FA/hour / OOF 재현 | NOT_RUN: 데이터·가중치 미제공 |
| 과거 render_demo CLI 실영상 | NOT_RUN: 영상·가중치·Windows 폰트 미제공 |
| git diff --check | PASS |
| 연구 209개 파일 / detector Compose 보존 | 시작 커밋과 동일 |

테스트용 가상환경은 Python 3.12, CPU PyTorch 2.15와 torchvision 0.29.2, Node 24.22.1을 사용했다. 운영 lock 파일은 바꾸지 않았다. 테스트 초기 실행 경로 오류 및 xgboost 의존성 누락은 환경을 수정해 재실행했다. 정적 물체 검사는 계산 시간이 길어 첫 실행을 중단하고 제한 시간을 늘려 전체 통과했다.

## 재실행

서비스별 의존성과 pytest를 설치한 가상환경에서 저장소 루트 기준:

```bash
python -m compileall -q backend/app storeops_ai storeops-demand storeops_qna video_research
PYTHONPATH=backend python -m pytest -q backend/tests
STACK_MODE=a_c1 PYTHONPATH=storeops_ai python -m pytest -q storeops_ai/tests
PYTHONPATH=storeops-demand python -m pytest -q storeops-demand/tests
PYTHONPATH=storeops_qna python -m pytest -q storeops_qna/tests
PYTHONPATH=storeops_ai python -m pytest -q analysis/tests
```

프런트는 `frontend/`에서 `npm ci`, `npm run lint`, `npm run build`를 실행한다.
서비스 가중치를 확보한 뒤 기본 C1P 모드와 실제 카메라, VLM, PostgreSQL 및 3PC 전체 흐름을 별도로 확인해야 한다. 단위 테스트는 UUID, 재전송 중복 방지, 미디어 업로드, VLM 요청/게이트, 인증된 스트리밍을 검사하지만 실서비스 성공을 뜻하지 않는다.

## 변경 범위

서비스 5개 경로의 main에 존재하는 파일은 모두 동일하다. 연구 브랜치만의 `storeops_ai/tools/` 분석 도구 4개는 보존했다. 연구 자산 이동은 없으며 이유를 `analysis/README.md`에 기록했다. 일반 Compose는 main 기능을 가져오되 연구 브랜치의 Qwen 컨테이너 이름과 loopback 포트 바인딩을 유지했다. novlm 구성은 보존하고 최신 VLM 게이트도 끄도록 보완했다. 프런트 전용 Compose를 추가했다.

main 브랜치 변경·main push·PR 병합·강제 push는 수행하지 않는다.
