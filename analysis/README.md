# StoreOpsAI 분석 자료 안내

연구 보존 상태: 기존 파일 이동 없음. 최종 결과·입력 캐시·분할·OOF·학습 코드는 `video_research/`에 그대로 보존한다.
배지는 자동 CI 결과가 연결되기 전까지 표시하지 않는다.

## 설치 및 검증

저장소 루트에서 서비스별 의존성을 별도 가상환경에 설치한다. 추가 정합성 검사는 `numpy`, `pytest`, CPU용 `torch`가 필요하다.

```bash
PYTHONPATH=storeops_ai python -m pytest -q analysis/tests
```

## 자료 분류 및 이동하지 않은 이유

| 위치 | 분류 및 판단 |
| --- | --- |
| `video_research/storeops_ai_fr_evt/pipelines/` | 모델 정의, 학습, 분할, OOF, 최종 평가와 중간 실험이 서로 import한다. 재현성 보존을 위해 통째로 유지 |
| `video_research/analysis/` | 이름은 analysis지만 전도 MLP 실행기와 config·label map·학습/검증 결과를 함께 사용하는 패키지. 임시 자료로 취급하지 않음 |
| `video_research/research_logs/` | 실패 사례, 정상 검수 CSV, 공식 평가·학습 차이 로그는 연구 결론의 근거이므로 유지 |
| `video_research/*FINAL.md`, 연구 Markdown | 최종 명세·프로토콜·의사결정·가중치 참조 기록. 변경 없이 유지 |
| `storeops_ai/tools/record_webcam.py`, `sweep_tracker.py`, `measure_track_fragments.py` | 비최종 추적 실험 도구이나 연구 결정 문서와 tracker 설정이 기존 CLI 경로를 참조한다. 이동 이익보다 기존 실행·참조 보존을 우선 |
| `storeops_ai/tools/render_demo.py` | 과거 4클래스 데모 시각화 도구. 최신 웹캠 내부 메서드에 의존하고 Windows 폰트·영상이 필요해 이동 전 실행 동등성을 확인할 수 없음 |

기존 CLI는 그대로다. `storeops_ai/`에서 `python tools/record_webcam.py 출력.mp4 60 0`, `python tools/sweep_tracker.py 영상.mp4`, `python tools/measure_track_fragments.py 영상.mp4 config/botsort_storeops.yaml`로 실행한다.
`render_demo`의 기존 명령은 `python -m tools.render_demo --video 영상.mp4 --out 결과.mp4`이며 최신 서비스와의 실행 호환성은 NOT_RUN이다.

새로운 일회성 분석은 이 폴더에 추가하되 최종 모델의 학습 입력이나 공식 validation을 변경하지 않는다.

## 기여 및 라이선스

루트 `AGENTS.md`와 연구 데이터 분할 규칙을 따른다. 데이터·가중치·비밀값을 커밋하지 않는다.
이 문서는 별도의 라이선스를 부여하지 않으며 저장소 및 원본 데이터의 이용 조건을 따른다.
