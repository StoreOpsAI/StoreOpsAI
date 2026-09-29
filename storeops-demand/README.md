# StoreOps 수요 예측·발주 초안

Python으로 구현한 독립 실행 모듈입니다. M5로 XGBoost를 실제 학습하고 내일·모레 예측, 박스 단위 추천량, 점주 승인 초안과 승인 이력을 제공합니다. 승인해도 실제 주문 전송이나 재고 변경은 없습니다.

## 바로 실행

현재 작업 환경에서는 이 폴더에서 `bash run_demo.sh`를 실행합니다. 기본 주소는 http://127.0.0.1:8765 입니다. 포트가 이미 사용 중이면 `PORT=8766 bash run_demo.sh`로 실행하세요. 화면에서 시연 토큰 `demo-owner-s01`을 입력합니다.

새 환경(검증 환경 Python 3.12)에서는:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install --no-deps -e .
bash run_demo.sh
```

- **M5 예측으로 추천 생성**: 실제 학습된 모델로 공개 자료 예측 → 추천 생성.
- **P001 명세서 모의 예시 생성**: 명세서의 16/12개 예측값을 사용한 합성·모의 시연. 모델 성능 자료가 아닙니다.
- 수량을 6개 배수로 수정하고 승인하면 SQLite에 초안과 승인 이력이 남습니다.
- OpenAPI: `/docs`. 기본 DB: `runtime/orders.sqlite3`.

토큰은 로컬 시연용입니다. 서버는 127.0.0.1에만 바인딩합니다. 실제 서비스에 통합할 때 `STOREOPS_DEMO`를 해제하고 기존 로그인 시스템의 검증된 사용자/매장 정보를 연결해야 합니다. 기본값은 토큰이 없어 요청을 거부합니다. 테스트/통합용 토큰 매핑은 환경변수 `STOREOPS_TOKENS_JSON`에서 받을 수 있습니다. 클라이언트가 보낸 매장/승인자 값으로 권한을 결정하지 않습니다.

## 재학습

```bash
python -m storeops.train \
  --data-dir /home/user/Downloads/m5-forecasting-accuracy \
  --store CA_1 --max-products 120 --history-days 730 --output artifacts
```

`--max-products 0`이면 지정한 매장의 전 상품을 학습합니다. 이번 검증은 CA_1에서 시드 42로 뽑은 120개 상품입니다. 전 매장/전 상품 학습이나 M5 대회 전체 성능 검증으로 표현하면 안 됩니다. 실행 시간과 메모리를 제한하기 위해 한 매장 표본을 기본값으로 삼았습니다.

- 우선 `sales_train_evaluation.csv`를 사용하고, 없으면 validation 파일을 사용합니다. 두 파일을 합치지 않습니다.
- `calendar.csv`의 날짜·미국 National 이벤트·행사 정보를 사용합니다. 휴일 변수는 M5 National 이벤트 기준이며 모든 미국 연방휴일의 완전한 목록을 뜻하지 않습니다.
- 한국 날씨/한국 휴일을 미국 M5에 붙이지 않습니다. `sell_prices.csv`는 현재 입력 변수에 사용하지 않습니다.
- 모델 두 개가 같은 예측 기준일에서 d1/d2를 각각 직접 예측합니다. 모레 예측에 내일 실제 판매량을 사용하지 않습니다.
- 일별 결측은 0으로 채우지 않습니다. `observed`, `closed`, `stockout`, `missing`을 보존하고, 관측 이외 상태는 수요 학습·입력에서 제외합니다. 최근 7일이 불완전하면 예측 요청을 거부합니다.
- M5는 휴무/품절 원인을 제공하지 않으므로 기존 0을 품절/휴무로 추정하지 않습니다. 실제 판매량이 잠재 수요와 같다는 보장은 없습니다.

`artifacts/evaluation.json`, `validation_predictions.csv`에 동일 기간 XGBoost/7일 평균 비교를 저장합니다. 홀드아웃을 평가한 뒤 제공 모델은 전체 관측 기간으로 다시 학습합니다. 따라서 저장된 제공 모델을 과거 홀드아웃에 다시 적용해 성능을 주장하면 안 됩니다.

## API 연결

모든 `/api` 요청에 `Authorization: Bearer <token>`이 필요합니다.

| 경로 | 용도 |
|---|---|
| POST `/api/forecasts` | 한 상품의 판매 이력과 미래 달력으로 d1/d2 예측 |
| POST `/api/orders/forecast-draft` | 예측 후 추천 초안 생성 |
| POST `/api/orders/drafts` | 외부 모듈 또는 모의 예측값으로 추천 초안 생성 |
| GET `/api/orders/drafts` | 로그인 점주의 매장 초안 목록 |
| GET `/api/orders/drafts/{id}` | 소속 매장의 초안 상세 |
| POST `/api/orders/drafts/{id}/approve` | 수정 수량으로 승인 초안 저장 |

초안 생성에는 `Idempotency-Key` 헤더가 필수입니다. 동일 키·같은 본문 재요청은 기존 초안을 돌려주고, 같은 키에 다른 본문은 409로 거부합니다. 승인 본문은 `{"qty":18}`입니다. 승인자는 토큰에서 결정하고 시각은 한국 시간(+09:00)으로 저장합니다. 같은 승인 재시도는 이력을 중복 생성하지 않습니다. 다른 수량으로 재승인하려면 새 초안을 만듭니다.

예측 요청 예제: `artifacts/example_forecast_request.json`. M5 기반 미국 데이터 예제이며 S01 UI에는 시연 목적으로 연결합니다. 국내 매장 학습 모델로 표현하지 않습니다. 모델 제공 기간 이전의 `as_of`, 미래 판매 이력, 불완전한 최근 이력, 한국 매장 요청은 거부합니다. 한국 모델은 국내 판매 자료와 한국 공휴일을 결합한 별도 학습이 필요합니다.

## 발주 계산과 미정 항목의 결정

```
need = max(0, d1 + d2 + safety_stock - on_hand - incoming)
boxes = ceil(need / pack_size)
recommended_qty = boxes * pack_size
```

기본 안전재고 10, 박스 6은 `config.json`에서 변경합니다. 승인 수량은 0 이상의 박스 배수만 허용합니다. 데이터는 초안에 스냅샷으로 남아 설정 변경 후에도 기존 계산 근거가 유지됩니다.

**도착 전 부족량은 SRS 미정 항목에 대한 구현 가정입니다.** 신규 발주가 도착하기 전 일별 누적 재고의 최대 부족량을 계산합니다. 기존 입고는 지정 날짜 시작에 적용합니다. 기본 신규 도착은 모레 영업 종료 후(`arrival_after_days=2`), 기존 입고는 내일 시작(`incoming_day=1`)입니다. 예측이 2일뿐이므로 2일 초과 리드타임은 거부합니다. 시간대별 판매·입고는 모델링하지 않습니다. SRS 예제는 `28-15=13`으로 일치합니다.

P001 합성 판매 이력은 `examples/P001_synthetic.csv`이며 최근 7일 평균 반올림 13.6, 지난주 같은 요일 12, 3월 15일 실제 17을 재현합니다. 모의 예측 16/12는 모델 출력과 구분합니다. 국내 공휴일 API 연동은 이 합성 예제에 구현하지 않았습니다.

## 검증

```bash
python -m pytest -q
```

테스트는 발주 경계·음수·NaN·박스 배수·입고 시점·결측·미래 데이터 누수·모델 저장/재로딩·매장 권한·중복/동시 승인·영속성을 확인합니다. 실제 M5 모델 연계 테스트는 `artifacts`가 없으면 건너뛰므로 학습 후 다시 실행하세요. 제공 폴더에는 검증된 모델이 포함되어 있습니다.

- [검증 보고서](reports/VERIFICATION.md)
- [SRS 대응표](SRS_TRACEABILITY.md)
- [SRS 원문](https://claude.ai/artifact/Wix3vUyy2P5Yu1TiwNzmgU)
- [M5 데이터 출처](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

실행 중에 원본 Kaggle CSV를 변경하지 않습니다. GPU, 외부 LLM, 유료 API는 필요하지 않습니다. SQLite는 SRS가 허용한 개발용 선택이며 PostgreSQL 연동/배포용 로그인은 팀 서버와 합칠 때 별도로 구현해야 합니다.
