# 밤샘 진행 기록 (2026-10-07 시작 02:20)

목표: 10-07 14:00 시연. 최종 A 모델 + 서비스 정렬 + 전도 GRU, 이후 물건 탐지기 추가 학습.
금지: 공식 validation 평가, 커밋, 임의 값(임계값·길이는 OOF 탐색). 시도한 방법 수를 아래에 센다(과적합 위험 참고).

## 돌고 있는 작업
- 유기 확인 로직(새 물건 + 사람 떠남) OOF 탐색: output/abandon_gate.log (48조합)
- 최종 A 모델 학습(--fold -1): output/final_a_train.log -> output/finetune_s3d/s3d_fold-1.pt

## 완료한 수정
- 서비스: 사람이 추적되지 않은 창은 분류하지 않던 조건 제거(path1_behavior._infer_full) -> 전도 놓침 원인
- 서비스: 기본 임계값 0.60 -> 쓰러짐 0.998 / 쓰레기 투기 0.991 / 절도 0.896 (OOF 6.23/4.70/2.15의 sigmoid)

## 30분 점검 로그
- 02:48 유기 확인 로직 45/48, 최종 A 학습 3/3 에폭 step 1650 (곧 완료)
- 02:55 최종 A 학습 완료(s3d_fold-1.pt). 서비스 training/runs/ours_a_final/best.pt로 복사, config 기본 I3D_WEIGHT_PATH를 ours_a_final로 변경(원래 mc_stage2는 환경변수로 복귀 가능). 서비스 로드·채점 확인. 주의: 이 모델은 모든 학습 클립을 봤으므로 위 클립 점수는 평가가 아님(전도 점수 최대 ~12.7로 fold 모델(~8)보다 극단) -> 임계값은 fold 모델 OOF 기준이라 최종 모델에 그대로 맞는지 보장 없음(공식 validation 한 번 평가 때 확인).
- 03:00 [유기 확인 로직 결과: 효과 없음] S3D 유기 점수 단독 재현율 0.902에서 오경보 2.97/시간. 새 물건(N=3~24)+사람 떠남(H) AND 게이트(48조합): 재현율>=0.9 만족 조합 4개뿐(모두 H=0, 사람 떠남 조건 없음), 오경보 16.9~25.2/시간으로 악화. H>0 조합은 재현율 0.9 미달. 해석: 현재 교차 적합 물건 탐지기(12에폭)로는 창 끝 물건/새 물건 판정이 불안정해 진짜 유기 창을 많이 버린다 -> 탐지기 개선 선행 필요. 시도한 방법 수: 게이트 48조합(같은 OOF).
- 03:10 [A 로짓만 쓰는 GRU(서비스 호환), 재현율>=0.9 기준 오경보/시간] 전도 K=1 1.37 -> K=15 0.85 / 유기 5.02 -> 3.44 / 절도 5.54 -> 5.49(이득 없음). 재현율>=0.95: 전도 1.89 -> 0.94, 유기 13.0 -> 10.2, 절도 30 -> 21. 판단: 전도만 소폭 이득, 서비스에 15창 상태·GRU 가중치 추가 비용 대비 작아 시연에는 넣지 않고 한 창 임계값 규칙 유지(필요하면 후속). 시도한 방법 수 누적: 게이트 48조합 + GRU 변형(A+C1, A, 사람채널 유무 × K 4~5종).
- 03:15 물건 탐지기 교차 적합 30에폭 재학습 시작(detached): C:/data/aihub/run_objcross30.sh, 로그 C:/data/aihub/objtrain30_{A,B}.log, 결과 output/object_ft/cross{A,B}30/weights, 완료 표시 C:/data/aihub/objcross30.done. 이전 12에폭은 에폭당 약 4분이어서 모델당 약 2시간(총 약 4시간) 예상 -> 약 07:15 완료. 이후: 전체 영상 물건 탐지 재추론 + relwin 재생성 + 유기 게이트/관계 특징 재평가(탐지 품질이 오르면 게이트가 달라지는지).
- 03:18 점검: 물건 탐지기 crossA30 6/30 에폭 진행 중(정상). 다른 작업 없음(최종 A 학습·유기 게이트는 완료).
- 03:48 점검: crossA30 13/30 에폭(약 4.3분/에폭 -> A 약 05:00, B 약 07:30 완료 예상). 이상 없음.
- 04:18 점검: crossA30 20/30 에폭, 검증 mAP50 0.60 부근(12에폭 때 0.50 -> 개선 중). 이상 없음.
- 04:48 점검: crossA30 27/30 에폭, mAP50 약 0.62~0.64(12에폭 0.50 대비 상승). A 약 05:05 완료 후 B 시작. 이상 없음.
- 05:18 점검: crossA30 완료(30에폭), crossB30 4/30 진행. B 약 07:20 완료 예상.
  crossA30 최고 에폭 20: 검증 mAP50 0.649, 재현율 0.607(12에폭 crossA mAP50 0.50 대비 상승). 검증은 탐지기 체크포인트 선택용 날짜 10%이며 평가가 아님.
- 05:48 점검: crossB30 11/30 에폭, 이상 없음.
- 06:17 점검: crossB30 17/30 에폭, 이상 없음.
- 06:48 점검: crossB30 24/30 에폭, 이상 없음.
- 07:20 crossB30 완료: 최고 에폭 14, 검증 mAP50 0.801, 재현율 0.753 (crossA30은 0.649/0.607). 다음 단계 시작(detached): C:/data/aihub/run_objinfer30.sh = 30에폭 모델로 전체 영상 추론(objdet30) -> 관계 특징(relwin30) -> 유기 게이트 재평가(output/abandon_gate30.log), 완료 표시 C:/data/aihub/objinfer30.done. 기존 objdet/relwin은 보존. 환경변수 OBJDET_SUFFIX 추가(object_infer, relation_windows, abandon_gate).
- 07:47 점검: objdet30 추론 978/1708, relwin30 0
- 08:18 점검: objdet30 1708/1708, relwin30 1708
- 08:35 [시연 정렬] storeops_ai/webcam_i3d.py 수정: 기본 가중치 ours_a_final(config I3D_WEIGHT_PATH), 창 4초(config I3D_WINDOW_SEC; 기존 8초), 카테고리별 임계값(config CATEGORY_THRESHOLDS; 기존 0.60 하나), 화면에 절도 추가·싸움/파손 제거, 추론 시작 조건을 '프레임 24장 이상'에서 '창이 찬 12장'으로(기존 조건이면 4초 창은 추론이 시작되지 않음). 영상 파일을 카메라로 대신 넣어 검증: 입력 12장, 약 2초 간격 추론 동작. 실행: cd storeops_ai; python webcam_i3d.py [--camera N]
- 08:47 점검: abandon_gate30 진행 중
- 09:12 행동인식_분석.md §13(유기 정리) 추가. 최종 C1 학습 시작(C:/data/aihub/run_c1_final.sh, 로그 c1_final.log, 완료 c1_final.done).
- 09:59 [유기 게이트 재평가(30에폭 탐지기) 결과: 여전히 효과 없음] 재현율>=0.9 만족 조합 4개뿐(모두 H=0), 오경보 17.1~54.3/시간(12에폭 탐지기 16.9~25.2보다 오히려 나쁨, S3D 단독 2.97). 탐지기 mAP 상승(B 0.80)에도 '새 물건 AND' 방식은 쓸 수 없음 -> 게이트(AND) 방식 폐기, 탐지 특징은 학습형 헤드 입력으로만 재검토. 최종 C1 학습 완료(output/finetune_s3d_c1/s3d_fold-1.pt). 다음: A+C1 서비스 연결(OOF 유기 7.62 -> 4.46).
- 10:50 [A+C1 스택을 서비스에 연결] 연구: pipelines/export_stack.py -> storeops_ai/training/runs/ours_stack/stack.json (클래스별 표준화·계수·임계값(OOF, 재현율>=0.9): 전도 thr 3.02(1.42 FA/h), 유기 3.12(4.46), 절도 2.07(8.10) 로그오즈; 계수는 _fit_linear 확률과 일치 검증). 서비스: training/runs/ours_c1_final/best.pt(최종 C1), models/stack_head.py(크롭 규칙·스택), I3DClassifier.predict_pair, path1_behavior(창 사람 박스 deque -> 원본 프레임 크롭), config(I3D_CROP_WEIGHT_PATH/I3D_STACK_PATH/I3D_USE_STACK, 기본 임계값 = 스택 OOF 임계값의 sigmoid: 쓰러짐 0.954 / 쓰레기 투기 0.958 / 절도 0.888), webcam_i3d.py(YOLO 사람 박스 -> 크롭 -> predict_pair). 끄기: 환경변수 I3D_USE_STACK=0 (A 단독, 임계값은 별도 지정 필요: 쓰러짐 0.998/쓰레기 투기 0.991/절도 0.896). 검증: 파일을 카메라로 넣은 웹캠 경로와 CLI 경로 모두 동작, 전도 클립에서 34초 사건 발급(A 단독 38초).
- 주의/미결: (1) Qwen VLM 서버(127.0.0.1:8001)가 꺼져 있어 사건 발급 후 VLM 호출이 3회 재시도 후 RuntimeError(사건 파일은 저장됨). 시연 때 vLLM을 띄울지(VRAM 14.7GB, GPU 공유 충돌) 또는 VLM을 끌지 결정 필요. (2) 스택은 fold 모델 OOF로 학습했고 서비스에는 전체 데이터 최종 모델이 들어감 -> 점수 분포 차이 가능, 공식 validation 한 번 평가 때 확인. (3) 서비스 창은 학습 창과 약 0.3초 위상 차.
- 10:18 점검: 돌고 있는 학습·추론 작업 없음(모두 완료). 남은 건 위험도 단계 서비스 반영과 시연 점검.
- 10:21 사용자 결정: 위험도 단계 대신 재현율 0.9 기준 이진 경보. RISK_LEVELS=1 환경변수일 때만 사건에 단계 표기(기본 끔). 단계 규칙의 0.95/0.90/0.42는 임의 목표값(근거 약함)이었음. stack.json의 levels는 남겨 둠.
- 10:23 공식 validation 평가 준비: 특징 추출 체인 시작(C:/data/aihub/run_val_features.sh; 완료 valfeatures.done). 설정 동결 항목: 최종 A(s3d_fold-1.pt)·최종 C1, 스택 계수·임계값(stack.json, 재현율>=0.9 OOF), 이진 경보. 평가는 1회만.
- 11:00 [관계 특징(물건 탐지기) 스택 결과] A+C1+rel(12/30에폭) 오경보/시간: 전도 1.42/1.42/1.46(변화 없음), 유기 4.46->4.92/5.57(악화), 절도 8.10->4.67/4.44(약 45% 감소). 주의: 탐지기가 유기·절도 영상 물건으로만 학습해 '장면 표시'로 쓰였을 가능성(shortcut) -> 정상 영상/절도 영상 사건 밖 창으로 나눠 확인 필요.
- 11:00 프레임 대조 새 물건/없어진 물건 창 특징(물건 탐지기 없음) 스크립트 pipelines/new_object_windows.py 작성(파라미터는 이전 TEMPORARY, OOF 탐색 전). 학습 영상 추출 시작(C:/data/aihub/run_nobj.sh, 완료 nobj.done). validation은 persons 끝난 뒤 한 번 더 실행. 주의(공식 validation 1회 규칙): 스택에 새 특징을 넣을 거라면 validation 평가 전에 설정을 동결해야 함 -> 13:00 전후 동결 여부 결정, 현재 배포 스택(A+C1)은 그대로 유지.
- 10:48 점검: 공식 validation 특징 추출 중(persons 1722/2050 = val 14/342 완료, 매장이동 val 101개는 각 300초라 오래 걸림, 추정 11:00~13:00), nobj(프레임 대조) 학습 영상 69/1708(약 10클립/분, CPU 경합, 추정 3~4시간).
- 10:55 사용자 지시: 프레임 대조 특징을 먼저. 공식 validation 특징 추출은 일시 중지(이어받기). 자동 체인 시작: C:/data/aihub/run_after_nobj.sh = nobj.done 대기 -> OOF 평가(pipelines/eval_nobj_stack.py -> output/nobj_stack_eval.log) -> validation 특징 재개(run_val_features.sh) -> validation 프레임 대조 -> val_ready.done. 공식 validation 평가는 자동 실행하지 않음. 12:17 확인 크론 등록.
- 11:20 [프레임 대조 특징 검수 결과: 현재 파라미터로는 부적합] 805개 클립 시점 집계: added 확정 시간당 유기 241 / 반품(정상) 406 / 매장이동(정상) 306, removed 112 / 219 / 84. 거의 모든 클립(정상 포함)에서 켜짐. 화면 확인(output/nobj_check/*.jpg: 감지=빨강, 정답 물건 박스=초록): 유기 클립 4개 중 3개에서 사건 끝 무렵 added가 정답 박스와 IoU 0.72~0.86으로 겹치나, 사건과 무관한 확정(카운터 점원, 먼 선반 위 작은 영역)이 훨씬 많음. 판단: 파라미터(밝기 차 25, 유지 3초, 최소 넓이 300 — 이전 TEMPORARY)가 과민. 조치: 전체 추출·자동 체인 중단. 다음: 정답 물건 박스(XML)를 기준으로 후보 단위 재현율/정상 영상 시간당 확정 수를 재서 날짜 fold OOF로 파라미터 탐색(일부 클립의 절반 해상도 회색 프레임 캐시) -> 좋은 값으로 전체 재추출 -> 스택 평가. 손 지정 구역(ROI) 제외는 규칙상 사용 안 함. 공식 validation 특징 추출은 재개(11:20).
- 11:16 프레임 대조 파라미터 탐색 재시작(48설정, 평가 수정: removed는 정답 박스 첫 프레임과 비교, 최대 넓이 비율 고정 0.05). output/tune_new_object.log, 예상 약 70분(12:30경). 공식 validation 특징 추출 병행(GPU 사람 추적).
- 11:18 점검: 프레임 대조 파라미터 탐색 1/48(약 1.5분/설정, 12:30경 완료 예상), 공식 validation 사람 추적 134/342(1842/2050).
- 11:23 자동 체인 등록: C:/data/aihub/run_pipeline_nobj.sh = 탐색 완료 대기 -> 고른 파라미터(output/new_object_selected_env.txt)로 전체 프레임 대조 특징(nobj_tuned, 12병렬) -> OOF 평가(output/nobj_stack_eval.log) -> valfeatures.done 대기 -> validation 영상 특징 -> nobj_tuned_val.done. 만족 설정이 없으면 pipeline_nobj.stop을 만들고 멈춤. 공식 validation 평가는 자동 실행 안 함.
- 11:48 점검: 프레임 대조 탐색 28/48, 공식 validation 사람 추적 208/342.
- 12:17 확인: 탐색 45/48(곧 완료 후 자동 체인이 전체 추출 시작), nobj_tuned 0개, 공식 validation 사람 추적 238/342(매장이동 val 긴 영상 구간), 크롭 0/342. 자동 체인 정상 대기 중. OOF 결과 아직 없음.
- 12:18 점검: 탐색 46/48, nobj_tuned 0/1708, 공식 validation 사람 추적 239/342.
- 12:31 확인: 탐색 48/48 완료. 선택(재현율>=0.7에서 오경보 최소): 밝기 차 15, 유지 6초, 최소 넓이 1000 -> 선택 fold0-2 유기 added 재현율 0.71 / 후보 오경보 159/시간, 보고 fold3-4 유기 0.64 / 절도 removed 0.21 / 170/시간. 신호 구분력 약함(재현율-오경보 trade-off). 자동 체인이 전체 추출 시작(nobj_tuned 58/1708). 공식 validation 사람 추적 246/342, 크롭 0.
- 12:40 방향 전환: 프레임 대조를 '파라미터 선택'이 아니라 '정답 물건 박스로 후보 분류기 학습'으로. 약한 설정 창 집계 추출(nobj_tuned)과 자동 체인(run_pipeline_nobj.sh) 중단.
- 12:45 후보 분류기 파이프라인: pipelines/dump_candidates.py(후보 수집, 느슨한 설정, 12병렬 -> C:/data/features/cand), pipelines/train_candidates.py(정답 물건 박스 라벨 + 물건 탐지기 점수 2개 + 16특징, sklearn HistGradientBoosting, 날짜 fold OOF -> 창 특징 nobj_cand). 후보 수집은 CPU 경합(사람 추적 프로세스 2개가 스레드 과다)으로 약 14클립/분, 추적 종료 후 가속 예상.
- 12:46 자동 체인 C:/data/aihub/run_pipeline_cand.sh: cand_dev.done 대기 -> train_candidates(output/cand_train.log) -> A+C1 스택 OOF 평가(output/cand_stack_eval.log, cand_eval.done) -> valfeatures.done 대기 -> validation 후보 수집(cand_val.done). 공식 validation 평가는 자동 실행 안 함.
- 12:50 [사용자 지시로 방향 정리] S3D(A+C1)가 놓고 감/가져감을 행동과 같이 학습하고 배경은 사람 크롭이 줄인다. 프레임 대조·후보 분류기·탐지기 파인튜닝은 시연에 필요 없다고 판단해 중단(후보 수집/체인 종료, 코드·부분 결과는 보존: pipelines/dump_candidates.py, train_candidates.py, tune_new_object.py, new_object_windows.py). 시연은 강의실이라 매장 영상으로 맞춘 부가 특징은 의미 없음. 설정 동결: 최종 A + 최종 C1 + stack.json(A+C1, 재현율>=0.9 OOF 임계값, 이진 경보). 공식 validation 평가 체인 C:/data/aihub/run_official_eval.sh = valfeatures.done 대기 -> A·C1 채점 -> 1회 평가(output/official_val/report.md, output/official_val_eval.log, official_eval.done).
- 12:48 점검: 공식 validation 사람 추적 299/342, 크롭 0/342. 후보 수집 체인은 중단됨. 공식 평가 체인(run_official_eval.sh)은 valfeatures.done 대기 중.
- 12:50 서비스에서 학습하지 않은 카테고리(싸움·파손) 제거: config.CATEGORIES=[정상,쓰러짐,쓰레기 투기,절도], CATEGORY_THRESHOLDS·웹캠 표시·CATEGORY_EN 정리, EXPANDED_CROP_CATEGORIES=빈 집합, tests/test_i3d_classifier.py 4카테고리 기준으로 수정(통과). 전도·절도 클립으로 CLI 사건 발급 재확인(점수 키 4개). 남은 기존 테스트 오류: test_path2_connection(import 경로 문제, 무관).
- 12:57 [물건 탐지기 8:2 파인튜닝 시작] 사용자 선택(정답 물건 박스로 YOLO 파인튜닝). 근거: 평가용 크기 분석(crossB30@1280) 한 변 0~20px 재현율 0.47, 20~35px 0.77, 35px+ 1.00 -> 작은 물건 놓침(Akyon ICIP 2022 SAHI 계열과 같은 방향, 해상도 상향). 구성 H: 학습 11,835 / 검증 1,974 / 평가 2,552 이미지(평가는 fold 3·4 날짜 20%, 기존 A 모델이 본 적 없음, 날짜 단위). 1920 원본 해상도, crossA30 가중치에서 이어서 12에폭 batch 2(예상 2시간 안팎). 체인 C:/data/aihub/run_objft_h.sh(export -> train -> eval, objft_h.done), 평가 결과 output/objft_h_eval.log(기존 crossA30@1280 vs H1920). 이 평가용 날짜는 최종 서비스 탐지기를 만들 때 다시 합쳐 학습 가능.
- 13:05 [13:00 확인] 공식 validation 특징 추출 완료(persons 342/342, crops 342/342, valfeatures.done). 공식 평가 체인(run_official_eval.sh)이 A 채점을 마치고 C1 채점 진행 중 -> report는 곧 생성(1회). 프레임 대조·후보 체인은 중단됨. 시연 서비스 점검 통과: 기본 가중치 ours_a_final/ours_c1_final/ours_stack, 스택 사용, 카테고리 4개, 임계값 쓰러짐 0.954/쓰레기 투기 0.958/절도 0.888, QWEN_VLM_BASE_URL= 빈 값에서 CLI 24초에 쓰러짐 사건 E002(0.975) 발급, 재시도 오류 없음, 테스트 출력 삭제. 물건 탐지기 8:2 파인튜닝 데이터 내보내기 진행(train 819/test 222 클립 단위 증가 중).
- 13:15 [공식 validation 평가 완료(1회, 동결 스택 A+C1, 임계값 전도 3.02/유기 3.12/절도 2.07)] output/official_val/report.md. 이벤트 단위 엄격 기준: 전도 클립 재현율 0.852 / 오경보 0.00/시간 / 지연 4초, 유기 0.887 / 0.54 / 4초(p90 10초), 절도 0.887 / 4.49 / 2초. 창 단위: 전도 P1.000 R0.792 F1 0.884 AUC 1.000, 유기 P0.983 R0.636 F1 0.773 AUC 0.991, 절도 P0.796 R0.608 F1 0.689 AUC 0.964, 정상 P0.967 R0.994. 혼동행렬(정상,전도,유기,절도): 정상행 18381/0/3/108, 전도행 127/484/0/0, 유기행 184/0/357/20, 절도행 318/0/3/498. OOF 대비 재현율 -3~7%p. 한계: val 정상은 매장이동 101개뿐(반품 같은 어려운 정상 없음)이라 오경보가 OOF보다 낮게 나옴(쉬운 평가), 같은 매장·다른 날짜라 새 장소 일반화는 측정 안 됨. 이 결과로 설정을 바꾸지 않는다.
- 13:15 물건 탐지기 8:2 파인튜닝: 1920 데이터셋 내보내기 중.
- 13:32 MVP 기준 명시·기록: 행동인식_분석.md §14(구성·평가 설정·엄격 기준 표·정상=걷기/서 있기 보조 표·한계·다음 평가), paper_table.md 부록, 메모리 progress-1007에 기준 수치 추가.
- 13:37 [13:37 확인] 프레임 대조 체인은 사용자 결정(S3D 기준, 프레임 대조 중단)으로 12:50에 중단됨. nobj_tuned 부분 결과만 남음(사용 안 함). 학습 창 채점(train-val 차이용) 진행 중, 물건 탐지기 8:2 데이터 내보내기 진행 중.
- 13:38 사용자 지시: 프레임 대조 체인 잔여물 삭제. 삭제: C:/data/features/{nobj,nobj_tuned,nobj_gray(14GB),cand}, C:/data/aihub의 해당 체인 로그·셸 스크립트·pick_nobj_params.py, output/{new_object_selected_env.txt,tune_new_object.log,nobj_check}. 보존: 소스 코드(pipelines/{new_object_windows,tune_new_object,dump_candidates,train_candidates,eval_nobj_stack}.py, models/new_object_detector.py), 9월 이전 평가 output/new_object_eval, output/abandon_label_examples.jpg.
- 13:38 train-val 차이 표 자동 체인: 학습 창 채점(A→C1) 완료 후 pipelines/eval_train_gap.py(이벤트 엄격 기준 + 정상=걷기·서 있기 기준, train/OOF/val 같은 임계값) -> output/official_val/train_val_gap.md, output/train_val_gap.log, gap.done.
- 13:43 사용자 지시: 프레임 대조 코드 삭제(new_object_windows, tune_new_object, dump_candidates, train_candidates, eval_nobj_stack). 결과 비교와 선택 이유는 행동인식_분석.md §15에 기록.
- 13:48 점검: 학습 창 채점 진행 중(A 0/2 파일), 탐지기 8:2 데이터 내보내기 진행. 의사결정_정리.md 작성 완료.
- 14:19 서비스 서버 실전 확인: websockets 17.2 설치 + requirements uvicorn[standard], VLM 스위치 VLM_ENABLED=0 추가(Windows에서 빈 환경변수는 변수를 지워 QWEN_VLM_BASE_URL= 방식은 VLM을 못 끔). 이 컴퓨터 외부 주소(<내부 주소>:8000)로 서버 접속 테스트: WS connection.ready, /path1/analyze 14초 응답, WS event.created/updated 푸시, GET /path1/events 정상, VLM 재시도 없음. 다른 컴퓨터 프론트로의 전달은 그쪽 주소 필요(미확인).
- 14:21 train-val 차이 표 완료(output/official_val/train_val_gap.md; 평균 Event F1 엄격 train 0.974 / OOF 0.890 / val 0.869, 걷기·서 있기 기준 0.992 / 0.954 / 0.891), 행동인식_분석.md §14-추가에 기록. 탐지기 8:2 학습 1에폭 진행 중. GitHub StoreOpsAI/StoreOpsAI 읽기 전용 확인: 사건은 탐지기 -> POST 백엔드 /api/internal/events(Bearer EVENT_INGEST_TOKEN) -> PostgreSQL -> 프론트가 백엔드 REST(/api/events, X-Session-ID)로 조회, 미디어는 /detector 프록시. 프론트는 탐지기 WebSocket을 쓰지 않음.
- 14:50 [결합 방식 비교(개발 OOF, 재현율>=0.9에서 오경보/시간)] 현재 스택(로지스틱 입력 8개)이 단순 방식보다 나쁨. 전도: A 단독(원 점수) 1.27, C1 1.37, 평균 1.09, 최댓값 0.99, 로지스틱 2입력 1.42, 로지스틱 8입력(현재) 1.42, GB 1.98. 유기: A 2.88, C1 4.74, 평균 1.86, 최댓값 2.55, LR2 2.37, LR8 4.46, GB 3.25. 절도: A 3.93, C1 4.35, 평균 3.11, 최댓값 3.29, LR2 3.66, LR8 8.10, GB 4.12. => 학습 없는 '두 로그오즈 평균'이 3행동 모두 현재 스택보다 낫거나 비슷(전도 1.09 vs 1.42, 유기 1.86 vs 4.46, 절도 3.11 vs 8.10). 이전 'A 단독 7.6/8.6' 값은 A 로짓 4개를 로지스틱으로 다시 맞춘 값이라 원 점수(2.88/3.93)보다 나빴음 -> 앞선 'A+C1 스택이 A보다 낫다'는 비교는 과장. 7종 비교로 고른 것이라 선택 편향 가능. 공식 validation은 이미 1회 사용했으므로 평균 방식으로 바꾸면 그 평가는 사후 재평가(블라인드 아님)가 됨. 사용자 결정 대기.
- 14:49 [동적 결합 채택·재평가] 서비스 fusion.json(동적), 평가 스크립트 fused_z로 교체, 공식 validation 재평가(FORCE_REEVAL, 사후·블라인드 아님): 평균 Event F1 0.871, 전도 0.883/절도 0.818/유기 0.913, 오경보 0.00/1.28/0.81, 재현율 0.790/0.775/0.887. 이전 로지스틱 보고서는 *_logistic_stack.md로 보존. 행동인식_분석.md §16, 의사결정_정리.md §9, 메모리 progress-1007 기록. 물체 탐지기 학습은 그대로 진행 중.
- 14:50 점검: 물건 탐지기 H1920 2/12 에폭 완료(검증 mAP50 0.618 -> 0.636, 재현율 0.589 -> 0.616, 에폭당 약 18분), 3에폭째 30%. 예상 종료 약 18시 안팎. 기존 crossA30 검증 mAP50 0.649 / 재현율 0.607(다른 검증 날짜라 직접 비교 불가, 평가용 날짜 비교는 학습 종료 후).
- 15:03 전도 임계값 4.91(OOF 재현율 0.95 목표), fusion.json 갱신, 공식 validation 3회째(블라인드 아님) 평균 Event F1 엄격 0.891 / 걷기·서 있기 0.918, train-val +0.097, OOF-val +0.045. 이전 결과 *_dynamic_fall624.md, *_logistic_stack.md 보존. 행동인식_분석.md §16-추가, 의사결정_정리.md §9-추가, 메모리 기록.
- 15:13 탐지기 놓친 사례 분석(output/detector_misses.jpg): 선반 위 정답 물건은 이웃과 외형 구별 불가(1920 해상도로도 개선 한계), 검정 비닐/검정 옷 대비 문제. 저장된 C1 크롭은 224 축소로 물건 13px(전체 1280 화면 39px)이고 크롭 영역 중앙값이 화면 높이 전체(1080)라 확대 효과 없음. 사용자 제안: 가려짐 유지(carried 상태) 논문 근거로 설계 -> 행동인식_분석.md §17에 문헌(ByteTrack ECCV22, SORT, DeepSORT, TCOW CVPR23, Mitzel&Leibe ECCV12 계열) 및 GT 확인(라벨은 보이는 구간만, occluded 0%, 유지 시간은 라벨로 못 정함) 기록. 탐지기 학습 종료(18시) 후 근처/선반 분해 평가 -> 운반 상태 규칙 오프라인 구현.
- 15:15 사용자 지시로 물건 탐지기 H1920 파인튜닝 중단(완료 에폭까지 가중치 output/object_ft/H1920/weights 보존, 평가는 미실행). 합의한 로직: 운반됨(carried) 상태로 등장 -> 물건을 두고 사람이 화면에서 사라짐 = 유기, 운반됨 상태로 사람과 함께 화면 밖으로 나감 = 절도.
- 15:18 점검: 운반됨 규칙 평가 진행 중(클립 993개: 유기 642, 절도 319, 매장이동 32; 정상 영상이 32개뿐(격리 제외)이라 정상 시간 약 2.7h).
- 15:24 손 크롭 소규모 시험 시작: 1단계 포즈 추출(pipelines/hand_pilot_pose.py, 유기 160 + 매장이동 32 + 반품 30, 절도는 기존 캐시 176) -> 2단계 손 크롭/얼린 S3D 특징/선형 프로브 비교 예정. 근거: 절도 정답 물건 프레임 5,372개에서 손목 가시 100%, 물건 150px 이내 99.9%.
- 15:25 손 크롭 시험 자동 체인: C:/data/aihub/run_hand_pilot.sh (포즈 추출 완료 -> hand_pilot_probe --extract -> --eval, 결과 output/hand_pilot_eval.log, hand_pilot.done).
- 16:15 [이동 평균·큰 인물 크롭] 동적 결합 점수를 과거 m개 창 평균으로 평활: 전도·유기 m=3(개발 OOF nested 오경보/시간 전도 1.89→0.94, 유기 2.60→1.72, 절도는 이득이 잡음 수준이라 m=1). 임계값 재선정(로그오즈): 전도 4.98(평활, 재현율 0.95 목표) / 유기 2.78(평활) / 절도 3.28. 서비스 StackHead가 행동별 점수 이력(카메라 하나 기준)을 보관, 지연 +4초. 가장 큰 인물만 크롭하는 MVP 규칙 실측(사람 2명 이상 창이 46.6%): C1 정답률 97.3%(합집합) → 80.5%(가장 큰 인물, 학습 창 기준이라 합집합 쪽은 낙관적). 기본값은 학습 규칙(합집합)으로 되돌리고 CROP_LARGEST_PERSON=1로 켜는 옵션. 공식 validation 재평가는 이 변경에 대해 하지 않음.
- 15:48 점검: 손 크롭 시험 포즈 추출 완료(222), 특징 추출 진행 19/391 영상.
- 15:53 이동 평균 반영 후 공식 validation(4회째, 비블라인드): 평균 Event F1 0.892(전도 0.927 / 유기 0.931 / 절도 0.818), 걷기·서 있기 0.918, train-val +0.094, OOF-val +0.048. 이전 결과 *_dynamic_fall491.md 보존. 행동인식_분석.md §16-추가2 기록.
- 16:09 첫 기록 시연(output/demo_recordings/20261007_160707.mp4 + _scores.csv, 추론 57회, 영상 102초/경과 122초): 이동 평균 끈 구성(임계값 로그오즈 4.91/4.51/3.28). 최대 점수: 쓰러짐 확률 0.0057(로그오즈 -5.2, 임계값 +10 부족), 쓰레기 투기 0.9694(로그오즈 3.46, 임계값 4.51에 1.05 부족, 83초경), 절도 0.159(로그오즈 -1.7). 장면 관찰: 창가 역광으로 화면 오른쪽이 하얗게 날아감, 사람이 카메라에 가깝고 화면 가장자리에서 잘림, 실제 표본 속도 약 2.5fps(영상 305프레임/122초; GPU에서 손 크롭 특징 추출이 같이 돌아 느려짐), 학습 영상(매장 천장 시점)과 시점·조명이 크게 다름.
- 16:18 점검: 손 크롭 시험 특징 추출 370/391(곧 선형 프로브 평가). 2차 시연 진행 중(유기 임계값 0.90 환경 보정, 기록 20261007_161214.mp4).
- 16:25 2차 시연 기록(output/demo_recordings/20261007_161214.mp4, 13분, 샘플링 약 2.5fps): 카메라 위치 개선 후 쓰러짐 경보 2구간(경과 115-119초, 398-404초) 모두 실제 눕기와 일치(영상 확인), 쓰러짐 오경보 0. 유기: 컵이 바닥에 있는 구간(140-183초, 565-595초) 경보, 임계값 0.90에서 경보 구간 14개 / 기본 0.989에서 7개(그 중 일부는 컵 사건 아님 추정, 사용자 시각 메모 필요). 절도 경보 2구간(426초: 검은 모자 쓴 사람이 카메라 쪽으로 걸어옴, 607초 사무실 책상)은 사용자 동작 확인 필요.
- 16:31 강의실 환경 보정(2차 시연 기록 기준, 사용자 지시: 절도는 제외): 영상으로 확인한 컵 유기 사건 2번(경과 약 140초·566초 놓음), 눕기 2번(약 112-125초, 396-406초). 임계값 정함(환경변수, 기본 설정 파일은 불변): 전도 0.944(로그오즈 사건 최소 피크 5.84와 사건 밖 최대 -0.19의 중간 2.83), 유기 0.9996(사건 최소 피크 9.21과 사건 밖 최대 6.64의 중간 7.93). 이 기록에서 사건 2/2, 사건 밖 경보 0. 근거 약함(사건 각 2개, 13분 기록 1개). 절도 경보 2번(426초 모자 쓴 사람이 걸어옴, 605-610초 사무실에서 앉아서 일하는 장면)은 오경보. 이 임계값으로 3차 시연 진행(블라인드 확인용).
- 16:32 손 크롭 시험 결과(output/hand_pilot_eval.log, 행동인식_분석.md §18): A+C1+H 평균이 A+C1보다 유기 AUROC 0.909→0.925 / 오경보 39.7→28.0, 절도 0.934→0.952 / 22.2→12.3(얼린 특징 선형 프로브, 부분집합, 근거 약함). 사용자 결정 대기: H 스트림 본 학습 진행 여부(포즈 추출 약 1.5h + 크롭 + 5-fold + 최종 학습 약 8-10h, 런타임 포즈 모델 추가).
- 16:46 웹캠 표시에 히스테리시스 추가(webcam_i3d.py): 임계값을 넘어 켜진 행동은 점수(확률)가 RELEASE_SCORE 0.5 아래로 내려갈 때까지 유지(누워 있는 동안 경보 유지, 근거: 2차 기록에서 121초 전도 점수 0.9907이 임계값 0.9927에 못 미쳐 누운 채로 꺼짐, 사건 밖 최대 0.45). 서비스 사건 발급은 쿨다운 30초 그대로.
- 16:48 점검: 돌고 있는 학습·추출 작업 없음(손 크롭 시험 완료·기록됨). 누운 직후 4~5초 지연 분석 완료(구조적, --interval-sec 1 제안). 사용자 응답 대기.
- 17:18 점검: 돌고 있는 작업 없음. 전도 경보 유지(깜빡임 0, 시작 지연 4~5초), 유기는 컵이 보이는 동안 대체로 유지되나 보장 안 됨(컵① 147-151초·172-176초 점수 급락). 시간 유지(30초) 제안, 사용자 결정 대기.

## 이어하기 메모 (2026-10-07, 손 크롭 H 본 학습 진행 중)
- (로컬 환경의 경로·세션 정보라 공개본에서 삭제함)
- (로컬 환경의 경로·세션 정보라 공개본에서 삭제함)
- 30분 점검(cron)은 세션 안에서만 살아 있다: 세션을 다시 열면 점검을 다시 등록해야 한다.
- 학습 체인: `C:/data/aihub/run_hand_full.sh` (포즈 추출 -> 크롭 캐시 -> H 5-fold + 최종 -> 완료 표시 `C:/data/aihub/hand_crops.done`, `hand_train.done`). 이미 만든 결과는 건너뛰므로 죽었으면 `bash C:/data/aihub/run_hand_full.sh`로 다시 실행.
- 결과 위치: 포즈 `output/pose_cache/<행동>/train`, 크롭 `C:/data/features/crops_h`, 모델 `output/finetune_s3d_h/`.
- 남은 일: `eval_hand_fusion.py` 작성(`ensemble_eval.merged`에 H 추가, A+C1 대 A+C1+H OOF 비교) -> 문서 기록 -> 서비스 총 파이프라인 확정 -> 영상·중간 산출물 정리 목록 제시.
- 규칙: 반품 영상 사용 안 함(사용자 지시), 서비스 환경 유지, 공식 validation 튜닝에 사용 금지, 커밋하지 않음.
- 지운 것(2026-10-07): output/fire_cv*, object_ft의 이미지·가중치, hf_cache. 영상(organized)과 crops_c1·frames224는 보존.
