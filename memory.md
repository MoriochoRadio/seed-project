# Memory — Trajectory-aware Sperm Motility Analysis

> 이 파일은 프로젝트 진행 상황·결정·새로 추가/변경된 사항을 누적 기록합니다.
> 새로운 변경이 있을 때마다 항상 갱신합니다.

## 🎯 프로젝트 개요
- **논문 제목**: *Trajectory-aware Deep Learning System for Robust Sperm Motility Analysis under High-Density Microscopy Conditions*
- **모체 논문**: Shahali et al., Human Reproduction, 2026 (DCT smoothing + InceptionTime)
- **차별점**:
  1. Tracking-aware trajectory classification (extracted trajectory가 아니라 **trajectory quality / uncertainty**까지 입력)
  2. Clinical domain robustness (high-density / fps / length / viscosity shift)
  3. System-level validity (분류 정확도 + tracking quality 동시 평가)
- **데이터셋**: [VISEM-Tracking (Kaggle)](https://www.kaggle.com/datasets/vlbthambawita/visemtracking)
  - 20개의 30초 영상 (30 fps) + bounding box annotation + WHO motility class label

## 🧱 프로젝트 구조
```
sperm/
├── memory.md                   # ← 본 파일 (모든 변경 기록)
├── README.md                   # 사용법 / 실험 안내
├── requirements.txt            # 의존성
├── configs/
│   └── default.yaml            # 기본 하이퍼파라미터
├── scripts/
│   └── download_visem.py       # Kaggle 다운로드 스크립트
├── src/
│   ├── data/
│   │   ├── visem_dataset.py    # VISEM-Tracking 로더
│   │   └── trajectory_utils.py # smoothing / padding / aug
│   ├── detection/
│   │   └── sperm_detector.py   # YOLO wrapper
│   ├── tracking/
│   │   ├── kalman_tracker.py   # Kalman + IoU/distance association
│   │   └── trajectory_builder.py
│   ├── features/
│   │   ├── casa_features.py    # VCL/VSL/VAP/LIN/STR/WOB/ALH/BCF/PAW
│   │   └── trajectory_features.py
│   ├── models/
│   │   ├── inception_time.py   # InceptionTime baseline
│   │   ├── hybrid_model.py     # CNN + Transformer (제안 모델)
│   │   └── multi_task_head.py
│   ├── train/
│   │   └── trainer.py
│   └── eval/
│       ├── metrics.py
│       └── tracking_metrics.py # IDF1, ID switch, MOTA-like
├── experiments/
│   ├── exp1_crowded_robustness.py  # 핵심 차별점 검증
│   ├── exp3_fps_length.py          # FPS / length sensitivity
│   └── exp4_ablation.py            # Tracking-aware ablation
└── tests/
    └── test_pipeline.py            # dummy 데이터 동작 검증
```

## 📐 설계 결정 사항
- **언어/프레임워크**: Python 3.10+, PyTorch 2.x, ultralytics(YOLO), torchmetrics
- **로컬 학습 가능 구조**: GPU 자동 감지, batch_size 자동 축소, 작은 dataset에서도 동작
- **모델**: Hybrid CNN+Transformer — InceptionTime 모듈 + Transformer encoder를 결합
- **Multi-task**: (1) WHO motility class (3-class), (2) hyperactivation(binary), (3) kinematic regression
- **Tracking**: Kalman filter + 거리/IoU 기반 association + identity reassignment (Hassani et al. 2025 참고)
- **Tracking confidence**: track length, IoU 평균, gap 빈도, 속도 jitter로 계산 → 모델에 추가 입력
- **Early stopping**: `EarlyStopping`(patience/min_delta/mode/monitor) — 검증 메트릭 개선이 patience epoch 동안 없으면 학습 자동 종료. 기본 monitor=`f1_macro`(max), patience=10, min_delta=1e-4
- **Best checkpoint**: `Trainer.fit()` 도중 개선이 발생할 때마다 `<save_dir>/best.pt` 에 즉시 저장. `Trainer.save(path)` 는 마지막 epoch이 아닌 *best* state를 기록 → 실험 결과 파일도 `*_best.pt` 로 저장됨

## 🧪 실험 매핑 (PDF → 코드)
| PDF 실험 | 파일 | 비교 모델 | 핵심 지표 |
|---|---|---|---|
| Exp1 Crowded Robustness | `experiments/exp1_crowded_robustness.py` | Baseline(InceptionTime) vs Proposed(Hybrid+confidence) | F1, IDF1, ID-switch |
| Exp3 FPS/Length | `experiments/exp3_fps_length.py` | 동일 모델, 입력 변형 | Accuracy drop ratio, ECE |
| Exp4 Ablation | `experiments/exp4_ablation.py` | proposed의 컴포넌트 제거 | F1, F1@high-density |

> Exp2(점도/PVP)는 VISEM-Tracking에 viscosity 라벨이 없어 보류. 추후 별도 데이터 추가 시 작성.

## 📝 변경 이력
- **2026-05-06**: 초기 프로젝트 스캐폴딩, memory.md 생성
- **2026-05-06**: VISEM-Tracking 다운로드 스크립트 추가
- **2026-05-06**: 데이터 로더 / trajectory utils / CASA feature / Kalman tracker / Hybrid 모델 / 학습 루프 / Exp1·3·4 작성
- **2026-05-06**: `__init__` 들에 lazy import 적용 (scipy/torch 부재 환경에서도 부분 테스트 가능)
- **2026-05-06**: 검증 결과 — `tests/test_minimal.py` 통과, DCT smoothing+CASA, trajectory_features, IDF1/IDF1-proxy/track-duration, classification_report+ECE 모두 정상. 전체 dummy pipeline (`tests/test_pipeline.py`)은 torch 설치된 환경에서 실행.
- **2026-05-06**: `python -m compileall src tests experiments scripts` 전체 통과 (구문 오류 없음)
- **2026-05-07**: **trajectory-level kinematic labelling + 단일 클래스 metric 안전 처리**
  - 기존: `_who_label_from_motility` 가 video-level majority class를 그 비디오의 *모든* trajectory에 동일하게 부여 → video-level split 시 val/test가 단일 클래스만 갖게 되어 trivial 100% accuracy + cohen_kappa nan
  - `VISEMTrackingDataset`에 `label_mode` 옵션 추가 (`trajectory_kinematic`이 default)
    - `trajectory_kinematic`: 각 trajectory의 VCL/LIN을 직접 계산 → VCL p33 미만은 immotile, 그 이상에서 LIN≥0.5는 progressive, 나머지는 non_progressive
    - `video_majority`: 기존 비디오 majority 라벨 (legacy)
  - 인덱싱 후 class distribution 자동 출력 → 학습 전에 라벨 균형 한 눈에 확인 가능
  - `src/eval/metrics.py`: 단일 클래스/빈 배열 시 sklearn warnings 억제, 모든 known label을 `labels=`로 명시 전달, cohen_kappa는 단일 클래스에서 0.0으로 정의 (nan 방지)
  - `experiments/exp1_crowded_robustness.py` / `experiments/exp4_ablation.py` / `configs/default.yaml`: `--label_mode` CLI 노출, config 키 추가
  - 검증 — 합성 데이터(progressive/non-prog/immotile 8개씩)로 모든 클래스 균등 분리 확인 (VCL p33=321, LIN cutoff=0.5), 단일 클래스에서도 metric warning 없이 동작 ✓
- **2026-05-07**: YAML 1.1 스칼라 파싱 이슈 수정 — `weight_decay: 1e-4` 가 문자열로 읽혀 옵티마이저가 터지던 버그
  - `configs/default.yaml`: `1e-4` → `1.0e-4` 로 변경 (코멘트 추가)
  - `experiments/_common.py`: `load_config` 가 알려진 numeric 키(`lr / weight_decay / min_delta / patience / ...`)를 자동으로 float/int 캐스팅 (방어적)
- **2026-05-07**: **VISEM-Tracking 로더 다중 레이아웃 자동 인식**
  - `src/data/_visem_layout.py` (torch-free) 신설: `find_video_dirs / split_for / label_dir_of / frame_index_from_filename`
  - `VISEMTrackingDataset` 가 (1) `<root>/Train|Test/<id>/labels` (2) `<root>/<id>/labels` (flat) (3) `<root>/.../<id>/labels` (wrapper) 모두 자동 인식
  - 파일 패턴은 `*_frame_<N>.txt`, `frame_<N>.txt`, `<N>.txt` 모두 허용
  - 빈 데이터셋 시 안내 메시지 + 발견된 video dir / 폴더 구조 진단 출력
  - `experiments/exp1_crowded_robustness.py`: Test 폴더 부재 시 video-level 70/15/15 split으로 자동 폴백
  - `scripts/inspect_visem.py` 신설: 사용자 디스크 구조 진단용 디버깅 스크립트
  - 검증: 3개 합성 레이아웃(L1 split / L2 flat / L3 wrapper) 모두 정상 인식 ✓
- **2026-05-07**: **Early stopping 추가 및 best-model 저장 정책 도입**
  - `src/train/trainer.py`: `EarlyStopping` 클래스(`patience`/`min_delta`/`mode`) + `TrainConfig` 에 4개 필드(`early_stopping_patience/min_delta/monitor/mode`) + `best_ckpt_name` 추가
  - `Trainer.fit()`: epoch마다 monitor 메트릭 확인 → 개선 시 `best_state` 캐시 + `<save_dir>/best.pt` 즉시 저장; patience 초과 시 `Early stopping at epoch ...` 로그 출력 후 종료. 학습 종료 직전 best state로 복원
  - `Trainer.save(path)`: 항상 *best* state 저장 (이전엔 최신 epoch 모델이 저장되었음)
  - `experiments/exp1_crowded_robustness.py`, `experiments/exp4_ablation.py`: `--patience / --min_delta / --monitor / --mode` CLI 인자 노출, baseline/proposed 별 `save_dir` 분리 (`runs/exp1/baseline/best.pt`, `runs/exp1/proposed/best.pt`), 최종 ckpt 이름은 `{arch}_best.pt`
  - `experiments/exp3_fps_length.py`: docstring을 `*_best.pt` 로 갱신 (평가 전용이라 early stopping 미적용)
  - `configs/default.yaml`: `train.early_stopping` 섹션 추가

### 검증 로그 (2026-05-06)
| 모듈 | 결과 |
|---|---|
| `compute_casa_features` (raw) | VCL prog>nonprog>immotile = 168.8 / 68.3 / 19.9 ✓ |
| `smooth_trajectory_dct` + CASA (smoothed) | ALH=5.19 (≈ sin amp 5), BCF=16.78 (≈ 8 Hz × 2) ✓ |
| `velocity_features / curvature / turning_angles / spectral_energy` | shape 검증 모두 통과 ✓ |
| `KalmanTracker` + `TrajectoryBuilder` | 3 trajectory, mean_len=40 frames ✓ |
| `idf1` (perfect=1.0, split=0.67), `idf1_proxy`=0.47, `mean_track_duration`=40 | ✓ |
| `classification_report` + `expected_calibration_error` | acc=0.83, ECE=0.033 ✓ |
| `compileall` (전체 src/tests/experiments/scripts) | 모두 컴파일 성공 ✓ |

> Hybrid 모델 / InceptionTime / Trainer는 torch 의존 → 사용자 로컬에서 `pip install -r requirements.txt` 후 `python tests/test_pipeline.py` 로 검증.

### 검증 로그 (2026-05-07, Early stopping 도입 후)
| 모듈 | 결과 |
|---|---|
| `EarlyStopping(max, patience=3)` | 0.5→0.6 후 plateau 3회 시 should_stop=True ✓ |
| `EarlyStopping(min, patience=2)` | 1.0→0.9 후 plateau 2회 시 should_stop=True ✓ |
| `EarlyStopping` NaN 처리 | NaN을 'no improvement'로 카운트 → 정상 종료 ✓ |
| `min_delta` 게이팅 | 0.5 → 0.51(<0.05) 무시, → 0.6 갱신 ✓ |
| `tests/test_minimal.py` | numpy/scipy 부분 회귀 통과 ✓ |
| `python -m compileall src tests experiments scripts` | 모두 컴파일 성공 ✓ |
| `EarlyStopping` import 경로 | `from src.train import EarlyStopping` (torch 없이 import 가능) ✓ |

## ⚠️ 알려진 제한 / TODO
- VISEM-Tracking은 hyperactivation 라벨이 명시되지 않음 → progressive/non-progressive/immotile 만 사용. hyperactivation은 합성 라벨(고VCL+고ALH 휴리스틱) 또는 추후 외부 라벨링 필요.
- viscosity domain shift 실험(Exp2)는 데이터 부재로 보류
- 외부 multi-center validation 데이터 미확보

