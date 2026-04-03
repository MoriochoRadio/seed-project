# 시스템 설계 문서 (Architecture Document)

## 1. 프로젝트 개요

### 목표
정자 현미경 영상을 입력받아 AI 기반으로 운동성을 분석하고 WHO 기준 기반 설명형 결과를 출력하는 병원 전 단계 보조 분석 시스템.

### 최종 비전 (AI-CASA)
```
Phase 1  ✅  운동성 분석 (현재 완성)
Phase 2  🔜  품질 평가 강화
Phase 3  📋  형태 분석 (Morphology)
Phase 4  🎯  AI-CASA 통합 시스템
```

---

## 2. 현재 시스템 구성 (Phase 1)

### 2-1. 전체 파이프라인

```
[입력]
현미경 영상 (.mp4 / .avi)
640×480, 45~50fps

        │
        ▼

[모듈 1] SpermDetector (detector.py)
├── YOLO11 medium 기반
├── 신뢰도 임계값: conf=0.3
├── 탐지 클래스: sperm(0), cluster(1), small(2)
└── 출력: 전체 정자 수 N (10프레임 중앙값)

        │
        ▼

[모듈 2] SpermTracker (tracker.py)
├── ByteTrack 알고리즘
├── 설정: track_buffer=60, match_thresh=0.8
├── 분석 구간: 최대 5초 (250프레임)
└── 출력: 정자별 이동 경로 + 11개 운동성 특징

        │
        ▼

[모듈 3] MotilityAnalyzer (analyzer.py)
├── Ridge + RandomForest 앙상블
├── 입력: 11개 특징 벡터
├── 전처리: StandardScaler
└── 출력: [전진%, 비전진%, 비운동성%]

        │
        ▼

[모듈 4] interpreter.py
├── WHO 6판 (2021) 기준 적용
├── 판정: 정상/경계/주의
├── 신뢰도 점수 (0~100점)
└── 재촬영 권고 여부

        │
        ▼

[출력]
SpermAnalysisPipeline.print_report()
→ 설명형 보고서
```

### 2-2. 모듈별 상세

#### SpermDetector

```python
class SpermDetector:
    # YOLO11 기반 정자 탐지기
    def detect_frame(frame)         # 단일 프레임 탐지
    def estimate_sperm_count(video) # 전체 정자 수 N 추정
```

**핵심 결정사항:**
- cluster 클래스 탐지 포기 → sperm에 집중 (논문도 동일 한계)
- conf=0.3 (낮춰서 더 많은 정자 탐지)
- 10프레임 중앙값으로 N 추정 (아웃라이어 제거)

#### SpermTracker

```python
class SpermTracker:
    def track_video(video_path)      # 영상 추적 실행
    def extract_features(track_hist) # 11개 특징 추출
```

**핵심 결정사항:**
- ByteTrack 유지 (교체보다 보정이 현실적)
- track_buffer=60 (비운동성 정자 추적 유지 시간 증가)
- 5초 분석으로 충분한 데이터 확보

**추출 특징 (11개):**
```
속도 계열: speed_mean, speed_median, speed_75, speed_90
직진성:    lin_mean, lin_75, straight_mean
비율:      ratio_fast, ratio_medium, ratio_slow
수량:      n_tracks
```

#### MotilityAnalyzer

```python
class MotilityAnalyzer:
    def predict(features) # 특징 → 운동성 % 예측
```

**모델 구성:**
```
Ridge(alpha=1.0)    ← 직선적 패턴
RandomForest(n=100) ← 복잡한 패턴
→ 두 예측의 평균 = 앙상블
→ 합이 100%가 되도록 정규화
```

**학습 데이터:** VISEM 원본 85명 (5-Fold CV MAE: 6.9%p)

**v1/v2 자동 대응:**
```python
self.calibrators = data.get('calibrators', None)
if self.calibrators:  # v2 보정 레이어 (조건부)
    pred = calibrate(pred)
```

#### interpreter.py

```python
def interpret_motility(prog, non_prog, immotile)
# WHO 기준 해석 → 판정 + 권고사항

def assess_confidence(counts, track_history, N)
# 신뢰도 평가 → 점수 + 재촬영 권고
```

**WHO 기준 (6판, 2021):**
```python
WHO_CRITERIA = {
    'progressive_normal':    32,  # 전진 운동성 정상 기준 (%)
    'total_motility_normal': 40,  # 총 운동성 정상 기준 (%)
    'immotile_warning':      50,  # 비운동성 주의 기준 (%)
    'immotile_severe':       70,  # 비운동성 심각 기준 (%)
}
```

**신뢰도 감점 로직:**
```
100점 시작
- 탐지 불안정 (CV > 0.3):   -20점
- 탐지 불안정 (CV > 0.15):  -10점
- 정자 수 매우 부족 (N<5):  -40점
- 정자 수 부족 (N<10):      -25점
- 정자 수 적음 (N<20):      -10점
- 추적 안정성 낮음 (<0.3):  -20점
- 추적 안정성 보통 (<0.6):  -10점

80점↑: 높음 ✅
60~79점: 보통 ⚠️
60점↓: 낮음 🔴 → 재촬영 권고
```

---

## 3. 데이터 파이프라인

### 3-1. YOLO11 학습 데이터 구성

```
VISEM-Tracking 원본 (20명)
        │
        ▼
전처리 (03_prepare_dataset.ipynb)
├── 프레임 추출
├── YOLO 형식 레이블 변환
└── 저장: data/processed/yolo_dataset/

        │
        ▼
Train/Val 분할
├── Train: 16명 (참가자 단위 분할)
└── Val: 4명 (11, 14, 22, 23)
```

### 3-2. 회귀 모델 학습 데이터 구성

```
VISEM 원본 (85명, .avi + CSV)
        │
        ▼
YOLO11으로 자동 탐지 + ByteTrack 추적
        │
        ▼
11개 특징 추출
        │
        ▼
운동성 CSV와 매핑 (세미콜론 구분자)
        │
        ▼
저장: outputs/visem_features.json
```

---

## 4. 핵심 의사결정 기록

| 결정 | 선택 | 이유 |
|---|---|---|
| YOLO 버전 | YOLO11 (v8 아님) | 작은 객체 탐지 강화 |
| 가상환경 | conda | CUDA 자동 관리 |
| cluster 탐지 | 포기 | 데이터 극도 불균형 (논문도 동일) |
| 분류 방식 | 회귀 모델 | 샘플별 특성 반영 (고정 임계값 실패) |
| 학습 데이터 | VISEM 85명 | VISEM-Tracking 16명으로는 과적합 |
| 앙상블 구성 | Ridge + RF | 단독보다 0.4%p 개선 |
| ByteTrack 유지 | 유지 | 교체보다 보정이 현실적 |
| v2 보정 레이어 | 미적용 | 데이터 누출 문제 확인 |

---

## 5. 향후 확장 설계 (Phase 2~4)

### Phase 2: 품질 평가 강화

```
추가 예정:
- 영상 흔들림 감지 (optical flow 기반)
- 조명 균일성 평가
- 배경 잡음 수준 측정
- 더 세밀한 재촬영 가이드
```

### Phase 3: 형태 분석 (Morphology)

```
새로 필요한 것:
- 형태 어노테이션 데이터셋
  (HSMA-DS, HuSHeM, MHSMA 등)
- 정자 머리/꼬리 세분화 모델
- 형태 이상 분류 모델

판정 항목:
- 머리 이상 (두부 결함)
- 중편 이상 (경부/중간부 결함)
- 꼬리 이상 (미부 결함)
- Teratozoospermia index
```

### Phase 4: AI-CASA 통합

```
최종 출력:
- 운동성 % (현재 구현)
- 형태 정상률 %
- 정자 농도 (선택적)
- 종합 판정
- 설명형 보고서
- PDF 출력
```

---

## 6. 파일 구조 상세

```
src/
├── __init__.py
│   └── SpermAnalysisPipeline 노출
│
├── detector.py
│   └── class SpermDetector
│       ├── __init__(model_path, conf)
│       ├── detect_frame(frame) → dict
│       └── estimate_sperm_count(video, n_frames) → (N, counts)
│
├── tracker.py
│   └── class SpermTracker
│       ├── __init__(detector, config_path)
│       ├── track_video(video_path, duration_sec) → dict
│       └── extract_features(track_history) → dict | None
│
├── analyzer.py
│   └── class MotilityAnalyzer
│       ├── __init__(model_path)
│       └── predict(features) → dict
│
├── interpreter.py
│   ├── WHO_CRITERIA (상수)
│   ├── interpret_motility(prog, non_prog, immotile) → dict
│   └── assess_confidence(counts, track_history, N) → dict
│
└── pipeline.py
    └── class SpermAnalysisPipeline
        ├── __init__(yolo_path, model_path, tracker_config)
        ├── analyze(video_path, verbose) → dict
        └── print_report(result, participant_id) → None
```
