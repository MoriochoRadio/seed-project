# 🔬 AI-CASA: AI 기반 정자 종합 분석 시스템

> 정자 현미경 영상을 입력받아 **운동성 · 운동 패턴 · 형태**를 자동 분석하고
> WHO 기준 기반 설명형 결과를 출력하는 병원 전 단계 보조 분석 시스템

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/YOLO-11-darkgreen)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-3.0-black)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-5%20Complete-brightgreen)]()
[![Demo](https://img.shields.io/badge/Demo-Live-2DD4BF)]()
[![Version](https://img.shields.io/badge/Version-v1.4.0-blueviolet)]()

---

## 📌 프로젝트 소개

이 프로젝트는 **정자 현미경 영상** 하나만 있으면:

1. **YOLO11** 로 정자를 프레임별 자동 탐지
2. **ByteTrack** 으로 정자 이동 경로 추적
3. **앙상블 회귀 모델** 로 운동성 수치 예측
4. **CASA 키네마틱** (VCL, VSL, VAP, LIN, STR, WOB, ALH) 계산
5. **EfficientNet-B3** 로 정자 형태 분류 (머리/첨체/공포/꼬리)
6. **WHO 6판 기준** 으로 종합 해석 및 설명형 보고서 출력
7. **Flask 웹 데모** 로 누구나 영상 업로드 후 인터랙티브 결과 확인

을 자동으로 수행합니다.

> ⚠️ 본 시스템은 의학적 진단 도구가 아닙니다. 병원 방문 전 참고용 보조 분석 도구입니다.

---

## ⚡ AI-CASA의 차별점

기존 CASA 시스템과 비교했을 때 본 시스템의 핵심 가치:

| 항목 | 기존 CASA | **AI-CASA (본 시스템)** |
|---|---|---|
| **장비** | 전용 장비 필요 (수천만 원) | **일반 현미경 영상만 있으면 OK** |
| **분석 항목** | 운동성 위주 (형태는 별도 검사) | **운동성 + 키네마틱 + 형태 통합** |
| **인력** | 전문 검사 인력 필요 | **AI 자동 분석** |
| **결과 해석** | 수치 위주, 해석 어려움 | **일반인용 / 전문가용 토글** |
| **분석 시간** | 수동 검토 시간 소요 | **즉시 결과 (수분 내)** |
| **접근성** | 병원·연구실 한정 | **웹 브라우저 어디서나** |
| **투명성** | 블랙박스 | **AI 신뢰도 + 판정 근거 명시** |

---

## 🏆 핵심 성과

| 지표 | 본 시스템 | 논문 최고 (motilitAI) |
|---|---|---|
| 운동성 MAE (5-Fold CV) | **6.9%p** | 7.31%p |
| YOLO11 mAP50 | **0.677** | ~0.65 (YOLOv5l) |
| 형태 분류 AUC (평균) | **0.725** | — |
| 판정 방향 정확도 | **75~100%** | — |

> 동일 데이터셋(VISEM) 기준으로 운동성 분석은 논문 최고 성능을 초과 달성

---

## 🏗️ 시스템 아키텍처

```
사용자 (브라우저)
        │
        │  영상 업로드
        ▼
┌─────────────────────────────┐
│       Flask Web Server      │  app.py
│       (webapp/routes.py)    │  → API 엔드포인트
└─────────────────────────────┘
        │
        │  백그라운드 분석 (Threading)
        ▼
┌─────────────────────────────┐
│       SpermDetector         │  YOLO11 정자 탐지
│       (detector.py)         │  → 전체 정자 수 N 추정
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│       SpermTracker          │  ByteTrack 정자 추적
│       (tracker.py)          │  → 이동 경로 + CASA 키네마틱
└─────────────────────────────┘
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
┌──────────────────────┐     ┌──────────────────────────┐
│  MotilityAnalyzer    │     │   MorphologyAnalyzer     │
│  (analyzer.py)       │     │   (morphology.py)        │
│  Ridge+RF 앙상블      │     │   EfficientNet-B3 v3     │
│  → 운동성 % 예측      │     │   → 형태 정상/비정상     │
└──────────────────────┘     └──────────────────────────┘
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
         ┌─────────────────────────┐
         │     interpreter.py      │
         │  WHO 6판 기준 해석       │
         │  → 판정 + 신뢰도 + 권고  │
         └─────────────────────────┘
                       │
                       ▼
         📋 인터랙티브 웹 보고서
         (일반인용 / 전문가용 토글)
```

---

## 🎨 웹 데모 주요 기능

### 메인 페이지
- 영상 드래그 앤 드롭 업로드
- MP4 / AVI / MOV 형식 지원 (최대 500MB)
- 분석 파이프라인 5단계 미리보기

### 분석 진행 화면
- 실시간 진행률 + 5단계 시각화
- 단계별 사용 모델 표시 (YOLO11, ByteTrack, Ridge+RF, EfficientNet-B3, WHO)
- 평균 30초~3분 소요

### 결과 보고서 (핵심)
- **종합 판정 배너** — 정상 / 일부 주의 / 전문의 상담 권고 / 재검사 필요
- **모드 토글** — 일반인용(친절한 설명) ↔ 전문가용(수치 중심)
- **AI 신뢰도 박스** — 각 모델의 검증 결과와 한계점 투명 공개
- **운동성 분석** — 도넛 차트 + 막대 그래프 + WHO 기준 검토
- **CASA 키네마틱** — 7가지 파라미터 (속도 + 운동 패턴)
- **형태 분석** — 4부위(머리/첨체/공포/꼬리) 이상 비율
- **종합 판정 근거** — 왜 이 판정이 나왔는지 항목별 설명

> 🎨 다크 네이비 + 청록 의료 도메인 디자인 / 인쇄 가능

---

## 📊 출력 보고서 예시 (CLI 버전)

```
============================================================
  🔬 AI 정자 분석 보고서
============================================================

【 분석 품질 】
  ✅ 신뢰도: 높음 (100점 / 100점)

────────────────────────────────────────────────────────────
【 정자 운동성 분석 】  (탐지된 정자: 50개)
────────────────────────────────────────────────────────────

  앞으로 잘 나아가는 정자  (전진 운동성):   21.4%  ████
  조금 움직이는 정자      (비전진 운동성): 26.2%  ░░░░░
  움직이지 않는 정자      (비운동성):      52.4%  ··········
  ─────────────────────────────────────────────
  전체 움직이는 정자      (총 운동성):     47.6%

  📋 WHO 기준 검토 결과
     🔴 전진 운동성 21.4% — WHO 정상 기준보다 현저히 낮음
     ✅ 총 운동성 47.6% — 정상 (기준 40% 이상)
     ⚠️  비운동성 52.4% — 움직이지 않는 정자 비율이 높음

────────────────────────────────────────────────────────────
【 정자 형태 분석 】
────────────────────────────────────────────────────────────

  분석된 정자: 1677개  |  정상 형태: 139개 (8.3%)
  ⚠️  머리    (두부)          이상  86.2%  █████████████████
  ⚠️  첨체    (머리 앞부분)    이상  80.7%  ████████████████
  ✅  공포    (머리 내 공간)   이상  31.2%  ██████
  ✅  꼬리    (미부)          이상  18.8%  ███

============================================================
【 종합 판정 】
============================================================
  🔴 최종 판정: [전문의 상담 권고]
  💬 정확한 진단을 위해 비뇨의학과 방문을 강력히 권고합니다.
============================================================
```

웹 버전에서는 동일 정보가 **인터랙티브 차트와 친절한 설명**으로 표시됩니다.

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# conda 가상환경 생성
conda create -n sperm-ai python=3.10
conda activate sperm-ai

# PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 나머지 패키지
pip install -r requirements.txt
```

### 2-A. 웹 데모 실행 (추천)

```bash
python app.py
```

브라우저에서 `http://localhost:5000` 접속 → 영상 업로드 → 자동 분석

### 2-B. Python 코드로 직접 분석 (3줄)

```python
from src.pipeline import SpermAnalysisPipeline

pipeline = SpermAnalysisPipeline()
result   = pipeline.analyze('video.mp4')
pipeline.print_report(result)
```

---

## 📁 프로젝트 구조

```
sperm-ai/
├── app.py                       # Flask 진입점
├── webapp/                      # 웹 애플리케이션
│   ├── routes.py                #   API 라우트 (/api/analyze, /api/status)
│   ├── tasks.py                 #   백그라운드 분석 작업 관리
│   ├── templates/               #   Jinja2 HTML 템플릿
│   │   ├── base.html
│   │   ├── index.html           #     메인/업로드 페이지
│   │   ├── analyzing.html       #     분석 진행 화면
│   │   └── result.html          #     결과 보고서
│   └── static/                  #   CSS / JS
│       ├── css/style.css
│       └── js/
│           ├── upload.js
│           ├── progress.js
│           └── result.js
│
├── src/                         # AI 분석 코어
│   ├── detector.py              #   SpermDetector (YOLO11)
│   ├── tracker.py               #   SpermTracker (ByteTrack + CASA)
│   ├── analyzer.py              #   MotilityAnalyzer (앙상블 회귀)
│   ├── morphology.py            #   MorphologyAnalyzer (EfficientNet-B3)
│   ├── interpreter.py           #   WHO 해석 + 신뢰도 + 종합 판정
│   └── pipeline.py              #   SpermAnalysisPipeline (통합)
│
├── models/                      # 학습된 모델 (Git 미포함, 별도 다운로드)
│   ├── yolo11_sperm_v2/                  #   YOLO11 탐지 모델
│   ├── motility_ensemble.pkl             #   운동성 회귀 모델
│   └── morphology_efficientnet_b3_v3.pt  #   형태 분류 모델 (v3)
│
├── notebooks/                   # 실험 기록 노트북 (01~16)
├── docs/                        # 문서
│   ├── architecture.md          #   시스템 설계
│   └── performance.md           #   상세 성능 분석
│
├── README.md
├── requirements.txt
└── bytetrack_custom.yaml
```

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|---|---|
| **백엔드** | Flask 3.0 / Werkzeug / Threading |
| **프론트엔드** | HTML5 / CSS3 / Vanilla JavaScript (Canvas API) |
| **AI 프레임워크** | PyTorch 2.6 / torchvision / scikit-learn |
| **객체 탐지** | YOLO11 (ultralytics) |
| **객체 추적** | ByteTrack (ultralytics 내장) |
| **형태 분류** | EfficientNet-B3 (멀티태스크) |
| **운동성 회귀** | Ridge + RandomForest 앙상블 |
| **영상 처리** | OpenCV / NumPy |
| **GPU 가속** | CUDA 12.4 / cuDNN |
| **버전 관리** | Git / GitHub (main + feature 브랜치 워크플로우) |

---

## 🔬 사용 데이터셋

| 데이터셋 | 참가자 | 용도 | 특징 |
|---|---|---|---|
| VISEM-Tracking | 20명 | YOLO11 학습 | 바운딩박스 + 추적ID |
| VISEM 원본 | 85명 | 회귀 모델 학습 | 영상 + 운동성 CSV |
| MHSMA | 235명 (1,540장) | 형태 모델 학습 | 4부위 정상/비정상 레이블 |

---

## ⚙️ 개발 환경

| 항목 | 스펙 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 3080 Ti Laptop (16GB VRAM) |
| RAM | 32GB |
| CPU | Intel i7-12800HX |
| Python | 3.10 |
| PyTorch | 2.6.0 + CUDA 12.4 |

---

## 📈 개발 로드맵

```
Phase 1  ✅ 완료   운동성 분석 (MAE 6.9%p, 논문 초과)         v1.0.0
Phase 2  ✅ 완료   CASA 키네마틱 (VCL/VSL/VAP/LIN/STR/WOB/ALH) v1.1.0
Phase 3  ✅ 완료   형태 분석 (EfficientNet-B3, AUC 0.725)     v1.2.0
Phase 4  ✅ 완료   AI-CASA 통합 보고서 (일반인 친화적)         v1.3.0
Phase 5  ✅ 완료   Flask 웹 데모 (인터랙티브 보고서)           v1.4.0  ← 현재
Phase 6  🔜 예정   외부 공유 (ngrok / 클라우드 배포)
Phase 7  🔜 예정   도메인 적응 개선 (Virtual Staining)
Phase 8  🔜 예정   실데이터 검증 / 학술 논문 제출
```

---

## 🏷️ Releases

| 버전 | 주요 내용 |
|---|---|
| **v1.4.0** | Flask 웹 데모 + 인터랙티브 보고서 (일반인/전문가 모드) |
| v1.3.0 | 일반인 친화적 보고서 + 종합 판정 시스템 |
| v1.2.0 | Phase 3 형태 분석 (EfficientNet-B3 v3) 통합 |
| v1.1.0 | Phase 2 CASA 키네마틱 7종 추가 |
| v1.0.0 | Phase 1 운동성 분석 (논문 초과 달성) |

---

## ⚠️ 주의사항 및 한계

- 이 시스템은 **의학적 진단 도구가 아닙니다**
- 형태 분석은 MHSMA(염색)↔VISEM(비염색) **도메인 차이**로 비정상 비율이 다소 과대 측정될 수 있음
- VISEM 데이터셋 환경에 최적화되어 있어 다른 현미경 기종에서는 성능이 달라질 수 있음
- 정자 농도가 매우 낮은 샘플(N < 10)은 분석 신뢰도가 낮음
- 최종 판단은 반드시 **전문 의료진**에게 받으시기 바랍니다

---

## 📚 참고 문헌

```bibtex
@article{thambawita2023visem,
  author  = {Thambawita, Vajira and Hicks, Steven A. and others},
  title   = {VISEM-Tracking, a human spermatozoa tracking dataset},
  journal = {Scientific Data},
  year    = {2023},
  doi     = {10.1038/s41597-023-02173-4}
}

@article{ottl2022motilitai,
  author  = {Ottl, Sandra and Amiriparian, Shahin and others},
  title   = {motilitAI: A machine learning framework for automatic
             prediction of human sperm motility},
  journal = {iScience},
  year    = {2022},
  doi     = {10.1016/j.isci.2022.104644}
}

@article{javadi2019mhsma,
  author  = {Javadi, Shahin and Mirroshandel, Seyed Abolghasem},
  title   = {A novel deep learning method for automatic assessment
             of human sperm images},
  journal = {Computers in Biology and Medicine},
  year    = {2019},
  doi     = {10.1016/j.compbiomed.2019.04.030}
}

@manual{who2021,
  title  = {WHO laboratory manual for the examination and
            processing of human semen, Sixth Edition},
  author = {{World Health Organization}},
  year   = {2021}
}
```

---

## 📝 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

---

*1인 개인 프로젝트 · 2026 · 문의: [GitHub Issues](https://github.com/MoriochoRadio/sperm-ai/issues)*
