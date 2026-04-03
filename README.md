# 🔬 AI-Based Sperm Motility Analysis System

> **병원 전 단계 보조 분석 시스템** — 정자 현미경 영상을 입력받아 AI로 운동성을 자동 분석하고 WHO 기준 기반 설명형 결과를 출력합니다.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/YOLO-11-darkgreen)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 프로젝트 소개

이 프로젝트는 **정자 현미경 영상**을 입력받아:

1. **YOLO11**로 정자를 프레임별로 탐지
2. **ByteTrack**으로 정자의 이동 경로를 추적
3. **앙상블 회귀 모델**로 운동성 수치 예측
4. **WHO 6판 기준**으로 결과 해석 및 설명형 보고서 출력

을 수행하는 **병원 전 단계 보조 분석 시스템**입니다.

> ⚠️ 본 시스템은 의학적 진단 도구가 아닙니다. 병원 방문 전 참고용 보조 분석 도구입니다.

---

## 🏆 핵심 성과

| 지표 | 본 시스템 | 논문 최고 (motilitAI) |
|---|---|---|
| 5-Fold CV MAE | **6.9%p** | 7.31%p |
| YOLO mAP50 | **0.677** | ~0.65 (YOLOv5l) |
| 판정 방향 정확도 | **75%** | — |

> 동일 데이터셋(VISEM) 기준으로 논문 최고 성능을 초과 달성

---

## 🏗️ 시스템 아키텍처

```
현미경 영상 입력 (.mp4 / .avi)
         │
         ▼
┌─────────────────────────────────┐
│         SpermDetector           │
│         (detector.py)           │
│  YOLO11 기반 정자 탐지           │
│  → 프레임별 정자 위치 검출        │
│  → 전체 정자 수 N 추정 (중앙값)   │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│         SpermTracker            │
│         (tracker.py)            │
│  ByteTrack 기반 정자 추적        │
│  → 정자별 이동 경로 (trajectory) │
│  → 11개 운동성 특징 추출         │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│       MotilityAnalyzer          │
│        (analyzer.py)            │
│  앙상블 회귀 모델                 │
│  Ridge + RandomForest 평균       │
│  → 전진 운동성 %                 │
│  → 비전진 운동성 %               │
│  → 비운동성 %                    │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│       interpreter.py            │
│  WHO 6판 기준 해석               │
│  → 정상/경계/주의 판정            │
│  → 신뢰도 점수 (0~100점)         │
│  → 재촬영 권고 여부              │
└─────────────────────────────────┘
         │
         ▼
    📋 설명형 보고서 출력
```

---

## 📊 WHO 기준 판정 로직

```
WHO 6판 (2021) 기준:

전진 운동성 (Progressive Motility)
  ≥ 32%  →  ✅ 정상
  22~32% →  ⚠️  경계
  < 22%  →  🔴 주의

총 운동성 (Total Motility)
  ≥ 40%  →  ✅ 정상
  < 40%  →  ⚠️  주의

비운동성 (Immotile)
  < 50%  →  ✅ 정상
  50~70% →  ⚠️  주의
  ≥ 70%  →  🔴 심각
```

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

### 2. 영상 분석 (3줄)

```python
from src.pipeline import SpermAnalysisPipeline

pipeline = SpermAnalysisPipeline()
result   = pipeline.analyze("your_video.mp4")
pipeline.print_report(result)
```

### 3. 출력 예시

```
=======================================================
  정자 운동성 분석 보고서
=======================================================
📊 분석 신뢰도: ✅ 높음 (100점)

📈 운동성 분석 결과 (탐지 정자 수: 50개)
   전진 운동성:    21.4%
   비전진 운동성:  26.2%
   비운동성:       52.4%
   총 운동성:      47.6%

🔍 WHO 기준 해석
   🔴 전진 운동성 21.4% — WHO 정상 기준(32%)보다 현저히 낮음
   ✅ 총 운동성 47.6%   — WHO 정상 기준(40%) 충족
   ⚠️  비운동성 52.4%  — 주의 필요 (기준 50% 초과)

🔴 종합 판정: [주의 필요]

💬 권고사항:
   AI 분석 결과 일부 지표가 WHO 기준 이하입니다.
   정밀 검사를 위해 병원 방문을 권고합니다.
   이 결과는 보조 분석이며 의학적 진단을 대체하지 않습니다.

-------------------------------------------------------
   ※ 이 결과는 AI 보조 분석이며 의학적 진단이 아닙니다.
=======================================================
```

---

## 📁 프로젝트 구조

```
sperm-ai/
├── data/
│   ├── raw/
│   │   ├── VISEM-Tracking/       # 20명, 바운딩박스 어노테이션
│   │   └── VISEM/                # 85명, 영상 + CSV
│   └── processed/
│       ├── yolo_dataset/         # YOLO 학습 데이터
│       └── yolo_balanced/        # 균형 데이터
├── models/
│   ├── yolo11_sperm_v2/
│   │   └── weights/best.pt       # YOLO11 탐지 모델
│   └── motility_ensemble.pkl     # 앙상블 회귀 모델
├── notebooks/                    # 실험 기록 노트북
│   ├── 01_check_environment.ipynb
│   ├── 04_train_yolo11.ipynb
│   ├── 09_motility_regression.ipynb
│   ├── 10_visem_features.ipynb
│   ├── 13_final_system.ipynb
│   └── ...
├── outputs/
│   ├── visem_features.json       # 85명 추출 특징
│   ├── final_validation_report.csv
│   └── result_participant_11.mp4 # 시각화 결과 영상
├── src/                          # 핵심 모듈
│   ├── __init__.py
│   ├── detector.py               # SpermDetector 클래스
│   ├── tracker.py                # SpermTracker 클래스
│   ├── analyzer.py               # MotilityAnalyzer 클래스
│   ├── interpreter.py            # WHO 해석 + 신뢰도
│   └── pipeline.py               # 통합 파이프라인 (메인)
├── docs/
│   ├── performance.md            # 상세 성능 분석
│   └── architecture.md           # 시스템 설계 문서
├── README.md
├── requirements.txt
└── bytetrack_custom.yaml
```

---

## 🔬 사용 데이터셋

### VISEM-Tracking
- **출처**: Thambawita et al. (2023), SimulaMet / OsloMet
- **참가자**: 20명
- **내용**: 각 30초 영상 (640×480, 50fps) + 바운딩박스 + 추적 ID
- **클래스**: sperm(0), cluster(1), small/pinhead(2)
- **용도**: YOLO11 탐지 모델 학습
- **라이센스**: CC BY 4.0

### VISEM 원본
- **출처**: Haugen et al. (2019), Simula Research Laboratory
- **참가자**: 85명
- **내용**: .avi 영상 + 운동성 CSV
- **용도**: 앙상블 회귀 모델 학습 (바운딩박스는 YOLO11로 자동 생성)
- **라이센스**: CC BY 4.0 (비상업적 사용)

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
| YOLO | YOLO11 (ultralytics) |
| 추적기 | ByteTrack |

---

## 🗺️ 개발 로드맵

```
Phase 1  ✅ 완료    운동성 분석 시스템
                   YOLO11 탐지 + ByteTrack 추적
                   앙상블 회귀 모델 (MAE 6.9%p)
                   WHO 기준 해석 + 신뢰도 평가

Phase 2  🔜 진행 예정   품질 평가 + 설명형 강화
                        신뢰도 로직 정교화
                        사용자 친화적 출력 개선

Phase 3  📋 계획 중    형태 분석 (Morphology)
                       정자 형태 이상 판별
                       형태 + 운동성 결합 분석

Phase 4  🎯 최종 목표  AI-CASA 통합 시스템
                       운동성 + 형태 + 농도 + 신뢰도
                       종합 보고서 자동 생성
```

---

## 📚 참고 문헌

```bibtex
@article{thambawita2023visem,
  author  = {Thambawita, Vajira and Hicks, Steven A. and
             Storås, Andrea M. and Nguyen, Thu and others},
  title   = {VISEM-Tracking, a human spermatozoa tracking dataset},
  journal = {Scientific Data},
  volume  = {10},
  year    = {2023},
  doi     = {10.1038/s41597-023-02173-4}
}

@article{ottl2022motilitai,
  author  = {Ottl, Sandra and Amiriparian, Shahin and
             Gerczuk, Maurice and Schuller, Björn},
  title   = {motilitAI: A machine learning framework for automatic
             prediction of human sperm motility},
  journal = {iScience},
  year    = {2022},
  doi     = {10.1016/j.isci.2022.104644}
}

@article{haugen2019visem,
  author  = {Haugen, Trine B. and Hicks, Steven A. and
             Andersen, Jorunn M. and others},
  title   = {VISEM: A multimodal video dataset of human spermatozoa},
  journal = {ACM Multimedia Systems Conference (MMSys)},
  year    = {2019}
}

@manual{who2021,
  title  = {WHO laboratory manual for the examination
            and processing of human semen, 6th edition},
  author = {{World Health Organization}},
  year   = {2021}
}
```

---

## 📝 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

---

## 👤 개발자

1인 개인 프로젝트 | 문의: GitHub Issues
