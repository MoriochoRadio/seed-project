# 🔬 AI-CASA: AI 기반 정자 종합 분석 시스템

> 정자 현미경 영상을 입력받아 **운동성 · 운동 패턴 · 형태**를 자동 분석하고
> WHO 기준 기반 설명형 결과를 출력하는 병원 전 단계 보조 분석 시스템

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/YOLO-11-darkgreen)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Phase](https://img.shields.io/badge/Phase-3%20Complete-brightgreen)]()

---

## 📌 프로젝트 소개

이 프로젝트는 **정자 현미경 영상** 하나만 있으면:

1. **YOLO11** 로 정자를 프레임별 자동 탐지
2. **ByteTrack** 으로 정자 이동 경로 추적
3. **앙상블 회귀 모델** 로 운동성 수치 예측
4. **CASA 키네마틱** (VCL, VSL, VAP, LIN, STR, WOB, ALH) 계산
5. **EfficientNet-B3** 로 정자 형태 분류 (머리/첨체/공포/꼬리)
6. **WHO 6판 기준** 으로 종합 해석 및 설명형 보고서 출력

을 자동으로 수행합니다.

> ⚠️ 본 시스템은 의학적 진단 도구가 아닙니다. 병원 방문 전 참고용 보조 분석 도구입니다.

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
현미경 영상 입력 (.mp4 / .avi)
         │
         ▼
┌─────────────────────────────┐
│      SpermDetector          │  YOLO11 정자 탐지
│      (detector.py)          │  → 전체 정자 수 N 추정
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│      SpermTracker           │  ByteTrack 정자 추적
│      (tracker.py)           │  → 이동 경로 + CASA 키네마틱
└─────────────────────────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌──────────────────────┐     ┌──────────────────────────┐
│   MotilityAnalyzer   │     │   MorphologyAnalyzer     │
│   (analyzer.py)      │     │   (morphology.py)        │
│  Ridge+RF 앙상블      │     │  EfficientNet-B3 v3      │
│  → 운동성 % 예측      │     │  → 형태 정상/비정상 분류  │
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
               📋 AI-CASA 통합 보고서
```

---

## 📊 출력 보고서 예시

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

### 2. 분석 실행 (3줄)

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
├── src/
│   ├── detector.py       # SpermDetector  (YOLO11)
│   ├── tracker.py        # SpermTracker   (ByteTrack + CASA)
│   ├── analyzer.py       # MotilityAnalyzer (앙상블 회귀)
│   ├── morphology.py     # MorphologyAnalyzer (EfficientNet-B3)
│   ├── interpreter.py    # WHO 해석 + 신뢰도 + 종합 판정
│   └── pipeline.py       # SpermAnalysisPipeline (통합)
├── models/
│   ├── yolo11_sperm_v2/              # YOLO11 탐지 모델
│   ├── motility_ensemble.pkl         # 운동성 회귀 모델
│   └── morphology_efficientnet_b3_v3.pt  # 형태 분류 모델 (v3)
├── notebooks/            # 실험 기록 노트북 (01~16번)
├── docs/
│   ├── performance.md    # 상세 성능 분석
│   └── architecture.md   # 시스템 설계 문서
├── README.md
├── requirements.txt
└── bytetrack_custom.yaml
```

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
| Detection | YOLO11 (ultralytics) |
| Tracking | ByteTrack |
| Morphology | EfficientNet-B3 (torchvision) |

---

## 📈 개발 로드맵

```
Phase 1  ✅ 완료   운동성 분석 (MAE 6.9%p, 논문 초과)
Phase 2  ✅ 완료   CASA 키네마틱 (VCL/VSL/VAP/LIN/STR/WOB/ALH)
Phase 3  ✅ 완료   형태 분석 (EfficientNet-B3, AUC 0.725)
Phase 4  ✅ 완료   AI-CASA 통합 보고서 (일반인 친화적)
Phase 5  🔜 예정   도메인 적응 개선 / 실데이터 검증
```

---

## ⚠️ 주의사항 및 한계

- 이 시스템은 **의학적 진단 도구가 아닙니다**
- 형태 분석은 MHSMA↔VISEM **도메인 차이**로 수치 과대 추정 가능성 있음
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

*1인 개인 프로젝트 | 문의: GitHub Issues*
