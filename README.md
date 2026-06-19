<div align="center">

# 🌱 SEED Project — Team T.O.P

### AI 기반 정자 자동 탐지 · 형태 · 운동성 통합 분석 시스템

**SEED** — *Sperm Evaluation and Embryo Development*

AI가 현미경 영상 속 정자를 자동으로 검출·추적하여
**운동성**과 **형태**를 동일 객체 기준으로 정량 평가하는 통합 분석 시스템

<br>

`2026-1 융합캡스톤디자인 I` · `Team T.O.P (Technology Of Prognosis)` · `송기원 교수님` · `v1.0.0`

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF)
![Flask](https://img.shields.io/badge/Flask-API-000000?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📑 목차

1. [Introduction — 프로젝트 소개 · 팀 · 개발방법론](#1-introduction)
2. [Problem — 문제 인식 · 정의 · 해결 방안](#2-problem)
3. [How — Actor · 개발환경 · 시스템 아키텍처](#3-how)
4. [Outcome — 데이터셋 · 처리 흐름 · 성능 · 기대효과](#4-outcome)
5. [Artifacts — 산출물 · 향후 계획](#5-artifacts)
6. [Quick Start · 프로젝트 구조](#-quick-start)
7. [Reference](#-reference)

---

## 1. Introduction

### 💡 프로젝트 소개

> **T.O.P** (*Technology Of Prognosis*) — "예측 기술"이라는 의미로, **AI를 활용한 질병 예측 실현** 을 지향한다.

**SEED** 는 AI가 일반 현미경 영상 속 정자를 자동으로 검출·추적하여, 정자의 상태(운동성·형태)를 **정량적**으로 평가하는 시스템이다. 검사자의 주관과 고가 장비에 의존하던 기존 정자 분석을, 누구나 접근 가능한 객관적·통합적 분석으로 전환하는 것을 목표로 한다.

### 👥 팀 구성 (조직도)

| 역할 | 이름 | 담당 업무 | 개발 파트 |
|---|---|---|---|
| **PM** | 김민지 | 프로젝트 총괄 · 일정 / 진척도 관리 | 운동성 분석 모델 |
| **CM** | 지승현 | 개발환경 표준화 · 형상 관리 · 소스 통합 | 데이터 수집 · 전처리 |
| **QA** | 서현준 | 산출물 품질 관리 · 시스템 위험 관리 · 성능 모니터링 | 객체 탐지 모델 |
| **ENG1** | 김태경 | 시스템 구조 설계 · 통합 관리 · 성능 개선 | 형태 분석 모델 |
| **ENG2** | 김혜현 | 시스템 구조 설계 · 통합 관리 · 성능 개선 | 결과 웹 페이지 |

### 🔄 개발 방법론

단계별 산출물과 검증 절차를 적용하는 **폭포수(Waterfall) 개발 방법론** 을 채택하였다.

```
제안 → 분석 → 설계 → 구현 → 시험 → 완료
```

---

## 2. Problem

### 🔍 문제 인식

남성 요인은 전체 불임의 약 **50%** 를 차지하며, 진단 수도 빠르게 증가하고 있다. (2020 → 2024, 국내 남성 난임 진단 **+36.9%**)
그럼에도 기존 정자 분석 방식에는 다음과 같은 구조적 한계가 존재한다.

| 방식 | 한계 |
|---|---|
| **수동 판독** | 검사자 숙련도·컨디션에 따른 결과 편차, 30분~1시간 소요, 정량화 어려움 |
| **CASA** (자동 분석) | 형태 판독은 여전히 전문가 수동 의존, 형태·운동성 **분리 분석**, 고가 장비(3~4만 달러)로 낮은 접근성 |
| **자가검사 키트** | 정자 농도 확인 위주, 형태·정밀 파라미터 측정 불가 |

### 🎯 문제 정의

> **객관적·통합적이며 접근이 용이한 AI 정자 분석 시스템의 부재**

| # | 문제 |
|---|---|
| 1 | **결과 객관성 부족** — 수동 판독 편차, CASA 형태 판독의 주관성 |
| 2 | **형태·운동성 통합 분석 부재** — 동일 정자 객체 기준의 통합 평가 불가 |
| 3 | **낮은 접근성** — 고가 장비·전문 인력 요구로 도입 제약 |

### ✅ 해결 방안

일반 현미경 영상만으로 동작하는 **AI 기반 정자 탐지·형태·운동성 통합 분석 시스템** 을 웹 기반으로 구축한다.

| # | 기여 | 핵심 기술 | 목표 성능 |
|---|---|---|---|
| 1 | 정량화된 운동성 분석 | YOLO11 탐지 + ByteTrack 추적 + 키네마틱(VCL/VSL/ALH 등) 산출 | mAP@50 ≥ 0.65 · MAE ≤ 7.0 |
| 2 | 형태·운동성 통합 분석 | EfficientNet-B3 형태 분류 + 동일 정자 기준 통합 파이프라인 | 부위별 평균 AUC ≥ 0.72 |
| 3 | 분석 접근성 향상 | 일반 현미경 영상 입력 · Flask 기반 웹 결과 제공 | — |

---

## 3. How

### 🎭 Actor 정의

| Actor | 상황 | 목표 |
|---|---|---|
| **남성 사용자** | 임신 계획 등으로 불임 여부를 사전 확인하고 싶음 | 전문 장비·인력 없이 자가 점검 |
| **의료진** | 정액 분석 보조 참고 도구가 필요함 | 객관적·정량적 수치로 판독 보조 |

> ⚠️ 본 시스템은 **보조·참고 도구**이며 의학적 진단을 대체하지 않는다.

### 🛠 개발 환경

| 구분 | 기술 |
|---|---|
| 개발 언어 | Python 3.10 |
| AI 프레임워크 | PyTorch · Ultralytics (YOLO11) · scikit-learn |
| 개발 도구 | JupyterLab · Cursor IDE |
| API 서버 | Flask · Gunicorn |
| 외부 배포 | Render · ngrok |
| 버전 관리 | Git · GitHub |

### 🏗 System Architecture

입력부터 결과 출력까지 **4계층 AI 분석 파이프라인** 으로 구성된다.

```
┌─────────────────────────────────────────────────────────────┐
│  [입력 계층]   현미경 영상 (.mp4/.avi) → 품질 평가 · 정규화      │
│                quality.py · normalizer.py → 640×480 / 50fps   │
├─────────────────────────────────────────────────────────────┤
│  [탐지·추적]   YOLO11 정자 탐지 → ByteTrack ID·궤적 복원        │
│                detector.py · tracker.py                       │
├─────────────────────────────────────────────────────────────┤
│  [분석 계층]   키네마틱 산출 · 운동성 회귀 · 형태 분류           │
│                casa_features.py · analyzer.py · morphology.py  │
├─────────────────────────────────────────────────────────────┤
│  [해석·출력]   WHO 기준 판정 · 신뢰도 점수 · 주석 영상 · 웹 리포트│
│                interpreter.py · annotator.py · webapp/        │
└─────────────────────────────────────────────────────────────┘
```

### 📋 Use Case

```
영상 업로드 → (품질 검증) → AI 분석 → 결과 리포트 제공
```

사용자가 웹에서 현미경 영상을 업로드하면, 서버가 비동기로 분석을 수행하고
운동성·형태·키네마틱 지표와 신뢰도 점수가 담긴 결과 화면을 제공한다.

---

## 4. Outcome

### 🗂 데이터셋 (3종)

공신력 있는 공개 데이터셋을 용도별로 활용하였다. (모두 비염색 위상차 현미경 영상 → 실제 적용 도메인과 일치)

| 용도 | 데이터셋 | 제공처 | 데이터 |
|---|---|---|---|
| 객체 탐지·추적 | **VISEM-Tracking** | SimulaMet · OsloMet (2023) | 현미경 영상(640×480, 50fps) + 박스·추적 ID |
| 운동성 분석 | **VISEM** (원본) | SimulaMet · OsloMet (2019) | 참가자 85명 영상 + 임상 운동성 측정값(CSV) |
| 형태 분석 | **MHSMA** | Javadi & Mirroshandel (2019) | 크롭 이미지(128×128) 1,540장 + 4부위 레이블 |

> VISEM 계열은 *Nature Scientific Data* 게재 · CC BY 4.0, MHSMA는 정자 형태 분석 분야의 표준 벤치마크 데이터셋이다.

### ⚙️ 모델 처리 흐름 (5개 핵심 알고리즘)

| # | 단계 | 모델 / 알고리즘 | 역할 |
|---|---|---|---|
| 1 | 객체 탐지 | **YOLO11** | 프레임별 정자 탐지 |
| 2 | 객체 추적 | **ByteTrack** | 동일 정자 ID·궤적 복원 |
| 3 | 키네마틱 계산 | CASA 지표 산출 | VCL·VSL·VAP·LIN·ALH 등 운동학 지표 |
| 4 | 운동성 분석 | **Ridge + RandomForest 앙상블** | 전진(PR)·비전진(NP)·비운동(IM) 비율 예측 |
| 5 | 형태 분석 | **EfficientNet-B3** | 머리·첨체·공포·꼬리 4부위 정상 여부 분류 |

### 📊 모델 성능 — 목표치 전 항목 초과 달성 ✅

| 항목 | 지표 | 결과 | 목표 | 비고 |
|---|---|---|---|---|
| 운동성 분석 | MAE | **6.90 %p** | ≤ 7.0 | 선행연구 motilitAI(7.31%p) 대비 0.41%p↓ · 5-Fold CV |
| 객체 탐지 | mAP@50 | **0.677** | ≥ 0.65 | 정자를 올바른 위치에서 탐지한 비율 |
| 형태 분석 | 평균 AUC | **0.727** | ≥ 0.72 | 4부위 정상/비정상 구분 능력 |

> 📈 상세 평가(모델 버전별 비교·교차검증·도메인 적응 실험)는 [`docs/performance.md`](docs/performance.md) 참고.

추가로 분석 결과의 신뢰도를 **0~100점**으로 정량화하여, 정자 수 부족·탐지 불안정 시 감점하고 60점 미만이면 **재촬영을 권고**한다. (WHO 6판 정상 기준: 총 운동성 PR+NP ≥ 40%, 전진 PR ≥ 32%)

### 🖥 웹 페이지 UI/UX

```
① 업로드 화면  →  ② 분석 화면 (비동기 진행)  →  ③ 결과 화면 (지표 · 주석 영상)
```

Flask 기반 웹 애플리케이션으로 영상 업로드부터 결과 제공까지 사용자 중심 흐름을 제공한다. (`webapp/`)

### 🌟 기대 효과

| # | 효과 | 내용 |
|---|---|---|
| 1 | **분석 객관성 확보** | 검사자 컨디션과 무관한 일관된 결과 · 정량적 지표 기반 판독 보조 |
| 2 | **통합 분석 실현** | 동일 정자 기준 형태·운동성 동시 평가 → 종합적 판단 가능 |
| 3 | **분석 접근성 향상** | 고가 장비 없이 일반 현미경 영상만으로 누구나 분석 가능 |

---

## 5. Artifacts

### 📦 산출물

프로젝트 전 단계에 걸쳐 **총 21종**의 공식 산출물을 작성하였으며, **MS Project 기준 진행률 100%** 로 완료되었다.
최종 발표 자료 및 통합 산출물(v1.0.0)은 [`deliverables/`](deliverables/) 폴더에 보존되어 있다.

| 파일 | 종류 |
|---|---|
| `[T.O.P]Seed_최종발표_v1.0.0.pptx` | 최종 발표 슬라이드 |
| `4학년_융합캡스톤디자인I_SEED 판넬_v1.0.0.pptx` | 캡스톤 전시 판넬 |
| `[CM]T.O.P_통합산출물_v1.0.0.hwp` / `.pdf` | 통합 산출물 문서 (21종 통합) |
| `[PM]T.O.P_Ms_Project_v1.0.0.mpp` | 일정·진척도 관리 |

> 자세한 내용은 [`deliverables/README.md`](deliverables/README.md) 참고.

### 🚀 향후 개발 계획

| # | 방향 | 내용 |
|---|---|---|
| 1 | 처리 속도 향상 | 영상 전처리 병렬화 · 실시간 처리 → 분석 대기 시간 단축 |
| 2 | 저농도 샘플 대응 | 정자 수 < 10개 시 정확도 보완 · 재촬영 가이드 고도화 |
| 3 | 파라미터 확장 | 농도·총정자수·생존율 등 WHO 6판 기준 분석 항목 확장 |

---

## 🚀 Quick Start

```bash
# 1. 레포 클론
git clone https://github.com/MoriochoRadio/seed-project.git
cd seed-project

# 2. 가상환경 생성 및 의존성 설치
conda create -n seed python=3.10
conda activate seed
pip install -r requirements.txt
# PyTorch는 환경에 맞게 별도 설치 (requirements.txt 상단 주석 참고)

# 3. 모델 가중치 준비
#   - 운동성/형태 모델은 models/ 에 포함
#   - YOLO11 가중치(best.pt)는 Google Drive에서 자동 다운로드
#     (GDRIVE_* 환경변수 설정 후) python setup_models.py

# 4. 웹 데모 실행
python app.py          # http://localhost:5000
```

> 배포는 [`render.yaml`](render.yaml) 기반으로 Render.com에 구성되어 있다.

### 📂 프로젝트 구조

```
seed-project/
├── app.py            ← Flask 진입점
├── src/              ← 핵심 분석 모듈
│   ├── detector.py       (YOLO11 정자 탐지)
│   ├── tracker.py        (ByteTrack 추적 + 운동성 특징)
│   ├── casa_features.py  (CASA 키네마틱 지표)
│   ├── analyzer.py       (운동성 앙상블 회귀)
│   ├── morphology.py     (EfficientNet-B3 형태 분석)
│   ├── interpreter.py    (WHO 기준 판정 · 신뢰도)
│   ├── normalizer.py     (영상 정규화) · quality.py (품질 평가)
│   ├── annotator.py      (주석 영상 생성) · pipeline.py (통합 파이프라인)
│   └── ...
├── webapp/           ← Flask 웹 애플리케이션 (routes · templates · static)
├── models/           ← 학습된 모델 가중치 (운동성 앙상블, 형태 v3)
├── notebooks/        ← 실험·학습 노트북 (01~16)
├── docs/             ← 프로젝트 문서 (architecture · performance · guides)
├── deliverables/     ← 최종 발표 자료 및 공식 산출물 (v1.0.0)
└── data/             ← 데이터셋 (gitignore 처리)
```

### 📚 더 보기

- **시스템 설계** — [`docs/architecture.md`](docs/architecture.md)
- **성능 평가 상세** — [`docs/performance.md`](docs/performance.md)
- **사용자 가이드** — [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) · [`docs/ADMIN_GUIDE.md`](docs/ADMIN_GUIDE.md)

---

## 📖 Reference

주요 참고 문헌 (전체 목록은 발표 자료 참고):

- Thambawita, V., et al. (2023). *VISEM-Tracking, a human spermatozoa tracking dataset.* Data in Brief, 47, 108944.
- Hicks, S. A., et al. (2019). *Machine learning-based analysis of sperm videos for male fertility prediction.* Scientific Reports, 9, 16770.
- Javadi, S., & Mirroshandel, S. A. (2019). *A novel deep learning method for automatic assessment of human sperm morphology.* Computers in Biology and Medicine, 109, 182–194.
- Zhang, Y., et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022.
- Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML 2019.
- World Health Organization. (2021). *WHO laboratory manual for the examination and processing of human semen* (6th ed.).

---

## 📄 License

본 프로젝트는 **학술 목적의 캡스톤 프로젝트**이며, **임상 진단 도구가 아님**을 명시한다.
배포 라이선스는 [`LICENSE`](LICENSE) 참고.

<div align="center">

**Team T.O.P** · Technology Of Prognosis · 2026

</div>
