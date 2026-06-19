# Team T.O.P — SEED Project

> **AI 기반 정자 자동 탐지·운동성·형태 분석 시스템**  
> 2026-1 융합캡스톤디자인 I · Team T.O.P (Technology Of Prognosis)

> 본 레포는 26년 1학기 캡스톤 팀 프로젝트입니다.

---

## 📌 Project Overview

**SEED** (*Sperm Evaluation and Embryo Development*) — AI가 현미경 영상 속 정자를 자동으로 검출·추적하여 운동성과 형태를 정량적으로 평가하는 통합 분석 시스템.

---

## 🎯 Problem

남성 불임은 전체 불임 원인의 약 **45%** 를 차지하며, 국내 진단 수는 5년간 **36.9% 증가**했다. 그러나 기존 정자 분석 방식에는 세 가지 구조적 한계가 존재한다.

| # | 한계 | 내용 |
|---|---|---|
| 1 | **결과 객관성 부족** | 수동 판독은 검사자 숙련도·컨디션에 따라 편차 발생, CASA의 형태 판독도 전문가의 주관적 판단에 의존 |
| 2 | **형태·운동성 통합 분석 부재** | 기존 CASA는 두 분석을 분리하여 동일 정자에 대한 통합 평가가 불가 |
| 3 | **낮은 접근성** | 고가 장비(3~4만 달러)와 전문 인력 요구로 일반 의료기관 도입 제약, 자가검사 키트는 정밀 분석 불가 |

---

## 💡 Solution

| # | 기여 | 핵심 기술 | 목표 성능 |
|---|---|---|---|
| 1 | 정량화된 운동성 분석 | YOLO11 탐지 + ByteTrack 추적 + 키네마틱(VCL/VSL/ALH 등) 산출 | mAP50 ≥ 0.65 / MAE ≤ 7.0 |
| 2 | 형태·운동성 통합 분석 | EfficientNet-B3 기반 형태 분류 + 동일 정자 기준 통합 파이프라인 | 부위별 평균 AUC ≥ 0.72 |
| 3 | 분석 접근성 향상 | 일반 현미경 영상 입력, Flask 기반 웹 결과 제공 | — |

## 🌟 Expected Impact

- **분석 객관성 확보** — 판독 편차 감소 및 정량적 지표 제공
- **통합 분석 실현** — 형태·운동성·키네마틱을 단일 파이프라인에서 동시 산출
- **진단 접근성 향상** — 고가 장비 없이 누구나 활용 가능한 분석 환경

---

## 👥 Team

| 역할 | 담당 업무 | 개발 파트 |
|---|---|---|
| **PM** | 프로젝트 총괄 · 일정 / 진척도 관리 | 정자 운동성 분석 |
| **CM** | 개발환경 표준화 · 형상 관리 · 소스 통합 | 전처리 최적화 |
| **QA** | 산출물 품질 관리 · 시스템 위험 관리 · 모델 성능 모니터링 | 객체 탐지 — 정자 개수 분석 |
| **ENG1** | 시스템 구조 설계 및 제작 · 통합 관리 · 성능 개선 | 객체 탐지 · 형태 분석 |
| **ENG2** | 시스템 구조 설계 및 제작 · 통합 관리 · 성능 개선 | 형태 분석 · 결과 웹 페이지 |

> 본 프로젝트는 **폭포수(Waterfall) 개발 방법론** 에 따라 단계별 산출물과 검증 절차를 적용한다.

---

## 🛠 Tech Stack

| 구분 | 기술 |
|---|---|
| 개발 언어 | Python 3.10 |
| AI 프레임워크 | PyTorch · ultralytics (YOLO11) · scikit-learn |
| 개발 도구 | JupyterLab · Cursor IDE |
| API 서버 | Flask |
| 버전 관리 | Git · GitHub |

---

## 🌿 Branch Strategy

```
main          ← 안정 버전 (직접 push 금지, PR로만 merge)
└── develop   ← 통합 테스트 브랜치
    ├── feature/preprocessing   (CM)
    ├── feature/detection       (ENG1, QA)
    ├── feature/motility        (PM)
    ├── feature/morphology      (ENG1, ENG2)
    └── feature/webapp          (ENG2)
```

**작업 흐름**
1. 각자 `feature/*` 브랜치에서 작업
2. 완성되면 `develop`으로 Pull Request 생성
3. 다른 팀원 1명 이상의 코드 리뷰 후 merge
4. `develop`에서 통합 검증 후 `main`으로 merge

---

## 📚 Base System

본 팀 프로젝트의 출발점이 된 1인 프로토타입(`sperm-ai` v1.5.0)의 상세 개발 기록:

- **성능 평가** — [`docs/performance.md`](docs/performance.md)
- **시스템 구조** — [`docs/architecture.md`](docs/architecture.md)

---

## 📂 Project Structure

```
seed-project/
├── src/              ← 핵심 모듈 (detector, tracker, analyzer, morphology, pipeline)
├── webapp/           ← Flask 웹 애플리케이션
├── models/           ← 학습된 모델 가중치 (YOLO11, 운동성 앙상블, 형태 v3)
├── data/             ← 데이터셋 (gitignore 처리)
├── notebooks/        ← 실험 및 분석 노트북
├── docs/             ← 프로젝트 문서 (architecture, performance, guides)
├── deliverables/     ← 최종 발표 자료 및 공식 산출물 (v1.0.0)
└── README.md
```

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

# 3. 자신의 작업 브랜치로 이동
git checkout feature/[자신의-파트명]
```

---

## 📦 Deliverables

프로젝트 최종 발표 자료 및 공식 산출물(v1.0.0)은 [`deliverables/`](deliverables/) 폴더에 보존되어 있습니다.

- 최종 발표 슬라이드 · 캡스톤 판넬 (`.pptx`)
- 통합 산출물 문서 (`.hwp` / `.pdf`)
- MS Project 일정·진척도 관리 (`.mpp`)

자세한 내용은 [`deliverables/README.md`](deliverables/README.md) 참고.

---

## 📄 License

본 프로젝트는 학술 목적의 캡스톤 프로젝트이며, 임상 진단 도구가 아님을 명시한다.