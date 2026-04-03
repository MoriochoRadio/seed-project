# GitHub 업로드 가이드

## 1. 사전 준비

### Git 설치 확인
```bash
git --version
# git version 2.x.x 이상이면 OK
```

Git이 없으면: https://git-scm.com/downloads 에서 설치

### Git 초기 설정 (처음 한 번만)
```bash
git config --global user.name "your_github_username"
git config --global user.email "your@email.com"
```

---

## 2. GitHub 저장소 생성

1. https://github.com 접속 후 로그인
2. 우측 상단 **"+"** → **"New repository"** 클릭
3. 설정:
   - **Repository name**: `sperm-ai` (또는 원하는 이름)
   - **Description**: `AI-based sperm motility analysis system`
   - **Visibility**: Public (포트폴리오용) 또는 Private
   - **Initialize**: 체크 해제 (우리가 직접 올릴 거니까)
4. **Create repository** 클릭

---

## 3. 로컬 저장소 초기화 및 업로드

Anaconda Prompt에서 실행:

```bash
# 프로젝트 폴더로 이동
cd C:\Users\neo62\sperm-ai

# Git 초기화
git init

# 원격 저장소 연결 (your_username을 실제 GitHub 아이디로 변경)
git remote add origin https://github.com/MoriochoRadio/sperm-ai.git

# 파일 추가
git add README.md
git add requirements.txt
git add LICENSE
git add .gitignore
git add bytetrack_custom.yaml
git add docs/
git add src/

# 커밋
git commit -m "feat: Add sperm motility analysis system (Phase 1 complete)

- YOLO11-based sperm detection (mAP50: 0.677)
- ByteTrack-based sperm tracking
- Ensemble regression model (MAE: 6.9%p, surpasses motilitAI)
- WHO 6th edition criteria interpretation
- Confidence scoring and retake recommendation
- Modular src/ architecture"

# GitHub에 업로드
git branch -M main
git push -u origin main
```

---

## 4. 데이터/모델 파일 처리

데이터와 모델 파일은 용량이 커서 GitHub에 직접 올리지 않아요.
대신 아래 방법을 사용해요:

### 방법 1: 다운로드 링크 README에 명시 (권장)
README.md에 이미 데이터셋 출처가 명시되어 있어요:
- VISEM-Tracking: https://zenodo.org/record/7293726
- VISEM: https://datasets.simula.no/visem/

### 방법 2: 모델 가중치 별도 공유 (선택)
- Google Drive에 업로드 후 링크 공유
- 또는 GitHub Releases에 업로드

---

## 5. GitHub 저장소 꾸미기 (선택)

### Topics 추가
저장소 페이지 → About 옆 ⚙️ → Topics:
```
computer-vision, sperm-analysis, yolo, deep-learning,
medical-imaging, python, pytorch, motility-analysis
```

### README 미리보기 확인
GitHub에서 README.md가 제대로 렌더링되는지 확인

---

## 6. 최종 저장소 구조 확인

```
GitHub 저장소 (공개):
sperm-ai/
├── README.md          ← 프로젝트 소개 (메인)
├── requirements.txt   ← 환경 설정
├── LICENSE            ← MIT 라이센스
├── .gitignore
├── bytetrack_custom.yaml
├── src/               ← 핵심 모듈
│   ├── __init__.py
│   ├── detector.py
│   ├── tracker.py
│   ├── analyzer.py
│   ├── interpreter.py
│   └── pipeline.py
└── docs/
    ├── performance.md  ← 상세 성능 분석
    └── architecture.md ← 시스템 설계

로컬에만 보관:
├── data/              ← 용량 큰 데이터 (gitignore)
├── models/            ← 모델 가중치 (gitignore)
├── outputs/           ← 결과물 (gitignore)
└── notebooks/         ← 실험 기록 (선택적 공개)
```
