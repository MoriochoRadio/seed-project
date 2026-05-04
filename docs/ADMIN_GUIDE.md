# AI-CASA 관리자/개발자 가이드

> 시스템 운영, 디버깅, 확장을 위한 기술 문서

---

## 🏗️ 시스템 아키텍처

### 전체 구조

```
┌──────────────────────────────────────────────────────┐
│                  사용자 (브라우저)                      │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│              Flask Web Server (app.py)                │
│                                                        │
│  routes.py — API 엔드포인트                            │
│    GET  /                — 메인 페이지                  │
│    POST /api/analyze     — 영상 업로드                  │
│    GET  /api/status/<id> — 진행 상태 폴링               │
│    GET  /analyzing/<id>  — 분석 진행 페이지             │
│    GET  /result/<id>     — 결과 보고서                  │
│    GET  /rejected/<id>   — 거부 페이지                  │
└────────────────────┬─────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────┐
│         Background Thread (tasks.py)                  │
│                                                        │
│  Step 0: 영상 품질 평가 (quality.py)                   │
│  Step 1: 영상 정규화 (normalizer.py, 조건부)            │
│  Step 2: 정자 탐지 (detector.py - YOLO11)              │
│  Step 3: 정자 추적 (tracker.py - ByteTrack + CASA)     │
│  Step 4: 운동성 분석 (analyzer.py - Ridge+RF)           │
│  Step 5: 형태 분석 (morphology.py - EfficientNet-B3)   │
│  Step 6: WHO 해석 (interpreter.py)                     │
└──────────────────────────────────────────────────────┘
```

### 데이터 흐름

```
1. 사용자 영상 업로드
   → tmp_uploads/{timestamp}_{filename}.mp4 저장

2. 백그라운드 분석 시작 (Threading)
   → JOBS 딕셔너리에 작업 등록 (메모리 기반 큐)
   → 별도 스레드로 분석 실행

3. 프론트엔드 1.5초마다 폴링
   → /api/status/{job_id}
   → 진행 상태 + 단계 메시지 갱신

4. 분석 완료 시 결과 저장
   → JOBS[job_id]['result'] = result_dict

5. 사용자 결과 페이지로 자동 이동
   → /result/{job_id}
```

---

## 📁 프로젝트 구조

```
sperm-ai/
├── app.py                    # Flask 진입점
├── webapp/                   # 웹 애플리케이션
│   ├── __init__.py
│   ├── routes.py             # API 라우트
│   ├── tasks.py              # 백그라운드 작업 + 스마트 분기
│   ├── templates/            # Jinja2 HTML
│   │   ├── base.html         # 공통 레이아웃
│   │   ├── index.html        # 메인/업로드
│   │   ├── analyzing.html    # 분석 진행
│   │   ├── result.html       # 결과 보고서
│   │   └── rejected.html     # 거부 안내
│   └── static/
│       ├── css/style.css
│       └── js/
│           ├── upload.js     # 드래그앤드롭 + 업로드
│           ├── progress.js   # 폴링 + 진행 표시
│           └── result.js     # 도넛 차트 + 모드 토글
│
├── src/                      # 분석 코어
│   ├── detector.py           # YOLO11
│   ├── tracker.py            # ByteTrack + CASA
│   ├── analyzer.py           # 운동성 회귀
│   ├── morphology.py         # 형태 분류
│   ├── interpreter.py        # WHO 해석
│   ├── pipeline.py           # 통합 파이프라인
│   ├── quality.py            # 영상 품질 평가 (Phase 6)
│   └── normalizer.py         # 영상 정규화 (Phase 6)
│
├── models/                   # 학습된 모델 (Git 미포함)
│   ├── yolo11_sperm_v2/
│   ├── motility_ensemble.pkl
│   └── morphology_efficientnet_b3_v3.pt
│
├── notebooks/                # 실험 기록
├── docs/                     # 문서
├── tmp_uploads/              # 임시 업로드 (Git 미포함)
└── requirements.txt
```

---

## 🚀 서버 실행

### 개발 모드 (debug=True)

```bash
cd C:\Users\neo62\sperm-ai
conda activate sperm-ai
python app.py
```

특징:
- 코드 변경 시 자동 재시작
- 상세 에러 페이지 표시
- localhost:5000 바인딩

### 외부 공유 (ngrok)

```bash
# 별도 명령창에서
ngrok.exe http 5000
```

발급된 URL을 외부에 공유 가능. 노트북/Wi-Fi가 켜져 있어야 동작.

---

## 🛠️ 운영 작업

### 임시 파일 정리

업로드된 영상이 누적되어 디스크 점유 → 주기적 정리 필요:

```bash
# tmp_uploads 폴더 정리 (.gitkeep만 유지)
del tmp_uploads\*.mp4
del tmp_uploads\*.avi
```

### 메모리 누수 모니터링

```
JOBS 딕셔너리는 메모리 기반:
  - 서버 재시작 시 모든 작업 정보 소실
  - 장기 운영 시 누적 가능

권장: 일정 시간 경과 후 자동 정리 로직 추가
   (현재는 미구현, 데모 목적)
```

### 모델 파일 관리

`.pt` 모델 파일은 Git에 올리지 않음 (용량 큼). 다른 PC 배포 시:

```
1. models/ 폴더 생성
2. 다음 파일 복사:
   - yolo11_sperm_v2/weights/best.pt
   - motility_ensemble.pkl
   - morphology_efficientnet_b3_v3.pt
```

---

## 🧪 디버깅

### Flask 콘솔 로그 확인

서버 실행 중 콘솔에 표시되는 정보:

```
127.0.0.1 - - [날짜] "GET /api/status/xxx HTTP/1.1" 200 -
  ↑                       ↑                          ↑
  요청 IP                  엔드포인트                    HTTP 코드
```

### 분석 진행 상태 확인

특정 작업의 상세 상태가 궁금할 때 (개발자 도구):

```javascript
// 브라우저 콘솔에서
fetch('/api/status/<job_id>')
  .then(r => r.json())
  .then(d => console.log(d));
```

### 디버그 출력 추가

`tasks.py`의 `run_analysis` 함수에 print 추가하면 콘솔에 출력됨:

```python
print(f"[DEBUG] 영상 메타: {quality['metadata']}")
```

### 자주 발생하는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| 분석이 오래 걸림 | 영상이 너무 큼 또는 풀 정규화 발동 | 영상 사양 확인, 짧게 자르기 |
| "정규화 실패" 오류 | OpenCV 코덱 문제 | mp4v 코덱 설치 확인 |
| 모델 로딩 실패 | models/ 폴더 누락 | 모델 파일 경로 확인 |
| 포트 5000 사용 중 | 다른 프로세스가 점유 | 다른 포트 사용 또는 프로세스 종료 |

---

## ⚙️ 핵심 모듈 설명

### quality.py — 영상 품질 평가

```python
class VideoQualityAnalyzer:
    """6차원 가중 평가 + 하드 룰"""
    
    가중치:
        resolution  0.20
        fps         0.15
        duration    0.05  (정규화로 해결되므로 낮음)
        stability   0.20
        brightness  0.15
        sharpness   0.25  (실질 품질 비중 높음)
    
    하드 룰 (등급과 무관, 무조건 거부):
        - duration < 5초
        - 해상도 < 320×240
        - fps < 10
        - 프레임 수 < 30
```

### normalizer.py — 영상 정규화

```python
class VideoNormalizer:
    """3-tier 스마트 분기"""
    
    get_normalization_type():
        'none' — 변환 불필요
        'trim' — 길이만 자르기 (5~15초)
        'full' — 해상도/FPS 변환 (30~60초)
    
    fast_trim():
        - 처음 MAX_DURATION(180초)만 사용
        - CLAHE 대비 향상 적용
        - 원본 코덱 유지
    
    normalize():
        - 풀 변환
        - 해상도 → 640×480
        - FPS → 50
        - 그레이스케일 + CLAHE
```

### tasks.py — 작업 관리

```python
JOBS = {}  # 메모리 기반 작업 큐

create_job():
    - UUID 발급
    - 백그라운드 스레드 시작

run_analysis():
    Step 0: 품질 평가 → 거부 가능
    Step 1: 정규화 (스마트 분기)
    Step 2~6: 기존 파이프라인
    
표시 조건:
    show_quality_card = (
        등급 B/C 이하 OR
        풀 정규화 적용 OR
        심각한 이슈 존재
    )
```

---

## 🔐 보안 고려사항

### 현재 구현

```
✅ 파일 확장자 검증 (.mp4/.avi/.mov)
✅ 파일 크기 제한 (500MB)
✅ secure_filename으로 경로 인젝션 방지
✅ UUID로 작업 ID 생성 (예측 불가)

⚠️ 미구현 (데모 목적):
- 사용자 인증
- 영상 업로드 횟수 제한
- 분석 결과 영구 저장
- 로그 기록
- HTTPS (ngrok이 자동 처리)
```

### 운영 시 추가 필요

```python
# Rate limiting 예시
from flask_limiter import Limiter

limiter = Limiter(app=app, key_func=get_remote_address)

@bp.route('/api/analyze', methods=['POST'])
@limiter.limit("10 per hour")
def api_analyze():
    ...
```

---

## 📈 성능 최적화

### 현재 병목

```
1. 영상 정규화 (CPU 인코딩)
   - VideoWriter mp4v 코덱
   - 1500 프레임 × 50ms ≈ 75초

2. YOLO11 탐지
   - GPU 사용 (RTX 3080 Ti)
   - 1 프레임 ≈ 30ms

3. 추적 + 키네마틱 계산
   - CPU 작업
   - 의외로 빠름

병목 순위:
   정규화 > YOLO 탐지 > 형태 분석 > 추적
```

### 개선 가능한 부분

```
1. NVENC GPU 인코딩 사용
   - cv2.VideoWriter → NVIDIA NVENC
   - 5~10배 빨라짐
   - 코드 복잡도 증가

2. 분석 결과 캐싱
   - 같은 영상 재분석 시 즉시 반환
   - 해시 기반 중복 검사

3. 작업 큐 분리 (Redis)
   - 메모리 → Redis
   - 다중 워커 가능
   - 장기 운영 안정성
```

---

## 🌐 배포 옵션

### 옵션 1: 로컬 + ngrok (현재)

```
장점:
  ✅ GPU 모델 빠르게 동작
  ✅ 무료
  ✅ 5분 설정

단점:
  ⚠️ 노트북 + Wi-Fi 의존
  ⚠️ 발표 끝나면 URL 죽음
```

### 옵션 2: 클라우드 GPU (유료)

```
- AWS EC2 (g4dn.xlarge): 시간당 $0.5~
- Google Cloud (T4 인스턴스): 비슷한 가격
- RunPod / Vast.ai: 더 저렴

장점:
  ✅ 24시간 운영
  ✅ GPU 가속

단점:
  ⚠️ 비용 발생
  ⚠️ 설정 복잡
```

### 옵션 3: 모델 서빙 분리

```
프론트엔드/Flask: 무료 클라우드 (Render/Railway)
모델 서빙: GPU 가능한 별도 인프라

→ 비용 효율적
→ 프론트엔드만 24시간 운영
→ 모델은 호출 시에만 동작
```

---

## 📊 모니터링

### 운영 중 확인할 지표

```
1. 분석 성공률
   - 거부율 / 오류율 / 정상 완료율

2. 평균 분석 시간
   - 시나리오별 (S/A/B/C 등급)

3. 메모리 사용량
   - JOBS 딕셔너리 크기
   - 임시 파일 누적량

4. GPU 활용도
   - nvidia-smi로 확인
```

### 로그 추가 권장

```python
import logging

logging.basicConfig(
    filename='aicasa.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 사용 예
logging.info(f"분석 시작: {job_id}, 영상: {video_name}")
```

---

## 🚧 향후 개발 계획

### Phase 7 (예정)

```
A. 옵션 B 도전: GAN 도메인 변환
   - 자가키트 영상 → VISEM 스타일 변환
   - CycleGAN / Pix2Pix 적용
   - 별도 학술 연구 프로젝트

B. 결과 영구 저장
   - SQLite/PostgreSQL 도입
   - 사용자 계정 + 분석 이력

C. 영상 처리 최적화
   - GPU 인코딩 (NVENC)
   - 배치 처리

D. 모바일 친화 UI
   - PWA 도입
   - 카메라 직접 촬영 지원
```

---

## 📞 개발자 문의

GitHub Issues: https://github.com/MoriochoRadio/sperm-ai/issues
