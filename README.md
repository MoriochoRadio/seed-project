# Trajectory-aware Sperm Motility Analysis (TA-SMA)

연구 주제 *"Trajectory-aware Deep Learning System for Robust Sperm Motility Analysis under High-Density Microscopy Conditions"*의 실험 코드 저장소입니다.

모체 논문 Shahali et al. (Human Reproduction, 2026) 대비 **차별점 3가지** —
1) tracking-aware trajectory classification, 2) clinical domain robustness, 3) system-level validity — 를 검증하는 코드를 포함합니다.

데이터셋은 [VISEM-Tracking](https://www.kaggle.com/datasets/vlbthambawita/visemtracking) 을 사용합니다.

## 🚀 빠른 시작
```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 데이터 다운로드 (Kaggle 인증 필요)
python scripts/download_visem.py --target ./data

# 3) 동작 확인 (dummy data)
python tests/test_pipeline.py

# 4) 실험 실행
python experiments/exp1_crowded_robustness.py --data_root ./data/visem_tracking --epochs 30
python experiments/exp3_fps_length.py        --ckpt ./runs/exp1/proposed.pt
python experiments/exp4_ablation.py          --data_root ./data/visem_tracking
```

## 📂 디렉토리
- `src/` — 모듈 코드(데이터, 검출, 추적, 피처, 모델, 학습, 평가)
- `experiments/` — 논문 실험과 1:1 매핑된 스크립트
- `scripts/` — 데이터 다운로드 등 유틸
- `tests/` — dummy 데이터로 코드가 살아있는지 검증
- `memory.md` — 변경/결정 기록 (필수 참고)

## 🧪 실험 매핑
| PDF 실험 | 스크립트 |
|---|---|
| Exp1 Crowded Robustness | `experiments/exp1_crowded_robustness.py` |
| Exp3 FPS / Length Sensitivity | `experiments/exp3_fps_length.py` |
| Exp4 Tracking-aware Ablation | `experiments/exp4_ablation.py` |

> Exp2 (viscosity) 는 VISEM-Tracking에 점도 라벨이 없어 보류 (memory.md 참고).
