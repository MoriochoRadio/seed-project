"""
setup_models.py — Render.com 배포 시 모델 파일 자동 다운로드
빌드 커맨드: pip install -r requirements.txt && python setup_models.py
"""

import os
import gdown

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'models')
YOLO_DIR  = os.path.join(MODEL_DIR, 'yolo11_sperm_v2', 'weights')

os.makedirs(YOLO_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 다운로드 대상 (경로: Google Drive file ID) ──────────────
MODELS = {
    os.path.join(YOLO_DIR,  'best.pt'):                          os.environ.get('GDRIVE_YOLO_ID'),
    os.path.join(MODEL_DIR, 'motility_ensemble.pkl'):            os.environ.get('GDRIVE_MOTILITY_ID'),
    os.path.join(MODEL_DIR, 'motility_ensemble_v2.pkl'):         os.environ.get('GDRIVE_MOTILITY_V2_ID'),
    os.path.join(MODEL_DIR, 'morphology_efficientnet_b3_v3.pt'): os.environ.get('GDRIVE_MORPHOLOGY_ID'),
}

# ── 다운로드 실행 ────────────────────────────────────────────
for path, file_id in MODELS.items():
    fname = os.path.basename(path)
    if os.path.exists(path):
        print(f'[skip] {fname} 이미 존재함')
        continue
    print(f'[download] {fname} ...')
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, path, quiet=False)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f'[ok] {fname} ({size_mb:.1f} MB)')
    else:
        print(f'[error] {fname} 다운로드 실패 — file ID 또는 공유 설정 확인 필요')
