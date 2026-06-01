"""
app.py — Flask 진입점
AI-CASA 웹 데모 서버
"""

import os
import gdown
from flask import Flask
from webapp.routes import bp


# ── Render.com 배포 시 모델 자동 다운로드 ───────────────────
def _download_models():
    BASE      = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE, 'models')
    YOLO_DIR  = os.path.join(MODEL_DIR, 'yolo11_sperm_v2', 'weights')

    os.makedirs(YOLO_DIR,  exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    MODELS = {
        os.path.join(YOLO_DIR,  'best.pt'):                          '1qwFllSftrTtYTEJRpbgXFpZn2Swfbkj4',
        os.path.join(MODEL_DIR, 'motility_ensemble.pkl'):            '1bvRXqMtt6JLG1uNiF-ewmxKOoFIqVzZq',
        os.path.join(MODEL_DIR, 'motility_ensemble_v2.pkl'):         '1Zc8X9ItiADYp6zo3g142yhX9zH8PbC6D',
        os.path.join(MODEL_DIR, 'morphology_efficientnet_b3_v3.pt'): '11xgc2WJ31MKxMA_2z90fzhjgPTxKKgew',
    }

    for path, file_id in MODELS.items():
        fname = os.path.basename(path)
        if os.path.exists(path):
            print(f'[model] {fname} 이미 존재함 — skip')
            continue
        print(f'[model] {fname} 다운로드 중...')
        gdown.download(f'https://drive.google.com/uc?id={file_id}', path, quiet=False)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f'[model] {fname} 완료 ({size_mb:.1f} MB)')
        else:
            print(f'[model] {fname} 다운로드 실패 — Drive 공유 설정 확인 필요')

_download_models()


# ── Flask 앱 생성 ────────────────────────────────────────
app = Flask(__name__,
            template_folder='webapp/templates',
            static_folder='webapp/static')

# 업로드 크기 제한 (500MB)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Blueprint 등록
app.register_blueprint(bp)


if __name__ == '__main__':
    print("=" * 60)
    print("  🔬 AI-CASA 웹 서버 시작")
    print("=" * 60)
    print("  접속 주소: http://localhost:5000")
    print("  종료: Ctrl+C")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)