"""
app.py — Flask 진입점
AI-CASA 웹 데모 서버
"""

import os
import gdown
from flask import Flask
from webapp.routes import bp
from dotenv import load_dotenv
load_dotenv()  # .env 파일 자동 로드


# ── Render.com 배포 시 모델 자동 다운로드 ───────────────────
def _download_models():
    BASE      = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE, 'models')
    YOLO_DIR  = os.path.join(MODEL_DIR, 'yolo11_sperm_v2', 'weights')

    os.makedirs(YOLO_DIR,  exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    REQUIRED_ENVS = [
        'GDRIVE_YOLO_ID',
        'GDRIVE_MOTILITY_ID',
        'GDRIVE_MOTILITY_V2_ID',
        'GDRIVE_MORPHOLOGY_ID',
    ]

    missing = [env for env in REQUIRED_ENVS if not os.environ.get(env)]

    if missing:
        print(f'[error] 누락된 환경변수: {missing}')
        print('[error] Render Dashboard → Environment에서 추가하세요.')
        exit(1)

    MODELS = {
        os.path.join(YOLO_DIR, 'best.pt'):
            os.environ['GDRIVE_YOLO_ID'],

        os.path.join(MODEL_DIR, 'motility_ensemble.pkl'):
            os.environ['GDRIVE_MOTILITY_ID'],

        os.path.join(MODEL_DIR, 'motility_ensemble_v2.pkl'):
            os.environ['GDRIVE_MOTILITY_V2_ID'],

        os.path.join(MODEL_DIR, 'morphology_efficientnet_b3_v3.pt'):
            os.environ['GDRIVE_MORPHOLOGY_ID'],
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


# app.py 마지막 블록 수정
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # 환경변수 PORT 없으면 8080
    print("=" * 60)
    print("  🔬 AI-CASA 웹 서버 시작")
    print("=" * 60)
    print(f"  접속 주소: http://localhost:{port}") 
    print("  종료: Ctrl+C")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=True)
