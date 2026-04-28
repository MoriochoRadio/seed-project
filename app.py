"""
app.py — Flask 진입점
AI-CASA 웹 데모 서버
"""

from flask import Flask
from webapp.routes import bp


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