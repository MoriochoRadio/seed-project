"""
routes.py — Flask API 라우트
"""

import os
import time
from flask import (Blueprint, request, render_template,
                   jsonify, redirect, url_for)
from werkzeug.utils import secure_filename

from .tasks import create_job, get_job


# ── Blueprint ────────────────────────────────────────────
bp = Blueprint('main', __name__)


# ── 업로드 폴더 ──────────────────────────────────────────
UPLOAD_FOLDER = r'C:\Users\neo62\sperm-ai\tmp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXT = {'.mp4', '.avi', '.mov'}
MAX_SIZE_MB = 500


# ── 메인 페이지 ──────────────────────────────────────────
@bp.route('/')
def index():
    """랜딩 페이지"""
    return render_template('index.html')


# ── 영상 업로드 + 분석 시작 ──────────────────────────────
@bp.route('/api/analyze', methods=['POST'])
def api_analyze():
    """영상 업로드 후 백그라운드 분석 시작"""

    # 파일 확인
    if 'video' not in request.files:
        return jsonify({'error': '영상 파일이 없습니다'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '파일을 선택해주세요'}), 400

    # 확장자 검증
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({
            'error': 'MP4, AVI, MOV 형식만 지원됩니다'}), 400

    # 안전한 파일명 + 타임스탬프
    safe_name = secure_filename(file.filename)
    timestamp = int(time.time())
    save_name = f"{timestamp}_{safe_name}"
    save_path = os.path.join(UPLOAD_FOLDER, save_name)

    file.save(save_path)

    # 분석 작업 생성
    job_id = create_job(save_path, file.filename)

    return jsonify({
        'job_id':  job_id,
        'message': '분석을 시작합니다',
    })


# ── 진행 상태 조회 (폴링) ────────────────────────────────
@bp.route('/api/status/<job_id>')
def api_status(job_id: str):
    """작업 진행 상태 반환 (JS 폴링용)"""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': '작업을 찾을 수 없습니다'}), 404

    return jsonify({
        'job_id':    job['id'],
        'status':    job['status'],
        'progress':  job['progress'],
        'stage':     job['stage'],
        'stage_idx': job['stage_idx'],
        'error':     job['error'],
    })


# ── 분석 진행 페이지 ─────────────────────────────────────
@bp.route('/analyzing/<job_id>')
def analyzing(job_id: str):
    """분석 진행 화면 (Step 5에서 만들 예정)"""
    job = get_job(job_id)
    if not job:
        return redirect(url_for('main.index'))

    # 이미 완료됐으면 결과 페이지로
    if job['status'] == 'done':
        return redirect(url_for('main.result', job_id=job_id))

    return render_template('analyzing.html',
                           job_id=job_id,
                           video_name=job['video_name'])


# ── 결과 페이지 ──────────────────────────────────────────
@bp.route('/result/<job_id>')
def result(job_id: str):
    """결과 보고서 페이지 (Step 6에서 만들 예정)"""
    job = get_job(job_id)
    if not job or job['status'] != 'done':
        return redirect(url_for('main.index'))

    return render_template('result.html',
                           job_id=job_id,
                           result=job['result'],
                           video_name=job['video_name'])