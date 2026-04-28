"""
tasks.py — 백그라운드 분석 작업 관리
영상 업로드 후 별도 스레드에서 분석을 실행하고
진행 상태를 메모리에 저장
"""

import os
import sys
import uuid
import time
import threading
import traceback


# ── 작업 큐 (메모리 기반) ─────────────────────────────────
# 실제 운영에선 Redis 등을 쓰지만 데모는 메모리로 충분
JOBS = {}


# ── 분석 파이프라인 (지연 로드) ───────────────────────────
_pipeline = None

def get_pipeline():
    """파이프라인 싱글톤 (첫 요청 시 로드)"""
    global _pipeline
    if _pipeline is None:
        sys.path.insert(0, r'C:\Users\neo62\sperm-ai')
        from src.pipeline import SpermAnalysisPipeline
        _pipeline = SpermAnalysisPipeline()
    return _pipeline


# ── 작업 생성 ─────────────────────────────────────────────
def create_job(video_path: str, video_name: str) -> str:
    """새 분석 작업 생성"""
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        'id':         job_id,
        'video_path': video_path,
        'video_name': video_name,
        'status':     'pending',     # pending / running / done / error
        'progress':   0,             # 0 ~ 100
        'stage':      '대기 중',
        'stage_idx':  0,             # 0 ~ 5
        'result':     None,
        'error':      None,
        'created_at': time.time(),
    }

    # 백그라운드 스레드로 분석 시작
    thread = threading.Thread(
        target=run_analysis,
        args=(job_id,),
        daemon=True)
    thread.start()

    return job_id


# ── 작업 상태 조회 ────────────────────────────────────────
def get_job(job_id: str) -> dict:
    """작업 상태 반환"""
    return JOBS.get(job_id)


# ── 분석 실행 (백그라운드) ────────────────────────────────
def run_analysis(job_id: str) -> None:
    """실제 분석 수행 (별도 스레드에서 실행)"""
    job = JOBS.get(job_id)
    if not job:
        return

    try:
        job['status']    = 'running'
        job['stage']     = '파이프라인 초기화 중'
        job['stage_idx'] = 0
        job['progress']  = 5

        # 파이프라인 로드 (최초 1회)
        pipeline = get_pipeline()

        # 단계별 진행 상태 업데이트
        # (실제 분석 함수가 verbose=True로 print하지만
        #  진행률은 추정값으로 표시)
        stages = [
            (10, '정자 탐지 중',    'YOLO11'),
            (30, '정자 추적 중',    'ByteTrack'),
            (50, '운동성 분석 중',  'Ridge + RF'),
            (75, '형태 분석 중',    'EfficientNet-B3'),
            (90, '보고서 생성 중',  'WHO 기준 해석'),
        ]

        # 진행률 업데이트 스레드 (단순 시간 기반)
        def update_progress():
            for prog, stage_name, _ in stages:
                if job['status'] != 'running':
                    return
                job['progress']  = prog
                job['stage']     = stage_name
                job['stage_idx'] = stages.index((prog, stage_name, _)) + 1
                time.sleep(2)

        progress_thread = threading.Thread(
            target=update_progress, daemon=True)
        progress_thread.start()

        # 실제 분석 실행
        result = pipeline.analyze(job['video_path'], verbose=True)

        if result is None:
            raise RuntimeError(
                "분석 실패: 파이프라인이 결과를 반환하지 못했습니다.")

        # 완료
        job['status']    = 'done'
        job['progress']  = 100
        job['stage']     = '분석 완료'
        job['stage_idx'] = 5
        job['result']    = result

    except Exception as e:
        job['status'] = 'error'
        job['error']  = str(e)
        traceback.print_exc()

    finally:
        # 업로드 영상 정리 (옵션)
        # try:
        #     os.remove(job['video_path'])
        # except Exception:
        #     pass
        pass