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
import subprocess
import shutil
from src.annotator import create_annotated_video

from src.pipeline import SpermAnalysisPipeline

def _find_ffmpeg() -> str:
    """ffmpeg 실행파일 경로 탐색 (PATH → conda Library\bin → conda Scripts 순)."""
    f = shutil.which('ffmpeg')
    if f:
        return f
    # conda 환경: CONDA_PREFIX 또는 현재 인터프리터 위치로 추정
    conda_prefix = os.environ.get('CONDA_PREFIX') or os.path.dirname(sys.executable)
    for subdir in (os.path.join('Library', 'bin'), 'bin', 'Scripts'):
        candidate = os.path.join(conda_prefix, subdir, 'ffmpeg.exe')
        if os.path.isfile(candidate):
            print(f"[ffmpeg] 발견: {candidate}")
            return candidate
    return 'ffmpeg'   # 마지막 fallback — 없으면 FileNotFoundError 발생

# ── 작업 큐 (메모리 기반) ─────────────────────────────────
# 실제 운영에선 Redis 등을 쓰지만 데모는 메모리로 충분
JOBS = {}


# ── 분석 파이프라인 (지연 로드) ───────────────────────────
_pipeline = None

def get_pipeline():
    """파이프라인 싱글톤 (첫 요청 시 로드)"""
    global _pipeline
    if _pipeline is None:
        #sys.path.insert(0, r'C:\Users\neo62\sperm-ai')
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
        'status':     'pending',
        'progress':   0,
        'stage':      '대기 중',
        'stage_idx':  0,
        'result':     None,
        'quality':        None,    # 추가
        'normalization':  None,    # 추가
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
        job['stage']     = '영상 품질 평가 중'
        job['stage_idx'] = 0
        job['progress']  = 5

        # ── Step 0: 영상 품질 평가 ───────────────────────
        from src.quality import VideoQualityAnalyzer
        quality_analyzer = VideoQualityAnalyzer()
        quality = quality_analyzer.analyze(job['video_path'])

        job['quality'] = quality

        # F등급은 분석 거부
        if not quality['can_analyze']:
            job['status']  = 'rejected'
            job['stage']   = '품질 검증 실패'
            job['error']   = '영상 품질이 분석 기준에 미달합니다.'
            return

       # ── Step 1: 영상 정규화 (스마트 분기) ──────────
        analysis_video_path = job['video_path']

        from src.normalizer import VideoNormalizer

        # 어떤 정규화가 필요한지 판단
        norm_type = VideoNormalizer.get_normalization_type(
            quality['metadata'])

        if norm_type != 'none':
            # 정규화 진행
            normalizer = VideoNormalizer()

            base, _ = os.path.splitext(job['video_path'])
            normalized_path = f"{base}_normalized.mp4"

            # 진행률 콜백 정의
            def on_progress(current, total, pct):
                job['progress'] = 8 + int(pct * 0.04)
                job['stage'] = (
                    f'영상 정규화 중 ({pct}% — '
                    f'{current}/{total} 프레임)')

            # 분기: 빠른 컷팅 vs 풀 정규화
            if norm_type == 'trim':
                # 길이만 다름 → 빠른 컷팅 (1~5초)
                job['stage'] = '영상 길이 조정 중 (빠른 변환)'
                job['progress'] = 8
                norm_result = normalizer.fast_trim(
                    job['video_path'],
                    normalized_path,
                    progress_callback=on_progress)
            else:
                # full: 해상도/FPS 변환 → 전체 정규화
                job['stage'] = '영상 정규화 중'
                job['progress'] = 8
                norm_result = normalizer.normalize(
                    job['video_path'],
                    normalized_path,
                    progress_callback=on_progress)

            if norm_result['success']:
                analysis_video_path = normalized_path
                job['normalization'] = norm_result
                job['progress'] = 12
            else:
                job['status'] = 'error'
                job['error'] = (
                    f"정규화 실패: {norm_result['error']}")
                return
        else:
            # 정규화 불필요 → 즉시 진행
            job['normalization'] = None
            job['progress'] = 12

        # ── Step 2~6: 기존 분석 파이프라인 ──────────────
        pipeline = get_pipeline()

        stages = [
            (15, '정자 탐지 중',    'YOLO11'),
            (35, '정자 추적 중',    'ByteTrack'),
            (55, '운동성 분석 중',  'Ridge + RF'),
            (75, '형태 분석 중',    'EfficientNet-B3'),
            (90, '보고서 생성 중',  'WHO 기준 해석'),
        ]

        # 진행률 업데이트 스레드
        def update_progress():
            for prog, stage_name, _ in stages:
                if job['status'] != 'running':
                    return
                job['progress']  = prog
                job['stage']     = stage_name
                job['stage_idx'] = stages.index(
                    (prog, stage_name, _)) + 1
                time.sleep(2)

        progress_thread = threading.Thread(
            target=update_progress, daemon=True)
        progress_thread.start()

        # 실제 분석 실행 (정규화된 영상으로!)
        result = pipeline.analyze(
            analysis_video_path, verbose=True)

        if result is None:
            raise RuntimeError(
                "분석 실패: 파이프라인이 결과를 반환하지 못했습니다.")

        # annotated 영상 생성 (track_history 재활용)
        try:
            base, _ = os.path.splitext(analysis_video_path)
            ann_path = f"{base}_annotated.mp4"
            track_history = result.get('track_history', {})
            if track_history:
                pipeline.render_annotated_video(
                    analysis_video_path,
                    track_history,
                    ann_path)
                result['annotated_video'] = ann_path
        except Exception as ann_err:
            # annotated 영상 생성 실패해도 분석 결과는 유지
            print(f"[annotated] 영상 생성 실패 (무시): {ann_err}")
            result['annotated_video'] = None

        # 품질 정보를 결과에 추가
        result['quality']       = quality
        result['normalization'] = job.get('normalization')

        # ── 품질 박스 표시 여부 판단 ──────────────────────
        # 다음 경우에만 사용자에게 품질 박스를 표시:
        # 1. 등급이 B 이하 (실질적으로 분석에 영향)
        # 2. 풀 정규화가 적용됨 (해상도/FPS 변환 = 큰 변환)
        # 3. 분석에 영향 줄 수 있는 심각한 이슈가 있음
        norm = job.get('normalization')

        # 풀 정규화 = 해상도 또는 FPS 변환이 있었던 경우
        is_full_norm = False
        if norm and norm.get('changes'):
            for change in norm['changes']:
                if '해상도' in change or '프레임률' in change:
                    is_full_norm = True
                    break

        # 심각한 이슈만 카운트 (선명도, 흔들림 등은 본질 문제 아니면 제외)
        critical_issues = [
            issue for issue in quality.get('issues', [])
            if any(keyword in issue for keyword in [
                '해상도가 낮음',
                '프레임률이 낮음',
                '너무 짧음',
                '영상이 불안정',
                '조명/밝기',
            ])
        ]

        result['show_quality_card'] = (
            quality['grade'] in ('B', 'C') or
            is_full_norm or
            len(critical_issues) > 0
        )
        # ── annotated + 원본(분석 구간) 영상 생성 ───────────────
        # 원본 패널에는 분석 구간(≈5초)과 길이/프레임이 일치하는 clean_path
        # (_view.mp4)를 쓴다. annotator가 이를 H.264+faststart로 변환하며,
        # 변환 실패 시 clean_path를 제거하므로 아래 fallback이 동작한다.
        try:
            base, _ = os.path.splitext(job['video_path'])
            annotated_path = f"{base}_annotated.mp4"
            clean_path     = f"{base}_view.mp4"   # 브라우저 재생용 '원본' 복사본
            track_history  = result.get('_track_history', {})

            # HUD용 종합 수치 (정자 단위가 아닌 샘플 전체 값)
            hud_stats = {
                'N':               result.get('N'),
                'progressive':     result.get('progressive'),
                'non_progressive': result.get('non_progressive'),
                'immotile':        result.get('immotile'),
                'normal_rate':     (result.get('morphology')
                                    or {}).get('normal_rate'),
                'confidence_score': result.get('confidence_score'),
            }

            success = create_annotated_video(
                video_path        = analysis_video_path,
                output_path       = annotated_path,
                clean_output_path = clean_path,
                track_history     = track_history,
                motility_grades   = result.get('motility_grades'),
                stats             = hud_stats,
            )
            if success:
                result['annotated_video_path'] = annotated_path
        except Exception as e:
            print(f"⚠️  annotated video 생성 실패: {e}")
            traceback.print_exc()
            clean_path = None

        # ── 원본 패널 영상 결정 (브라우저 호환 H.264 보장) ───────
        # 1순위: clean_path(_view.mp4) — 분석 구간 5초, 이미 H.264 변환됨
        if clean_path and os.path.exists(clean_path):
            result['original_video_path'] = clean_path
        else:
            # 2순위: 분석 영상 전체를 H.264+faststart로 직접 변환
            original_h264 = (os.path.splitext(job['video_path'])[0]
                             + '_original_h264.mp4')
            ffmpeg_bin = _find_ffmpeg()
            converted = False
            if ffmpeg_bin:
                try:
                    ret = subprocess.run([
                        ffmpeg_bin, '-y', '-i', analysis_video_path,
                        '-vcodec', 'libx264', '-crf', '23',
                        '-preset', 'fast', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart',
                        original_h264
                    ], capture_output=True)
                    if ret.returncode == 0 and os.path.exists(original_h264):
                        converted = True
                    else:
                        print(f"[경고] 원본 H.264 변환 실패:\n"
                              f"{ret.stderr.decode(errors='ignore')[-500:]}")
                except Exception as e:
                    print(f"[경고] ffmpeg 실행 오류: {e}")
            else:
                print("[경고] ffmpeg를 찾을 수 없습니다. 원본을 그대로 사용합니다.")
            # 3순위: 변환 실패 시 분석 영상 원본 그대로 (재생 안 될 수 있음)
            result['original_video_path'] = (
                original_h264 if converted else analysis_video_path)

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
        pass
