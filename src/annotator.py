"""
annotator.py
추적 결과 + 운동성 등급을 분석 영상에 그려서 저장한다.

설계 원칙 (5초 분석 윈도우 시각화):
  - tracker가 분석한 구간(최대 250프레임 ≈ 5초)까지만 렌더한다.
    분석되지 않은 뒷부분을 그리면 주석 없는 원본만 흘러가므로 잘라낸다.
  - 정자 단위로 그릴 수 있는 것만 영상 위에 직접 그린다:
      박스 + 추적 ID + 궤적 꼬리 + 운동성 등급 색(PR/NP/IM).
  - 정자 단위 매핑이 없는 값(전체 운동성 %, 정상형태율, 정자 수, 신뢰도)은
    화면 모서리 종합 수치(HUD)로 표시한다.
  - OpenCV putText는 한글을 렌더하지 못하므로 오버레이 텍스트는 ASCII만 사용한다.
"""
import os
import cv2
import numpy as np
from collections import defaultdict


# 등급별 색상 (BGR) — PR 초록 / NP 주황 / IM 회색
GRADE_COLORS = {
    'PR':  (0, 200, 80),     # 초록 — 전진
    'NP':  (0, 140, 255),    # 주황 — 비전진 (BGR: R=255,G=140,B=0)
    'IM':  (110, 110, 110),  # 회색 — 비운동
    None:  (180, 180, 180),  # 미분류
}

GRADE_LABELS = {'PR': 'PR', 'NP': 'NP', 'IM': 'IM', None: '?'}


def _fmt(v, suffix=''):
    """None 안전 포맷."""
    if v is None:
        return '--'
    if isinstance(v, float):
        return f'{v:.1f}{suffix}'
    return f'{v}{suffix}'


def _draw_panel(frame, x, y, w, h, alpha=0.45):
    """반투명 어두운 패널 — 텍스트 가독성용."""
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    x, y = max(0, x), max(0, y)
    roi = frame[y:y2, x:x2]
    if roi.size == 0:
        return
    dark = np.zeros_like(roi)
    cv2.addWeighted(dark, alpha, roi, 1 - alpha, 0, roi)


def _draw_legend(frame):
    """좌상단 등급 범례 (ASCII)."""
    items = [('PR (progressive)', GRADE_COLORS['PR']),
             ('NP (non-prog.)',   GRADE_COLORS['NP']),
             ('IM (immotile)',    GRADE_COLORS['IM'])]
    _draw_panel(frame, 6, 6, 168, 16 * len(items) + 8)
    for i, (text, color) in enumerate(items):
        y = 20 + i * 16
        cv2.circle(frame, (16, y - 4), 4, color, -1, cv2.LINE_AA)
        cv2.putText(frame, text, (26, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (240, 240, 240), 1, cv2.LINE_AA)


def _draw_hud(frame, stats):
    """우상단 종합 수치 HUD — 정자 단위가 아닌 샘플 전체 값."""
    if not stats:
        return
    W = frame.shape[1]
    lines = [
        f"Count        : {_fmt(stats.get('N'))}",
        f"Motility(reg): PR {_fmt(stats.get('progressive'), '%')}"
        f"  NP {_fmt(stats.get('non_progressive'), '%')}"
        f"  IM {_fmt(stats.get('immotile'), '%')}",
        f"Normal morph : {_fmt(stats.get('normal_rate'), '%')}",
        f"Confidence   : {_fmt(stats.get('confidence_score'))}",
    ]
    panel_w = 296
    x = W - panel_w - 6
    _draw_panel(frame, x, 6, panel_w, 16 * len(lines) + 8)
    for i, text in enumerate(lines):
        y = 20 + i * 16
        cv2.putText(frame, text, (x + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (240, 240, 240), 1, cv2.LINE_AA)


def _open_writer(path, fps, size, isColor=True):
    """브라우저 호환 코덱(H.264 계열) 우선으로 VideoWriter를 연다.
    mp4v는 마지막 fallback. (normalizer._open_browser_writer와 동일 전략)"""
    for fourcc_str in ('mp4v', 'H264', 'X264', 'avc1'):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_str),
                            fps, size, isColor)
        if w.isOpened():
            print(f"[annotator] 코덱: {fourcc_str} -> {path}")
            return w
    return None


def create_annotated_video(video_path: str,
                           output_path: str,
                           track_history: dict,
                           motility_grades: dict = None,
                           stats: dict = None,
                           clean_output_path: str = None,
                           fps: float = 50.0,
                           trail_len: int = 20,
                           max_frames: int = 250) -> bool:
    """
    분석 영상에 추적 궤적 + 운동성 등급 + 종합 수치를 그려 저장한다.

    Args:
        video_path:      분석에 사용된 영상 경로(정규화본)
        output_path:     출력 영상 저장 경로
        track_history:   {tid: [(frame_idx, cx, cy), ...]}
        motility_grades: grade_tracks() 반환값 (tid_grades 포함). 없으면 색만.
        stats:           HUD용 종합 수치 dict
                         (N, progressive, non_progressive, immotile,
                          normal_rate, confidence_score)
        fps:             출력 fps fallback
        trail_len:       궤적 표시 길이(프레임 수)
        max_frames:      렌더 상한(분석 윈도우와 일치, 기본 250 ≈ 5초)

    Returns:
        성공 여부
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames

    # ── 렌더 종료 프레임: 분석된 구간까지만 ─────────────────
    # tracker가 첫 max_frames(≈250)만 추적하므로 그 이후는 그릴 게 없다.
    if track_history:
        max_track_frame = max(
            (f for pts in track_history.values() for f, *_ in pts),
            default=0)
        last_frame = min(max_track_frame, max_frames - 1)
    else:
        last_frame = min(total - 1, max_frames - 1)

    # ── 출력 라이터 (브라우저 호환 코덱 우선) ───────────────
    out = _open_writer(output_path, orig_fps, (W, H))
    if out is None:
        cap.release()
        return False
    print(f"[annotator] 렌더 프레임: 0~{last_frame} ({last_frame + 1}장)")

    # 오버레이 없는 '원본' 표시용 복사본 (같은 디코드 루프에서 동시 생성).
    # 입력이 .avi/mjpeg/mp4v 등 브라우저 비호환이어도 항상 재생 가능한
    # h264 mp4를 '원본' 패널에 제공하기 위함.
    clean_out = None
    if clean_output_path:
        clean_out = _open_writer(clean_output_path, orig_fps, (W, H))

    # ── 사전 인덱스 구성 (매 프레임 재계산 방지) ────────────
    tid_grade = {}
    if motility_grades and 'tid_grades' in motility_grades:
        tid_grade = dict(motility_grades['tid_grades'])

    frame_index = defaultdict(list)        # frame_idx -> [(tid, cx, cy)]
    tid_frames  = defaultdict(set)         # tid -> {frame_idx} (박스 표시 판정용)
    for tid, pts in track_history.items():
        for frame_idx, cx, cy, *_ in pts:
            frame_index[frame_idx].append((tid, cx, cy))
            tid_frames[tid].add(frame_idx)

    trail_buffer = defaultdict(list)       # tid -> 최근 좌표 버퍼

    # ── 프레임 루프 ─────────────────────────────────────────
    frame_idx = 0
    while frame_idx <= last_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # 오버레이 그리기 전에 '원본' 복사본 저장 (frame은 이후 in-place로 그려짐)
        if clean_out is not None:
            clean_out.write(frame)

        # 이번 프레임에 등장한 정자 → 궤적 버퍼 갱신
        for tid, cx, cy in frame_index.get(frame_idx, []):
            buf = trail_buffer[tid]
            buf.append((int(cx), int(cy)))
            if len(buf) > trail_len:
                buf.pop(0)

        # 활성 정자 그리기 (사라진 정자도 꼬리는 잠시 유지)
        for tid, pts_list in trail_buffer.items():
            if not pts_list:
                continue
            grade = tid_grade.get(tid)
            color = GRADE_COLORS[grade]

            # 궤적 꼬리 (오래된 점일수록 어둡게 페이드)
            for i in range(1, len(pts_list)):
                alpha = i / len(pts_list)
                c = tuple(int(v * alpha) for v in color)
                cv2.line(frame, pts_list[i - 1], pts_list[i],
                         c, 2, cv2.LINE_AA)

            # 박스 + ID + 등급 라벨 (이 프레임에 실제 등장할 때만)
            if frame_idx in tid_frames.get(tid, ()):
                cx, cy = pts_list[-1]
                bh = 10
                cv2.rectangle(frame, (cx - bh, cy - bh),
                              (cx + bh, cy + bh), color, 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)
                cv2.putText(frame, f"{tid}:{GRADE_LABELS[grade]}",
                            (cx - bh, cy - bh - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            color, 1, cv2.LINE_AA)

        # 오버레이: 범례(좌상단) + 종합 수치(우상단)
        _draw_legend(frame)
        _draw_hud(frame, stats)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    if clean_out is not None:
        clean_out.release()

    # mp4v → H.264 변환 (브라우저 호환)
    import subprocess
    h264_path = output_path.replace('.mp4', '_h264.mp4')
    ret = subprocess.run([
        'ffmpeg', '-y', '-i', output_path,
        '-vcodec', 'libx264',
        '-crf', '23',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        h264_path
    ], capture_output=True)

    if ret.returncode == 0:
        os.remove(output_path)
        os.rename(h264_path, output_path)

    return True
