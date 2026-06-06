"""
annotator.py
추적 결과 + 운동성 등급을 원본 영상에 그려서 저장
"""
import os
import cv2
import subprocess
import numpy as np
from collections import defaultdict


# 등급별 색상 (BGR)
GRADE_COLORS = {
    'PR': (0, 200, 80),    # 초록 — 전진
    'NP': (200, 120, 0),   # 주황 — 비전진
    'IM': (80, 80, 80),    # 회색 — 비운동
    None: (180, 180, 180), # 미분류
}

GRADE_LABELS = {
    'PR': 'PR', 'NP': 'NP', 'IM': 'IM', None: '?'
}


def create_annotated_video(video_path: str,
                           output_path: str,
                           track_history: dict,
                           motility_grades: dict = None,
                           fps: float = 50.0,
                           trail_len: int = 20) -> bool:
    """
    원본 영상에 추적 궤적 + 운동성 등급을 그려서 저장

    Args:
        video_path:       원본 영상 경로
        output_path:      출력 영상 저장 경로
        track_history:    {tid: [(frame_idx, cx, cy), ...]}
        motility_grades:  grade_tracks() 반환값 (없으면 색상만)
        fps:              출력 fps
        trail_len:        궤적 표시 길이 (프레임 수)

    Returns:
        성공 여부
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 코덱 순서대로 시도
    for fourcc_str in ('H264', 'X264', 'DIVX', 'mp4v'):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        out = cv2.VideoWriter(output_path, fourcc, orig_fps, (W, H))
        if out.isOpened():
            print(f"[annotator] 코덱: {fourcc_str}")
            break
    else:
        cap.release()
        return False

    # tid → 등급 매핑 (motility_grades에서 역산)
    tid_grade = {}
    if motility_grades and 'tid_grades' in motility_grades:
        for tid, grade in motility_grades['tid_grades'].items():
            tid_grade[tid] = grade

    # frame_idx → [(tid, cx, cy)] 인덱스 구성
    frame_index = defaultdict(list)
    for tid, pts in track_history.items():
        for frame_idx, cx, cy in pts:
            frame_index[frame_idx].append((tid, cx, cy))

    # tid → 최근 포인트 버퍼
    # tid → 마지막으로 등장한 프레임 인덱스
    tid_last_frame = {}
    for tid, pts in track_history.items():
        if pts:
            tid_last_frame[tid] = max(f for f, _, _ in pts)

    # tid → 최근 포인트 버퍼
    trail_buffer = defaultdict(list)

    # 추적 종료 프레임 (track_history의 최대 frame_idx)
    max_track_frame = max(
        (f for pts in track_history.values() for f, _, _ in pts),
        default=0
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 이번 프레임에 등장하는 정자 처리
        for tid, cx, cy in frame_index.get(frame_idx, []):
            trail_buffer[tid].append((int(cx), int(cy)))
            if len(trail_buffer[tid]) > trail_len:
                trail_buffer[tid].pop(0)

        # 모든 활성 tid 처리 (사라진 정자도 trail은 유지)
        for tid, pts_list in trail_buffer.items():
            if not pts_list:
                continue

            grade = tid_grade.get(tid, None)
            color = GRADE_COLORS[grade]

            # 이동 경로 그리기
            for i in range(1, len(pts_list)):
                alpha = i / len(pts_list)
                c = tuple(int(v * alpha) for v in color)
                cv2.line(frame,
                         pts_list[i-1], pts_list[i],
                         c, 2, cv2.LINE_AA)

            # 마지막 위치에 바운딩 박스 (해당 프레임에 등장할 때만)
            if frame_idx in {f for f, _, _ in track_history.get(tid, [])}:
                cx, cy = pts_list[-1]
                box_half = 10
                x1, y1 = cx - box_half, cy - box_half
                x2, y2 = cx + box_half, cy + box_half
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              color, 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy),
                           2, color, -1, cv2.LINE_AA)
                label = f"{tid}:{GRADE_LABELS[grade]}"
                cv2.putText(frame, label,
                            (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, color, 1, cv2.LINE_AA)

        # 범례
        legend_items = [
            ('PR 전진', GRADE_COLORS['PR']),
            ('NP 비전진', GRADE_COLORS['NP']),
            ('IM 비운동', GRADE_COLORS['IM']),
        ]
        for i, (text, color) in enumerate(legend_items):
            y = 16 + i * 16
            cv2.circle(frame, (10, y), 4, color, -1)
            cv2.putText(frame, text, (18, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, color, 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    return True