"""
normalizer.py
입력 영상을 VISEM 표준(640×480, 50fps)으로 정규화
─────────────────────────────────────────────────
다양한 해상도/프레임률/색상의 입력을 받아
우리 시스템이 학습된 도메인에 가깝게 변환
"""

import os
import cv2
import numpy as np


# ── VISEM 표준 사양 ───────────────────────────────────────
TARGET_WIDTH    = 640
TARGET_HEIGHT   = 480
TARGET_FPS      = 50
MAX_DURATION    = 180   # 최대 3분 (초과 시 잘라냄)


class VideoNormalizer:
    """입력 영상을 VISEM 표준으로 변환"""

    # ── 정규화 필요성 판단 ────────────────────────────────
    @staticmethod
    def needs_normalization(metadata: dict,
                            tolerance: dict = None) -> tuple:
        """
        영상이 실제로 정규화가 필요한지 판단

        Returns:
            (needs: bool, reasons: list)
        """
        if tolerance is None:
            tolerance = {
                'width_diff':    50,    # 50px 이내면 OK
                'height_diff':   50,
                'fps_diff':      8,     # 8fps 이내면 OK
                'duration_max':  MAX_DURATION,
            }

        reasons = []

        # 해상도 차이
        if abs(metadata['width'] - TARGET_WIDTH) > tolerance['width_diff']:
            reasons.append(
                f"해상도 차이 ({metadata['width']}×{metadata['height']} → "
                f"{TARGET_WIDTH}×{TARGET_HEIGHT})")
        if abs(metadata['height'] - TARGET_HEIGHT) > tolerance['height_diff']:
            if not reasons:  # 위에서 안 잡혔다면
                reasons.append(
                    f"해상도 차이 ({metadata['width']}×{metadata['height']} → "
                    f"{TARGET_WIDTH}×{TARGET_HEIGHT})")

        # FPS 차이
        if abs(metadata['fps'] - TARGET_FPS) > tolerance['fps_diff']:
            reasons.append(
                f"프레임률 차이 ({metadata['fps']:.0f}fps → {TARGET_FPS}fps)")

        # 길이
        if metadata['duration'] > tolerance['duration_max']:
            reasons.append(
                f"영상 길이 ({metadata['duration']:.0f}초 → {tolerance['duration_max']}초로 자름)")

        needs = len(reasons) > 0
        return needs, reasons

    # ── 정규화 유형 판단 (스마트 분기) ────────────────────
    @staticmethod
    def get_normalization_type(metadata: dict) -> str:
        """
        어떤 종류의 정규화가 필요한지 판단

        Returns:
            'none'      - 정규화 불필요 (그대로 사용)
            'trim'      - 길이만 자름 (빠름, 1~5초)
            'full'      - 전체 변환 (느림, 30~120초)
        """
        # 해상도/FPS 차이 확인
        width_diff  = abs(metadata['width']  - TARGET_WIDTH)
        height_diff = abs(metadata['height'] - TARGET_HEIGHT)
        fps_diff    = abs(metadata['fps']    - TARGET_FPS)

        need_full = (
            width_diff  > 50 or
            height_diff > 50 or
            fps_diff    > 8
        )
        need_trim = metadata['duration'] > MAX_DURATION

        if need_full:
            return 'full'
        if need_trim:
            return 'trim'
        return 'none'

    # ── 빠른 컷팅 (CLAHE 포함, 재인코딩 1회) ─────────────
    def fast_trim(self, input_path: str,
                  output_path: str = None,
                  progress_callback=None) -> dict:
        """
        길이만 다른 영상의 빠른 변환
        - 처음 MAX_DURATION 초만 사용
        - 해상도/FPS 변환은 안 함
        - CLAHE 대비 향상은 적용 (정자 가시성 ↑)
        """
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_normalized.mp4"

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return self._error("영상을 열 수 없습니다.")

        orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps    = float(cap.get(cv2.CAP_PROP_FPS))
        orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_duration = orig_frames / orig_fps if orig_fps > 0 else 0

        original = {
            'width':    orig_width,
            'height':   orig_height,
            'fps':      round(orig_fps, 1),
            'n_frames': orig_frames,
            'duration': round(orig_duration, 1),
        }

        # 처음 MAX_DURATION 초만 사용
        max_frames = int(MAX_DURATION * orig_fps)
        max_frames = min(max_frames, orig_frames)

        # VideoWriter 준비
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path, fourcc, orig_fps,
            (orig_width, orig_height), isColor=True)

        if not out.isOpened():
            cap.release()
            return self._error("출력 영상 생성 실패")

        # CLAHE 한 번만 생성
        clahe = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8))

        last_pct = -1
        for i in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # CLAHE 대비 향상 (정자 가시성 ↑)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)
            frame_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            out.write(frame_out)

            if progress_callback:
                pct = int((i + 1) / max_frames * 100)
                if pct != last_pct and pct % 5 == 0:
                    progress_callback(i + 1, max_frames, pct)
                    last_pct = pct

        cap.release()
        out.release()

        if progress_callback:
            progress_callback(max_frames, max_frames, 100)

        normalized = {
            'width':    orig_width,
            'height':   orig_height,
            'fps':      round(orig_fps, 1),
            'n_frames': max_frames,
            'duration': round(max_frames / orig_fps, 1),
        }

        return {
            'success':     True,
            'output_path': output_path,
            'original':    original,
            'normalized':  normalized,
            'changes':     [
                f"길이 {orig_duration:.0f}초 → "
                f"{MAX_DURATION}초로 자름",
                "대비 향상 (CLAHE 적용)",
            ],
            'error':       None,
        }

    def normalize(self, input_path: str,
                  output_path: str = None,
                  progress_callback=None) -> dict:
        """
        영상 정규화 후 결과 반환

        Args:
            input_path: 입력 영상 경로
            output_path: 출력 경로 (None이면 자동 생성)
            progress_callback: 진행률 보고 콜백 함수
                callback(current_frame, total_frames, percent)
        """
        # 출력 경로 자동 생성
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_normalized.mp4"

        # 입력 영상 열기
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return self._error("영상을 열 수 없습니다.")

        # 원본 메타데이터
        orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps    = float(cap.get(cv2.CAP_PROP_FPS))
        orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_duration = orig_frames / orig_fps if orig_fps > 0 else 0

        original = {
            'width':    orig_width,
            'height':   orig_height,
            'fps':      round(orig_fps, 1),
            'n_frames': orig_frames,
            'duration': round(orig_duration, 1),
        }

        changes = []

        # ── 변환 결정 ────────────────────────────────────
        # 해상도 변경 필요 여부
        need_resize = (orig_width != TARGET_WIDTH or
                       orig_height != TARGET_HEIGHT)

        # FPS 변경 필요 여부 (5fps 이상 차이나면)
        need_fps_change = abs(orig_fps - TARGET_FPS) >= 5

        # 길이 자르기 필요 여부
        need_trim = orig_duration > MAX_DURATION

        # 그레이스케일 변환은 항상 수행 (색상 정규화)
        # → 위상차 현미경처럼 보이게

        if need_resize:
            changes.append(
                f"해상도 {orig_width}×{orig_height} "
                f"→ {TARGET_WIDTH}×{TARGET_HEIGHT}")
        if need_fps_change:
            changes.append(
                f"프레임률 {orig_fps:.0f}fps → {TARGET_FPS}fps")
        if need_trim:
            changes.append(
                f"길이 {orig_duration:.0f}초 → {MAX_DURATION}초로 자름")
        changes.append("그레이스케일 변환 (위상차 시뮬레이션)")

        # ── VideoWriter 준비 ─────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path, fourcc, TARGET_FPS,
            (TARGET_WIDTH, TARGET_HEIGHT),
            isColor=True)  # 그레이스케일이지만 3채널로 저장

        if not out.isOpened():
            cap.release()
            return self._error("출력 영상 생성 실패")

        # ── 프레임 처리 루프 (순차 읽기로 최적화) ───────
        # FPS 변환을 위한 비율
        fps_ratio = orig_fps / TARGET_FPS if orig_fps > 0 else 1.0

        max_frames_to_read = orig_frames
        if need_trim:
            max_frames_to_read = int(MAX_DURATION * orig_fps)

        # 어느 원본 프레임이 출력에 포함될지 미리 계산
        target_orig_indices = set()
        target_frame_idx = 0
        while True:
            orig_idx = int(target_frame_idx * fps_ratio)
            if orig_idx >= max_frames_to_read:
                break
            target_orig_indices.add(orig_idx)
            target_frame_idx += 1

        clahe = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8))

        frames_written = 0
        orig_frame_idx = 0
        last_reported_pct = -1

        # 순차 읽기 (점프 없이)
        while orig_frame_idx < max_frames_to_read:
            ret, frame = cap.read()
            if not ret:
                break

            # 이 프레임이 출력 대상인가?
            if orig_frame_idx in target_orig_indices:
                if need_resize:
                    frame = cv2.resize(
                        frame, (TARGET_WIDTH, TARGET_HEIGHT),
                        interpolation=cv2.INTER_AREA
                        if orig_width > TARGET_WIDTH
                        else cv2.INTER_LINEAR)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = clahe.apply(gray)

                frame_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                out.write(frame_out)
                frames_written += 1

            orig_frame_idx += 1

            # 진행률 콜백 (5%마다)
            if progress_callback and max_frames_to_read > 0:
                pct = int(orig_frame_idx / max_frames_to_read * 100)
                if pct != last_reported_pct and pct % 5 == 0:
                    progress_callback(
                        orig_frame_idx, max_frames_to_read, pct)
                    last_reported_pct = pct

        cap.release()
        out.release()

        # 최종 콜백
        if progress_callback:
            progress_callback(
                max_frames_to_read, max_frames_to_read, 100)

        # ── 결과 메타데이터 ─────────────────────────────
        normalized = {
            'width':    TARGET_WIDTH,
            'height':   TARGET_HEIGHT,
            'fps':      TARGET_FPS,
            'n_frames': frames_written,
            'duration': round(frames_written / TARGET_FPS, 1),
        }

        return {
            'success':     True,
            'output_path': output_path,
            'original':    original,
            'normalized':  normalized,
            'changes':     changes,
            'error':       None,
        }

    # ── 오류 결과 ─────────────────────────────────────────
    def _error(self, msg: str) -> dict:
        return {
            'success':     False,
            'output_path': None,
            'original':    None,
            'normalized':  None,
            'changes':     [],
            'error':       msg,
        }