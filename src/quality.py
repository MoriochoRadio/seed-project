"""
quality.py
입력 영상의 품질을 자동 평가하는 모듈
─────────────────────────────────────
VISEM 데이터셋(640×480, 50fps, 위상차 현미경)을 기준으로
입력 영상이 얼마나 우리 시스템에 적합한지 평가
"""

import cv2
import numpy as np


# ── VISEM 표준 사양 ───────────────────────────────────────
VISEM_STANDARD = {
    'width':       640,
    'height':      480,
    'fps':         50,
    'duration_min': 5,    # 최소 5초
    'duration_max': 240,  # 최대 4분
}


# ── 평가 등급 ─────────────────────────────────────────────
GRADE_THRESHOLDS = {
    'S': 90,   # Standard - VISEM 표준 영상
    'A': 75,   # Acceptable - 약간의 변환 후 분석 가능
    'B': 60,   # Borderline - 변환 후 분석은 가능하나 신뢰도 낮음
    'C': 45,   # Critical - 분석 시도는 가능하나 결과 보장 못함
    'F': 0,    # Fail - 분석 거부
}


class VideoQualityAnalyzer:
    """영상 품질을 자동 평가하는 클래스"""

    def __init__(self):
        self.standard = VISEM_STANDARD

    def analyze(self, video_path: str) -> dict:
        """
        영상 분석 후 평가 결과 반환

        Returns:
            dict with keys:
              - metadata:        원본 영상 메타데이터
              - scores:          세부 점수
              - total_score:     종합 점수 (0~100)
              - grade:           등급 (S/A/B/C/F)
              - issues:          발견된 문제점 목록
              - recommendations: 사용자 안내사항
              - can_analyze:     분석 가능 여부 (bool)
        """
        # 1. 영상 메타데이터 추출
        metadata = self._extract_metadata(video_path)
        if metadata is None:
            return self._make_failure_result(
                "영상 파일을 열 수 없습니다.")

        # 1-1. 본질적 한계 검사 (하드 룰) ─────────────────
        hard_fail_reason = self._check_hard_limits(metadata)
        if hard_fail_reason:
            return self._make_failure_result_with_meta(
                hard_fail_reason, metadata)

        # 2. 세부 평가 항목별 점수 계산
        scores = {
            'resolution':  self._score_resolution(metadata),
            'fps':         self._score_fps(metadata),
            'duration':    self._score_duration(metadata),
            'stability':   self._score_stability(video_path),
            'brightness':  self._score_brightness(video_path),
            'sharpness':   self._score_sharpness(video_path),
        }

        # 3. 종합 점수 (가중 평균)
        weights = {
            'resolution': 0.20,
            'fps':        0.15,
            'duration':   0.05,   # 길이 영향 축소 (정규화로 해결)
            'stability':  0.20,
            'brightness': 0.15,
            'sharpness':  0.25,   # 실질 품질 비중 강화
        }
        total = sum(scores[k] * weights[k] for k in scores)
        total = round(total)

        # 4. 등급 결정
        grade = self._determine_grade(total)

        # 5. 이슈 + 권고사항 생성
        issues, recommendations = self._diagnose(scores, metadata)

        return {
            'metadata':        metadata,
            'scores':          scores,
            'total_score':     total,
            'grade':           grade,
            'issues':          issues,
            'recommendations': recommendations,
            'can_analyze':     grade != 'F',
        }

    # ── 메타데이터 추출 ───────────────────────────────────
    def _extract_metadata(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps      = float(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = n_frames / fps if fps > 0 else 0
        cap.release()

        return {
            'width':    width,
            'height':   height,
            'fps':      round(fps, 1),
            'n_frames': n_frames,
            'duration': round(duration, 1),
        }

    # ══════════════════════════════════════════════════════
    #   세부 평가 함수들
    # ══════════════════════════════════════════════════════

    def _score_resolution(self, meta: dict) -> int:
        """해상도 점수 (640×480 기준)"""
        w, h = meta['width'], meta['height']
        target_w, target_h = 640, 480

        # 너무 작으면 거의 0점
        if w < 320 or h < 240:
            return 20

        # 표준 정확히 일치
        if w == target_w and h == target_h:
            return 100

        # 너무 큰 4K급 (다운샘플링 필요)
        if w >= 1920 or h >= 1080:
            return 70

        # HD급
        if w >= 1280:
            return 85

        # 표준 근처
        return 90

    def _score_fps(self, meta: dict) -> int:
        """프레임률 점수 (50fps 기준)"""
        fps = meta['fps']

        if fps < 15:        return 20  # 너무 낮음
        if fps < 24:        return 50
        if 24 <= fps <= 30: return 75   # 일반 비디오
        if 30 < fps < 45:   return 85
        if 45 <= fps <= 60: return 100  # 표준
        if fps > 60:        return 90   # 너무 빠름 → 다운샘플링
        return 50

    def _score_duration(self, meta: dict) -> int:
        """
        길이 점수 (관대한 기준)
        길이는 정규화로 해결되므로 등급에 큰 영향 없음
        """
        d = meta['duration']

        if d < 3:              return 30   # 너무 짧음
        if d < 10:             return 70   # 약간 짧음
        if 10 <= d <= 600:     return 100  # 10초~10분 모두 OK
        if d > 600:            return 80   # 10분 초과
        return 70

    def _score_stability(self, video_path: str) -> int:
        """영상 안정성 (프레임 간 차이로 흔들림 측정)"""
        cap = cv2.VideoCapture(video_path)
        prev_gray = None
        diffs = []

        for _ in range(min(30, int(cap.get(
                cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 120))
            if prev_gray is not None:
                diff = float(np.mean(
                    np.abs(gray.astype(np.int16) -
                           prev_gray.astype(np.int16))))
                diffs.append(diff)
            prev_gray = gray
        cap.release()

        if not diffs:
            return 50

        avg_diff = float(np.mean(diffs))
        # 정자 움직임으로 인한 자연스러운 차이는 5~15
        # 흔들림 심하면 30 이상
        if avg_diff < 5:    return 90   # 너무 안정 = 정지 영상?
        if avg_diff < 15:   return 100  # 정상
        if avg_diff < 25:   return 80
        if avg_diff < 40:   return 60
        return 30  # 흔들림 심함

    def _score_brightness(self, video_path: str) -> int:
        """밝기 적정성"""
        cap = cv2.VideoCapture(video_path)
        means = []

        for _ in range(min(10, int(cap.get(
                cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            means.append(float(np.mean(gray)))
        cap.release()

        if not means:
            return 50

        avg = float(np.mean(means))
        # 위상차 현미경은 보통 회색조 (90~180)
        if 90 <= avg <= 180:
            return 100
        if 70 <= avg < 90 or 180 < avg <= 210:
            return 75
        if 50 <= avg < 70 or 210 < avg <= 230:
            return 50
        return 25  # 너무 어둡거나 너무 밝음

    def _score_sharpness(self, video_path: str) -> int:
        """
        선명도 (라플라시안 분산)
        VISEM 영상 기준에 맞춘 관대한 임계값
        """
        cap = cv2.VideoCapture(video_path)
        sharpness_values = []

        for _ in range(min(10, int(cap.get(
                cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness_values.append(float(laplacian.var()))
        cap.release()

        if not sharpness_values:
            return 50

        avg = float(np.mean(sharpness_values))
        # VISEM 위상차 영상의 일반적 분포는 20~80 사이
        # 위상차 현미경 특성상 자연스럽게 흐릿함
        if avg > 80:    return 100
        if avg > 40:    return 90    # VISEM 기본 영상 대부분
        if avg > 20:    return 75
        if avg > 10:    return 55
        return 30  # 매우 흐림

    # ══════════════════════════════════════════════════════
    #   등급 결정
    # ══════════════════════════════════════════════════════

    def _determine_grade(self, score: int) -> str:
        for grade in ['S', 'A', 'B', 'C']:
            if score >= GRADE_THRESHOLDS[grade]:
                return grade
        return 'F'

    # ══════════════════════════════════════════════════════
    #   진단 + 권고
    # ══════════════════════════════════════════════════════

    def _diagnose(self, scores: dict, meta: dict) -> tuple:
        issues = []
        recs   = []

        if scores['resolution'] < 70:
            issues.append(
                f"해상도가 낮음 ({meta['width']}×{meta['height']})")
            recs.append(
                "더 높은 해상도(최소 640×480)로 촬영해주세요")

        if scores['fps'] < 70:
            issues.append(f"프레임률이 낮음 ({meta['fps']} fps)")
            recs.append("초당 30프레임 이상으로 촬영해주세요")

        if scores['duration'] < 70:
            if meta['duration'] < 10:
                issues.append(
                    f"영상이 너무 짧음 ({meta['duration']}초)")
                recs.append("최소 30초 이상 촬영해주세요")

        if scores['stability'] < 70:
            issues.append("영상이 불안정 (흔들림 감지)")
            recs.append("삼각대 등으로 카메라를 고정해주세요")

        if scores['brightness'] < 70:
            issues.append("조명/밝기가 부적합")
            recs.append("밝고 균일한 조명에서 촬영해주세요")

        if scores['sharpness'] < 70:
            issues.append("초점이 맞지 않음 (흐림)")
            recs.append("정자에 정확히 초점을 맞춰주세요")

        return issues, recs

    # ══════════════════════════════════════════════════════
    #   실패 결과
    # ══════════════════════════════════════════════════════

    # ── 하드 룰 검사 (본질적 한계) ────────────────────────
    def _check_hard_limits(self, meta: dict) -> str:
        """
        분석 자체가 불가능한 본질적 한계 검사
        조건 하나라도 해당되면 무조건 F등급
        """
        # 너무 짧은 영상
        if meta['duration'] < 5:
            return (f"영상이 너무 짧음 ({meta['duration']:.1f}초) "
                    f"— 최소 5초 이상 필요")

        # 너무 작은 해상도
        if meta['width'] < 320 or meta['height'] < 240:
            return (f"해상도가 너무 낮음 "
                    f"({meta['width']}×{meta['height']}) "
                    f"— 최소 320×240 이상 필요")

        # 너무 낮은 프레임률
        if meta['fps'] < 10:
            return (f"프레임률이 너무 낮음 ({meta['fps']:.1f}fps) "
                    f"— 최소 10fps 이상 필요")

        # 프레임 수 자체가 너무 적음
        if meta['n_frames'] < 30:
            return (f"프레임 수가 너무 적음 ({meta['n_frames']}프레임) "
                    f"— 최소 30프레임 이상 필요")

        return None  # 통과

    # ── 메타데이터 포함된 실패 결과 ─────────────────────────
    def _make_failure_result_with_meta(self, reason: str,
                                        metadata: dict) -> dict:
        return {
            'metadata':        metadata,
            'scores':          {},
            'total_score':     0,
            'grade':           'F',
            'issues':          [reason],
            'recommendations': [
                "권장 사양에 맞는 영상을 업로드해주세요"
            ],
            'can_analyze':     False,
        }

    def _make_failure_result(self, reason: str) -> dict:
        return {
            'metadata':        None,
            'scores':          {},
            'total_score':     0,
            'grade':           'F',
            'issues':          [reason],
            'recommendations': ["다른 영상을 업로드해주세요"],
            'can_analyze':     False,
        }