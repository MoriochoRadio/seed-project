"""
tracker.py
ByteTrack 기반 정자 추적 및 운동성 특징 추출 모듈
"""

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from .casa_features    import compute_casa_features, CASA_NAMES
from .trajectory_utils import smooth_trajectory_dct


class SpermTracker:
    """ByteTrack 기반 정자 추적기"""

    def __init__(self, detector, config_path: str):
        """
        Args:
            detector: SpermDetector 인스턴스
            config_path: ByteTrack yaml 설정 파일 경로
        """
        self.detector    = detector
        self.config_path = config_path

    def track_video(self, video_path: str,
                    duration_sec: float = 5.0) -> dict:
        """
        영상 추적 실행

        Returns:
            {tid: [(frame_idx, cx, cy), ...]}
        """
        cap        = cv2.VideoCapture(video_path)
        fps        = cap.get(cv2.CAP_PROP_FPS)
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = min(int(fps * duration_sec), total, 250)

        track_history = defaultdict(list)

        for fidx in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            res = self.detector.model.track(
                frame,
                persist      = True,
                tracker      = self.config_path,
                verbose      = False,
                conf         = self.detector.conf
            )

            if res[0].boxes.id is not None:
                for box, tid, cls in zip(
                    res[0].boxes.xywh.cpu().numpy(),
                    res[0].boxes.id.cpu().numpy().astype(int),
                    res[0].boxes.cls.cpu().numpy().astype(int)
                ):
                    if cls == 0:
                        track_history[tid].append(
                            (fidx, float(box[0]), float(box[1]),
                             float(box[2]), float(box[3])))

        cap.release()
        return dict(track_history)

    def extract_features(self, track_history: dict) -> dict:
        """
        추적 결과에서 운동성 특징 11개 추출

        Returns:
            특징 딕셔너리 또는 None (추적 데이터 부족 시)
        """
        speeds, lins, straight_dists = [], [], []

        for tid, pts in track_history.items():
            if len(pts) < 5:
                continue

            coords = np.array([(cx, cy) for _, cx, cy, *_ in pts])
            dists  = np.sqrt(
                np.sum(np.diff(coords, axis=0)**2, axis=1))
            total_dist    = float(np.sum(dists))
            straight_dist = float(np.sqrt(
                (coords[-1][0] - coords[0][0])**2 +
                (coords[-1][1] - coords[0][1])**2))

            avg_speed = total_dist / len(pts)
            linearity = straight_dist / (total_dist + 1e-6)

            speeds.append(avg_speed)
            lins.append(linearity)
            straight_dists.append(straight_dist)

        if not speeds:
            return None

        speeds = np.array(speeds)

        return {
            'speed_mean':    float(np.mean(speeds)),
            'speed_median':  float(np.median(speeds)),
            'speed_75':      float(np.percentile(speeds, 75)),
            'speed_90':      float(np.percentile(speeds, 90)),
            'lin_mean':      float(np.mean(lins)),
            'lin_75':        float(np.percentile(lins, 75)),
            'straight_mean': float(np.mean(straight_dists)),
            'ratio_fast':    float(np.mean(speeds > 1.5)),
            'ratio_medium':  float(np.mean(
                                 (speeds > 0.5) & (speeds <= 1.5))),
            'ratio_slow':    float(np.mean(speeds <= 0.5)),
            'n_tracks':      len(speeds),
        }

    def compute_sample_kinematics(self,
                           track_history: dict,
                           fps: float = 50.0,
                           um_per_px: float = 0.7031) -> dict:
        """
        전체 샘플의 CASA 키네마틱 파라미터 계산 (WHO 6판 기준)
        - 스무딩: DCT-12 저역통과 필터 (Shahali 2026)
        - ALH: 접선 수직 성분 최댓값 (WHO 정의)
        - 추가 지표: BCF (편모 박동 Hz), PAW (측방 진폭 평균)
        """
        results = []

        for tid, pts in track_history.items():
            if len(pts) < 10:
                continue

            raw_xy = np.array([[cx, cy] for _, cx, cy, *_ in pts],
                            dtype=np.float32)

            smooth_xy = smooth_trajectory_dct(raw_xy, n_keep=12)

            casa = compute_casa_features(smooth_xy,
                                        raw_xy=raw_xy,
                                        fps=int(fps))

            for i in (0, 1, 2, 6, 8):
                casa[i] *= um_per_px

            results.append(casa)

        if not results:
            return {}

        arr = np.stack(results, axis=0)
        out = {
            f"{name}_mean": round(float(arr[:, i].mean()), 3)
            for i, name in enumerate(CASA_NAMES)
        }

        return out

    def grade_tracks(self,
                     track_history: dict,
                     fps: float = 50.0,
                     um_per_px: float = 0.7031,
                     str_min: float = 0.60,
                     vap_min: float = 15.0,
                     vcl_immotile: float = 18.0,
                     min_track_len: int = 10) -> dict:
        """정자별 운동성 3등급(PR/NP/IM) 분포 — 운동학 임계 기반 '보조 view'.

        각 트랙(≥min_track_len)의 per-track CASA(VCL/VAP/STR)를
        compute_sample_kinematics와 동일한 방식으로 계산해 등급화한다.

        등급 규칙 (임계는 production 클린 궤적 85개에서 % MAE 최소화로 보정한 값):
            IM : VCL < vcl_immotile
            PR : (VAP ≥ vap_min) and (STR ≥ str_min)   (그 외 motile)
            NP : 나머지

        ※ 회귀 운동성 %가 주 추정치이며, 본 등급은 참고용(개별 트랙 임계 기반).
        """
        counts = {'PR': 0, 'NP': 0, 'IM': 0}
        tid_grades = {}  

        for tid, pts in track_history.items():
            if len(pts) < min_track_len:
                continue

            coords = np.array([(cx, cy) for _, cx, cy, *_ in pts])
            n          = len(coords)
            total_time = (n - 1) * (1.0 / fps)

            dists      = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
            vcl        = (float(np.sum(dists)) * um_per_px) / total_time

            straight = float(np.sqrt(
                (coords[-1][0]-coords[0][0])**2 +
                (coords[-1][1]-coords[0][1])**2))
            vsl = (straight * um_per_px) / total_time

            w = min(5, n)
            smoothed = np.array([
                coords[max(0, i-w//2):i+w//2+1].mean(axis=0)
                for i in range(n)
            ])
            avg_path = float(np.sum(
                np.sqrt(np.sum(np.diff(smoothed, axis=0)**2, axis=1))))
            vap = (avg_path * um_per_px) / total_time
            str_ = vsl / (vap + 1e-6)

            if vcl < vcl_immotile:
                counts['IM'] += 1
                tid_grades[tid] = 'IM'   # ← 추가
            elif (vap >= vap_min) and (str_ >= str_min):
                counts['PR'] += 1
                tid_grades[tid] = 'PR'   # ← 추가
            else:
                counts['NP'] += 1
                tid_grades[tid] = 'NP'   # ← 추가

        n_graded = sum(counts.values())
        if n_graded == 0:
            return {}

        return {
            'grade_progressive':     round(100.0 * counts['PR'] / n_graded, 1),
            'grade_non_progressive': round(100.0 * counts['NP'] / n_graded, 1),
            'grade_immotile':        round(100.0 * counts['IM'] / n_graded, 1),
            'n_graded':              n_graded,
            'tid_grades':            tid_grades, 
            'thresholds': {'STR_min': str_min, 'VAP_min': vap_min,
                           'VCL_immotile': vcl_immotile},
        }