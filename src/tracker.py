"""
tracker.py
ByteTrack 기반 정자 추적 및 운동성 특징 추출 모듈
"""

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO


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
                            (fidx, float(box[0]), float(box[1])))

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

            coords = np.array([(cx, cy) for _, cx, cy in pts])
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
        전체 샘플의 CASA 키네마틱 파라미터 계산

        Returns:
            샘플 수준 평균 키네마틱 파라미터 딕셔너리
        """
        VCLs, VSLs, VAPs = [], [], []
        LINs, STRs, WOBs = [], [], []
        ALHs = []

        for tid, pts in track_history.items():
            if len(pts) < 10:
                continue

            coords = np.array([(cx, cy) for _, cx, cy in pts])
            n      = len(coords)
            dt     = 1.0 / fps

            # VCL
            dists       = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
            total_path  = float(np.sum(dists))
            total_time  = (n - 1) * dt
            vcl = (total_path * um_per_px) / total_time

            # VSL
            straight = float(np.sqrt(
                (coords[-1][0]-coords[0][0])**2 +
                (coords[-1][1]-coords[0][1])**2))
            vsl = (straight * um_per_px) / total_time

            # VAP (5점 이동 평균 경로)
            w = min(5, n)
            smoothed = np.array([
                coords[max(0, i-w//2):i+w//2+1].mean(axis=0)
                for i in range(n)
            ])
            avg_path = float(np.sum(
                np.sqrt(np.sum(np.diff(smoothed, axis=0)**2, axis=1))))
            vap = (avg_path * um_per_px) / total_time

            lin = vsl / (vcl + 1e-6)
            str_ = vsl / (vap + 1e-6)
            wob = vap / (vcl + 1e-6)

            # ALH
            devs = np.sqrt(np.sum((coords - smoothed)**2, axis=1))
            alh  = float(np.mean(devs)) * um_per_px

            VCLs.append(vcl); VSLs.append(vsl); VAPs.append(vap)
            LINs.append(lin); STRs.append(str_); WOBs.append(wob)
            ALHs.append(alh)

        if not VCLs:
            return {}

        return {
            'VCL_mean': round(float(np.mean(VCLs)), 1),
            'VSL_mean': round(float(np.mean(VSLs)), 1),
            'VAP_mean': round(float(np.mean(VAPs)), 1),
            'LIN_mean': round(float(np.mean(LINs)), 3),
            'STR_mean': round(float(np.mean(STRs)), 3),
            'WOB_mean': round(float(np.mean(WOBs)), 3),
            'ALH_mean': round(float(np.mean(ALHs)), 2),
            'n_analyzed': len(VCLs),
        }