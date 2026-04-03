"""
detector.py
YOLO11 기반 정자 탐지 모듈
"""

from ultralytics import YOLO
import numpy as np
import cv2


class SpermDetector:
    """YOLO11 정자 탐지기"""

    def __init__(self, model_path: str, conf: float = 0.3):
        """
        Args:
            model_path: YOLO11 모델 경로
            conf: 탐지 신뢰도 임계값
        """
        self.model = YOLO(model_path)
        self.conf  = conf
        print(f"✅ SpermDetector 로드 완료: {model_path}")

    def detect_frame(self, frame: np.ndarray) -> dict:
        """
        단일 프레임에서 정자 탐지

        Returns:
            {
              'count': 탐지된 정자 수,
              'boxes': [[cx, cy, w, h], ...],
              'confs': [신뢰도, ...],
            }
        """
        results = self.model(frame, verbose=False, conf=self.conf)
        boxes_raw = results[0].boxes

        sperm_mask = boxes_raw.cls.cpu().numpy().astype(int) == 0
        boxes = boxes_raw.xywh.cpu().numpy()[sperm_mask]
        confs  = boxes_raw.conf.cpu().numpy()[sperm_mask]

        return {
            'count': int(sperm_mask.sum()),
            'boxes': boxes.tolist(),
            'confs': confs.tolist(),
        }

    def estimate_sperm_count(self, video_path: str,
                              n_frames: int = 10) -> tuple:
        """
        영상 첫 n_frames에서 정자 수 추정

        Returns:
            (중앙값, 프레임별 카운트 리스트)
        """
        cap    = cv2.VideoCapture(video_path)
        counts = []

        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            det = self.detect_frame(frame)
            counts.append(det['count'])

        cap.release()

        N = int(np.median(counts)) if counts else 0
        return N, counts