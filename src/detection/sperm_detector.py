"""Sperm head detector wrapper.

A thin layer around `ultralytics.YOLO` that exposes a uniform
``detect(frame_bgr) -> np.ndarray (N, 5) [x1, y1, x2, y2, score]`` API.
The actual model checkpoint can be:
    * a public YOLOv8 weight (``yolov8n.pt``) for testing the pipeline
    * a fine-tuned weight on VISEM-Tracking (the dataset includes YOLO labels,
      so a fine-tune is a one-liner — see ``train_detector.py`` if added later)

If ``ultralytics`` is missing the class falls back to a label-file detector that
reads pre-computed VISEM bboxes directly. This keeps the whole pipeline runnable
without GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class SpermDetector:
    def __init__(
        self,
        weights: Optional[str] = "yolov8n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        labels_dir: Optional[Path] = None,
        image_size: tuple[int, int] = (1080, 1920),
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.device = device
        self._model = None
        if weights and labels_dir is None:
            try:
                from ultralytics import YOLO  # type: ignore

                self._model = YOLO(weights)
            except Exception as exc:  # pragma: no cover
                print(f"[SpermDetector] ultralytics unavailable ({exc}); using label-file fallback.")

    # ------------------------------------------------------------------ #

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Detect on a single BGR frame → (N, 5) [x1, y1, x2, y2, score]."""
        if self._model is None:
            raise RuntimeError("No YOLO model loaded. Provide weights or use detect_from_label.")
        result = self._model.predict(
            frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy().reshape(-1, 1)
        return np.concatenate([xyxy, conf], axis=1).astype(np.float32)

    # ------------------------------------------------------------------ #

    def detect_from_label(self, video_id: str, frame_idx: int) -> np.ndarray:
        """Read pre-computed YOLO label file as if it were a detector output."""
        if self.labels_dir is None:
            raise RuntimeError("labels_dir not set on SpermDetector")
        label_path = self.labels_dir / f"{video_id}_frame_{frame_idx}.txt"
        if not label_path.is_file():
            return np.zeros((0, 5), dtype=np.float32)
        H, W = self.image_size
        boxes = []
        with label_path.open() as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h, *rest = parts
                cx, cy, w, h = map(float, (cx, cy, w, h))
                x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
                x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
                boxes.append([x1, y1, x2, y2, 1.0])
        return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 5), dtype=np.float32)
