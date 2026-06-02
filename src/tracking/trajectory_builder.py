"""Run a video through detector + tracker and return per-track trajectories.

This is the bridge between (Detection→Tracking) and the trajectory-aware
classifier.  Outputs are compatible with ``VISEMTrackingDataset.__getitem__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .kalman_tracker import KalmanTracker, Track


@dataclass
class BuiltTrack:
    track_id: int
    frames: np.ndarray         # (T,)
    xy: np.ndarray             # (T, 2)
    fragmented: bool
    hits: int
    age: int

    @property
    def length(self) -> int:
        return self.frames.shape[0]


class TrajectoryBuilder:
    def __init__(
        self,
        tracker: Optional[KalmanTracker] = None,
        min_hits: int = 5,
    ) -> None:
        self.tracker = tracker or KalmanTracker()
        self.min_hits = min_hits
        # snapshots after the loop ends so we can return long completed tracks
        self._completed: List[Track] = []

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self.tracker = KalmanTracker(
            max_age=self.tracker.max_age,
            min_hits=self.tracker.min_hits,
            max_distance=self.tracker.max_distance,
        )
        self._completed = []

    def feed(self, detections_per_frame: Iterable[np.ndarray]) -> None:
        for f, det in enumerate(detections_per_frame):
            self.tracker.step(det, f)
            # Snapshot tracks that just disappeared so we keep their history
            for tid, t in list(self.tracker.tracks.items()):
                if t.time_since_update >= self.tracker.max_age and t.hits >= self.min_hits:
                    self._completed.append(t)

    # ------------------------------------------------------------------ #

    def build(self) -> List[BuiltTrack]:
        out: List[BuiltTrack] = []
        seen: set[int] = set()
        for t in list(self.tracker.tracks.values()) + self._completed:
            if t.track_id in seen or t.hits < self.min_hits:
                continue
            seen.add(t.track_id)
            frames = np.array([f for f, _ in t.history], dtype=np.int32)
            xy = np.stack([p for _, p in t.history], axis=0)
            out.append(
                BuiltTrack(
                    track_id=t.track_id,
                    frames=frames,
                    xy=xy,
                    fragmented=t.fragmented,
                    hits=t.hits,
                    age=t.age,
                )
            )
        return out
