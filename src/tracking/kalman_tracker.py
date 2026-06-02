"""Lightweight constant-velocity Kalman tracker for sperm heads.

State vector: [cx, cy, vx, vy].  Measurement: [cx, cy].
Association is greedy by Euclidean distance; for crowded scenes the
``TrajectoryBuilder`` may swap this for ``lap`` / Hungarian assignment.

This is intentionally NOT a re-implementation of BoT-SORT — for the experiments
we mainly need: (i) identity preservation across frames, (ii) per-track lifecycle
metadata (age, hits, fragments) so the model can compute tracking confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Track:
    track_id: int
    x: np.ndarray            # state [cx, cy, vx, vy]
    P: np.ndarray            # state covariance (4x4)
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    history: List[Tuple[int, np.ndarray]] = field(default_factory=list)
    fragmented: bool = False
    id_switches: int = 0


class KalmanTracker:
    """Multi-object tracker with constant-velocity Kalman predictions."""

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 3,
        max_distance: float = 25.0,
        process_noise: float = 1.0,
        measurement_noise: float = 1.0,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self._next_id = 0
        self.tracks: Dict[int, Track] = {}

        dt = 1.0
        self.F = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.H = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise

    # ------------------------------------------------------------------ #

    def _new_track(self, xy: np.ndarray, frame: int) -> Track:
        x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=np.float32)
        P = np.eye(4, dtype=np.float32) * 10.0
        t = Track(track_id=self._next_id, x=x, P=P)
        t.history.append((frame, xy.astype(np.float32).copy()))
        t.hits = 1
        self.tracks[self._next_id] = t
        self._next_id += 1
        return t

    def _predict(self) -> None:
        for t in self.tracks.values():
            t.x = self.F @ t.x
            t.P = self.F @ t.P @ self.F.T + self.Q
            t.age += 1
            t.time_since_update += 1

    def _update_track(self, t: Track, z: np.ndarray, frame: int) -> None:
        y = z - self.H @ t.x
        S = self.H @ t.P @ self.H.T + self.R
        K = t.P @ self.H.T @ np.linalg.inv(S)
        t.x = t.x + K @ y
        t.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ t.P
        t.hits += 1
        t.time_since_update = 0
        t.history.append((frame, np.array([t.x[0], t.x[1]], dtype=np.float32)))

    # ------------------------------------------------------------------ #

    def step(self, detections: np.ndarray, frame: int) -> List[Track]:
        """Update with one frame's detections (xyxy format) and return active tracks.

        Parameters
        ----------
        detections : np.ndarray (N, 4) or (N, 5) — xyxy [+ score]
        frame      : int — frame index
        """
        self._predict()

        if detections.size == 0:
            self._cull_dead()
            return [t for t in self.tracks.values() if t.time_since_update == 0]

        # detection centroids
        xy_dets = np.stack(
            [(detections[:, 0] + detections[:, 2]) / 2,
             (detections[:, 1] + detections[:, 3]) / 2],
            axis=1,
        ).astype(np.float32)

        # cost matrix between predicted track centroids and detections
        track_ids = list(self.tracks.keys())
        if track_ids:
            preds = np.stack([self.tracks[tid].x[:2] for tid in track_ids], axis=0)
            cost = np.linalg.norm(preds[:, None, :] - xy_dets[None, :, :], axis=2)
        else:
            cost = np.zeros((0, xy_dets.shape[0]), dtype=np.float32)

        # greedy association (low-cost first)
        assigned_dets: set[int] = set()
        assigned_tracks: set[int] = set()
        if cost.size:
            order = np.argsort(cost, axis=None)
            for flat in order:
                ti, di = np.unravel_index(flat, cost.shape)
                if ti in assigned_tracks or di in assigned_dets:
                    continue
                if cost[ti, di] > self.max_distance:
                    break
                t = self.tracks[track_ids[ti]]
                if t.time_since_update > 0 and t.hits >= self.min_hits:
                    t.fragmented = True
                self._update_track(t, xy_dets[di], frame)
                assigned_tracks.add(ti)
                assigned_dets.add(di)

        # spawn new tracks for unassigned detections
        for di in range(xy_dets.shape[0]):
            if di in assigned_dets:
                continue
            self._new_track(xy_dets[di], frame)

        self._cull_dead()
        return [t for t in self.tracks.values() if t.time_since_update == 0]

    # ------------------------------------------------------------------ #

    def _cull_dead(self) -> None:
        dead = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in dead:
            del self.tracks[tid]

    # ------------------------------------------------------------------ #

    def all_tracks(self) -> List[Track]:
        return list(self.tracks.values())
