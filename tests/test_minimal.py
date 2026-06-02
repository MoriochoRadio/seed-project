"""Minimal smoke tests using only numpy.

Verifies the components that don't depend on scipy/torch/sklearn:
    * CASA features
    * Kalman tracker + TrajectoryBuilder
The full ``tests/test_pipeline.py`` covers torch/scipy code paths and is
expected to be run in an environment with all `requirements.txt` deps installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.features.casa_features import compute_casa_features, CASA_NAMES
from src.tracking.kalman_tracker import KalmanTracker
from src.tracking.trajectory_builder import TrajectoryBuilder


def synth(n: int, mode: str = "progressive", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 30.0
    if mode == "progressive":
        x = 100 + 80 * t + rng.normal(0, 0.3, n)
        y = 100 + 5 * np.sin(2 * np.pi * 8 * t) + rng.normal(0, 0.3, n)
    elif mode == "non_progressive":
        x = 100 + 8 * np.sin(2 * np.pi * 1.5 * t) + rng.normal(0, 0.5, n)
        y = 100 + 5 * np.cos(2 * np.pi * 1.5 * t) + rng.normal(0, 0.5, n)
    else:
        x = 100 + rng.normal(0, 0.4, n)
        y = 100 + rng.normal(0, 0.4, n)
    return np.stack([x, y], axis=1).astype(np.float32)


def test_casa() -> None:
    print("[1/2] CASA features (no smoothing)")
    for mode in ("progressive", "non_progressive", "immotile"):
        xy = synth(60, mode=mode, seed=hash(mode) & 0xFF)
        # Use raw xy as both 'smoothed' and 'raw' (no scipy required)
        casa = compute_casa_features(xy, raw_xy=xy, fps=30)
        assert casa.shape == (9,), casa.shape
        assert np.all(np.isfinite(casa)), f"non-finite CASA for {mode}"
        named = dict(zip(CASA_NAMES, [round(float(v), 3) for v in casa]))
        print(f"    {mode}: {named}")
    # progressive should have largest VCL and largest LIN
    p = compute_casa_features(synth(60, "progressive", 1))
    n = compute_casa_features(synth(60, "non_progressive", 1))
    i = compute_casa_features(synth(60, "immotile", 1))
    assert p[0] > n[0] > i[0], f"VCL ordering violated: {p[0]:.1f} {n[0]:.1f} {i[0]:.1f}"
    print("    OK")


def test_tracker() -> None:
    print("[2/2] Kalman tracker + TrajectoryBuilder")
    rng = np.random.default_rng(0)
    builder = TrajectoryBuilder(KalmanTracker(max_distance=30.0, min_hits=2), min_hits=2)
    detections = []
    for f in range(30):
        rows = []
        for k, (vx, vy) in enumerate([(2.0, 0.2), (-1.5, 1.0), (0.0, 0.0)]):
            cx = 100 + k * 60 + vx * f + rng.normal(0, 0.2)
            cy = 100 + k * 30 + vy * f + rng.normal(0, 0.2)
            rows.append([cx - 5, cy - 5, cx + 5, cy + 5, 1.0])
        detections.append(np.asarray(rows, dtype=np.float32))
    builder.feed(detections)
    tracks = builder.build()
    assert len(tracks) >= 3, f"expected ≥3 tracks, got {len(tracks)}"
    avg_len = float(np.mean([t.length for t in tracks]))
    print(f"    tracks={len(tracks)}, mean_len={avg_len:.1f}")
    assert avg_len > 10
    print("    OK")


if __name__ == "__main__":
    test_casa()
    test_tracker()
    print("\nMinimal smoke tests passed ✓")
