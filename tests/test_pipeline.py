"""Smoke tests — exercise every public module on synthetic data.

Run:
    python tests/test_pipeline.py

The test set never touches the real VISEM-Tracking data; instead it generates
random sperm-like trajectories, builds a tiny in-memory dataset and verifies
that:
    * CASA features compute without errors
    * Kalman tracker links detections across frames
    * Hybrid model performs forward + backward
    * InceptionTime baseline performs forward + backward
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.trajectory_utils import (
    smooth_trajectory_dct,
    augment_trajectory,
    pad_or_crop,
)
from src.features.casa_features import compute_casa_features, CASA_NAMES
from src.features.trajectory_features import (
    velocity_features,
    curvature,
    turning_angles,
    spectral_energy,
)
from src.tracking.kalman_tracker import KalmanTracker
from src.tracking.trajectory_builder import TrajectoryBuilder
from src.models import HybridTrajectoryModel, InceptionTime
from src.train import Trainer
from src.train.trainer import TrainConfig
from src.eval.metrics import classification_report


# --------------------------------------------------------------------------- #

def synth_trajectory(n: int, fps: int = 30, mode: str = "progressive", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    if mode == "progressive":
        x = 100 + 80.0 * t + rng.normal(0, 0.3, n)
        y = 100 + 5.0 * np.sin(2 * np.pi * 8 * t) + rng.normal(0, 0.3, n)
    elif mode == "non_progressive":
        x = 100 + 8.0 * np.sin(2 * np.pi * 1.5 * t) + rng.normal(0, 0.5, n)
        y = 100 + 5.0 * np.cos(2 * np.pi * 1.5 * t) + rng.normal(0, 0.5, n)
    else:  # immotile
        x = 100 + rng.normal(0, 0.4, n)
        y = 100 + rng.normal(0, 0.4, n)
    return np.stack([x, y], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #

def test_smoothing_and_features() -> None:
    print("\n[1/5] smoothing + CASA + trajectory features")
    xy = synth_trajectory(60, mode="progressive")
    sm = smooth_trajectory_dct(xy, n_keep=12)
    assert sm.shape == xy.shape
    casa = compute_casa_features(sm, raw_xy=xy, fps=30)
    print("    CASA:", dict(zip(CASA_NAMES, [round(float(v), 3) for v in casa])))
    assert casa.shape == (9,)
    vf = velocity_features(xy, fps=30)
    assert vf.shape == (60, 5)
    cu = curvature(xy)
    assert cu.shape == (60,)
    ta = turning_angles(xy)
    assert ta.shape == (58,)
    se = spectral_energy(xy[:, 0], n_bins=8)
    assert se.shape == (8,)
    print("    OK")


def test_augmentation_and_padding() -> None:
    print("\n[2/5] augmentation + padding")
    xy = synth_trajectory(20, mode="non_progressive")
    aug = augment_trajectory(xy)
    assert aug.shape == xy.shape
    pad = pad_or_crop(xy, 30)
    assert pad.shape == (30, 2)
    short = pad_or_crop(xy, 10)
    assert short.shape == (10, 2)
    print("    OK")


def test_tracker() -> None:
    print("\n[3/5] Kalman tracker + trajectory builder")
    rng = np.random.default_rng(0)
    builder = TrajectoryBuilder(KalmanTracker(max_distance=30.0, min_hits=2), min_hits=2)
    detections = []
    # Simulate 3 moving sperm across 30 frames (xyxy boxes)
    for f in range(30):
        rows = []
        for k, (vx, vy) in enumerate([(2.0, 0.2), (-1.5, 1.0), (0.0, 0.0)]):
            cx = 100 + k * 50 + vx * f + rng.normal(0, 0.2)
            cy = 100 + k * 30 + vy * f + rng.normal(0, 0.2)
            rows.append([cx - 5, cy - 5, cx + 5, cy + 5, 1.0])
        detections.append(np.asarray(rows, dtype=np.float32))
    builder.feed(detections)
    tracks = builder.build()
    assert len(tracks) >= 3, f"expected ≥3 tracks, got {len(tracks)}"
    avg_len = np.mean([t.length for t in tracks])
    print(f"    tracks={len(tracks)}, mean_len={avg_len:.1f}")
    assert avg_len > 10
    print("    OK")


# --------------------------------------------------------------------------- #
# Dummy in-memory dataset compatible with HybridTrajectoryModel inputs
# --------------------------------------------------------------------------- #

class DummyDataset(Dataset):
    def __init__(self, n: int = 30, T: int = 30) -> None:
        self.n = n
        self.T = T
        self.modes = ["progressive", "non_progressive", "immotile"]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        label = idx % 3
        mode = self.modes[label]
        xy = synth_trajectory(self.T, mode=mode, seed=idx)
        sm = smooth_trajectory_dct(xy)
        casa = compute_casa_features(sm, raw_xy=xy, fps=30)
        v = np.zeros_like(xy); v[1:] = np.diff(xy, axis=0)
        mask = np.ones((self.T,), dtype=np.float32)
        density = np.full((self.T,), float(label * 0.5 + 0.1), dtype=np.float32)
        return {
            "trajectory": torch.from_numpy(xy),
            "velocity": torch.from_numpy(v),
            "mask": torch.from_numpy(mask),
            "density": torch.from_numpy(density),
            "confidence": torch.tensor(0.7, dtype=torch.float32),
            "casa": torch.from_numpy(casa.astype(np.float32)),
            "label": torch.tensor(label, dtype=torch.long),
            "hyper": torch.tensor(int(label == 0), dtype=torch.long),
        }


def test_models_forward_backward() -> None:
    print("\n[4/5] models forward/backward")
    ds = DummyDataset(n=12, T=30)
    loader = DataLoader(ds, batch_size=6, shuffle=True)

    # Hybrid
    hybrid = HybridTrajectoryModel(num_classes=3, use_tracking_confidence=True)
    trainer = Trainer(hybrid, TrainConfig(epochs=2, batch_size=6, num_workers=0, save_dir=Path("/tmp")))
    trainer.fit(ds, ds)
    metrics, _ = trainer.evaluate(loader)
    print("    hybrid:", metrics.as_dict())
    assert 0.0 <= metrics.accuracy <= 1.0

    # InceptionTime baseline
    base = InceptionTime(in_channels=4, num_classes=3, depth=3)
    trainer2 = Trainer(base, TrainConfig(epochs=2, batch_size=6, num_workers=0, save_dir=Path("/tmp")))
    trainer2.fit(ds, ds)
    metrics2, _ = trainer2.evaluate(loader)
    print("    inception:", metrics2.as_dict())
    assert 0.0 <= metrics2.accuracy <= 1.0
    print("    OK")


def test_metrics() -> None:
    print("\n[5/5] eval metrics")
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.array([0, 1, 2, 0, 2, 2])
    pr = np.eye(3, dtype=np.float32)[p] * 0.7 + 0.1
    rep = classification_report(y, p, pr)
    assert 0 <= rep.accuracy <= 1
    print("    classification_report:", rep.as_dict())
    print("    OK")


# --------------------------------------------------------------------------- #

def main() -> None:
    test_smoothing_and_features()
    test_augmentation_and_padding()
    test_tracker()
    test_models_forward_backward()
    test_metrics()
    print("\nAll smoke tests passed ✓")


if __name__ == "__main__":
    main()
