"""Exp 3 — Trajectory length / FPS sensitivity (System-level robustness).

PDF 실험 3: 동일 모델을 입력 길이와 fps를 줄여가며 평가하여,
proposed 모델이 baseline보다 graceful degradation 하는지 검증.

조건
  Length: 1.0 s → 0.5 s → 0.25 s
  FPS   : 30 → 15 → 10  (VISEM-Tracking is 30 fps)

평가
  Accuracy / F1 / ECE — accuracy drop ratio = (acc_full − acc_reduced) / acc_full

Test-set selection
------------------
This script is **evaluation-only**.  It uses the test videos that Exp1 held out
during training, so the same model is never asked about a video it learned
from.  The mapping is recovered as follows (first match wins):

  1. ``--split_file <path>``                — explicit override
  2. ``<baseline_ckpt>.parent / split_videos.json`` — written by Exp1's
     video-level fallback (the common case for VISEM-Tracking)
  3. ``<root>/Test``                         — real Test split on disk
  4. Same-seed reconstruction                 — re-runs Exp1's fallback partition
     using ``--seed`` (default 42) so the test split matches Exp1's default

사용법 (Exp1에서 학습한 BEST 모델 사용)
  python experiments/exp3_fps_length.py \
      --data_root D:/sperm_data \
      --baseline_ckpt ./runs/exp1/baseline_best.pt \
      --proposed_ckpt ./runs/exp1/proposed_best.pt \
      --save_dir ./runs/exp3
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from _common import load_config, save_json  # type: ignore

from src.data import VISEMTrackingDataset
from src.train import Trainer, build_model_from_config
from src.train.trainer import TrainConfig


# --------------------------------------------------------------------------- #
# Down-sampling wrapper
# --------------------------------------------------------------------------- #

def downsample_dataset(ds: VISEMTrackingDataset, target_fps: int, target_window_s: float) -> VISEMTrackingDataset:
    """Create a copy of the dataset with reduced fps / window length.

    Frame stride = 30 / target_fps so a 30 fps source becomes target_fps.
    """
    ds2 = copy.copy(ds)
    ds2.fps = target_fps
    ds2.window = int(round(target_window_s * target_fps))
    ds2.augment = False
    ds2._stride = max(1, int(round(30.0 / target_fps)))
    orig_getitem = VISEMTrackingDataset.__getitem__

    def _strided_getitem(self, idx):
        rec = self._records[idx]
        old = rec.copy()
        s = getattr(self, "_stride", 1)
        if s > 1:
            rec_strided = {
                **rec,
                "frames": rec["frames"][::s],
                "xy": rec["xy"][::s],
            }
            self._records[idx] = rec_strided
            try:
                return orig_getitem(self, idx)
            finally:
                self._records[idx] = old
        return orig_getitem(self, idx)

    ds2.__class__ = type("StridedVISEM", (VISEMTrackingDataset,), {"__getitem__": _strided_getitem})
    return ds2


# --------------------------------------------------------------------------- #
# Test-set resolution
# --------------------------------------------------------------------------- #

def _resolve_test_videos(
    data_root: Path,
    cfg: dict,
    *,
    explicit_split: Optional[Path],
    ckpt: Path,
    seed: int,
    label_mode: str,
) -> List[str]:
    """Return the list of test video IDs to evaluate on, using the priority
    order documented in the module docstring."""

    # 1) explicit --split_file
    if explicit_split is not None and explicit_split.is_file():
        with explicit_split.open() as f:
            meta = json.load(f)
        ids = list(meta.get("test_videos", []))
        if ids:
            print(f"[exp3] using --split_file with test_videos={ids}")
            return ids

    # 2) split file next to the checkpoint
    auto = ckpt.parent / "split_videos.json"
    if auto.is_file():
        with auto.open() as f:
            meta = json.load(f)
        ids = list(meta.get("test_videos", []))
        if ids:
            print(f"[exp3] using {auto} (seed={meta.get('seed')}) → test_videos={ids}")
            return ids

    # 3) on-disk Test/ split
    test_ds = VISEMTrackingDataset(
        root=data_root, split="Test",
        fps=cfg["data"]["fps"], window_seconds=cfg["data"]["window_seconds"],
        augment=False, min_track_length=cfg["data"]["min_track_length"],
        verbose=False, label_mode=label_mode,
    )
    if len(test_ds) > 0:
        ids = sorted({rec["video_id"] for rec in test_ds._records})
        print(f"[exp3] using on-disk Test/ split, {len(ids)} videos")
        return ids

    # 4) same-seed reconstruction of Exp1's fallback
    train_ds = VISEMTrackingDataset(
        root=data_root, split="Train",
        fps=cfg["data"]["fps"], window_seconds=cfg["data"]["window_seconds"],
        augment=False, min_track_length=cfg["data"]["min_track_length"],
        verbose=False, label_mode=label_mode,
    )
    if len(train_ds) == 0:
        raise RuntimeError(
            "No training data found either. Confirm --data_root points at the "
            "right place (try `python scripts/inspect_visem.py --root ...`)."
        )
    video_ids = sorted({rec["video_id"] for rec in train_ds._records})
    rng = np.random.default_rng(seed)
    rng.shuffle(video_ids)
    n_v = len(video_ids)
    n_test = max(1, int(n_v * 0.15))
    n_val = max(1, int(n_v * 0.15))
    test_videos = video_ids[:n_test]
    print(
        f"[exp3] no split file found — reconstructing with seed={seed}: "
        f"test_videos={test_videos}  "
        f"(must match the seed Exp1 was run with!)"
    )
    return test_videos


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--baseline_ckpt", type=Path, required=True)
    parser.add_argument("--proposed_ckpt", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, default=Path("./runs/exp3"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Used to reconstruct the test split if no split_videos.json is found "
             "(must match the seed Exp1 was trained with).",
    )
    parser.add_argument(
        "--split_file",
        type=Path,
        default=None,
        help="Explicit path to a split_videos.json from Exp1.",
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        default=None,
        choices=["trajectory_kinematic", "video_majority"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    label_mode = args.label_mode or cfg["data"].get("label_mode", "trajectory_kinematic")

    # 1) Figure out which video IDs are the held-out test set
    test_video_ids = _resolve_test_videos(
        args.data_root, cfg,
        explicit_split=args.split_file,
        ckpt=args.baseline_ckpt,
        seed=args.seed,
        label_mode=label_mode,
    )

    # 2) Load the FULL trajectory pool (any split) and filter to test videos.
    #    Using split='All' avoids missing data on Train-only releases.
    full_ds = VISEMTrackingDataset(
        root=args.data_root, split="All",
        fps=cfg["data"]["fps"], window_seconds=cfg["data"]["window_seconds"],
        augment=False, min_track_length=cfg["data"]["min_track_length"],
        label_mode=label_mode,
    )
    test_ids = set(test_video_ids)
    test_indices = [i for i, rec in enumerate(full_ds._records) if rec["video_id"] in test_ids]
    if not test_indices:
        raise RuntimeError(
            f"After filtering for test_videos={sorted(test_ids)}, no trajectories remain. "
            f"Available video_ids in dataset = "
            f"{sorted({r['video_id'] for r in full_ds._records})}."
        )
    print(f"[exp3] test trajectory count = {len(test_indices)}")
    base_test = Subset(full_ds, test_indices)

    grids = [
        {"label": "1.0s_30fps", "fps": 30, "window_s": 1.0},
        {"label": "0.5s_30fps", "fps": 30, "window_s": 0.5},
        {"label": "0.25s_30fps", "fps": 30, "window_s": 0.25},
        {"label": "1.0s_15fps", "fps": 15, "window_s": 1.0},
        {"label": "1.0s_10fps", "fps": 10, "window_s": 1.0},
    ]

    results = {"baseline": {}, "proposed": {}}

    for arch_label, ckpt, model_cfg in (
        ("baseline", args.baseline_ckpt,
         {**cfg["model"], "arch": "inception", "use_tracking_confidence": False}),
        ("proposed", args.proposed_ckpt,
         {**cfg["model"], "arch": "hybrid", "use_tracking_confidence": True}),
    ):
        if not ckpt.is_file():
            raise RuntimeError(f"Checkpoint not found: {ckpt}")
        model = build_model_from_config(model_cfg, num_classes=full_ds.num_classes)
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        trainer = Trainer(model, TrainConfig(epochs=0, batch_size=args.batch_size))

        for cfg_grid in grids:
            ds_down = downsample_dataset(full_ds, cfg_grid["fps"], cfg_grid["window_s"])
            test_subset = Subset(ds_down, test_indices)
            loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
            metrics, ece = trainer.evaluate(loader)
            results[arch_label][cfg_grid["label"]] = {**metrics.as_dict(), "ece": ece}
            print(f"[{arch_label}|{cfg_grid['label']}] {results[arch_label][cfg_grid['label']]}")

    # Drop ratio relative to 1.0s_30fps
    for arch_label in ("baseline", "proposed"):
        ref = results[arch_label].get("1.0s_30fps", {}).get("accuracy", float("nan"))
        for cfg_grid in grids:
            cur = results[arch_label][cfg_grid["label"]].get("accuracy", float("nan"))
            ratio = (ref - cur) / ref if ref and not np.isnan(ref) and ref != 0 else float("nan")
            results[arch_label][cfg_grid["label"]]["accuracy_drop_ratio"] = ratio

    save_json(args.save_dir / "exp3_results.json", {
        "results": results, "config": cfg, "test_videos": sorted(test_ids),
    })

    # Pretty summary
    print("\n=== Exp3 summary ===")
    header = f"{'setting':<14} | {'baseline acc':>12} {'drop':>6} | {'proposed acc':>12} {'drop':>6}"
    print(header)
    print("-" * len(header))
    for g in grids:
        b = results["baseline"][g["label"]]
        p = results["proposed"][g["label"]]
        print(
            f"{g['label']:<14} | "
            f"{b['accuracy']:>12.3f} {b['accuracy_drop_ratio']:>6.2%} | "
            f"{p['accuracy']:>12.3f} {p['accuracy_drop_ratio']:>6.2%}"
        )


if __name__ == "__main__":
    main()
