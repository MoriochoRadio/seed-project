"""Exp 4 — Ablation: efficacy of tracking-aware information.

PDF 실험 4: proposed model에서 컴포넌트를 단계적으로 제거하며 성능 변화 측정.

Variants
  V1  : raw trajectory only (InceptionTime baseline equivalent)
  V2  : V1 + CASA features (no tracking confidence, no density)
  V3  : V2 + tracking confidence
  V4  : V3 + per-frame density   ← full proposed

각 variant를 동일 데이터로 학습 → test set의 high-density bucket에서 평가.

사용법
  python experiments/exp4_ablation.py \
      --data_root ./data/visem_tracking \
      --epochs 30 \
      --patience 10 \
      --monitor f1_macro \
      --save_dir ./runs/exp4

각 variant는 동일한 early-stopping 설정으로 학습되며, 학습 종료 시 자동으로
best 모델로 복원되어 평가됩니다. checkpoint는 ``<save_dir>/<variant>/best.pt``
와 ``<save_dir>/<variant>_best.pt`` 양쪽에 저장됩니다.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from _common import load_config, save_json, density_split  # type: ignore

from src.data import VISEMTrackingDataset
from src.train import Trainer, build_model_from_config
from src.train.trainer import TrainConfig
from src.models import HybridTrajectoryModel


# --------------------------------------------------------------------------- #
# Variant configurations
# --------------------------------------------------------------------------- #

VARIANTS = {
    "V1_traj_only":    {"use_casa": False, "use_tracking_confidence": False, "use_density": False},
    "V2_+casa":        {"use_casa": True,  "use_tracking_confidence": False, "use_density": False},
    "V3_+confidence":  {"use_casa": True,  "use_tracking_confidence": True,  "use_density": False},
    "V4_full":         {"use_casa": True,  "use_tracking_confidence": True,  "use_density": True},
}


class AblationModel(nn.Module):
    """Wraps HybridTrajectoryModel to gate side-channel inputs by variant."""

    def __init__(self, base: HybridTrajectoryModel, use_casa: bool, use_density: bool) -> None:
        super().__init__()
        self.base = base
        self.use_casa = use_casa
        self.use_density = use_density

    def forward(self, **kw):
        if not self.use_casa:
            kw["casa"] = torch.zeros_like(kw["casa"])
        if not self.use_density:
            kw["density"] = torch.zeros_like(kw["density"])
        return self.base(**kw)


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_dir", type=Path, default=Path("./runs/exp4"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Train→val split ratio (used to monitor early stopping).")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--monitor", type=str, default="f1_macro",
                        choices=["f1_macro", "accuracy", "auroc", "cohen_kappa", "loss", "ece"])
    parser.add_argument("--mode", type=str, default="max", choices=["max", "min"])
    parser.add_argument(
        "--label_mode",
        type=str,
        default=None,
        choices=["trajectory_kinematic", "video_majority"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    label_mode = args.label_mode or cfg["data"].get("label_mode", "trajectory_kinematic")
    train_ds = VISEMTrackingDataset(
        root=args.data_root, split="Train",
        fps=cfg["data"]["fps"], window_seconds=cfg["data"]["window_seconds"],
        augment=True, min_track_length=cfg["data"]["min_track_length"],
        label_mode=label_mode,
    )
    test_ds = VISEMTrackingDataset(
        root=args.data_root, split="Test",
        fps=cfg["data"]["fps"], window_seconds=cfg["data"]["window_seconds"],
        augment=False, min_track_length=cfg["data"]["min_track_length"],
        label_mode=label_mode,
    )

    # ------------------------------------------------------------------ #
    # Test-fold fallback (VISEM-Tracking public release is Train-only):
    # partition Train videos into train/val/test ~70/15/15 at the video
    # level so trajectories from one video never leak across splits.
    # ------------------------------------------------------------------ #
    if len(test_ds) == 0:
        print(
            "[exp4] No 'Test' split detected — partitioning Train videos "
            "into train/val/test (~70/15/15) at the video level."
        )
        video_ids = sorted({rec["video_id"] for rec in train_ds._records})
        rng = np.random.default_rng(args.seed)
        rng.shuffle(video_ids)
        n_v = len(video_ids)
        n_test = max(1, int(n_v * 0.15))
        n_val = max(1, int(n_v * 0.15))
        test_videos = set(video_ids[:n_test])
        val_videos = set(video_ids[n_test:n_test + n_val])
        train_videos = set(video_ids[n_test + n_val:])

        train_idx_train, val_idx_train, te_idx = [], [], []
        for i, rec in enumerate(train_ds._records):
            if rec["video_id"] in train_videos:
                train_idx_train.append(i)
            elif rec["video_id"] in val_videos:
                val_idx_train.append(i)
            elif rec["video_id"] in test_videos:
                te_idx.append(i)

        # non-augmenting view of the same dataset for val/test
        eval_ds = VISEMTrackingDataset(
            root=args.data_root, split="Train",
            fps=cfg["data"]["fps"], window_seconds=cfg["data"]["window_seconds"],
            augment=False, min_track_length=cfg["data"]["min_track_length"],
            verbose=False, label_mode=label_mode,
        )
        train_subset = Subset(train_ds, train_idx_train)
        val_subset = Subset(eval_ds, val_idx_train) if val_idx_train else None
        # density buckets are computed on the held-out test indices of eval_ds
        eval_records = [eval_ds._records[i] for i in te_idx]
        by_density = density_split(eval_records, eval_ds.image_size)
        rec_to_idx = {id(r): i for i, r in zip(te_idx, eval_records)}
        high_idx = [rec_to_idx[id(r)] for r in by_density.get("high", [])]
        low_idx = [rec_to_idx[id(r)] for r in by_density.get("low", [])]
        test_ds = eval_ds  # rebind so the per-density DataLoaders below work
        print(
            f"[exp4] split sizes: train={len(train_idx_train)}, "
            f"val={len(val_idx_train)}, test={len(te_idx)}  "
            f"(test density buckets: low={len(low_idx)}, high={len(high_idx)})"
        )
    else:
        # Standard path: Test split exists on disk.
        by_density = density_split(test_ds._records, test_ds.image_size)
        rec_to_idx = {id(r): i for i, r in enumerate(test_ds._records)}
        high_idx = [rec_to_idx[id(r)] for r in by_density.get("high", [])]
        low_idx = [rec_to_idx[id(r)] for r in by_density.get("low", [])]

        # Train→val split (held out for early-stopping signal)
        n_train = len(train_ds)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n_train)
        n_val = int(n_train * args.val_ratio)
        val_idx_train = perm[:n_val].tolist()
        train_idx_train = perm[n_val:].tolist()
        train_subset = Subset(train_ds, train_idx_train)
        val_subset = Subset(train_ds, val_idx_train) if val_idx_train else None

    results = {}

    for label, mods in VARIANTS.items():
        print(f"\n========== Ablation: {label} ==========")
        model_cfg = {**cfg["model"], "arch": "hybrid",
                     "use_tracking_confidence": mods["use_tracking_confidence"]}
        base_model = build_model_from_config(model_cfg, num_classes=train_ds.num_classes)
        wrapped = AblationModel(base_model, use_casa=mods["use_casa"], use_density=mods["use_density"])

        # Adapter trainer: HybridTrajectoryModel API expected
        # (Trainer detects HybridTrajectoryModel — wrap it as such)
        wrapped.__class__.__bases__ = (HybridTrajectoryModel,)  # duck-type for Trainer

        variant_save_dir = args.save_dir / label
        trainer = Trainer(
            wrapped,
            TrainConfig(
                epochs=args.epochs, batch_size=args.batch_size,
                lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
                seed=args.seed,
                save_dir=variant_save_dir,
                early_stopping_patience=args.patience,
                early_stopping_min_delta=args.min_delta,
                early_stopping_monitor=args.monitor,
                early_stopping_mode=args.mode,
            ),
        )
        history = trainer.fit(train_subset, val_subset)
        # Save BEST checkpoint (also ensured during fit) at a stable name
        ckpt = args.save_dir / f"{label}_best.pt"
        trainer.save(ckpt)

        per = {}
        if high_idx:
            loader = DataLoader(Subset(test_ds, high_idx), batch_size=args.batch_size, shuffle=False)
            m, ece = trainer.evaluate(loader)
            per["high"] = {**m.as_dict(), "ece": ece, "n": len(high_idx)}
        if low_idx:
            loader = DataLoader(Subset(test_ds, low_idx), batch_size=args.batch_size, shuffle=False)
            m, ece = trainer.evaluate(loader)
            per["low"] = {**m.as_dict(), "ece": ece, "n": len(low_idx)}

        results[label] = {
            "per_density": per,
            "history": history,
            "best_ckpt": str(ckpt),
            "best_metric": history.get("best_metric") if isinstance(history, dict) else None,
            "best_epoch": history.get("best_epoch") if isinstance(history, dict) else None,
            "stopped_early": history.get("stopped_early") if isinstance(history, dict) else None,
        }

    save_json(args.save_dir / "exp4_results.json", {"results": results, "config": cfg})

    # ------------------------------------------------------------------ #
    # Pretty summary — shows the ablation progression (V1→V2→V3→V4) at
    # high-density and low-density buckets so the contribution of each
    # added component is immediately visible.
    # ------------------------------------------------------------------ #
    print("\n=== Exp4 summary ===")
    header = f"{'variant':<18} {'low F1':>8} {'high F1':>8} {'low Acc':>8} {'high Acc':>8} {'best_ep':>7}"
    print(header)
    print("-" * len(header))
    for k, v in results.items():
        per = v.get("per_density", {}) or {}
        low = per.get("low", {}) or {}
        high = per.get("high", {}) or {}
        def fmt(x): return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "
        print(
            f"{k:<18} "
            f"{fmt(low.get('f1_macro')):>8} {fmt(high.get('f1_macro')):>8} "
            f"{fmt(low.get('accuracy')):>8} {fmt(high.get('accuracy')):>8} "
            f"{str(v.get('best_epoch', '-')):>7}"
        )

    # Quick deltas (V4 vs V1) — the headline claim "tracking-aware terms help"
    try:
        v1 = results["V1_traj_only"]["per_density"]
        v4 = results["V4_full"]["per_density"]
        for bucket in ("low", "high"):
            if bucket in v1 and bucket in v4:
                d = v4[bucket]["f1_macro"] - v1[bucket]["f1_macro"]
                print(f"\n  Δ F1 (V4_full − V1_traj_only) at {bucket}-density: {d:+.3f}")
    except (KeyError, TypeError):
        pass

    if not any(v.get("per_density") for v in results.values()):
        print(
            "\n  ! All per_density buckets are empty. This usually means the Test split"
            "\n    is missing AND too few Train videos to carve out a held-out test set."
            "\n    Tip: confirm `videos_found` > 5 in the dataset log above."
        )


if __name__ == "__main__":
    main()
