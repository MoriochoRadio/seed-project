"""Exp 1 — Crowded environment trajectory robustness.

핵심 차별점 검증 (PDF 실험 1):
  Trajectory quality가 낮아질수록 Shahali-style baseline 모델보다
  tracking-aware proposed 모델이 robust한가?

비교 모델
  Baseline : InceptionTime  (trajectory only, no tracking-aware terms)
  Proposed : HybridTrajectoryModel(use_tracking_confidence=True)

설계
  1) VISEM-Tracking 전체 track을 mean-density 3분위로 low/medium/high 로 분할
  2) 동일 train/test split에서 두 모델 학습 (train은 모든 density 혼합)
  3) low/med/high test set 각각에 대해 metric 측정 → 차이 분석
  4) tracking quality(IDF1 proxy) vs classification F1 의 상관계수 보고

사용법
  python experiments/exp1_crowded_robustness.py \
      --data_root ./data/visem_tracking \
      --epochs 30 \
      --patience 10 \
      --monitor f1_macro \
      --save_dir ./runs/exp1

학습은 항상 best 모델 기준으로 종료/저장되며, early stopping이 활성화되어 있어
검증 메트릭이 ``patience`` epoch 동안 개선되지 않으면 자동 종료됩니다.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from _common import load_config, save_json, density_split  # type: ignore

from src.data import VISEMTrackingDataset
from src.eval.metrics import classification_report, expected_calibration_error
from src.eval.tracking_metrics import idf1_proxy
from src.train import Trainer, build_model_from_config
from src.train.trainer import TrainConfig


# --------------------------------------------------------------------------- #

def make_subsets(
    dataset: VISEMTrackingDataset,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """Split records into train/val/test (60/20/20) deterministically."""
    n = len(dataset)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_ratio)
    n_test = int(n * 0.2)
    val_idx = perm[:n_val]
    test_idx = perm[n_val: n_val + n_test]
    train_idx = perm[n_val + n_test:]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def density_subsets(dataset: VISEMTrackingDataset, indices: list[int]) -> dict[str, list[int]]:
    records = [dataset._records[i] for i in indices]
    by_density = density_split(records, dataset.image_size)
    out = {k: [] for k in by_density}
    rec_to_idx = {id(r): i for i, r in zip(indices, records)}
    for k, recs in by_density.items():
        out[k] = [rec_to_idx[id(r)] for r in recs]
    return out


def evaluate_loader(trainer: Trainer, loader: DataLoader):
    return trainer.evaluate(loader)


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_dir", type=Path, default=Path("./runs/exp1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early-stopping patience (epochs without improvement).")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="Min improvement counted as 'better' (in monitor units).")
    parser.add_argument("--monitor", type=str, default="f1_macro",
                        choices=["f1_macro", "accuracy", "auroc", "cohen_kappa", "loss", "ece"])
    parser.add_argument("--mode", type=str, default="max", choices=["max", "min"])
    parser.add_argument(
        "--label_mode",
        type=str,
        default=None,
        choices=["trajectory_kinematic", "video_majority"],
        help="Override config's label_mode (trajectory_kinematic recommended).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["batch_size"] = args.batch_size
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
    if len(train_ds) == 0:
        raise RuntimeError(
            "Empty train dataset. Run `python scripts/inspect_visem.py --root <your-root>`"
            " to see what's on disk; --data_root must point at a folder whose subfolders"
            " each contain a labels/ directory of YOLO .txt files."
        )

    # If the dataset has no separate Test split (the public VISEM-Tracking release
    # is Train-only), partition videos from the Train fold into train/val/test
    # at the *video* level so trajectories from one video never leak across splits.
    if len(test_ds) == 0:
        print(
            "[exp1] No 'Test' split detected — partitioning Train videos "
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

        tr_idx, val_idx, te_idx = [], [], []
        for i, rec in enumerate(train_ds._records):
            if rec["video_id"] in train_videos:
                tr_idx.append(i)
            elif rec["video_id"] in val_videos:
                val_idx.append(i)
            elif rec["video_id"] in test_videos:
                te_idx.append(i)

        # Build a non-augmenting view of the same dataset for val/test
        eval_ds = VISEMTrackingDataset(
            root=args.data_root, split="Train",
            fps=cfg["data"]["fps"], window_seconds=cfg["data"]["window_seconds"],
            augment=False, min_track_length=cfg["data"]["min_track_length"],
            verbose=False, label_mode=label_mode,
        )
        train_subset = Subset(train_ds, tr_idx)
        val_subset = Subset(eval_ds, val_idx) if val_idx else None
        test_ds = eval_ds  # rebind so density buckets/eval below use the eval view
        test_indices = te_idx
    else:
        # train/val split inside the Train fold
        tr_idx, val_idx, _ = make_subsets(train_ds, val_ratio=0.15, seed=args.seed)
        train_subset = Subset(train_ds, tr_idx)
        val_subset = Subset(train_ds, val_idx) if val_idx else None
        test_indices = list(range(len(test_ds)))

    by_density = density_subsets(test_ds, test_indices) if test_indices else {}

    results = {}

    for arch_label, model_cfg in (
        ("baseline", {**cfg["model"], "arch": "inception", "use_tracking_confidence": False}),
        ("proposed", {**cfg["model"], "arch": "hybrid", "use_tracking_confidence": True}),
    ):
        print(f"\n========== Training {arch_label} ==========")
        model = build_model_from_config(model_cfg, num_classes=train_ds.num_classes)
        # Each arch gets its own save_dir so the per-arch best.pt doesn't collide
        arch_save_dir = args.save_dir / arch_label
        trainer = Trainer(
            model,
            TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=cfg["train"]["lr"],
                weight_decay=cfg["train"]["weight_decay"],
                seed=args.seed,
                save_dir=arch_save_dir,
                early_stopping_patience=args.patience,
                early_stopping_min_delta=args.min_delta,
                early_stopping_monitor=args.monitor,
                early_stopping_mode=args.mode,
            ),
        )
        history = trainer.fit(train_subset, val_subset)
        # `trainer.save()` always writes the BEST model (not the final epoch).
        ckpt = args.save_dir / f"{arch_label}_best.pt"
        trainer.save(ckpt)

        # Per-density evaluation
        per_density = {}
        for level, idx in by_density.items():
            if not idx:
                continue
            loader = DataLoader(Subset(test_ds, idx), batch_size=args.batch_size, shuffle=False)
            metrics, ece = trainer.evaluate(loader)
            per_density[level] = {**metrics.as_dict(), "ece": ece, "n": len(idx)}

        # Aggregate IDF1 proxy from track records
        idf1_per_density = {}
        for level, idx in by_density.items():
            recs = [test_ds._records[i] for i in idx]
            # build a pseudo-track object with .length / .fragmented for proxy
            pseudo = [
                type("T", (), {
                    "length": int(r["frames"].size),
                    "fragmented": bool(np.any(np.diff(r["frames"]) > 1)),
                })()
                for r in recs
            ]
            idf1_per_density[level] = idf1_proxy(pseudo)

        results[arch_label] = {
            "history": history,
            "per_density": per_density,
            "idf1_proxy_per_density": idf1_per_density,
            "best_ckpt": str(ckpt),
            "best_metric": history.get("best_metric") if isinstance(history, dict) else None,
            "best_epoch": history.get("best_epoch") if isinstance(history, dict) else None,
            "stopped_early": history.get("stopped_early") if isinstance(history, dict) else None,
        }

    # Comparative analysis: F1 gap proposed-vs-baseline at each density
    summary = {"f1_gap": {}, "tracking_correlation": {}}
    for level in ("low", "medium", "high"):
        b = results["baseline"]["per_density"].get(level, {}).get("f1_macro", float("nan"))
        p = results["proposed"]["per_density"].get(level, {}).get("f1_macro", float("nan"))
        summary["f1_gap"][level] = p - b

    # Correlation between tracking quality and classification F1 across density levels
    levels = [k for k in ("low", "medium", "high") if k in by_density]
    if len(levels) >= 2:
        for arch_label in ("baseline", "proposed"):
            x = [results[arch_label]["idf1_proxy_per_density"].get(l, np.nan) for l in levels]
            y = [results[arch_label]["per_density"].get(l, {}).get("f1_macro", np.nan) for l in levels]
            x, y = np.asarray(x), np.asarray(y)
            if np.isfinite(x).all() and np.isfinite(y).all() and np.std(x) > 0 and np.std(y) > 0:
                summary["tracking_correlation"][arch_label] = float(np.corrcoef(x, y)[0, 1])
            else:
                summary["tracking_correlation"][arch_label] = float("nan")

    save_json(args.save_dir / "exp1_results.json",
              {"results": results, "summary": summary, "config": cfg})
    print("\n=== Exp1 summary ===")
    print(summary)


if __name__ == "__main__":
    main()
