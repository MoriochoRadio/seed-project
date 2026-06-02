"""Run an experiment script multiple times with different seeds and
aggregate the resulting JSON outputs into a single summary table.

Works on Windows / macOS / Linux (no shell-loop syntax involved).

Usage examples
--------------
# Exp4 ablation with 5 seeds
python scripts/run_seeds.py exp4 ^
    --data_root D:/sperm_data ^
    --epochs 60 --patience 25 ^
    --seeds 0 1 2 3 4 ^
    --save_root ./runs/exp4

# Exp1 robustness with 3 seeds
python scripts/run_seeds.py exp1 ^
    --data_root D:/sperm_data ^
    --epochs 50 --patience 20 ^
    --seeds 0 1 2 ^
    --save_root ./runs/exp1
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO = Path(__file__).resolve().parents[1]

# Maps experiment name → (script path, results-json filename)
EXPERIMENTS: Dict[str, Dict[str, str]] = {
    "exp1": {
        "script": "experiments/exp1_crowded_robustness.py",
        "results": "exp1_results.json",
    },
    "exp4": {
        "script": "experiments/exp4_ablation.py",
        "results": "exp4_results.json",
    },
}


def run_one(experiment: str, seed: int, save_dir: Path, common_args: List[str]) -> Path:
    """Run a single seed and return the path to its results JSON."""
    spec = EXPERIMENTS[experiment]
    cmd = [
        sys.executable,
        str(REPO / spec["script"]),
        "--seed", str(seed),
        "--save_dir", str(save_dir),
        *common_args,
    ]
    print(f"\n>>> seed={seed}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO)
    if proc.returncode != 0:
        raise RuntimeError(f"seed={seed} failed with returncode {proc.returncode}")
    return save_dir / spec["results"]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _flatten_metrics(payload: dict) -> Dict[str, float]:
    """Flatten a results JSON into ``{path.to.metric: value}`` for scalar floats only."""
    out: Dict[str, float] = {}

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{prefix}[{i}]")
        else:
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                # skip noisy keys (epochs, history details) — keep only metric-like names
                if any(
                    seg in prefix
                    for seg in (
                        "f1_macro", "accuracy", "auroc", "cohen_kappa", "ece",
                        "f1_gap", "tracking_correlation", "drop_ratio",
                    )
                ):
                    out[prefix] = float(obj)

    walk(payload)
    return out


def aggregate(result_paths: List[Path]) -> None:
    runs = []
    for p in result_paths:
        if not p.is_file():
            print(f"[run_seeds] WARNING: missing {p}")
            continue
        with p.open() as f:
            runs.append(_flatten_metrics(json.load(f)))
    if not runs:
        print("[run_seeds] No results to aggregate.")
        return

    all_keys = sorted({k for r in runs for k in r})
    print("\n=== Multi-seed aggregate ===")
    header = f"{'metric':<60}  {'mean':>10}  {'std':>10}  {'n':>3}"
    print(header)
    print("-" * len(header))
    for k in all_keys:
        vals = [r[k] for r in runs if k in r]
        if not vals:
            continue
        mu = stats.fmean(vals)
        sd = stats.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"{k:<60}  {mu:>10.4f}  {sd:>10.4f}  {len(vals):>3}")


# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=sorted(EXPERIMENTS), help="Which experiment to run.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--save_root", type=Path, default=Path("./runs"),
                   help="Per-seed dirs will be created under here as <save_root>_seed<S>")
    p.add_argument("--data_root", type=Path, default=None,
                   help="Required unless --aggregate_only is used.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Any further args passed verbatim to the experiment script (e.g. --label_mode video_majority)",
    )
    p.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Skip training. Only read existing seed result JSONs under "
             "<save_root>_seed<S>/ and print the aggregate table.",
    )
    args = p.parse_args()

    spec = EXPERIMENTS[args.experiment]

    if args.aggregate_only:
        result_paths = [Path(f"{args.save_root}_seed{s}") / spec["results"] for s in args.seeds]
        missing = [p for p in result_paths if not p.is_file()]
        if missing:
            print(f"[run_seeds] WARNING: {len(missing)} seed(s) have no results JSON yet:")
            for m in missing:
                print(f"    {m}")
        aggregate(result_paths)
        return

    if args.data_root is None:
        p.error("--data_root is required unless --aggregate_only is set.")

    common = [
        "--data_root", str(args.data_root),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        *(args.extra or []),
    ]

    result_paths: List[Path] = []
    for s in args.seeds:
        seed_dir = Path(f"{args.save_root}_seed{s}")
        seed_dir.mkdir(parents=True, exist_ok=True)
        try:
            rp = run_one(args.experiment, s, seed_dir, common)
            result_paths.append(rp)
        except RuntimeError as e:
            print(f"[run_seeds] {e}", file=sys.stderr)

    aggregate(result_paths)


if __name__ == "__main__":
    main()
