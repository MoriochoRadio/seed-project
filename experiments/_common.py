"""Shared helpers for experiment scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make ``src`` importable regardless of how the script is launched
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402


# Keys whose values must be float — guards against PyYAML 1.1 reading "1e-4"
# (no decimal point) as a string instead of a number.
_FLOAT_KEYS = {
    "lr", "weight_decay", "min_delta", "min_lr",
    "rotation_deg", "jitter_std", "occlusion_prob", "crossing_prob",
    "iou_threshold", "max_distance", "dropout",
    "motility", "hyperactivation", "kinematic",
}
_INT_KEYS = {
    "batch_size", "epochs", "num_workers", "seed", "patience",
    "fps", "min_track_length", "max_age", "min_hits",
    "cnn_filters", "cnn_depth", "transformer_dim", "transformer_heads",
    "transformer_layers", "casa_dim", "in_channels",
}


def _coerce_numeric(d: Any) -> Any:
    """Recursively coerce known numeric keys to float / int.

    PyYAML 1.1 parses ``1e-4`` (no decimal) as a *string*. This walker fixes
    those silently so downstream code can do ``cfg['train']['weight_decay']``
    without worrying about the YAML version on the user's machine.
    """
    if isinstance(d, dict):
        for k, v in list(d.items()):
            if isinstance(v, (dict, list)):
                d[k] = _coerce_numeric(v)
            elif k in _FLOAT_KEYS and isinstance(v, str):
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
            elif k in _INT_KEYS and isinstance(v, str):
                try:
                    d[k] = int(float(v))
                except ValueError:
                    pass
        return d
    if isinstance(d, list):
        return [_coerce_numeric(x) for x in d]
    return d


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open() as f:
        cfg = yaml.safe_load(f)
    return _coerce_numeric(cfg) if isinstance(cfg, dict) else cfg


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, default=str)


def density_split(records: List[dict], image_size: tuple[int, int]) -> Dict[str, List[dict]]:
    """Partition trajectory records into low/medium/high density bins.

    ``records`` is expected to follow ``VISEMTrackingDataset._records`` schema —
    each entry has a per-frame ``density`` array. We summarise each track by
    *mean density during its lifespan*, then bin by tertiles.
    """
    import numpy as np

    if not records:
        return {"low": [], "medium": [], "high": []}
    summaries = []
    for r in records:
        if r["density"].size == 0:
            summaries.append(0.0)
        else:
            summaries.append(float(r["density"][r["frames"]].mean()))
    summaries = np.asarray(summaries)
    q1, q2 = np.quantile(summaries, [1 / 3, 2 / 3])
    out = {"low": [], "medium": [], "high": []}
    for r, s in zip(records, summaries):
        if s <= q1:
            out["low"].append(r)
        elif s <= q2:
            out["medium"].append(r)
        else:
            out["high"].append(r)
    return out
