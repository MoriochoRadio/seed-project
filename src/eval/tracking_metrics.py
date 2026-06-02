"""Tracking metrics: IDF1, ID switches, mean track duration.

These are *track-level* aggregates rather than full MOTA/HOTA implementations
(which depend on ground-truth identity assignments not always present in
VISEM-Tracking). When ground-truth is unavailable, IDF1 becomes a proxy based
on track-fragmentation rate — see ``idf1_proxy``.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np


def count_id_switches(tracks_per_frame: Sequence[Sequence[int]]) -> int:
    """A track id is considered switched when an object that was assigned id A
    in frame t gets a different id in frame t+1 — measured by greedy NN matching
    between consecutive frames. Caller passes track ids per frame ordered by the
    same matching used in the experiment.
    """
    switches = 0
    prev = None
    for ids in tracks_per_frame:
        if prev is not None:
            for i, j in zip(prev, ids):
                if i != j and i is not None and j is not None:
                    switches += 1
        prev = ids
    return switches


def mean_track_duration(tracks: Iterable) -> float:
    """Average length (in frames) of completed tracks."""
    lengths = [t.length for t in tracks if hasattr(t, "length")]
    return float(np.mean(lengths)) if lengths else 0.0


def idf1(
    pred_tracks: Sequence[Sequence[Tuple[int, int]]],   # list of (frame, gt_id)
    gt_tracks: Sequence[Sequence[Tuple[int, int]]],
) -> float:
    """Standard IDF1 implementation (Ristani et al. 2016, simplified).

    Each *track* is represented as a list of (frame, gt_id) pairs after
    resolving a 1-to-1 assignment between predicted and ground-truth tracks
    that maximises ID-true positives.
    """
    if not pred_tracks or not gt_tracks:
        return 0.0

    # Build cost matrix: -ID-TP (we want to maximise TP via Hungarian)
    cost = np.zeros((len(pred_tracks), len(gt_tracks)), dtype=np.float32)
    for i, p in enumerate(pred_tracks):
        p_set = set(p)
        for j, g in enumerate(gt_tracks):
            cost[i, j] = -len(p_set.intersection(set(g)))

    # Hungarian assignment
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
    except Exception:  # pragma: no cover
        # Fallback: greedy
        row_ind, col_ind = [], []
        used_p, used_g = set(), set()
        flat = np.argsort(cost, axis=None)
        for k in flat:
            i, j = np.unravel_index(k, cost.shape)
            if i in used_p or j in used_g:
                continue
            row_ind.append(i)
            col_ind.append(j)
            used_p.add(i)
            used_g.add(j)

    id_tp = -cost[row_ind, col_ind].sum()
    id_fp = sum(len(p) for p in pred_tracks) - id_tp
    id_fn = sum(len(g) for g in gt_tracks) - id_tp
    if id_tp + 0.5 * (id_fp + id_fn) == 0:
        return 0.0
    return float(id_tp / (id_tp + 0.5 * (id_fp + id_fn)))


def idf1_proxy(built_tracks: Iterable) -> float:
    """When GT track ids are unavailable: proxy based on fragmentation/length.

    Encourages long, unfragmented tracks. Range ≈ [0, 1].
    """
    tracks = list(built_tracks)
    if not tracks:
        return 0.0
    frag_rate = sum(1 for t in tracks if getattr(t, "fragmented", False)) / len(tracks)
    avg_len = mean_track_duration(tracks)
    length_score = min(1.0, avg_len / 60.0)        # 60 frames ≈ 2s @ 30fps
    return float(0.7 * length_score + 0.3 * (1 - frag_rate))
