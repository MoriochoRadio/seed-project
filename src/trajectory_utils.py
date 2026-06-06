"""Trajectory-level utilities: smoothing, padding, augmentation, density."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.fft import dct, idct

# --------------------------------------------------------------------------- #
# Smoothing
# --------------------------------------------------------------------------- #

def smooth_trajectory_dct(xy: np.ndarray, n_keep: int = 12) -> np.ndarray:
    """DCT-based low-pass smoothing (Shahali et al. 2026, DCT-12).

    Parameters
    ----------
    xy : np.ndarray, shape (T, 2)
    n_keep : int, number of leading DCT coefficients to retain.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (T, 2)")
    T = xy.shape[0]
    if T <= n_keep:
        return xy.copy()
    coefs = dct(xy, axis=0, norm="ortho")
    coefs[n_keep:] = 0.0
    return idct(coefs, axis=0, norm="ortho")

# --------------------------------------------------------------------------- #
# Padding / cropping (fixed-length input for DL models)
# --------------------------------------------------------------------------- #

def pad_or_crop(seq: np.ndarray, length: int, pad_value: float = 0.0) -> np.ndarray:
    """Pad with `pad_value` at the end, or right-crop to `length` frames."""
    T = seq.shape[0]
    if T == length:
        return seq.astype(np.float32, copy=False)
    if T > length:
        return seq[:length].astype(np.float32, copy=False)
    pad_shape = (length - T, *seq.shape[1:])
    return np.concatenate(
        [seq, np.full(pad_shape, pad_value, dtype=seq.dtype)], axis=0
    ).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Augmentation (training-time)
# --------------------------------------------------------------------------- #

@dataclass
class AugmentConfig:
    rotation_deg: float = 15.0
    jitter_std: float = 0.5         # pixels
    occlusion_prob: float = 0.1     # per-frame drop probability
    crossing_prob: float = 0.05     # add a synthetic ID-switch-like jump


def augment_trajectory(
    xy: np.ndarray,
    cfg: Optional[AugmentConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply rotation, jitter, occlusion (frame zero-out), and crossing."""
    cfg = cfg or AugmentConfig()
    rng = rng or np.random.default_rng()
    out = xy.astype(np.float32).copy()

    # 1) random rotation around centroid
    theta = math.radians(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    centroid = out.mean(axis=0, keepdims=True)
    out = (out - centroid) @ R.T + centroid

    # 2) Gaussian jitter
    out += rng.normal(0.0, cfg.jitter_std, size=out.shape).astype(np.float32)

    # 3) random occlusion: zero some frames (model handles missing via mask)
    if cfg.occlusion_prob > 0.0:
        drop = rng.random(out.shape[0]) < cfg.occlusion_prob
        out[drop] = 0.0

    # 4) crossing-like jump: insert one ID-switch by swapping a tail
    if cfg.crossing_prob > 0.0 and rng.random() < cfg.crossing_prob and out.shape[0] > 8:
        cut = rng.integers(low=4, high=out.shape[0] - 4)
        delta = rng.normal(0.0, 5.0, size=(2,)).astype(np.float32)
        out[cut:] += delta
    return out


# --------------------------------------------------------------------------- #
# Density estimation
# --------------------------------------------------------------------------- #

def estimate_density(
    bboxes_per_frame: list[np.ndarray],
    frame_area: float,
) -> np.ndarray:
    """Average #bboxes per frame normalised by area (cells per unit area).

    Returns
    -------
    np.ndarray, shape (T,) — density per frame
    """
    return np.array(
        [(b.shape[0] / frame_area) if frame_area > 0 else 0.0 for b in bboxes_per_frame],
        dtype=np.float32,
    )