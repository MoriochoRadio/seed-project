"""Trajectory-derived features used by the learned-feature branch."""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft


def velocity_features(xy: np.ndarray, fps: int = 30) -> np.ndarray:
    """Per-frame (vx, vy, |v|, ax, ay) features → shape (T, 5)."""
    T = xy.shape[0]
    out = np.zeros((T, 5), dtype=np.float32)
    if T < 2:
        return out
    v = np.zeros_like(xy)
    v[1:] = np.diff(xy, axis=0) * fps
    a = np.zeros_like(xy)
    a[1:] = np.diff(v, axis=0) * fps
    out[:, :2] = v
    out[:, 2] = np.linalg.norm(v, axis=1)
    out[:, 3:] = a
    return out


def curvature(xy: np.ndarray) -> np.ndarray:
    """Discrete curvature κ_t ≈ (x'y'' - y'x'') / (x'^2 + y'^2)^{3/2} → (T,)."""
    T = xy.shape[0]
    if T < 3:
        return np.zeros((T,), dtype=np.float32)
    dx = np.gradient(xy[:, 0])
    dy = np.gradient(xy[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx ** 2 + dy ** 2) ** 1.5 + 1e-9
    return ((dx * ddy - dy * ddx) / denom).astype(np.float32)


def turning_angles(xy: np.ndarray) -> np.ndarray:
    """Frame-to-frame turning angles (radians) → (T-2,)."""
    if xy.shape[0] < 3:
        return np.zeros((0,), dtype=np.float32)
    v = np.diff(xy, axis=0)
    a = v[:-1]
    b = v[1:]
    num = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    den = (a * b).sum(axis=1)
    return np.arctan2(num, den).astype(np.float32)


def spectral_energy(signal: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Energy in `n_bins` evenly-spaced frequency bins of a 1-D signal."""
    if signal.ndim != 1:
        raise ValueError("spectral_energy expects a 1-D signal")
    if signal.size < n_bins:
        return np.zeros((n_bins,), dtype=np.float32)
    spec = np.abs(rfft(signal)) ** 2
    edges = np.linspace(0, spec.size, n_bins + 1, dtype=int)
    return np.array(
        [spec[edges[i]: edges[i + 1]].sum() for i in range(n_bins)],
        dtype=np.float32,
    )
