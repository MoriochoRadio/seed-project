"""Classical CASA kinematic features + PAW (Path Average Width).

Definitions follow Shahali et al. 2026 / WHO 2010:
    VCL  : curvilinear velocity        — total path length / duration  (raw xy)
    VSL  : straight-line velocity      — start→end displacement / duration
    VAP  : average-path velocity       — smoothed path length / duration
    LIN  : linearity                   — VSL / VCL
    STR  : straightness                — VSL / VAP
    WOB  : wobble                      — VAP / VCL
    ALH  : amplitude of lateral head displacement (smoothed-path frame)
    BCF  : beat-cross frequency (zero-crossings of lateral component)
    PAW  : path average width (Shahali 2026): mean lateral excursion of raw
           trajectory around the smoothed average path.

All velocity outputs are in pixels/second; convert externally to µm/s using
microscope calibration if needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


CASA_NAMES = ("VCL", "VSL", "VAP", "LIN", "STR", "WOB", "ALH", "BCF", "PAW")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > 1e-9 else 0.0


def _path_length(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _project_lateral(raw: np.ndarray, smooth: np.ndarray) -> np.ndarray:
    """Signed lateral distance of raw points from the smoothed local tangent."""
    n = min(raw.shape[0], smooth.shape[0])
    raw, smooth = raw[:n], smooth[:n]
    # local tangent of the smoothed path (forward differences, last copies prev)
    tangent = np.zeros_like(smooth)
    tangent[:-1] = np.diff(smooth, axis=0)
    tangent[-1] = tangent[-2] if n >= 2 else 0
    norm = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-9
    tangent = tangent / norm
    # lateral = perpendicular component
    diff = raw - smooth
    perp = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    lateral = (diff * perp).sum(axis=1)
    return lateral


def compute_casa_features(
    smooth_xy: np.ndarray,
    raw_xy: Optional[np.ndarray] = None,
    fps: int = 30,
) -> np.ndarray:
    """Return CASA features in the order of ``CASA_NAMES``.

    Parameters
    ----------
    smooth_xy : (T, 2) — smoothed (e.g., DCT-12) trajectory used for VAP/path
    raw_xy    : (T, 2) — original (un-smoothed) coordinates. If ``None``,
                ``smooth_xy`` is used (PAW will then be 0).
    fps       : sampling rate, used to convert per-frame distances to per-second
    """
    if smooth_xy.ndim != 2 or smooth_xy.shape[1] != 2:
        raise ValueError("smooth_xy must be (T, 2)")
    if raw_xy is None:
        raw_xy = smooth_xy

    T = smooth_xy.shape[0]
    duration = max(T - 1, 1) / float(fps)

    # VCL — based on raw trajectory length
    vcl = _path_length(raw_xy) / duration

    # VSL — start to end of raw trajectory
    vsl = _safe_div(
        float(np.linalg.norm(raw_xy[-1] - raw_xy[0])) if T >= 2 else 0.0,
        duration,
    )

    # VAP — based on smoothed path
    vap = _path_length(smooth_xy) / duration

    lin = _safe_div(vsl, vcl)
    stra = _safe_div(vsl, vap)
    wob = _safe_div(vap, vcl)

    # ALH — max lateral deviation; PAW — mean of |lateral|
    lateral = _project_lateral(raw_xy, smooth_xy) if T >= 2 else np.zeros(0)
    alh = float(np.max(np.abs(lateral))) if lateral.size else 0.0
    paw = float(np.mean(np.abs(lateral))) if lateral.size else 0.0

    # BCF — zero-crossings of lateral signal per second
    if lateral.size >= 2:
        zc = np.sum(np.sign(lateral[:-1]) * np.sign(lateral[1:]) < 0)
        bcf = float(zc) / duration
    else:
        bcf = 0.0

    return np.array(
        [vcl, vsl, vap, lin, stra, wob, alh, bcf, paw],
        dtype=np.float32,
    )
