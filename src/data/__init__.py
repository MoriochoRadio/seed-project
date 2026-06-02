"""Data loading utilities.

``VISEMTrackingDataset`` requires PyTorch, so we expose it lazily for
environments where torch is unavailable (smoke tests, CI without torch).
"""

from .trajectory_utils import (  # noqa: F401
    smooth_trajectory_dct,
    pad_or_crop,
    augment_trajectory,
    estimate_density,
)


def __getattr__(name):
    if name in {"VISEMTrackingDataset", "TrajectoryRecord"}:
        from . import visem_dataset as _vd
        return getattr(_vd, name)
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
