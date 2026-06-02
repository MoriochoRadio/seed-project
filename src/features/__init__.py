"""Public feature API.

CASA features depend only on NumPy. The richer trajectory features (curvature,
turning angles, spectral energy) require ``scipy.fft``; we import them lazily
so that ``compute_casa_features`` is usable in environments without scipy
(e.g., for quick smoke tests).
"""

from .casa_features import compute_casa_features, CASA_NAMES  # noqa: F401


def __getattr__(name):  # PEP 562 lazy attribute access
    if name in {"velocity_features", "curvature", "spectral_energy", "turning_angles"}:
        from . import trajectory_features as _tf  # local import requires scipy
        return getattr(_tf, name)
    raise AttributeError(f"module 'src.features' has no attribute {name!r}")
