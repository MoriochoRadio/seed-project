"""Evaluation utilities.

``metrics`` requires scikit-learn; ``tracking_metrics`` only needs
NumPy/SciPy. We import the two lazily so the cheap path stays cheap.
"""

from .tracking_metrics import idf1, count_id_switches, mean_track_duration, idf1_proxy  # noqa: F401


def __getattr__(name):
    if name in {"classification_report", "expected_calibration_error"}:
        from . import metrics as _m
        return getattr(_m, name)
    raise AttributeError(f"module 'src.eval' has no attribute {name!r}")
