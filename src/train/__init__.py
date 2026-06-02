"""Training utilities.

`EarlyStopping` is torch-free and importable without torch installed.
The full Trainer/TrainConfig pull in torch lazily.
"""

from .early_stopping import EarlyStopping  # noqa: F401


def __getattr__(name):
    if name in {"Trainer", "TrainConfig", "build_model_from_config"}:
        from . import trainer as _t
        return getattr(_t, name)
    raise AttributeError(f"module 'src.train' has no attribute {name!r}")
