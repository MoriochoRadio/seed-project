"""Torch-free layout-detection helpers for VISEM-Tracking.

Kept in a separate module so they can be imported / unit-tested without
PyTorch installed (the rest of `visem_dataset.py` requires torch).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


_FRAME_PATTERNS = [
    re.compile(r"^(?P<vid>.+?)_frame_(?P<n>\d+)$"),
    re.compile(r"^frame_?(?P<n>\d+)$"),
    re.compile(r"^(?P<n>\d+)$"),
]


def looks_like_video_dir(p: Path) -> bool:
    """A 'video directory' contains a ``labels/`` subdir of .txt files."""
    if not p.is_dir():
        return False
    for cand in ("labels", "Labels"):
        d = p / cand
        if d.is_dir() and any(d.glob("*.txt")):
            return True
    return False


def find_video_dirs(root: Path, max_depth: int = 4) -> List[Path]:
    """Walk under ``root`` and return all VISEM-style video directories."""
    out: List[Path] = []
    if not root.is_dir():
        return out

    def _walk(d: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            children = list(d.iterdir())
        except (PermissionError, OSError):
            return
        if looks_like_video_dir(d):
            out.append(d)
            return
        for c in children:
            if c.is_dir() and not c.name.startswith("."):
                _walk(c, depth + 1)

    _walk(root, 0)
    return sorted(out, key=lambda p: p.as_posix())


def split_for(video_dir: Path) -> str:
    """Infer split from any 'Train' / 'Test' / 'Val' parent in the path."""
    parts = [p.lower() for p in video_dir.parts]
    for s in ("train", "test", "val"):
        if s in parts:
            return s.capitalize()
    return "All"


def label_dir_of(video_dir: Path) -> Optional[Path]:
    for cand in ("labels", "Labels"):
        d = video_dir / cand
        if d.is_dir():
            return d
    return None


def frame_index_from_filename(stem: str) -> Optional[int]:
    for pat in _FRAME_PATTERNS:
        m = pat.match(stem)
        if m and "n" in m.groupdict():
            try:
                return int(m.group("n"))
            except (TypeError, ValueError):
                continue
    return None
