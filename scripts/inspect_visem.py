"""Inspect a local copy of the VISEM-Tracking dataset and report what was found.

Usage
-----
    python scripts/inspect_visem.py --root E:/visem_tracking

Prints
------
1) The deepest directory that looks like the dataset root
2) Detected splits (Train / Test / unsplit)
3) Per-split: #video folders, #label files, sample filenames
4) A first-row preview of one label file (so you can verify YOLO format)

Use this when ``VISEMTrackingDataset`` returns 0 records — the output tells you
exactly what the loader sees.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple


LABEL_PATTERNS = [
    re.compile(r".*_frame_(\d+)\.txt$"),
    re.compile(r"frame_?(\d+)\.txt$"),
    re.compile(r"^(\d+)\.txt$"),
]


def find_label_dirs(root: Path) -> List[Path]:
    """Walk under root and collect every directory that contains *.txt label files."""
    out: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        try:
            txts = list(p.glob("*.txt"))
        except (PermissionError, OSError):
            continue
        if txts:
            out.append(p)
    return out


def label_pattern_summary(label_dir: Path) -> Tuple[str, int]:
    """Return (matched-pattern-name, #matched-files) for the heaviest pattern."""
    txts = list(label_dir.glob("*.txt"))
    counts = []
    for pat in LABEL_PATTERNS:
        c = sum(1 for t in txts if pat.match(t.name))
        counts.append((pat.pattern, c))
    counts.sort(key=lambda x: -x[1])
    if counts and counts[0][1] > 0:
        return counts[0]
    return ("(no pattern)", 0)


def detect_splits(root: Path) -> List[str]:
    splits: List[str] = []
    for cand in ("Train", "train", "Test", "test", "Val", "val"):
        if (root / cand).is_dir():
            splits.append(cand)
    return splits


def first_video_id_for(label_dir: Path) -> str:
    """Try to recover a video id (stem prefix) from the first label file."""
    for f in label_dir.glob("*.txt"):
        stem = f.stem
        m = re.match(r"^(.*?)_frame_\d+$", stem)
        if m:
            return m.group(1)
        m = re.match(r"^frame_?(\d+)$", stem)
        if m:
            return label_dir.parent.name  # e.g. parent dir is the id
        return stem
    return "?"


def inspect(root: Path) -> None:
    print(f"== inspect VISEM-Tracking under: {root} ==\n")
    if not root.is_dir():
        print(f"  ! path does not exist or is not a directory")
        return

    # Top-level entries
    top = sorted([p.name for p in root.iterdir()])[:40]
    print(f"top-level entries ({len(top)} shown):")
    for name in top:
        print(f"  - {name}")

    splits = detect_splits(root)
    print(f"\ndetected split folders: {splits or '(none — likely flat layout)'}")

    # Inspect each split (or root if no splits)
    targets = [(root / s, s) for s in splits] or [(root, "(root)")]
    for tgt, name in targets:
        print(f"\n--- {name} : {tgt} ---")
        if not tgt.is_dir():
            print("  (missing)")
            continue
        video_dirs = [p for p in tgt.iterdir() if p.is_dir()]
        print(f"  video subfolders: {len(video_dirs)}")
        if not video_dirs:
            continue
        # Inspect first 3 video dirs
        for vd in video_dirs[:3]:
            print(f"  > {vd.name}/")
            children = [p.name for p in vd.iterdir()][:8]
            for c in children:
                print(f"      {c}")
            label_dirs = [p for p in vd.rglob("*") if p.is_dir() and any(p.glob("*.txt"))]
            for ld in label_dirs[:1]:
                pat, n = label_pattern_summary(ld)
                vid = first_video_id_for(ld)
                first = next(ld.glob("*.txt"), None)
                first_line = ""
                if first is not None:
                    try:
                        with first.open() as f:
                            first_line = f.readline().strip()
                    except Exception:
                        pass
                print(f"      labels:   {ld}")
                print(f"        pattern={pat!r}  matched={n}")
                print(f"        guessed video_id={vid!r}")
                if first is not None:
                    print(f"        sample file: {first.name}")
                    print(f"        sample row : {first_line[:120]}")

    # Aggregate label directory count
    all_label_dirs = find_label_dirs(root)
    print(f"\nTotal directories with .txt files: {len(all_label_dirs)}")
    if all_label_dirs:
        sample = all_label_dirs[0]
        print(f"  example label dir: {sample}")
        pat, n = label_pattern_summary(sample)
        print(f"  example pattern : {pat!r}  matched={n}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    inspect(args.root)


if __name__ == "__main__":
    main()
