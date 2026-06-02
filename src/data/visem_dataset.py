"""VISEM-Tracking dataset loader (auto-detecting layout).

The VISEM-Tracking dataset is distributed in several slightly-different layouts
depending on which release / source you used.  This loader auto-detects the
layout by looking for a video folder that contains a ``labels/`` directory of
YOLO ``.txt`` files.

Supported layouts
-----------------
1) Train/Test split with explicit folders::

       <root>/Train/<video_id>/labels/<video_id>_frame_<N>.txt
       <root>/Train/<video_id>/<video_id>.mp4
       <root>/Train/<video_id>/<video_id>.csv

2) Same as (1) but the user already extracted only one split, OR everything is
   under a single root::

       <root>/<video_id>/labels/<video_id>_frame_<N>.txt
       <root>/<video_id>/images/<video_id>_frame_<N>.jpg

3) The Kaggle release that wraps the data in an extra folder, e.g.::

       <root>/VISEM_Tracking_Train_v4/Train/<video_id>/labels/...

The auto-detection walks at most 3 levels under ``root`` searching for any
directory that contains a ``labels/`` subdirectory of ``.txt`` files.

YOLO bbox file rows: ``<class> <cx> <cy> <w> <h>`` (all normalised to [0, 1]).
An optional 6th column with track-id is honoured if present.

Each sample returned by ``__getitem__`` is a dict with:
    * trajectory   : torch.Tensor (T, 2)  — pixel-space x, y
    * velocity     : torch.Tensor (T, 2)
    * mask         : torch.Tensor (T,)    — 1 = real frame, 0 = padded
    * confidence   : torch.Tensor ()      — tracking confidence in [0, 1]
    * density      : torch.Tensor (T,)    — #detections per frame
    * label        : torch.Tensor ()      — WHO motility class id
    * hyper        : torch.Tensor ()      — synthetic hyperactivation flag (0/1)
    * casa         : torch.Tensor (9,)    — VCL/VSL/VAP/LIN/STR/WOB/ALH/BCF/PAW
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .trajectory_utils import pad_or_crop, augment_trajectory, AugmentConfig
from ._visem_layout import (
    find_video_dirs as _find_video_dirs,
    split_for as _split_for,
    label_dir_of as _label_dir_of,
    frame_index_from_filename as _frame_index_from_filename,
    looks_like_video_dir as _looks_like_video_dir,
)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

WHO_CLASSES = ("progressive", "non_progressive", "immotile")
WHO2IDX = {c: i for i, c in enumerate(WHO_CLASSES)}

# Layout detection helpers live in ``_visem_layout.py`` (torch-free).


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

@dataclass
class TrajectoryRecord:
    trajectory: np.ndarray
    velocity: np.ndarray
    mask: np.ndarray
    confidence: float
    density: np.ndarray
    label: int
    hyper: int
    casa: np.ndarray
    meta: Dict[str, str]


def _parse_yolo_label(path: Path) -> np.ndarray:
    """Parse a YOLO label file → (N, 6) [class, cx, cy, w, h, id]."""
    if not path.is_file():
        return np.zeros((0, 6), dtype=np.float32)
    rows: List[List[float]] = []
    try:
        with path.open() as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    rows.append([float(x) for x in parts[:6]])
                except ValueError:
                    continue
    except (OSError, UnicodeDecodeError):
        return np.zeros((0, 6), dtype=np.float32)
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    arr = np.zeros((len(rows), 6), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            arr[i, j] = v
        if len(r) < 6:
            arr[i, 5] = -1.0  # no track-id
    return arr


def _read_motility_csv(path: Path) -> Dict[str, float]:
    """Read VISEM motility csv → mapping of metric→value."""
    if not path.is_file():
        return {}
    try:
        with path.open() as f:
            reader = csv.reader(f)
            header = next(reader, None)
            row = next(reader, None)
    except (OSError, UnicodeDecodeError):
        return {}
    if not header or not row:
        return {}
    out: Dict[str, float] = {}
    for k, v in zip(header, row):
        try:
            out[k.strip().lower()] = float(v)
        except ValueError:
            out[k.strip().lower()] = v
    return out


def _who_label_from_motility(meta: Dict[str, float]) -> int:
    """Pick the WHO majority class for a video.

    VISEM-Tracking provides percent of progressive / non-progressive / immotile.
    We treat the *majority* class as the video-level label.
    """
    p = float(meta.get("progressive", meta.get("progressive motility", 0)) or 0)
    n = float(
        meta.get("non progressive", meta.get("non_progressive",
        meta.get("non-progressive", meta.get("non progressive motility", 0)))) or 0
    )
    i = float(meta.get("immotile", 0) or 0)
    arr = np.array([p, n, i])
    if arr.sum() == 0:
        return 2  # immotile fallback
    return int(arr.argmax())


# --------------------------------------------------------------------------- #
# Tracking-aware nearest-neighbour linker (used when track ids are absent)
# --------------------------------------------------------------------------- #

def _link_detections(
    detections: List[np.ndarray],
    max_distance: float = 0.05,
) -> List[List[Tuple[int, int]]]:
    """Naive frame-to-frame NN linker → list of tracks of (frame, idx) pairs."""
    tracks: List[List[Tuple[int, int]]] = []
    last_pos: List[Tuple[int, np.ndarray]] = []  # (track_index, last xy)

    for f, det in enumerate(detections):
        if det.shape[0] == 0:
            continue
        xy = det[:, 1:3]
        used: set = set()
        for t_idx, last_xy in list(last_pos):
            if not xy.size:
                break
            dists = np.linalg.norm(xy - last_xy, axis=1)
            j = int(np.argmin(dists))
            if j in used or dists[j] > max_distance:
                continue
            tracks[t_idx].append((f, j))
            used.add(j)
            last_pos[last_pos.index((t_idx, last_xy))] = (t_idx, xy[j])
        for j in range(xy.shape[0]):
            if j in used:
                continue
            tracks.append([(f, j)])
            last_pos.append((len(tracks) - 1, xy[j]))
    return tracks


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class VISEMTrackingDataset(Dataset):
    """Iterable of fixed-length sperm trajectories with tracking-aware features.

    Parameters
    ----------
    root : Path-like, dataset root.  Layout is auto-detected.
    split : "Train" | "Test" | "All"
        ``All`` returns every video found regardless of any Train/Test parent;
        any other value filters by parent directory name (case-insensitive).
    fps : effective fps (default 30, matches VISEM-Tracking).
    window_seconds : window length in seconds → used as fixed input length.
    image_size : (H, W) used to convert YOLO normalised coords to pixels.
    augment : enable augmentation (training only).
    min_track_length : minimum #frames before a trajectory is included.
    use_synthetic_hyperactivation : derive hyperactivation flag from VCL+ALH.
    label_mode : "trajectory_kinematic" | "video_majority"
        How class labels are assigned.

        * ``"trajectory_kinematic"`` (default & recommended) — derives a label
          per *individual trajectory* from its CASA-style features (VCL/LIN).
          This produces meaningful per-track classes which is what the paper's
          motility-classification experiments actually need.
        * ``"video_majority"`` — assigns each trajectory the WHO majority class
          of its parent video (legacy behaviour).  Use only if you know what
          you're doing; this collapses many distinct tracks to a single label.
    verbose : print discovered video directories + class distribution on construction.
    """

    def __init__(
        self,
        root: os.PathLike,
        split: str = "Train",
        fps: int = 30,
        window_seconds: float = 1.0,
        image_size: Tuple[int, int] = (1080, 1920),
        augment: bool = False,
        min_track_length: int = 15,
        use_synthetic_hyperactivation: bool = True,
        max_videos: Optional[int] = None,
        verbose: bool = True,
        label_mode: str = "trajectory_kinematic",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.fps = fps
        self.window = int(round(window_seconds * fps))
        self.image_size = image_size
        self.augment = augment
        self.min_track_length = min_track_length
        self.use_synthetic_hyper = use_synthetic_hyperactivation
        self.aug_cfg = AugmentConfig()
        self.verbose = verbose
        if label_mode not in ("trajectory_kinematic", "video_majority"):
            raise ValueError(
                f"label_mode must be 'trajectory_kinematic' or 'video_majority', got {label_mode!r}"
            )
        self.label_mode = label_mode

        self._records: List[Dict] = []
        self._discovered_video_dirs: List[Path] = []
        self._index_videos(max_videos=max_videos)
        if self.label_mode == "trajectory_kinematic":
            self._assign_trajectory_labels()
        if self.verbose:
            self._print_class_distribution()

    # --------------------------------------------------------------------- #

    @property
    def num_classes(self) -> int:
        return len(WHO_CLASSES)

    def __len__(self) -> int:
        return len(self._records)

    # --------------------------------------------------------------------- #
    # Indexing
    # --------------------------------------------------------------------- #

    def _video_dirs(self) -> List[Path]:
        all_dirs = _find_video_dirs(self.root)
        self._discovered_video_dirs = all_dirs
        if self.split.lower() in ("all", "*"):
            return all_dirs
        wanted = self.split.lower()
        out = [d for d in all_dirs if _split_for(d).lower() == wanted]
        # Heuristic fallback: if user asked for Train but no split folder exists
        # (flat layout), accept everything found.
        if not out:
            split_folders = {_split_for(d).lower() for d in all_dirs}
            if split_folders == {"all"}:
                if self.verbose:
                    print(
                        f"[VISEMTrackingDataset] split='{self.split}' requested but no "
                        f"Train/Test sub-folders found under {self.root}. Falling back "
                        f"to all {len(all_dirs)} discovered video folders."
                    )
                return all_dirs
        return out

    def _index_videos(self, max_videos: Optional[int]) -> None:
        from .trajectory_utils import estimate_density  # local import to avoid cycles

        H, W = self.image_size
        frame_area = float(H * W)

        video_dirs = self._video_dirs()
        if max_videos is not None:
            video_dirs = video_dirs[:max_videos]

        if self.verbose:
            print(
                f"[VISEMTrackingDataset] root={self.root}  split={self.split}  "
                f"videos_found={len(video_dirs)}  total_discovered={len(self._discovered_video_dirs)}"
            )
            if video_dirs[:3]:
                preview = "  ".join(str(p.relative_to(self.root)) for p in video_dirs[:3])
                print(f"[VISEMTrackingDataset] e.g. {preview}")

        if not video_dirs:
            self._print_empty_diagnostic()
            return

        for video_dir in video_dirs:
            video_id = video_dir.name
            label_dir = _label_dir_of(video_dir)
            if label_dir is None:
                continue

            # Collect per-frame detections in frame order.
            # Tolerate any of the three filename patterns above.
            candidates: List[Tuple[int, Path]] = []
            for fp in label_dir.glob("*.txt"):
                idx = _frame_index_from_filename(fp.stem)
                if idx is None:
                    continue
                candidates.append((idx, fp))
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[0])

            detections: List[np.ndarray] = []
            frame_indices: List[int] = []
            for idx, fp in candidates:
                detections.append(_parse_yolo_label(fp))
                frame_indices.append(idx)

            # Trajectories: prefer track-ids in column 5; otherwise NN linking.
            has_ids = any((det[:, 5] >= 0).any() for det in detections if det.shape[0])
            tracks: Dict[int, List[Tuple[int, np.ndarray]]] = {}
            if has_ids:
                for f, det in enumerate(detections):
                    for row in det:
                        tid = int(row[5])
                        if tid < 0:
                            continue
                        cx, cy = row[1] * W, row[2] * H
                        tracks.setdefault(tid, []).append(
                            (frame_indices[f], np.array([cx, cy], np.float32))
                        )
            else:
                linked = _link_detections(detections)
                for k, link in enumerate(linked):
                    pts = []
                    for f, j in link:
                        cx = detections[f][j, 1] * W
                        cy = detections[f][j, 2] * H
                        pts.append((frame_indices[f], np.array([cx, cy], np.float32)))
                    tracks[k] = pts

            density = estimate_density(detections, frame_area)
            # Try a few CSV name variants
            motility_csv = None
            for cand in (video_dir / f"{video_id}.csv",
                         video_dir / "motility.csv",
                         video_dir / f"{video_id}_motility.csv"):
                if cand.is_file():
                    motility_csv = cand
                    break
            meta = _read_motility_csv(motility_csv) if motility_csv else {}
            label = _who_label_from_motility(meta)

            for tid, pts in tracks.items():
                if len(pts) < self.min_track_length:
                    continue
                self._records.append(
                    {
                        "video_id": video_id,
                        "track_id": tid,
                        "frames": np.array([f for f, _ in pts], dtype=np.int32),
                        "xy": np.stack([p for _, p in pts], axis=0),
                        "density": density,
                        "label": label,
                        "split": _split_for(video_dir),
                    }
                )

        if self.verbose:
            print(
                f"[VISEMTrackingDataset] indexed {len(self._records)} trajectory records "
                f"(min_track_length={self.min_track_length})"
            )

    # --------------------------------------------------------------------- #

    # --------------------------------------------------------------------- #
    # Per-trajectory kinematic labelling
    # --------------------------------------------------------------------- #

    def _assign_trajectory_labels(self) -> None:
        """Replace the (video-majority) label on each record with a per-trajectory
        WHO-style label derived from kinematic features.

        Strategy (no µm-calibration assumed; uses dataset-relative thresholds):
            1. Compute VCL & LIN for every record from its smoothed trajectory.
            2. Define the *immotile* threshold as the 33rd percentile of VCL.
            3. Records below that threshold are immotile (class 2).
            4. Among the remaining (motile) records, those with LIN ≥ 0.5 are
               progressive (class 0); the rest are non-progressive (class 1).

        This produces a class-balanced labelling that lets a classifier learn
        progressive-vs-tumbling-vs-immotile from the trajectory shape itself.
        """
        if not self._records:
            return
        from .trajectory_utils import smooth_trajectory_dct
        from ..features.casa_features import compute_casa_features

        vcls: List[float] = []
        lins: List[float] = []
        for rec in self._records:
            xy = rec["xy"].astype(np.float32)
            sm = smooth_trajectory_dct(xy, n_keep=12)
            casa = compute_casa_features(sm, raw_xy=xy, fps=self.fps)
            vcls.append(float(casa[0]))
            lins.append(float(casa[3]))
        vcls_arr = np.asarray(vcls, dtype=np.float32)
        lins_arr = np.asarray(lins, dtype=np.float32)

        # Avoid degenerate quantile when all VCLs are identical
        if vcls_arr.std() < 1e-6:
            vcl_immotile_thr = vcls_arr.min() - 1.0
        else:
            vcl_immotile_thr = float(np.quantile(vcls_arr, 1 / 3))

        for rec, vcl, lin in zip(self._records, vcls_arr, lins_arr):
            if vcl < vcl_immotile_thr:
                rec["label"] = 2  # immotile
            elif lin >= 0.5:
                rec["label"] = 0  # progressive
            else:
                rec["label"] = 1  # non-progressive

        if self.verbose:
            print(
                f"[VISEMTrackingDataset] kinematic labelling: "
                f"VCL p33={vcl_immotile_thr:.2f}, LIN cutoff=0.5"
            )

    def _print_class_distribution(self) -> None:
        if not self._records:
            return
        from collections import Counter
        cnt = Counter(int(r["label"]) for r in self._records)
        total = sum(cnt.values())
        names = WHO_CLASSES
        parts = [f"{names[i]}={cnt.get(i, 0)} ({cnt.get(i, 0) / total * 100:.1f}%)" for i in range(len(names))]
        print(f"[VISEMTrackingDataset] class distribution ({total} tracks): {' | '.join(parts)}")

    def _print_empty_diagnostic(self) -> None:
        """Helpful message when no video directories were found."""
        msg = [
            f"[VISEMTrackingDataset] WARNING: no video directories found under {self.root}.",
            "  A 'video directory' is one that contains a 'labels/' sub-folder of .txt files.",
            "  Tip: run `python scripts/inspect_visem.py --root <your-root>` to see exactly",
            "  what is on disk; then pass that root (or a subfolder of it) to --data_root.",
        ]
        # Show a few top-level entries to help the user
        try:
            top = sorted([p.name for p in self.root.iterdir()])[:15]
            if top:
                msg.append(f"  top-level entries at root: {top}")
        except OSError:
            pass
        print("\n".join(msg))

    # --------------------------------------------------------------------- #
    # Tracking confidence
    # --------------------------------------------------------------------- #

    @staticmethod
    def _tracking_confidence(frames: np.ndarray, xy: np.ndarray) -> float:
        """Heuristic confidence in [0, 1]:
        - longer continuous tracks → higher
        - smaller velocity jitter  → higher
        - fewer frame gaps         → higher
        """
        if len(frames) < 2:
            return 0.0
        gaps = np.diff(frames)
        gap_score = 1.0 / (1.0 + (gaps - 1).clip(min=0).mean())
        v = np.diff(xy, axis=0)
        speeds = np.linalg.norm(v, axis=1)
        if speeds.size and speeds.std() > 1e-6:
            jitter_score = 1.0 / (1.0 + speeds.std() / (speeds.mean() + 1e-6))
        else:
            jitter_score = 1.0
        length_score = min(1.0, len(frames) / 30.0)
        return float(0.5 * length_score + 0.3 * gap_score + 0.2 * jitter_score)

    # --------------------------------------------------------------------- #

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from ..features.casa_features import compute_casa_features
        from .trajectory_utils import smooth_trajectory_dct

        rec = self._records[idx]
        xy = rec["xy"].astype(np.float32)
        frames = rec["frames"]

        # Optional augmentation (training)
        if self.augment:
            xy = augment_trajectory(xy, self.aug_cfg)

        # Smoothed copy used for CASA feature extraction
        xy_smooth = smooth_trajectory_dct(xy, n_keep=12)
        casa = compute_casa_features(xy_smooth, raw_xy=xy, fps=self.fps)

        # Synthetic hyperactivation flag (VCL > 150 µm/s & ALH > 7 µm)
        hyper = int(self.use_synthetic_hyper and casa[0] > 150.0 and casa[6] > 7.0)

        # Velocity & mask
        v = np.zeros_like(xy)
        if xy.shape[0] > 1:
            v[1:] = np.diff(xy, axis=0)

        # Density per frame, aligned to this trajectory's frame indices
        density_frames = rec["density"]
        if density_frames.size:
            safe_frames = np.clip(frames, 0, density_frames.size - 1)
            per_frame_density = density_frames[safe_frames]
        else:
            per_frame_density = np.zeros_like(frames, dtype=np.float32)

        # Pad / crop to fixed window
        T = self.window
        xy_p = pad_or_crop(xy, T)
        v_p = pad_or_crop(v, T)
        d_p = pad_or_crop(per_frame_density.astype(np.float32), T)
        mask = np.zeros((T,), dtype=np.float32)
        mask[: min(T, xy.shape[0])] = 1.0

        confidence = self._tracking_confidence(frames, xy)

        sample = {
            "trajectory": torch.from_numpy(xy_p),               # (T, 2)
            "velocity":   torch.from_numpy(v_p),                # (T, 2)
            "mask":       torch.from_numpy(mask),               # (T,)
            "density":    torch.from_numpy(d_p),                # (T,)
            "confidence": torch.tensor(confidence, dtype=torch.float32),  # ()
            "casa":       torch.from_numpy(casa.astype(np.float32)),      # (9,)
            "label":      torch.tensor(rec["label"], dtype=torch.long),
            "hyper":      torch.tensor(hyper, dtype=torch.long),
            "video_id":   rec["video_id"],
            "track_id":   rec["track_id"],
        }
        return sample
