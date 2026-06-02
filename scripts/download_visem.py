"""Download VISEM-Tracking dataset from Kaggle.

VISEM-Tracking layout (after extraction)
----------------------------------------
visem_tracking/
├── Train/
│   ├── 11/
│   │   ├── 11.mp4
│   │   ├── labels/                # YOLO-format bboxes per frame
│   │   │   ├── 11_frame_0.txt
│   │   │   └── ...
│   │   └── 11.csv                 # WHO motility class for the video
│   └── ...
└── Test/
    └── ...

Usage
-----
$ pip install kagglehub
$ export KAGGLE_USERNAME=...; export KAGGLE_KEY=...     # OR ~/.kaggle/kaggle.json
$ python scripts/download_visem.py --target ./data

Notes
-----
- 첫 다운로드는 ~수 GB가 소요됩니다.
- 인증 실패 시 https://www.kaggle.com/account → "Create New API Token" 으로
  kaggle.json 을 받아 ~/.kaggle/kaggle.json 에 두세요.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


KAGGLE_DATASET = "vlbthambawita/visemtracking"


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_via_kagglehub(target: Path) -> Path:
    """Download via the kagglehub package (preferred)."""
    import kagglehub  # type: ignore

    print(f"[download_visem] kagglehub.dataset_download('{KAGGLE_DATASET}') ...")
    cached = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"[download_visem] cached at: {cached}")

    _ensure_dir(target)
    # Copy / link the cached dataset to the desired target dir
    final = target / "visem_tracking"
    if final.exists():
        print(f"[download_visem] target already exists: {final}")
        return final
    print(f"[download_visem] copying to {final} ...")
    shutil.copytree(cached, final)
    return final


def download_via_kaggle_cli(target: Path) -> Path:
    """Fallback: use the kaggle CLI via subprocess."""
    import subprocess

    _ensure_dir(target)
    final = target / "visem_tracking"
    if final.exists():
        return final
    print("[download_visem] using `kaggle datasets download`")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            KAGGLE_DATASET,
            "-p",
            str(target),
            "--unzip",
        ],
        check=True,
    )
    # The CLI extracts directly into `target`; rename root if needed
    return final if final.exists() else target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("./data"),
        help="다운로드 대상 디렉토리 (default: ./data)",
    )
    parser.add_argument(
        "--method",
        choices=["kagglehub", "cli"],
        default="kagglehub",
        help="다운로드 방법",
    )
    args = parser.parse_args()

    try:
        if args.method == "kagglehub":
            path = download_via_kagglehub(args.target)
        else:
            path = download_via_kaggle_cli(args.target)
    except Exception as exc:  # pragma: no cover
        print(f"[download_visem] failed: {exc}", file=sys.stderr)
        print(
            "[download_visem] 인증을 확인하세요:"
            " ~/.kaggle/kaggle.json 또는 KAGGLE_USERNAME/KAGGLE_KEY",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[download_visem] done -> {path}")
    print(
        "다음 단계:"
        f"\n  python tests/test_pipeline.py"
        f"\n  python experiments/exp1_crowded_robustness.py --data_root {path}"
    )


if __name__ == "__main__":
    main()
