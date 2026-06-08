"""
hybrid_infer.py
HybridTrajectoryModel을 이용한 per-track 운동성 등급화.

grade_tracks()와 동일한 dict를 반환하므로 pipeline.py 하단은 수정 불필요.

의존:
  src/models/hybrid_model.py  — feature/pm 브랜치에서 복사
  proposed_best.pt            — 학습된 가중치 (HYBRID_CKPT 환경변수 또는 인자)

GPU 필수: CUDA가 없으면 RuntimeError 발생.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── feature/pm에서 복사한 모델 클래스 ─────────────────────────────────────
# 없으면 ImportError → 배포 전 반드시 src/models/hybrid_model.py 확인
from .models.hybrid_model import HybridTrajectoryModel  # type: ignore

# ── 상수 ───────────────────────────────────────────────────────────────────
GRADE_MAP   = {0: "PR", 1: "NP", 2: "IM"}   # cls_head argmax → 등급
CASA_DIM    = 9                               # casa_features.py CASA_NAMES 길이
SEQ_LEN     = 30                              # 1.0 s @ 30 fps (학습 기본값)
TRAJ_DIM    = 4                               # [cx, cy, vx, vy]


def _require_cuda() -> torch.device:
    """GPU가 없으면 즉시 RuntimeError."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[hybrid_infer] CUDA GPU가 필요합니다. "
            "GPU 없이는 HybridModel 추론을 실행할 수 없습니다."
        )
    return torch.device("cuda")


def _build_traj_tensor(pts: list,
                       seq_len: int,
                       fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    track_history 한 트랙 → (seq_len, 4) 궤적 텐서 + 유효 마스크.

    pts : [(frame_idx, cx, cy, w, h), ...]  (frame_idx 정렬 보장 불필요)
    반환: traj (seq_len, 4), mask (seq_len,) bool
    """
    pts_sorted = sorted(pts, key=lambda p: p[0])
    xy = np.array([[cx, cy] for _, cx, cy, *_ in pts_sorted], dtype=np.float32)

    # 속도 계산 (픽셀/프레임 → 픽셀/초)
    if len(xy) > 1:
        vel = np.diff(xy, axis=0, prepend=xy[:1]) * fps
    else:
        vel = np.zeros_like(xy)

    traj = np.concatenate([xy, vel], axis=1)  # (N, 4)

    # seq_len에 맞게 잘라내거나 패딩
    mask = np.zeros(seq_len, dtype=bool)
    n = len(traj)
    if n >= seq_len:
        traj_out = traj[-seq_len:]          # 최신 seq_len 프레임 사용
        mask[:] = True
    else:
        traj_out = np.zeros((seq_len, 4), dtype=np.float32)
        traj_out[:n] = traj
        mask[:n] = True

    return traj_out, mask


def _build_casa_tensor(pts: list,
                       fps: float,
                       um_per_px: float) -> np.ndarray:
    """
    단일 트랙 pts → CASA 9D 벡터 (casa_features.compute_casa_features 재사용).
    """
    from .casa_features import compute_casa_features
    from .trajectory_utils import smooth_trajectory_dct

    raw_xy  = np.array([[cx, cy] for _, cx, cy, *_ in pts], dtype=np.float32)
    smooth  = smooth_trajectory_dct(raw_xy, n_keep=12)
    casa    = compute_casa_features(smooth, raw_xy=raw_xy, fps=int(fps))

    # µm 단위 변환 (pipeline.py와 동일한 인덱스)
    for i in (0, 1, 2, 6, 8):
        casa[i] *= um_per_px

    return casa.astype(np.float32)  # (9,)


# ── 메인 클래스 ─────────────────────────────────────────────────────────────

class HybridGrader:
    """
    HybridTrajectoryModel을 로드하고 per-track 등급화를 수행.

    사용 예::

        grader = HybridGrader("proposed_best.pt")
        result = grader.grade(track_history, fps=50.0)
    """

    def __init__(self,
                 ckpt_path: str | Path,
                 num_classes: int = 3,
                 seq_len: int = SEQ_LEN,
                 fps_train: float = 30.0):
        """
        Args:
            ckpt_path:   proposed_best.pt 경로
            num_classes: 3 (PR / NP / IM)
            seq_len:     학습 시 사용한 시퀀스 길이 (프레임 수)
            fps_train:   학습 시 사용한 FPS (velocity 스케일 일치용)
        """
        self.device   = _require_cuda()
        self.seq_len  = seq_len
        self.fps_train = fps_train

        # 모델 생성 + 가중치 로드
        self.model = HybridTrajectoryModel(num_classes=num_classes).to(self.device)
        sd = torch.load(ckpt_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[hybrid_infer] ⚠ missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[hybrid_infer] ⚠ unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
        self.model.eval()
        print(f"[hybrid_infer] ✅ HybridModel 로드: {ckpt_path}  (device={self.device})")

    # ── 단일 배치 추론 ───────────────────────────────────────────────────────
    @torch.no_grad()
    def _infer_batch(self,
                     trajs:  torch.Tensor,   # (B, T, 4)
                     masks:  torch.Tensor,   # (B, T) bool
                     casas:  torch.Tensor,   # (B, 9)
                     confs:  torch.Tensor,   # (B, 1)
                     ) -> np.ndarray:
        """logits → argmax 등급 인덱스 (B,)"""
        out   = self.model(trajs, masks, casas, confs)
        # HybridModel 출력이 dict인 경우 / Tensor인 경우 모두 처리
        if isinstance(out, dict):
            logits = out.get("cls", out.get("logits", out[list(out.keys())[0]]))
        else:
            logits = out
        return logits.argmax(dim=-1).cpu().numpy()

    # ── 공개 인터페이스 ──────────────────────────────────────────────────────
    def grade(self,
              track_history: dict,
              fps: float = 50.0,
              um_per_px: float = 0.7031,
              min_track_len: int = 10,
              batch_size: int = 64) -> dict:
        """
        grade_tracks()와 동일한 dict 반환.

        Returns::

            {
                'grade_progressive':     float,   # PR %
                'grade_non_progressive': float,   # NP %
                'grade_immotile':        float,   # IM %
                'n_graded':              int,
                'tid_grades':            {tid: 'PR'/'NP'/'IM'},
                'thresholds':            {'model': 'HybridTrajectoryModel'},
            }
        """
        eligible = {
            tid: pts for tid, pts in track_history.items()
            if len(pts) >= min_track_len
        }
        if not eligible:
            return {}

        tids  = list(eligible.keys())
        trajs, masks, casas, confs = [], [], [], []

        for tid in tids:
            pts    = eligible[tid]
            tr, mk = _build_traj_tensor(pts, self.seq_len, self.fps_train)
            ca     = _build_casa_tensor(pts, fps, um_per_px)
            conf   = np.array([mk.mean()], dtype=np.float32)   # 유효 프레임 비율

            trajs.append(tr)
            masks.append(mk)
            casas.append(ca)
            confs.append(conf)

        # GPU 배치 추론
        all_preds: list[int] = []
        for i in range(0, len(tids), batch_size):
            sl = slice(i, i + batch_size)

            t_tr = torch.tensor(np.stack(trajs[sl]), dtype=torch.float32, device=self.device)
            t_mk = torch.tensor(np.stack(masks[sl]), dtype=torch.bool,    device=self.device)
            t_ca = torch.tensor(np.stack(casas[sl]), dtype=torch.float32, device=self.device)
            t_cf = torch.tensor(np.stack(confs[sl]), dtype=torch.float32, device=self.device)

            all_preds.extend(self._infer_batch(t_tr, t_mk, t_ca, t_cf).tolist())

        # 집계
        counts     = {"PR": 0, "NP": 0, "IM": 0}
        tid_grades = {}
        for tid, pred_idx in zip(tids, all_preds):
            grade = GRADE_MAP.get(pred_idx, "NP")
            counts[grade] += 1
            tid_grades[tid] = grade

        n = len(tids)
        return {
            "grade_progressive":     round(counts["PR"] / n * 100, 1),
            "grade_non_progressive": round(counts["NP"] / n * 100, 1),
            "grade_immotile":        round(counts["IM"] / n * 100, 1),
            "n_graded":              n,
            "tid_grades":            tid_grades,
            "thresholds":            {"model": "HybridTrajectoryModel"},
        }
