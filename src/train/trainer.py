"""Trainer for trajectory-aware sperm motility classifier.

특징
----
* `EarlyStopping` 컴포넌트 — 검증 메트릭 개선이 ``patience`` epoch 동안 없으면 학습 중단
* `best_state` / `best_metric` 추적 → 학습 종료 시 *best* 모델로 자동 복원
* `Trainer.save()` 는 항상 *best* 모델을 저장합니다 (마지막 epoch의 모델이 아닙니다).
* fit() 도중에도 개선이 발생할 때마다 ``<save_dir>/best.pt`` 에 즉시 저장 → 크래시
  시에도 best checkpoint가 보존됩니다.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..eval.metrics import classification_report, expected_calibration_error
from ..models import HybridTrajectoryModel, InceptionTime
from .early_stopping import EarlyStopping


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #

def build_model_from_config(model_cfg: Dict[str, Any], num_classes: int = 3) -> nn.Module:
    arch = model_cfg.get("arch", "hybrid")
    if arch == "inception":
        return InceptionTime(
            in_channels=model_cfg.get("in_channels", 4),
            num_classes=num_classes,
            n_filters=model_cfg.get("cnn_filters", 32),
            kernel_sizes=tuple(model_cfg.get("cnn_kernels", (9, 19, 39))),
            depth=model_cfg.get("cnn_depth", 6),
        )
    if arch == "hybrid":
        return HybridTrajectoryModel(
            num_classes=num_classes,
            in_channels=model_cfg.get("in_channels", 4),
            cnn_filters=model_cfg.get("cnn_filters", 32),
            cnn_kernels=tuple(model_cfg.get("cnn_kernels", (9, 19, 39))),
            cnn_depth=model_cfg.get("cnn_depth", 4),
            transformer_dim=model_cfg.get("transformer_dim", 64),
            transformer_heads=model_cfg.get("transformer_heads", 4),
            transformer_layers=model_cfg.get("transformer_layers", 2),
            casa_dim=model_cfg.get("casa_dim", 9),
            use_tracking_confidence=model_cfg.get("use_tracking_confidence", True),
            dropout=model_cfg.get("dropout", 0.1),
        )
    raise ValueError(f"Unknown arch: {arch}")


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"
    num_workers: int = 0
    seed: int = 42
    save_dir: Path = Path("./runs")
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "motility": 1.0, "hyperactivation": 0.5, "kinematic": 0.1,
    })
    # ---- Early stopping ----
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    early_stopping_monitor: str = "f1_macro"     # f1_macro/accuracy/auroc/cohen_kappa/loss/ece
    early_stopping_mode: str = "max"             # 'max' for f1/acc, 'min' for loss/ece
    # ---- Best checkpoint (saved during fit, NOT the final-epoch model) ----
    best_ckpt_name: str = "best.pt"


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: TrainConfig,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = self._resolve_device(cfg.device)
        self.model.to(self.device)

        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=max(cfg.epochs, 1),
        )
        self.cls_loss = nn.CrossEntropyLoss()
        self.bin_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.SmoothL1Loss()

        # State updated during fit()
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_metric: float = -math.inf if cfg.early_stopping_mode == "max" else math.inf
        self.best_epoch: int = -1
        self.best_ckpt_path: Optional[Path] = None

    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(name)

    # ------------------------------------------------------------------ #

    def _is_hybrid(self) -> bool:
        return isinstance(self.model, HybridTrajectoryModel)

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._is_hybrid():
            return self.model(
                trajectory=batch["trajectory"].to(self.device),
                velocity=batch["velocity"].to(self.device),
                mask=batch["mask"].to(self.device),
                confidence=batch["confidence"].to(self.device),
                casa=batch["casa"].to(self.device),
                density=batch["density"].to(self.device),
            )
        # InceptionTime baseline: stack [x, y, vx, vy] as channels
        x = torch.cat([batch["trajectory"], batch["velocity"]], dim=-1).transpose(1, 2)
        logits = self.model(x.to(self.device))
        return {"logits": logits}

    # ------------------------------------------------------------------ #

    def _step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self._forward(batch)
        labels = batch["label"].to(self.device)
        loss = self.cls_loss(out["logits"], labels) * self.cfg.loss_weights.get("motility", 1.0)
        log = {"loss_motility": float(loss.item())}
        if "hyper_logits" in out:
            l_h = self.bin_loss(out["hyper_logits"], batch["hyper"].to(self.device))
            loss = loss + l_h * self.cfg.loss_weights.get("hyperactivation", 0.0)
            log["loss_hyper"] = float(l_h.item())
        if "kinematic_pred" in out:
            l_r = self.reg_loss(out["kinematic_pred"], batch["casa"].to(self.device))
            loss = loss + l_r * self.cfg.loss_weights.get("kinematic", 0.0)
            log["loss_reg"] = float(l_r.item())
        return loss, log

    # ------------------------------------------------------------------ #

    def _select_monitor_value(
        self,
        train_loss: float,
        val_metrics: Optional[object],
        val_ece: float,
    ) -> float:
        """Pick the value to monitor for early-stopping based on cfg.

        Falls back to train loss (negated under 'max' mode) when no validation
        set is provided, so 'max' direction continues to make sense.
        """
        m = self.cfg.early_stopping_monitor
        if val_metrics is None:
            return -train_loss if self.cfg.early_stopping_mode == "max" else train_loss
        if m == "loss":
            return train_loss
        if m == "ece":
            return val_ece
        d = val_metrics.as_dict()
        if m in d:
            return float(d[m])
        return float(d.get("f1_macro", -math.inf))

    # ------------------------------------------------------------------ #

    def fit(
        self,
        train_ds: Dataset,
        val_ds: Optional[Dataset] = None,
    ) -> Dict[str, Any]:
        set_seed(self.cfg.seed)
        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.num_workers, drop_last=False,
        )
        val_loader = (
            DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False,
                       num_workers=self.cfg.num_workers)
            if val_ds is not None and len(val_ds) > 0 else None
        )

        # ---- Early stopping ---- #
        stopper = EarlyStopping(
            patience=self.cfg.early_stopping_patience,
            min_delta=self.cfg.early_stopping_min_delta,
            mode=self.cfg.early_stopping_mode,
        )
        best_ckpt_path = Path(self.cfg.save_dir) / self.cfg.best_ckpt_name
        best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path = best_ckpt_path

        history: List[Dict[str, Any]] = []
        for epoch in range(self.cfg.epochs):
            self.model.train()
            running = 0.0
            for batch in train_loader:
                loss, _log = self._step(batch)
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optim.step()
                running += float(loss.item())
            self.scheduler.step()
            train_loss = running / max(len(train_loader), 1)

            entry: Dict[str, Any] = {"epoch": epoch, "train_loss": train_loss}
            val_metrics = None
            val_ece = float("nan")
            if val_loader is not None:
                val_metrics, val_ece = self.evaluate(val_loader)
                entry.update({f"val_{k}": v for k, v in val_metrics.as_dict().items()})
                entry["val_ece"] = val_ece

            monitor_val = self._select_monitor_value(train_loss, val_metrics, val_ece)
            improved = stopper.step(monitor_val, epoch)
            entry["monitor"] = monitor_val
            entry["best_so_far"] = stopper.best
            entry["bad_epochs"] = stopper.num_bad_epochs
            entry["improved"] = improved

            if improved:
                self.best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.best_metric = stopper.best
                self.best_epoch = epoch
                # Persist immediately so a crash won't lose the best checkpoint
                torch.save(self.best_state, best_ckpt_path)
                entry["saved_best"] = str(best_ckpt_path)

            history.append(entry)
            print(f"[trainer] {entry}")

            if stopper.should_stop:
                print(
                    f"[trainer] Early stopping at epoch {epoch} "
                    f"(best={stopper.best:.4f} @ epoch {stopper.best_epoch}, "
                    f"no improvement for {stopper.num_bad_epochs} epochs)"
                )
                break

        # Restore best weights so subsequent .save()/.evaluate() use BEST model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        else:
            # Degenerate: no improvement ever recorded → snapshot final state
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }
            torch.save(self.best_state, best_ckpt_path)

        return {
            "history": history,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "best_ckpt": str(best_ckpt_path),
            "stopped_early": stopper.should_stop,
        }

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate(self, loader: DataLoader):
        self.model.eval()
        ys, ps, probs = [], [], []
        for batch in loader:
            out = self._forward(batch)
            logits = out["logits"]
            prob = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = prob.argmax(axis=1)
            ys.append(batch["label"].cpu().numpy())
            ps.append(pred)
            probs.append(prob)
        if not ys:
            return classification_report(np.array([]), np.array([])), float("nan")
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        pr = np.concatenate(probs, axis=0)
        return classification_report(y, p, pr), expected_calibration_error(y, pr)

    # ------------------------------------------------------------------ #

    def save(self, path: os.PathLike) -> None:
        """Always save the **best** model encountered during fit().

        If ``fit`` was not called or no improvement was ever recorded, this
        falls back to the current model state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.best_state if self.best_state is not None else self.model.state_dict()
        torch.save(state, path)
