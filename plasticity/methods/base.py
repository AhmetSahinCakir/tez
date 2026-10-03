"""Plug-in interface for plasticity-preserving methods.

A *method* owns the optimiser and can hook into the training step at four points::

    loss = CE(logits, y) + method.regularizer()      # extra loss terms (L2 Init, Parseval, ...)
    loss.backward()
    method.before_step(step)                          # gradient surgery (UPGD, scale-corrected sin)
    optimizer.step()
    method.after_step(step, features)                 # projections / clipping / re-initialisation
                                                      # (Weight Clipping, NaP, CBP, Shrink&Perturb)

All methods are defined for the *standard* parametrisation unless documented otherwise; the proposed
sin / tanh reparametrisation is a *model* option (``model.reparam``) and combines with the
``baseline`` method (plain SGD / Adam) in the main experiments.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from ..models import MLP


def build_optimizer(params: Iterable[torch.nn.Parameter], cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """SGD / Adam from the ``optimizer`` config section (``{'name': 'sgd', 'lr': 0.01, 'weight_decay': 0, ...}``)."""
    name = cfg.get("name", "sgd").lower()
    lr = float(cfg.get("lr", 0.01))
    wd = float(cfg.get("weight_decay", 0.0))
    params = list(params)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.0)), weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=tuple(cfg.get("betas", (0.9, 0.999))), eps=float(cfg.get("eps", 1e-8)), weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=tuple(cfg.get("betas", (0.9, 0.999))), eps=float(cfg.get("eps", 1e-8)), weight_decay=wd)
    raise ValueError(f"unknown optimizer {name!r}")


class Method:
    """Base class: plain gradient descent, no intervention."""

    name = "baseline"
    requires_features = False  # set True if ``after_step`` needs the hidden-layer features

    def __init__(self, model: MLP, cfg: Dict[str, Any], optimizer_cfg: Dict[str, Any], generator: Optional[torch.Generator] = None):
        self.model = model
        self.cfg = dict(cfg)
        self.optimizer_cfg = dict(optimizer_cfg)
        self.generator = generator
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr = float(optimizer_cfg.get("lr", 0.01))

    # --- lifecycle -------------------------------------------------------------------------------
    def build_optimizer(self) -> torch.optim.Optimizer:
        self.optimizer = build_optimizer(self.model.parameters(), self.optimizer_cfg)
        return self.optimizer

    def on_task_start(self, task_index: int) -> None:  # noqa: D401
        pass

    def on_task_end(self, task_index: int) -> None:
        pass

    # --- step hooks --------------------------------------------------------------------------------
    def regularizer(self) -> Optional[torch.Tensor]:
        return None

    def before_step(self, step: int) -> None:
        pass

    def after_step(self, step: int, features: Optional[List[torch.Tensor]] = None) -> None:
        pass

    # --- reporting ---------------------------------------------------------------------------------
    def state_summary(self) -> Dict[str, Any]:
        """Scalar diagnostics appended to each task's log row (e.g. number of units replaced)."""
        return {}

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"
