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

    # --- checkpointing -------------------------------------------------------------------------
    # Attributes that are references into the model / optimiser / config (restored elsewhere), not state.
    _STATE_SKIP = frozenset({"model", "cfg", "optimizer_cfg", "optimizer", "generator", "lr", "layers", "params",
                             "param_names", "layer_names"})

    def _model_storages(self) -> set:
        out = set()
        for t in list(self.model.parameters()) + list(self.model.buffers()):
            try:
                out.add(t.untyped_storage().data_ptr())
            except Exception:  # pragma: no cover - older torch
                out.add(t.storage().data_ptr())
        return out

    def _is_own_tensor(self, t: torch.Tensor, model_storages: set) -> bool:
        if isinstance(t, nn.Parameter):
            return False
        try:
            ptr = t.untyped_storage().data_ptr()
        except Exception:  # pragma: no cover
            ptr = t.storage().data_ptr()
        return ptr not in model_storages

    def state_dict(self) -> Dict[str, Any]:
        """Mutable state of the method (traces, counters, RNG) for run checkpoints.

        Generic: every instance attribute that is a scalar, a tensor owned by the method (not a model
        parameter / buffer or a view of one), a list of such tensors, or a list of scalars, plus the state of
        ``self.generator``.  Static attributes (bounds, anchors, config echoes) are included too; restoring
        them is harmless because they are recomputed identically at construction.
        """
        ms = self._model_storages()
        out: Dict[str, Any] = {}
        if self.generator is not None:
            out["__generator__"] = self.generator.get_state()
        for k, v in vars(self).items():
            if k in self._STATE_SKIP:
                continue
            if v is None or isinstance(v, (bool, int, float, str)):
                out[k] = v
            elif torch.is_tensor(v):
                if self._is_own_tensor(v, ms):
                    out[k] = v.detach().clone()
            elif isinstance(v, list) and v:
                if all(torch.is_tensor(x) for x in v):
                    if all(self._is_own_tensor(x, ms) for x in v):
                        out[k] = [x.detach().clone() for x in v]
                elif all(isinstance(x, (bool, int, float, str)) for x in v):
                    out[k] = list(v)
        return out

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Inverse of :meth:`state_dict` (tensors are copied in place so that views stay valid)."""
        state = dict(state)
        gen = state.pop("__generator__", None)
        if gen is not None and self.generator is not None:
            self.generator.set_state(gen)
        for k, v in state.items():
            cur = getattr(self, k, None)
            if torch.is_tensor(v):
                if torch.is_tensor(cur) and cur.shape == v.shape:
                    cur.copy_(v)
                else:
                    setattr(self, k, v.clone())
            elif isinstance(v, list) and v and all(torch.is_tensor(x) for x in v):
                if isinstance(cur, list) and len(cur) == len(v) and all(torch.is_tensor(c) and c.shape == x.shape for c, x in zip(cur, v)):
                    for c, x in zip(cur, v):
                        c.copy_(x)
                else:
                    setattr(self, k, [x.clone() for x in v])
            else:
                setattr(self, k, list(v) if isinstance(v, list) else v)

    # --- reporting ---------------------------------------------------------------------------------
    def state_summary(self) -> Dict[str, Any]:
        """Scalar diagnostics appended to each task's log row (e.g. number of units replaced)."""
        return {}

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"
