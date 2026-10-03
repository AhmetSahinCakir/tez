"""L2 Init -- regenerative regularisation toward the initial parameters (Kumar, Marklund & Van Roy, 2025).

Reference
---------
S. Kumar, H. Marklund, B. Van Roy (2025). *Maintaining Plasticity in Continual Learning via
Regenerative Regularization*. (arXiv:2308.11958; conference version 2025.)

Formulation implemented
-----------------------
L2 Init replaces the usual L2 penalty toward the origin (weight decay) by a penalty toward the
*initial* parameters ``θ0``, which "regenerates" the initial distribution instead of shrinking the
weights::

    L_total(θ) = L_task(θ) + lam * ||θ - θ0||_2^2            (sum over all regularised parameters)

Implementation: the term is **not** added to the loss. Its exact gradient is added to the data
gradient in :meth:`before_step` (i.e. after ``loss.backward()`` and before ``optimizer.step()``)::

    p.grad += 2 * lam * (p - p0)          for every regularised parameter p

Equivalence: ``∇_θ [lam * ||θ - θ0||_2^2] = 2 * lam * (θ - θ0)`` and back-propagation is linear in the
loss, so ``grad(L_task + lam ||θ-θ0||²) = grad(L_task) + 2 lam (θ - θ0)``. Any optimiser that only
consumes ``p.grad`` (SGD with / without momentum, Adam, AdamW) therefore performs exactly the same
update as if the penalty had been added to the loss, at the cost of two in-place fused vector operations and
without building an autograd graph over all parameters at every step. Note that this is the *coupled*
form (the penalty enters the gradient and thus Adam's moment estimates), as in the loss-based
definition of Kumar et al.; it is not a decoupled "decay toward θ0".

The method acts on the **free parameters** of the model, so it works unchanged for every
parametrisation: for ``W = A sin(Θ)`` (or tanh) ``θ0 = Θ0 = f⁻¹(W0 / A)`` and the penalty is measured in
Θ-space (the coordinates the optimiser moves in), not in effective-weight space. ``θ0`` is stored as
detached clones at construction time, i.e. the matched initialisation ``W0`` of the thesis.

Simplifications / choices
-------------------------
* All trainable parameters are regularised by default (weights, biases, LayerNorm affine parameters,
  learnable amplitudes if enabled), as in Kumar et al.; ``include_bias=False`` excludes every parameter
  whose name ends in ``bias`` (``theta_bias`` of the linear layers and LayerNorm biases).
* Parameters whose ``.grad`` is ``None`` when :meth:`before_step` runs (unused in the forward pass)
  receive a freshly allocated gradient equal to the penalty gradient, so the pull toward θ0 is applied
  to every regularised parameter.
* ``regularizer()`` returns ``None`` (the term is handled in gradient space); the current value of the
  penalty is available as :meth:`penalty` for logging.

Config keys (``method:`` section)
---------------------------------
``lam``          (float, default 1e-2)  regularisation strength λ.
``include_bias`` (bool,  default True)  regularise bias parameters too.

``state_summary()`` reports ``dist_to_init`` = ``||θ - θ0||_2`` over all regularised parameters and
``reg_loss`` = ``lam * dist_to_init**2`` (the penalty value).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch

from ..models import MLP
from .base import Method


class L2Init(Method):
    """``p.grad += 2*lam*(p - p0)`` in ``before_step`` (exact gradient of ``lam*||θ-θ0||²``)."""

    name = "l2_init"
    requires_features = False

    def __init__(self, model: MLP, cfg: Dict[str, Any], optimizer_cfg: Dict[str, Any], generator: Optional[torch.Generator] = None):
        super().__init__(model, cfg, optimizer_cfg, generator)
        self.lam = float(cfg.get("lam", 1e-2))
        self.include_bias = bool(cfg.get("include_bias", True))
        if self.lam < 0:
            raise ValueError(f"l2_init: lam must be >= 0, got {self.lam}")
        self.param_names: List[str] = []
        self.params: List[torch.nn.Parameter] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if not self.include_bias and name.endswith("bias"):
                continue
            self.param_names.append(name)
            self.params.append(p)
        # θ0: detached clones of the initial (matched) free parameters
        self.init_params: List[torch.Tensor] = [p.detach().clone() for p in self.params]
        self._coef = 2.0 * self.lam

    # --- hooks -----------------------------------------------------------------------------------
    def regularizer(self) -> Optional[torch.Tensor]:
        return None  # handled exactly in gradient space, see module docstring

    @torch.no_grad()
    def before_step(self, step: int) -> None:
        """Add ``2*lam*(p - p0)`` to every regularised parameter's gradient (two in-place foreach ops)."""
        if self._coef == 0.0 or not self.params:
            return
        grads: List[torch.Tensor] = []
        for p in self.params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grads.append(p.grad)
        # grad += c*p ; grad -= c*p0   (two in-place fused kernels, no temporary allocation; == c*(p - p0))
        torch._foreach_add_(grads, self.params, alpha=self._coef)
        torch._foreach_add_(grads, self.init_params, alpha=-self._coef)

    # --- helpers / reporting -------------------------------------------------------------------
    @torch.no_grad()
    def reset_init(self) -> None:
        """Re-anchor θ0 to the current parameters (not used by the protocol; provided for ablations)."""
        for p0, p in zip(self.init_params, self.params):
            p0.copy_(p.detach())

    @torch.no_grad()
    def dist_to_init(self) -> float:
        """``||θ - θ0||_2`` over all regularised parameters."""
        if not self.params:
            return 0.0
        diffs = torch._foreach_sub(self.params, self.init_params)
        return math.sqrt(sum(float((d * d).sum()) for d in diffs))

    @torch.no_grad()
    def penalty(self) -> float:
        """Current value of ``lam * ||θ - θ0||_2^2``."""
        return self.lam * self.dist_to_init() ** 2

    def state_summary(self) -> Dict[str, Any]:
        d = self.dist_to_init()
        return {"dist_to_init": d, "reg_loss": self.lam * d * d}

    def extra_repr(self) -> str:
        return f"lam={self.lam}, include_bias={self.include_bias}, n_params={len(self.params)}"
