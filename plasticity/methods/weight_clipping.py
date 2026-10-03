"""Weight Clipping (Elsayed, Lan, Lyle & Mahmood, 2024).

Reference
---------
M. Elsayed, Q. Lan, C. Lyle, A. R. Mahmood (2024). *Weight Clipping for Deep Continual and
Reinforcement Learning*. Reinforcement Learning Journal (RLJ), vol. 1.

Formulation implemented
-----------------------
Weight clipping is a projection applied **after every optimiser step**: each weight matrix ``W_l``
is clipped element-wise onto the box whose half-width is a multiple ``kappa`` (κ) of the bound
``b_l`` of the layer's *initial* uniform distribution::

    W_l <- clip(W_l, -kappa * b_l, +kappa * b_l)        for every layer l, after every update

With the Kaiming-uniform initialiser used in this repository, ``b_l = gain * sqrt(3 / fan_in)``
(:attr:`ReparamLinear.weight_init_bound`), so ``kappa = 1`` confines the weights to their initial
support and ``kappa > 1`` allows a κ-fold expansion of it (Elsayed et al. recommend κ in {2, 3}).
Biases are projected in the same way onto ``[-kappa * c_l, kappa * c_l]`` with ``c_l = 1 / sqrt(fan_in)``
(the ``nn.Linear`` bias bound, :attr:`ReparamLinear.bias_init_bound`) when ``clip_bias=True``.

Because the projection only depends on ``kappa`` and the initialiser, it adds no gradient term and
is independent of the optimiser (SGD / Adam); it bounds the effective weights by construction and
is therefore the closest published counterpart of the thesis' bounded reparametrisation
(``W = A sin(Θ)`` with ``A = gamma * b_l``), with ``kappa`` playing the role of ``gamma``.

Simplifications / choices
-------------------------
* Elsayed et al. derive ``b_l`` from the PyTorch default initialiser; here ``b_l`` is whatever bound
  the model's initialiser reports, so κ keeps its meaning (multiple of the initial support) for any
  of the supported initialisation schemes.
* LayerNorm affine parameters (when ``model.layer_norm``) have no initialisation bound and are not
  clipped.
* The projection is applied once at construction as well, so that ``|W| <= kappa * b_l`` holds along
  the whole trajectory; for ``kappa >= 1`` this is a no-op (initial weights lie inside ``[-b_l, b_l]``).
* Standard parametrisation only: clipping the free parameter Θ of a sin / tanh layer would not
  clip the effective weight, so ``__init__`` raises ``ValueError`` for any non-standard layer.

Config keys (``method:`` section)
---------------------------------
``kappa``     (float, default 2.0)  multiple of the initial bound defining the clipping box.
``clip_bias`` (bool,  default True) also clip the bias vectors.

``state_summary()`` reports ``clip_frac`` (fraction of weight-matrix entries at the bound,
``|w| >= kappa*b_l - 1e-8``) and, with ``clip_bias``, ``clip_frac_bias``; both are computed with a
full pass over the parameters and are meant to be called once per task, not every step.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ..models import MLP
from .base import Method

_AT_BOUND_TOL = 1e-8


class WeightClipping(Method):
    """Project every weight (and bias) onto ``[-kappa*b_l, kappa*b_l]`` after each optimiser step."""

    name = "weight_clipping"
    requires_features = False

    def __init__(self, model: MLP, cfg: Dict[str, Any], optimizer_cfg: Dict[str, Any], generator: Optional[torch.Generator] = None):
        super().__init__(model, cfg, optimizer_cfg, generator)
        self.kappa = float(cfg.get("kappa", 2.0))
        self.clip_bias = bool(cfg.get("clip_bias", True))
        if self.kappa <= 0:
            raise ValueError(f"weight_clipping: kappa must be > 0, got {self.kappa}")
        for li, lin in enumerate(model.linear_layers):
            if lin.mode != "standard":
                raise ValueError(
                    f"weight_clipping is defined for the standard parametrisation only; layer {li} has mode={lin.mode!r}. "
                    "Clipping the free parameter Theta of a sin/tanh layer would not bound the effective weight."
                )
        # Clipped tensors and their (symmetric) bounds, kept as flat lists for torch._foreach_* ops.
        self._weights: List[torch.Tensor] = []
        self._weight_bounds: List[float] = []
        self._biases: List[torch.Tensor] = []
        self._bias_bounds: List[float] = []
        for lin in model.linear_layers:
            self._weights.append(lin.theta)
            self._weight_bounds.append(self.kappa * float(lin.weight_init_bound))
            if self.clip_bias and lin.theta_bias is not None:
                self._biases.append(lin.theta_bias)
                self._bias_bounds.append(self.kappa * float(lin.bias_init_bound))
        self._tensors = self._weights + self._biases
        self._hi = self._weight_bounds + self._bias_bounds
        self._lo = [-b for b in self._hi]
        self.n_clipped_steps = 0
        self.project()  # no-op for kappa >= 1 (initial weights lie inside the initial support)

    # --- projection ----------------------------------------------------------------------------
    @torch.no_grad()
    def project(self) -> None:
        """``W <- clip(W, -kappa*b_l, kappa*b_l)`` for every clipped tensor (two fused foreach kernels)."""
        if not self._tensors:
            return
        torch._foreach_clamp_min_(self._tensors, self._lo)
        torch._foreach_clamp_max_(self._tensors, self._hi)

    def after_step(self, step: int, features: Optional[List[torch.Tensor]] = None) -> None:
        self.project()
        self.n_clipped_steps += 1

    # --- reporting -------------------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def _at_bound_fraction(tensors: List[torch.Tensor], bounds: List[float]) -> float:
        n_at, n_all = 0, 0
        for t, b in zip(tensors, bounds):
            n_at += int((t.abs() >= b - _AT_BOUND_TOL).sum())
            n_all += t.numel()
        return n_at / n_all if n_all else 0.0

    def state_summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"clip_frac": self._at_bound_fraction(self._weights, self._weight_bounds)}
        if self.clip_bias:
            out["clip_frac_bias"] = self._at_bound_fraction(self._biases, self._bias_bounds)
        return out

    def extra_repr(self) -> str:
        return f"kappa={self.kappa}, clip_bias={self.clip_bias}, bounds={[round(b, 4) for b in self._weight_bounds]}"
