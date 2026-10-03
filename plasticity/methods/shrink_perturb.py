"""Shrink & Perturb (Ash & Adams, 2020) in the continual form of Dohare et al. (2024).

References
----------
J. T. Ash, R. P. Adams (2020). *On Warm-Starting Neural Network Training*. NeurIPS 2020.
S. Dohare, J. F. Hernandez-Garcia, Q. Lan, P. Rahman, A. R. Mahmood, R. S. Sutton (2024).
*Loss of plasticity in deep continual learning*. Nature 632, 768-774 (Methods, "Shrink and perturb";
public implementation ``lop/algos/bp.py``, ``Backprop.perturb``).

Formulation implemented
-----------------------
Ash & Adams shrink the parameters of a warm-started network toward zero and add Gaussian noise once,
before re-training on the enlarged dataset. In the continual form used by Dohare et al. the same
operation is applied **after every optimiser step** (``every='step'``, default) or, as a cheaper
variant, once at every task boundary (``every='task'``, applied in :meth:`on_task_end`)::

    θ <- (1 - shrink) * θ + noise_std * ε,        ε ~ N(0, I) (independent per element and per application)

for every weight matrix of the network and, when ``perturb_bias=True`` (default), every bias vector
(the ``theta`` / ``theta_bias`` parameters of all :class:`~plasticity.models.ReparamLinear` layers,
hidden and output). The noise is drawn with the method's ``torch.Generator`` (``self.generator``), so
runs are reproducible for a given seed.

Relation to the public code of Dohare et al.: their ``Backprop.perturb`` adds ``N(0, perturb_scale²)``
noise to the weights and biases of every linear layer after each step, and realises the shrinking
through the optimiser's (coupled) L2 weight decay, i.e. a multiplicative factor ``1 - lr * weight_decay``
per step. Here the shrink factor is explicit and independent of the optimiser (``shrink`` plays the
role of ``lr * weight_decay``; set ``optimizer.weight_decay: 0`` to avoid shrinking twice).

Parametrisation
---------------
The operation acts on the **free parameters** and therefore runs unchanged for every parametrisation
(no ``ValueError`` for sin / tanh layers). It is, however, *meant for the standard parametrisation*
``W = Θ``, for which "shrink toward zero" has its intended meaning (weights pulled toward the origin,
norm reduced). For ``W = A sin(Θ)`` the same update shrinks the *angle* Θ toward 0, which also moves
``W`` toward 0 only on the principal branch (|angle| <= π/2), and the noise is added in Θ-space (its
effect on ``W`` is scaled by the Jacobian ``cos(angle)``). Such combinations are ablations, not the
main comparison of the thesis.

Simplifications / choices
-------------------------
* Only the linear layers' weights and biases are shrunk / perturbed; LayerNorm affine parameters
  (when ``model.layer_norm``) and learnable amplitudes are left untouched (they have no "warm-start"
  interpretation and Dohare et al. only perturb linear layers).
* With ``every='task'`` nothing happens inside :meth:`after_step`; the update is applied exactly once
  per task in :meth:`on_task_end` (i.e. *before* the next task starts), which is the point at which
  Ash & Adams apply it (between the old and the new training problem).
* Implementation: one fused ``_foreach_mul_`` (shrink) and one fused ``_foreach_add_`` (noise) over
  all affected tensors; the noise for all tensors is drawn with a single ``normal_`` call into a flat
  pre-allocated buffer whose views have the parameters' shapes (no per-step Python loop over tensors
  beyond the fused kernels, no per-step allocation).

Config keys (``method:`` section)
---------------------------------
``shrink``       (float, default 1e-4)   shrink factor ``p``; ``θ <- (1-p) θ``. Must be in ``[0, 1]``.
``noise_std``    (float, default 1e-3)   standard deviation of the additive Gaussian noise. ``>= 0``.
``every``        (str,   default 'step') ``'step'`` = after every optimiser step; ``'task'`` = at task end.
``perturb_bias`` (bool,  default True)   also shrink / perturb the bias vectors.

``state_summary()`` reports ``n_applications`` (number of shrink-and-perturb updates applied so far).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ..models import MLP
from .base import Method

_EVERY = ("step", "task")
_KNOWN_KEYS = {"shrink", "noise_std", "every", "perturb_bias"}


class ShrinkPerturb(Method):
    """``θ <- (1 - shrink) θ + noise_std ε`` after every step (or at every task end)."""

    name = "shrink_perturb"
    requires_features = False

    def __init__(self, model: MLP, cfg: Dict[str, Any], optimizer_cfg: Dict[str, Any], generator: Optional[torch.Generator] = None):
        super().__init__(model, cfg, optimizer_cfg, generator)
        unknown = set(cfg) - _KNOWN_KEYS
        if unknown:
            raise ValueError(f"shrink_perturb: unknown config keys {sorted(unknown)}; known: {sorted(_KNOWN_KEYS)}")
        self.shrink = float(cfg.get("shrink", 1e-4))
        self.noise_std = float(cfg.get("noise_std", 1e-3))
        self.every = str(cfg.get("every", "step")).lower()
        self.perturb_bias = bool(cfg.get("perturb_bias", True))
        if not 0.0 <= self.shrink <= 1.0:
            raise ValueError(f"shrink_perturb: shrink must be in [0, 1], got {self.shrink}")
        if self.noise_std < 0.0:
            raise ValueError(f"shrink_perturb: noise_std must be >= 0, got {self.noise_std}")
        if self.every not in _EVERY:
            raise ValueError(f"shrink_perturb: every must be one of {_EVERY}, got {self.every!r}")
        if generator is None:  # keep runs reproducible even when the caller forgot the generator
            self.generator = torch.Generator().manual_seed(0)

        # Affected tensors: weights (theta) of every linear layer, biases (theta_bias) optionally.
        self.params: List[torch.nn.Parameter] = []
        self.param_names: List[str] = []
        for li, lin in enumerate(model.linear_layers):
            self.params.append(lin.theta)
            self.param_names.append(f"layer{li}.theta")
            if self.perturb_bias and lin.theta_bias is not None:
                self.params.append(lin.theta_bias)
                self.param_names.append(f"layer{li}.theta_bias")
        # Flat noise buffer + shaped views (one RNG call per application, one fused add).
        total = sum(p.numel() for p in self.params)
        self._noise_flat = torch.empty(total, dtype=self.params[0].dtype if self.params else torch.float32)
        self._noise_views: List[torch.Tensor] = []
        offset = 0
        for p in self.params:
            self._noise_views.append(self._noise_flat[offset : offset + p.numel()].view(p.shape))
            offset += p.numel()
        self._keep = 1.0 - self.shrink
        self.n_applications = 0

    # --- the operation ---------------------------------------------------------------------------
    @torch.no_grad()
    def apply(self) -> None:
        """One shrink-and-perturb update of all affected parameters."""
        if self.params:
            if self.shrink > 0.0:
                torch._foreach_mul_(self.params, self._keep)
            if self.noise_std > 0.0:
                self._noise_flat.normal_(0.0, 1.0, generator=self.generator)
                torch._foreach_add_(self.params, self._noise_views, alpha=self.noise_std)
        self.n_applications += 1

    # --- hooks -------------------------------------------------------------------------------------
    def after_step(self, step: int, features: Optional[List[torch.Tensor]] = None) -> None:
        if self.every == "step":
            self.apply()

    def on_task_end(self, task_index: int) -> None:
        if self.every == "task":
            self.apply()

    # --- reporting ---------------------------------------------------------------------------------
    def state_summary(self) -> Dict[str, Any]:
        return {"n_applications": self.n_applications}

    def extra_repr(self) -> str:
        return (f"shrink={self.shrink}, noise_std={self.noise_std}, every={self.every!r}, "
                f"perturb_bias={self.perturb_bias}, n_tensors={len(self.params)}")
