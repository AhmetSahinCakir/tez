"""Scale-corrected update for the bounded reparametrisations (thesis variant, "ölçek-düzeltmeli güncelleme").

This is **not** a published method: it is the thesis' own secondary variant of the proposed
bounded-periodic reparametrisation ``W = A sin(Θ)`` (and of its non-periodic control ``W = A tanh(Θ)``).
It addresses the mechanism H3 of the thesis: the state-dependent effective learning rate of a bounded
reparametrisation vanishes at the bound.

Background (see the docstring of ``plasticity/models/reparam.py``)
-----------------------------------------------------------------
With ``c = layer.normalized_jacobian()`` (``c = cos(angle)`` for sin, ``c = 1 - tanh²(angle)`` for tanh,
element-wise) the chain rule gives ``∂L/∂Θ = (∂W/∂Θ) ∂L/∂W`` and one plain SGD step on the free
parameter moves the effective weight by (first order in η)::

    theta_scale = 'amplitude' :  ΔW ≈ -η c²    ∂L/∂W        (W = A sin(Φ/A), free parameter Φ)
    theta_scale = 'unit'      :  ΔW ≈ -η A² c² ∂L/∂W        (W = A sin(Θ),   free parameter Θ)

Near the bound ``|W| -> A`` one has ``c -> 0``: the effective step vanishes and the weight freezes
(for sin only until Θ crosses the turning point; for tanh permanently).

Formulation implemented
-----------------------
In :meth:`before_step` (after ``loss.backward()``, before ``optimizer.step()``, under ``no_grad``) the
gradient of every reparametrised free parameter is rescaled element-wise::

    s      = min(1 / c², max_scale)                       (s = max_scale where c = 0)
    Θ.grad <- s ⊙ Θ.grad

so that the plain-SGD effective step becomes::

    theta_scale = 'amplitude' :  ΔW ≈ -η    min(1, max_scale · c²) ∂L/∂W
    theta_scale = 'unit'      :  ΔW ≈ -η A² min(1, max_scale · c²) ∂L/∂W

i.e. the standard step ``-η ∂L/∂W`` is restored wherever ``c² >= 1/max_scale`` and the amplification is
capped at ``max_scale`` closer to the bound (``c² < 1/max_scale``: "capped" elements, whose effective
step is ``max_scale · c²`` times the standard one). The layer constant ``A²`` of the ``'unit'`` scaling is
deliberately *not* corrected: it is a fixed per-layer learning rate, not a saturation effect (the main
experiments use ``'amplitude'``, where it is absent). Exactly at the bound (``c = 0``) the effective step
stays zero whatever ``s`` is (``ΔW = c ΔΘ``); the correction unfreezes weights *near* the bound, it does
not move weights sitting *on* it.

Which parameters: every :class:`ReparamLinear` with ``mode != 'standard'`` (sin **and** tanh: the class
is named after the proposed sin map but the formula is the same for the tanh control, which allows the
correction to be compared across both bounded maps). ``theta`` is treated with
``layer.normalized_jacobian()``; ``theta_bias`` with ``layer.bias_normalized_jacobian()`` when the bias
is reparametrised too (``bias_mode != 'standard'``, i.e. ``compact_bias=True``). Standard layers
(e.g. the output layer with ``reparam.output = standard``), standard biases, LayerNorm parameters and
learnable amplitudes are left untouched. Parameters whose ``.grad`` is ``None`` are skipped.
``__init__`` raises ``ValueError`` if no layer of the model is reparametrised (the method is undefined
for the standard parametrisation, where ``c ≡ 1``).

Optimiser interaction: the formulas above are exact (to first order in η) for plain SGD, which is what
the main experiments use. With SGD momentum the rescaled gradient enters the momentum buffer; with
Adam / AdamW the per-element normalisation largely undoes a per-element rescaling, so the correction is
only approximate there. A weight decay set in the optimiser acts on Θ and is not rescaled.

Cost: three element-wise kernels per reparametrised tensor per step (``c²``, reciprocal + clamp, in-place
multiply); no per-step Python loop over elements.

Config keys (``method:`` section)
---------------------------------
``max_scale``  (float, default 10.0)  cap of the gradient amplification ``s``; must be finite and ``>= 1``
                                     (``1.0`` = no correction, plain SGD on Θ).

``state_summary()`` reports ``mean_scale`` (mean of ``s`` over all treated elements at the last step) and
``frac_capped`` (fraction of treated elements with ``s = max_scale``, i.e. ``c² <= 1/max_scale``, at the
last step); before the first step they are ``1.0`` and ``0.0``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..models import MLP, ReparamLinear
from .base import Method

_KNOWN_KEYS = {"max_scale"}


class ScaleCorrectedSin(Method):
    """``Θ.grad <- min(1/c², max_scale) ⊙ Θ.grad`` in ``before_step`` for every reparametrised layer."""

    name = "scale_corrected"
    requires_features = False

    def __init__(self, model: MLP, cfg: Dict[str, Any], optimizer_cfg: Dict[str, Any], generator: Optional[torch.Generator] = None):
        super().__init__(model, cfg, optimizer_cfg, generator)
        unknown = set(cfg) - _KNOWN_KEYS
        if unknown:
            raise ValueError(f"scale_corrected: unknown config keys {sorted(unknown)}; known: {sorted(_KNOWN_KEYS)}")
        self.max_scale = float(cfg.get("max_scale", 10.0))
        if not math.isfinite(self.max_scale) or self.max_scale < 1.0:
            raise ValueError(f"scale_corrected: max_scale must be finite and >= 1 (1 = no correction), got {self.max_scale}")
        self.layer_indices: List[int] = [li for li, lin in enumerate(model.linear_layers) if lin.mode != "standard"]
        self.layers: List[ReparamLinear] = [model.linear_layers[li] for li in self.layer_indices]
        if not self.layers:
            raise ValueError(
                "scale_corrected is defined for the bounded reparametrisations only (W = A sin(Θ) / A tanh(Θ)); "
                "every layer of the model has mode='standard' (set model.reparam.hidden / output to 'sin' or 'tanh')."
            )
        self._last_scales: List[torch.Tensor] = []  # s of the last step, one tensor per treated parameter
        self.n_steps = 0

    # --- helpers -------------------------------------------------------------------------------
    @torch.no_grad()
    def scale(self, c: torch.Tensor) -> torch.Tensor:
        """``s = min(1 / c², max_scale)`` element-wise (new tensor; ``c = 0`` gives ``max_scale``, no NaN)."""
        return c.square().reciprocal_().clamp_(max=self.max_scale)

    # --- the step --------------------------------------------------------------------------------
    @torch.no_grad()
    def before_step(self, step: int) -> None:
        """Rescale ``theta.grad`` (and ``theta_bias.grad`` if reparametrised) of every bounded layer by ``s``."""
        scales: List[torch.Tensor] = []
        for lin in self.layers:
            g = lin.theta.grad
            if g is not None:
                s = self.scale(lin.normalized_jacobian())
                g.mul_(s)
                scales.append(s)
            if lin.theta_bias is not None and lin.bias_mode != "standard":
                gb = lin.theta_bias.grad
                if gb is not None:
                    sb = self.scale(lin.bias_normalized_jacobian())
                    gb.mul_(sb)
                    scales.append(sb)
        self._last_scales = scales
        self.n_steps += 1

    # --- reporting -------------------------------------------------------------------------------
    @torch.no_grad()
    def state_summary(self) -> Dict[str, Any]:
        if not self._last_scales:
            return {"mean_scale": 1.0, "frac_capped": 0.0}
        n = sum(s.numel() for s in self._last_scales)
        total = sum(float(s.sum()) for s in self._last_scales)
        capped = sum(int((s >= self.max_scale).sum()) for s in self._last_scales)
        return {"mean_scale": total / n, "frac_capped": capped / n}

    def extra_repr(self) -> str:
        return f"max_scale={self.max_scale}, layers={self.layer_indices}"
