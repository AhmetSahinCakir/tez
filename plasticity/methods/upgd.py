"""UPGD -- Utility-based Perturbed Gradient Descent (Elsayed & Mahmood, 2024).

Reference
---------
M. Elsayed, A. R. Mahmood (2024). *Addressing Loss of Plasticity and Catastrophic Forgetting in
Continual Learning*. ICLR 2024. (Algorithm 1, "UPGD-W" = weight-wise first-order utility with global
scaling; public implementation ``core/optim/weight/search/first_order.py::FirstOrderGlobalUPGD`` of
github.com/mohmdelsayed/upgd.)

Formulation implemented (weight-space, first-order utility, global scaling)
--------------------------------------------------------------------------
For every parameter element ``θ_i`` with gradient ``g_i = ∂L/∂θ_i`` (after ``loss.backward()``)::

    U_i      = -g_i θ_i                                   instantaneous first-order utility
                                                          (Taylor estimate of L(θ with θ_i = 0) - L(θ))
    u_i     <- β u_i + (1 - β) U_i                        utility trace (u_0 = 0)
    û_i      = u_i / (1 - β^t)                            bias-corrected trace, t = number of steps so far
    s_i      = sigmoid( û_i / (max_j |û_j| + eps) )       scaled utility, global max over *all* treated
                                                          elements of *all* parameters
    θ_i     <- θ_i - lr * (g_i + σ ξ_i) * (1 - s_i),      ξ_i ~ N(0, 1)

A weight that is useful (large positive ``-g θ``: removing it would increase the loss) has ``s`` close
to its maximum and is protected (small step, little noise); a useless weight (``s`` small) receives
(almost) the full gradient step plus noise, which keeps it "alive" and preserves plasticity.

Implementation: everything happens in :meth:`before_step`, which **replaces** ``p.grad`` by
``(g + σ ξ) ⊙ (1 - s)``; the plain SGD optimiser (``optimizer.name: sgd``, no momentum) then performs
exactly the update above. With SGD-momentum, Adam or AdamW the gated / perturbed gradient enters the
moment estimates, so the resulting update is only an *approximation* of UPGD (the per-element
rescaling is partly undone by Adam's normalisation); the main experiments use plain SGD. A weight decay
set in the optimiser is applied by the optimiser on top of the UPGD step (coupled L2 for SGD).

Differences from the public implementation (documented, deliberately following the thesis' spec)
-------------------------------------------------------------------------------------------------
* Scaling: ``max_j |û_j|`` (maximum *absolute* bias-corrected utility) is used, so ``û/max ∈ [-1, 1]``
  and ``s ∈ [sigmoid(-1), sigmoid(1)] ≈ [0.269, 0.731]``. The public code divides by the maximum of
  the *signed, uncorrected* trace, which flips the sign of the ratio when every utility is negative.
* Step size: the public code uses ``alpha = -2 * lr`` so that a neutral weight (``s = 0.5``) takes the
  plain SGD step. Here the literal update ``-lr (g + σξ)(1-s)`` is used (a neutral weight takes *half*
  the SGD step); ``gate_scale`` (default 1.0) multiplies ``(1 - s)`` and ``gate_scale = 2.0`` reproduces
  the public code's convention. The learning rate is tuned per method in the pilot calibration anyway.
* ``eps`` is added to the global maximum (guards the all-zero-utility case, where ``s = 0.5`` everywhere).

Protected fraction: because ``s <= sigmoid(1) ≈ 0.731`` by construction, a threshold of ``s > 0.75``
(as first proposed for this diagnostic) can never be exceeded. ``frac_protected`` is therefore defined
as the fraction of elements whose bias-corrected utility exceeds ``protect_ratio`` (default 0.5) times
the global maximum, i.e. ``s > sigmoid(protect_ratio)`` (``≈ 0.622`` for the default).

Parametrisation
---------------
The utility ``-g ⊙ θ`` is the first-order estimate of the loss change when the *weight* is removed, so
the method is defined for the standard parametrisation ``W = Θ``. ``__init__`` raises ``ValueError``
if any layer has ``mode != 'standard'`` (for ``W = A sin(Θ)`` the same expression would estimate the
loss change of setting Θ to 0, which also zeroes ``W``, but this extension is not part of the thesis).

Which parameters: every trainable parameter of the model (weights, biases and, if present, LayerNorm
affine parameters), as in the public optimiser; ``include_bias=False`` excludes every parameter whose
name ends in ``bias`` (they then receive the plain gradient step). Parameters whose ``.grad`` is
``None`` at :meth:`before_step` (unused in the forward pass) are treated as having zero gradient:
their trace decays (``u <- β u``) and the optimiser skips them.

Complexity: ~8 fused ``torch._foreach_*`` kernels over the parameter list per step and a single
``normal_`` call into a flat pre-allocated noise buffer; no per-step Python loop over elements.

Config keys (``method:`` section)
---------------------------------
``beta``          (float, default 0.999) utility-trace decay β.
``sigma``         (float, default 1e-3)  standard deviation σ of the gradient perturbation.
``eps``           (float, default 1e-8)  added to the global max |û| in the scaling.
``include_bias``  (bool,  default True)  apply UPGD to bias parameters too.
``gate_scale``    (float, default 1.0)   multiplies ``(1 - s)`` (2.0 = convention of the public code).
``protect_ratio`` (float, default 0.5)   ``frac_protected`` counts elements with ``û > protect_ratio * max|û|``.

``state_summary()`` reports ``mean_scaled_utility`` (mean of ``s`` at the last step) and
``frac_protected`` (fraction of treated elements with ``s > sigmoid(protect_ratio)`` at the last step);
before the first step they are ``0.5`` and ``0.0``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch

from ..models import MLP
from .base import Method

_KNOWN_KEYS = {"beta", "sigma", "eps", "include_bias", "gate_scale", "protect_ratio"}


class UPGD(Method):
    """``p.grad <- (g + σξ) ⊙ (1 - s)`` in ``before_step`` with ``s`` the globally scaled utility trace."""

    name = "upgd"
    requires_features = False

    def __init__(self, model: MLP, cfg: Dict[str, Any], optimizer_cfg: Dict[str, Any], generator: Optional[torch.Generator] = None):
        super().__init__(model, cfg, optimizer_cfg, generator)
        unknown = set(cfg) - _KNOWN_KEYS
        if unknown:
            raise ValueError(f"upgd: unknown config keys {sorted(unknown)}; known: {sorted(_KNOWN_KEYS)}")
        self.beta = float(cfg.get("beta", 0.999))
        self.sigma = float(cfg.get("sigma", 1e-3))
        self.eps = float(cfg.get("eps", 1e-8))
        self.include_bias = bool(cfg.get("include_bias", True))
        self.gate_scale = float(cfg.get("gate_scale", 1.0))
        self.protect_ratio = float(cfg.get("protect_ratio", 0.5))
        if not 0.0 <= self.beta < 1.0:
            raise ValueError(f"upgd: beta must be in [0, 1), got {self.beta}")
        if self.sigma < 0.0:
            raise ValueError(f"upgd: sigma must be >= 0, got {self.sigma}")
        if self.eps <= 0.0:
            raise ValueError(f"upgd: eps must be > 0, got {self.eps}")
        if self.gate_scale <= 0.0:
            raise ValueError(f"upgd: gate_scale must be > 0, got {self.gate_scale}")
        for li, lin in enumerate(model.linear_layers):
            if lin.mode != "standard":
                raise ValueError(
                    f"upgd is defined for the standard parametrisation only (weight-space utility -g*w); "
                    f"layer {li} has mode={lin.mode!r}."
                )
        if generator is None:
            self.generator = torch.Generator().manual_seed(0)

        self.params: List[torch.nn.Parameter] = []
        self.param_names: List[str] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if not self.include_bias and name.endswith("bias"):
                continue
            self.params.append(p)
            self.param_names.append(name)
        # utility traces u (same shapes as the parameters), gate buffers (1 - s), flat noise buffer
        self.utility_trace: List[torch.Tensor] = [torch.zeros_like(p) for p in self.params]
        self._gate: List[torch.Tensor] = []  # gate_scale * (1 - s) of the last step (set in before_step)
        total = sum(p.numel() for p in self.params)
        self._noise_flat = torch.empty(total, dtype=self.params[0].dtype if self.params else torch.float32)
        self._noise_views: List[torch.Tensor] = []
        offset = 0
        for p in self.params:
            self._noise_views.append(self._noise_flat[offset : offset + p.numel()].view(p.shape))
            offset += p.numel()
        self.step_count = 0  # t: number of before_step calls (for the bias correction)
        self._has_gate = False
        self._protect_s = 1.0 / (1.0 + math.exp(-self.protect_ratio))  # sigmoid(protect_ratio)

    # --- helpers -------------------------------------------------------------------------------
    def bias_correction(self) -> float:
        return 1.0 - self.beta ** self.step_count if self.step_count > 0 else 1.0

    @torch.no_grad()
    def corrected_utility(self) -> List[torch.Tensor]:
        """``û = u / (1 - β^t)`` for every treated parameter (new tensors)."""
        return torch._foreach_div(self.utility_trace, self.bias_correction()) if self.params else []

    @torch.no_grad()
    def scaled_utility(self) -> List[torch.Tensor]:
        """``s`` of the last step (new tensors; ``0.5`` everywhere before the first step)."""
        if not self._has_gate:
            return [torch.full_like(p, 0.5) for p in self.params]
        s = torch._foreach_div(self._gate, self.gate_scale)  # (1 - s)
        torch._foreach_neg_(s)
        torch._foreach_add_(s, 1.0)
        return s

    # --- the step -----------------------------------------------------------------------------
    @torch.no_grad()
    def before_step(self, step: int) -> None:
        """Update the utility traces and replace ``p.grad`` by ``(g + σξ) ⊙ (1 - s)``."""
        if not self.params:
            return
        self.step_count += 1
        beta = self.beta
        # 1) trace update  u <- β u + (1-β) (-g θ)   (parameters without a gradient: u <- β u)
        torch._foreach_mul_(self.utility_trace, beta)
        have = [i for i, p in enumerate(self.params) if p.grad is not None]
        if not have:
            return
        grads = [self.params[i].grad for i in have]
        params = [self.params[i] for i in have]
        traces = [self.utility_trace[i] for i in have]
        torch._foreach_addcmul_(traces, grads, params, value=-(1.0 - beta))
        # 2) global max |û| = max |u| / (1 - β^t), computed on the traces (one fused inf-norm)
        bc = 1.0 - beta ** self.step_count
        all_norms = torch._foreach_norm(self.utility_trace, ord=float("inf"))
        max_abs_uhat = float(torch.stack(all_norms).max()) / bc
        # 3) gate = gate_scale * (1 - s),  s = sigmoid(û / (max|û| + eps));  1 - sigmoid(x) = sigmoid(-x)
        neg_scale = -1.0 / (bc * (max_abs_uhat + self.eps))
        gate = torch._foreach_mul(self.utility_trace, neg_scale)   # -û / (max|û| + eps)   (u / bc = û)
        torch._foreach_sigmoid_(gate)                               # = 1 - s
        if self.gate_scale != 1.0:
            torch._foreach_mul_(gate, self.gate_scale)
        self._gate = gate
        gates = [gate[i] for i in have]
        self._has_gate = True
        # 4) p.grad <- (g + σ ξ) ⊙ gate
        if self.sigma > 0.0:
            self._noise_flat.normal_(0.0, 1.0, generator=self.generator)
            noise = [self._noise_views[i] for i in have]
            torch._foreach_add_(grads, noise, alpha=self.sigma)
        torch._foreach_mul_(grads, gates)

    # --- reporting ----------------------------------------------------------------------------
    @torch.no_grad()
    def state_summary(self) -> Dict[str, Any]:
        if not self._has_gate or not self.params:
            return {"mean_scaled_utility": 0.5, "frac_protected": 0.0}
        n_all = sum(g.numel() for g in self._gate)
        # gate = gate_scale * (1 - s)  =>  s = 1 - gate / gate_scale ;  s > s* <=> gate < gate_scale * (1 - s*)
        thr = self.gate_scale * (1.0 - self._protect_s)
        sum_s = sum(float(g.numel()) - float(g.sum()) / self.gate_scale for g in self._gate)
        n_prot = sum(int((g < thr).sum()) for g in self._gate)
        return {"mean_scaled_utility": sum_s / n_all, "frac_protected": n_prot / n_all}

    def extra_repr(self) -> str:
        return (f"beta={self.beta}, sigma={self.sigma}, eps={self.eps}, include_bias={self.include_bias}, "
                f"gate_scale={self.gate_scale}, n_params={len(self.params)}")
