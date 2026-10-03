"""Parseval regularisation -- orthogonality penalty on the weight matrices (Chung et al., 2024).

Reference
---------
W. Chung, L. Cherif, D. Meger, D. Precup (2024). *Parseval Regularization for Continual Reinforcement
Learning*. NeurIPS 2024 (arXiv:2412.07224). The regulariser itself goes back to the Parseval networks
of M. Cisse, P. Bojanowski, E. Grave, Y. Dauphin, N. Usunier (2017), *Parseval Networks: Improving
Robustness to Adversarial Examples*, ICML 2017. Chung et al. show that keeping the weight matrices
(approximately) orthogonal throughout training preserves plasticity (gradient propagation, feature
rank) in sequences of RL tasks; here it is used as a published comparison method for the supervised
continual streams of the thesis.

Formulation implemented
-----------------------
For every regularised layer ``l`` with effective weight matrix ``W_l ∈ R^{out × in}``::

    R(W) = beta * sum_l || G_l - I ||_F^2,       G_l = W_l W_lᵀ   (out <= in: rows orthonormal)
                                                   G_l = W_lᵀ W_l   (out >  in: columns orthonormal)

The smaller of the two Gram matrices is used because the larger one is necessarily rank deficient and
``|| · - I||`` could never reach zero. The term is **added to the loss** (returned as a scalar tensor by
:meth:`regularizer`); its gradient is obtained by autograd, so the hooks ``before_step`` / ``after_step``
are not used. With plain SGD the update of a regularised layer is therefore
``ΔW = -η (∇_W L_task + 4 beta (W Wᵀ - I) W)`` (``out <= in``).

Scope: ``scope='hidden'`` (default, as in Chung et al., who leave the output / policy head
unregularised) applies the penalty to the hidden layers only; ``scope='all'`` also includes the output
layer. Biases, LayerNorm parameters and (learnable) amplitudes are never regularised.

Parametrisation: the penalty is a function of the **effective** weights ``W_l = f(Θ_l)``
(:meth:`ReparamLinear.effective_weight`), so it is the *same objective* for every parametrisation
(``standard``: ``W = Θ``; ``sin``: ``W = A sin(Θ)``; ``tanh``: ``W = A tanh(Θ)``). For the bounded maps the
autograd gradient in Θ-space picks up the Jacobian ``∂W/∂Θ`` exactly like the data gradient does, i.e.
``∇_Θ R = (∂W/∂Θ) ⊙ ∇_W R``. Orthonormal rows of a ``784``- or ``100``-input layer have entries of
order ``1/√in``, well inside the bound ``A = γ b`` of the thesis' matched initialisation, so the
regulariser is attainable in every parametrisation. No ``ValueError`` is raised for reparametrised
layers (the method is documented to work on effective weights).

Simplifications / choices
-------------------------
* Unit target ``I`` (literal Parseval constraint). With the matched Kaiming-uniform initialisation
  ``E[W0 W0ᵀ] = gain² I`` (``gain² = 2`` for ReLU), so at ``t = 0`` the penalty also *rescales* the rows
  from norm ``√2`` toward ``1``; no scale parameter / orthogonal initialisation is added (the thesis
  requires the common ``W0`` for all methods).
* The optional *diversity* regulariser of Chung et al. (on the hidden representations) is not part of
  this implementation; only the Parseval (orthogonality) term is compared.
* Dense layers only (the models of the thesis are MLPs); no convolutional variant.
* Cost: one ``out × out`` (or ``in × in``) Gram matrix per regularised layer per step plus its backward
  (≈ 3 small matmuls); for the 784-100-100-100-10 PMNIST network this is ~30 MFLOP per online step,
  the dominant cost of the method but still CPU-friendly.

Config keys (``method:`` section)
---------------------------------
``beta``   (float, default 1e-3)       regularisation strength β (``0`` disables the term).
``scope``  (str,   default 'hidden')   ``'hidden'`` | ``'all'`` (``'all'`` includes the output layer).

``state_summary()`` reports ``orth_residual`` = mean over the regularised layers of ``||G_l - I||_F``,
the per-layer values ``orth_residual/L{i}`` (``i`` = index in ``model.linear_layers``) and
``reg_loss`` = the current value of ``R(W)``. All are computed once per task without autograd.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch

from ..models import MLP, ReparamLinear
from .base import Method

_KNOWN_KEYS = {"beta", "scope"}
_SCOPES = ("hidden", "all")


class Parseval(Method):
    """``regularizer() = beta * sum_l ||W_l W_lᵀ - I||_F²`` on the effective weights (autograd)."""

    name = "parseval"
    requires_features = False

    def __init__(self, model: MLP, cfg: Dict[str, Any], optimizer_cfg: Dict[str, Any], generator: Optional[torch.Generator] = None):
        super().__init__(model, cfg, optimizer_cfg, generator)
        unknown = set(cfg) - _KNOWN_KEYS
        if unknown:
            raise ValueError(f"parseval: unknown config keys {sorted(unknown)}; known: {sorted(_KNOWN_KEYS)}")
        self.beta = float(cfg.get("beta", 1e-3))
        self.scope = str(cfg.get("scope", "hidden"))
        if not math.isfinite(self.beta) or self.beta < 0.0:
            raise ValueError(f"parseval: beta must be a finite number >= 0, got {self.beta}")
        if self.scope not in _SCOPES:
            raise ValueError(f"parseval: scope must be one of {_SCOPES}, got {self.scope!r}")
        all_layers = model.linear_layers
        n_hidden = len(model.hidden)
        self.layer_indices: List[int] = list(range(n_hidden)) if self.scope == "hidden" else list(range(len(all_layers)))
        self.layers: List[ReparamLinear] = [all_layers[i] for i in self.layer_indices]
        # Identity of the (smaller) Gram matrix of each layer, pre-allocated once.
        self._eyes: List[torch.Tensor] = [
            torch.eye(min(lin.out_features, lin.in_features), dtype=lin.theta.dtype, device=lin.theta.device) for lin in self.layers
        ]

    # --- helpers -------------------------------------------------------------------------------
    @staticmethod
    def gram(w: torch.Tensor) -> torch.Tensor:
        """``W Wᵀ`` if ``out <= in`` else ``Wᵀ W`` (the Gram matrix that can equal the identity)."""
        return w @ w.transpose(0, 1) if w.shape[0] <= w.shape[1] else w.transpose(0, 1) @ w

    def _residual(self, i: int, w: torch.Tensor) -> torch.Tensor:
        eye = self._eyes[i]
        if eye.dtype != w.dtype or eye.device != w.device:  # model moved / cast after construction
            eye = self._eyes[i] = eye.to(dtype=w.dtype, device=w.device)
        return self.gram(w) - eye

    # --- hooks -----------------------------------------------------------------------------------
    def regularizer(self) -> Optional[torch.Tensor]:
        """``beta * sum_l ||G_l - I||_F²`` as a scalar tensor attached to the autograd graph (``None`` if β = 0)."""
        if self.beta == 0.0 or not self.layers:
            return None
        total: Optional[torch.Tensor] = None
        for i, lin in enumerate(self.layers):
            r = self._residual(i, lin.effective_weight())
            term = (r * r).sum()
            total = term if total is None else total + term
        return self.beta * total

    # --- reporting ---------------------------------------------------------------------------------
    @torch.no_grad()
    def residual_norms(self) -> List[float]:
        """``||G_l - I||_F`` for every regularised layer (no autograd)."""
        return [float(self._residual(i, lin.effective_weight()).norm()) for i, lin in enumerate(self.layers)]

    @torch.no_grad()
    def penalty(self) -> float:
        """Current value of ``R(W)``."""
        return self.beta * sum(n * n for n in self.residual_norms())

    def state_summary(self) -> Dict[str, Any]:
        norms = self.residual_norms()
        out: Dict[str, Any] = {"orth_residual": (sum(norms) / len(norms)) if norms else 0.0}
        for li, n in zip(self.layer_indices, norms):
            out[f"orth_residual/L{li}"] = n
        out["reg_loss"] = self.beta * sum(n * n for n in norms)
        return out

    def extra_repr(self) -> str:
        return f"beta={self.beta}, scope={self.scope}, layers={self.layer_indices}"
