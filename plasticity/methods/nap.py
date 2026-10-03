"""Normalize-and-Project (NaP) -- constant effective learning rate under normalisation (Lyle et al., 2024).

Reference
---------
C. Lyle, Z. Zheng, E. Nikishin, B. Avila Pires, R. Pascanu, W. Dabney (2024). *Normalization and
effective learning rates in reinforcement learning*. Advances in Neural Information Processing
Systems 37 (NeurIPS 2024). arXiv:2407.01800.

Idea
----
When a normalisation layer follows a linear map, ``LN(alpha * (W x + b)) = LN(W x + b)`` for every
``alpha > 0`` (up to the LayerNorm ``eps``): the network output is invariant to the *scale* of that
layer's parameters. The gradient of a scale-invariant function is inversely proportional to the scale,
``grad_W L  ∝ 1 / ||W||``, so a gradient step of size ``eta`` rotates the weight vector by an angle that
scales like ``eta / ||W||^2``. Parameter-norm growth during training therefore silently *decays* the
effective learning rate (one of the mechanisms behind loss of plasticity); NaP removes this hidden
schedule by projecting each weight matrix back onto the sphere of its *initial* norm after every
optimiser step, so the effective learning rate stays constant throughout training.

Formulation implemented
-----------------------
Let ``theta_l`` denote the (standard-parametrisation) weight matrix of hidden layer ``l`` (every hidden
layer of the :class:`~plasticity.models.MLP` is followed by a LayerNorm when ``model.layer_norm=True``)
and ``r_l = ||theta_l(0)||`` the norm it had at construction time (the matched initialisation
``W0`` of the thesis). After every optimiser step::

    theta_l  <-  theta_l * r_l / ||theta_l||            for every projected layer l   (no_grad)

* ``scope='matrix'`` (default): ``||.||`` is the Frobenius norm of the **whole matrix**. This is the
  invariance that LayerNorm actually provides: it normalises *across* the output units, so only the
  joint scale of the matrix is irrelevant to the function (scaling one row alone changes the output).
* ``scope='row'``: ``||.||`` is taken per output unit (per row of ``theta_l``), i.e. every unit's
  incoming weight vector is projected to its own initial norm. This is the per-neuron variant that is
  exact for per-unit normalisers (e.g. BatchNorm); under LayerNorm it is a stronger constraint than
  required, so it is provided as an option and is **not** the default.
* ``project_bias=True``: the bias is projected **jointly** with the weights, i.e. the norm is taken over
  the concatenation ``[theta_l, theta_bias_l]`` (per matrix, or per row ``[theta_l[i, :], b_i]``) and
  both are scaled by the same factor. Joint scaling is the exact symmetry of ``LN(W x + b)``; projecting
  the bias to its *own* initial norm would be ill-defined for the default zero bias initialisation.
  Default ``False``: only the weight matrices are projected (as in the main experiments of Lyle et al.).
* ``include_output=True``: the output layer is projected as well. It is **not** followed by a
  normalisation, so its scale is *not* a symmetry of the network (the projection then constrains the
  logit scale); default ``False`` leaves the output layer free.

Simplifications / choices
-------------------------
* Standard parametrisation only: rescaling the free parameter Theta of a sin / tanh layer does not
  rescale the effective weight, so ``__init__`` raises ``ValueError`` for any non-standard layer.
* The LayerNorm affine parameters (gain / shift) are not touched. A learnable LayerNorm gain
  re-introduces a free scale in front of the next layer that NaP does not control, which is why the
  recommended configuration is ``model.layer_norm=true, model.ln_affine=false``; a warning is emitted
  when affine LayerNorms are present.
* The optimiser state (momentum / Adam moments) is left unchanged by the projection, as in the paper.
* The projection is a pure parameter operation (no gradient term, independent of the optimiser) and
  costs one fused ``_foreach_norm`` + one fused ``_foreach_mul_`` per step for ``scope='matrix'``.

Config keys (``method:`` section)
---------------------------------
``scope``          (str,  default 'matrix')  'matrix' (Frobenius norm of the whole matrix) or 'row'.
``project_bias``   (bool, default False)     project the bias jointly with the weight matrix.
``include_output`` (bool, default False)     also project the output layer (not scale-invariant).
``eps``            (float, default 1e-12)    lower clamp of the current norm in the division.

``state_summary()`` reports ``norm_growth`` = mean over projected layers of
``||theta_l|| (before the projection of the last step) / r_l`` -- i.e. how much a single update
inflates the norm (``1.0`` = no growth), computed from the norms the projection already calculates --
plus the same quantity per layer (``norm_growth/L{l}``) and ``n_projections``.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..models import MLP
from .base import Method

_SCOPES = ("matrix", "row")


class NormalizeAndProject(Method):
    """Project each hidden weight matrix back to its initial norm after every optimiser step."""

    name = "nap"
    requires_features = False

    def __init__(self, model: MLP, cfg: Dict[str, Any], optimizer_cfg: Dict[str, Any], generator: Optional[torch.Generator] = None):
        super().__init__(model, cfg, optimizer_cfg, generator)
        self.scope = str(cfg.get("scope", "matrix"))
        self.project_bias = bool(cfg.get("project_bias", False))
        self.include_output = bool(cfg.get("include_output", False))
        self.eps = float(cfg.get("eps", 1e-12))
        if self.scope not in _SCOPES:
            raise ValueError(f"nap: scope must be one of {_SCOPES}, got {self.scope!r}")
        if not bool(getattr(model, "layer_norm", False)):
            raise ValueError(
                "nap (Normalize-and-Project) requires a normalisation layer after every hidden linear map: "
                "the projection only preserves the function when the layer output is scale-invariant. "
                "Set model.layer_norm=true (and model.ln_affine=false, the recommended configuration)."
            )
        for li, lin in enumerate(model.linear_layers):
            if lin.mode != "standard":
                raise ValueError(
                    f"nap is defined for the standard parametrisation only; layer {li} has mode={lin.mode!r}. "
                    "Rescaling the free parameter Theta of a sin/tanh layer does not rescale the effective weight."
                )
        if any(isinstance(n, nn.LayerNorm) and n.elementwise_affine for n in model.norms):
            warnings.warn(
                "nap: the model's LayerNorms have affine parameters; their gain re-introduces a free scale that "
                "NaP does not control. The recommended configuration is model.ln_affine=false.",
                stacklevel=2,
            )

        self.layers = list(model.hidden) + ([model.output] if self.include_output else [])
        self.layer_names = [f"hidden.{i}" for i in range(len(model.hidden))] + (["output"] if self.include_output else [])
        self._weights: List[torch.Tensor] = [lin.theta for lin in self.layers]
        # biases projected jointly with their matrix (only where the layer has a bias)
        self._bias_index: List[int] = [i for i, lin in enumerate(self.layers) if self.project_bias and lin.theta_bias is not None]
        self._biases: List[torch.Tensor] = [self.layers[i].theta_bias for i in self._bias_index]
        self._bias_index_t = torch.tensor(self._bias_index, dtype=torch.long)

        with torch.no_grad():
            if self.scope == "matrix":
                init = self._matrix_norms()  # (L,)
            else:
                init = [self._row_norm(i) for i in range(len(self.layers))]  # list of (out_l,)
        for i, n in enumerate(init if self.scope == "row" else [init]):
            if not bool(torch.all(torch.isfinite(n))) or float(n.min()) <= 0.0:
                raise ValueError(f"nap: a projected layer has a zero / non-finite initial norm ({self.layer_names[i] if self.scope == 'row' else 'matrix scope'}); "
                                 "every projected weight (row) must have a positive initial norm.")
        self.init_norms = init.clone() if self.scope == "matrix" else [n.clone() for n in init]
        self._last_growth = torch.ones(len(self.layers))  # per-layer ||theta|| / r before the last projection
        self.n_projections = 0

    # --- norms ------------------------------------------------------------------------------------
    @torch.no_grad()
    def _matrix_norms(self) -> torch.Tensor:
        """Per-layer Frobenius norm of ``[theta_l, theta_bias_l]`` (bias only when projected), shape (L,)."""
        n = torch.stack(torch._foreach_norm(self._weights))
        if self._biases:
            sq = n * n
            sq[self._bias_index_t] += torch.stack(torch._foreach_norm(self._biases)) ** 2
            n = sq.sqrt()
        return n

    @torch.no_grad()
    def _row_norm(self, i: int) -> torch.Tensor:
        w = self._weights[i]
        sq = (w * w).sum(dim=1)
        if self.project_bias and self.layers[i].theta_bias is not None:
            b = self.layers[i].theta_bias
            sq = sq + b * b
        return sq.sqrt()

    # --- projection ---------------------------------------------------------------------------------
    @torch.no_grad()
    def project(self) -> None:
        """``theta_l <- theta_l * r_l / ||theta_l||`` for every projected layer (and jointly its bias)."""
        if self.scope == "matrix":
            cur = self._matrix_norms()
            ratio = self.init_norms / cur.clamp_min(self.eps)
            scalars = ratio.tolist()
            torch._foreach_mul_(self._weights, scalars)
            if self._biases:
                torch._foreach_mul_(self._biases, [scalars[i] for i in self._bias_index])
            self._last_growth = cur / self.init_norms
        else:
            for i, w in enumerate(self._weights):
                cur = self._row_norm(i)
                ratio = self.init_norms[i] / cur.clamp_min(self.eps)
                w.mul_(ratio.unsqueeze(1))
                if self.project_bias and self.layers[i].theta_bias is not None:
                    self.layers[i].theta_bias.mul_(ratio)
                self._last_growth[i] = (cur / self.init_norms[i]).mean()
        self.n_projections += 1

    def after_step(self, step: int, features: Optional[List[torch.Tensor]] = None) -> None:
        self.project()

    # --- reporting ----------------------------------------------------------------------------------
    @torch.no_grad()
    def current_norms(self) -> torch.Tensor:
        """Current per-layer norms in the projection's own convention (matrix scope: (L,); row scope: mean row norm per layer)."""
        if self.scope == "matrix":
            return self._matrix_norms()
        return torch.stack([self._row_norm(i).mean() for i in range(len(self.layers))])

    def state_summary(self) -> Dict[str, Any]:
        g = self._last_growth
        out: Dict[str, Any] = {"norm_growth": float(g.mean()), "n_projections": self.n_projections}
        for i in range(len(self.layers)):
            out[f"norm_growth/L{i}"] = float(g[i])
        return out

    def extra_repr(self) -> str:
        r = self.init_norms.tolist() if self.scope == "matrix" else [float(n.mean()) for n in self.init_norms]
        return (f"scope={self.scope}, project_bias={self.project_bias}, include_output={self.include_output}, "
                f"layers={self.layer_names}, init_norms={[round(x, 4) for x in r]}")
