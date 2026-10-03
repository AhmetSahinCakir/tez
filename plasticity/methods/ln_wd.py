"""Layer Normalisation + weight decay (LN+WD) -- the strong simple baseline of Lyle et al.

Reference
---------
C. Lyle, Z. Zheng, K. Khetarpal, H. van Hasselt, R. Pascanu, J. Martens, W. Dabney. *Disentangling
the Causes of Plasticity Loss in Neural Networks*. arXiv:2402.18762 (cited as Lyle et al., 2025 in the
thesis bibliography). The paper's empirical conclusion is that layer normalisation combined with
(a small amount of) weight decay addresses several of the identified plasticity-loss mechanisms at
once -- parameter-norm growth / effective-learning-rate decay (through LN's scale invariance plus the
decay pulling the norm back), preactivation shift and unit saturation (LN re-centres and re-scales
the pre-activations), and loss-landscape sharpening -- and is competitive with far more elaborate
interventions. It is therefore the "simple strong baseline" the proposed reparametrisation is compared
against.

Formulation implemented
-----------------------
The *model* provides the LayerNorm (``model.layer_norm=True`` is required: every hidden linear map of
the :class:`~plasticity.models.MLP` is followed by ``LayerNorm``); this method provides the weight decay
through the **optimiser**, with coefficient ``lam`` (config key ``weight_decay``) on a *decayed* parameter
group and ``0`` on a *non-decayed* group, reusing :func:`plasticity.methods.base.build_optimizer`::

    SGD  / Adam  : g <- g + lam * theta ; standard update with g          (coupled L2, i.e. the gradient of
                                                                            lam/2 * ||theta||^2 is added)
    AdamW        : theta <- theta - lr * lam * theta ; Adam update          (decoupled weight decay,
                                                                            Loshchilov & Hutter 2019)

i.e. exactly the semantics PyTorch's ``SGD`` / ``Adam`` / ``AdamW`` give to ``weight_decay``, selected by
``optimizer.name``. Parameter groups:

* decayed      : every ``ReparamLinear.theta`` (all weight matrices, hidden and output) and, when
                 ``decay_bias=True`` (default), every ``ReparamLinear.theta_bias``;
                 plus the LayerNorm affine parameters when ``decay_ln=True``.
* non-decayed  : the LayerNorm affine parameters (gain / shift) when ``decay_ln=False`` (default) --
                 decaying the LN gain towards zero would shrink the signal LN just normalised --
                 and the biases when ``decay_bias=False``. (Any other trainable parameter the model might
                 expose is placed here as well, so nothing is silently left out of the optimiser.)

Simplifications / choices
-------------------------
* ``weight_decay`` is a *method* hyper-parameter. The ``optimizer.weight_decay`` entry of the optimiser
  section must be ``0`` (or equal to the method value); otherwise ``__init__`` raises, so the decay
  strength is never specified twice with different values.
* Standard parametrisation only: decaying the free parameter Theta of a sin / tanh layer is not the
  published method (it would be a decay in Theta-space, not of the effective weight), so ``__init__``
  raises ``ValueError`` for any non-standard layer.
* No extra loss term: ``regularizer()`` returns ``None``; the decay lives entirely in the optimiser.

Config keys (``method:`` section)
---------------------------------
``weight_decay`` (float, default 1e-3)  decay coefficient ``lam`` of the decayed group.
``decay_bias``   (bool,  default True)  decay the linear-layer biases too.
``decay_ln``     (bool,  default False) decay the LayerNorm affine parameters too.

``state_summary()`` reports ``w_fro_norm`` = ``sqrt(sum_l ||theta_l||_F^2)`` over all weight matrices
(which are always decayed) and ``decayed_fro_norm`` = the same joint norm over *every* tensor of the
decayed group.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..models import MLP
from .base import Method, build_optimizer


class LayerNormWeightDecay(Method):
    """LayerNorm network (model side) + optimiser weight decay on the linear-layer parameters."""

    name = "ln_wd"
    requires_features = False

    def __init__(self, model: MLP, cfg: Dict[str, Any], optimizer_cfg: Dict[str, Any], generator: Optional[torch.Generator] = None):
        super().__init__(model, cfg, optimizer_cfg, generator)
        self.weight_decay = float(cfg.get("weight_decay", 1e-3))
        self.decay_bias = bool(cfg.get("decay_bias", True))
        self.decay_ln = bool(cfg.get("decay_ln", False))
        if self.weight_decay < 0:
            raise ValueError(f"ln_wd: weight_decay must be >= 0, got {self.weight_decay}")
        if not bool(getattr(model, "layer_norm", False)):
            raise ValueError(
                "ln_wd (LayerNorm + weight decay) requires a LayerNorm after every hidden layer: "
                "set model.layer_norm=true (weight decay alone, without normalisation, is a different baseline: "
                "use method.name=baseline with optimizer.weight_decay instead)."
            )
        for li, lin in enumerate(model.linear_layers):
            if lin.mode != "standard":
                raise ValueError(
                    f"ln_wd is defined for the standard parametrisation only; layer {li} has mode={lin.mode!r}. "
                    "Weight decay on the free parameter Theta of a sin/tanh layer is not the published method."
                )
        opt_wd = float(optimizer_cfg.get("weight_decay", 0.0) or 0.0)
        if opt_wd != 0.0 and not math.isclose(opt_wd, self.weight_decay, rel_tol=1e-9):
            raise ValueError(
                f"ln_wd: optimizer.weight_decay={opt_wd} conflicts with method.weight_decay={self.weight_decay}; "
                "set optimizer.weight_decay=0 and configure the decay through the method section."
            )

        decayed: List[torch.nn.Parameter] = []
        non_decayed: List[torch.nn.Parameter] = []
        self.weights: List[torch.nn.Parameter] = []
        for lin in model.linear_layers:
            decayed.append(lin.theta)
            self.weights.append(lin.theta)
            if lin.theta_bias is not None:
                (decayed if self.decay_bias else non_decayed).append(lin.theta_bias)
        self.ln_params: List[torch.nn.Parameter] = [p for n in model.norms if isinstance(n, nn.LayerNorm) for p in n.parameters()]
        (decayed if self.decay_ln else non_decayed).extend(self.ln_params)
        seen = {id(p) for p in decayed} | {id(p) for p in non_decayed}
        for p in model.parameters():  # anything else trainable (none for the plain MLP) is trained without decay
            if p.requires_grad and id(p) not in seen:
                non_decayed.append(p)
                seen.add(id(p))
        self.decayed, self.non_decayed = decayed, non_decayed

    # --- lifecycle ---------------------------------------------------------------------------------
    def build_optimizer(self) -> torch.optim.Optimizer:
        groups: List[Dict[str, Any]] = [{"params": self.decayed, "weight_decay": self.weight_decay}]
        if self.non_decayed:
            groups.append({"params": self.non_decayed, "weight_decay": 0.0})
        opt_cfg = dict(self.optimizer_cfg)
        opt_cfg["weight_decay"] = self.weight_decay  # optimiser-level default; every group sets its own value explicitly
        self.optimizer = build_optimizer(groups, opt_cfg)
        return self.optimizer

    def regularizer(self) -> Optional[torch.Tensor]:
        return None  # the decay is applied by the optimiser, see module docstring

    # --- reporting ----------------------------------------------------------------------------------
    @staticmethod
    @torch.no_grad()
    def _joint_norm(tensors: List[torch.Tensor]) -> float:
        if not tensors:
            return 0.0
        return float(torch.stack(torch._foreach_norm(tensors)).norm())

    def state_summary(self) -> Dict[str, Any]:
        return {"w_fro_norm": self._joint_norm(self.weights), "decayed_fro_norm": self._joint_norm(self.decayed)}

    def extra_repr(self) -> str:
        return (f"weight_decay={self.weight_decay}, decay_bias={self.decay_bias}, decay_ln={self.decay_ln}, "
                f"n_decayed={len(self.decayed)}, n_non_decayed={len(self.non_decayed)}")
