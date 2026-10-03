"""Continual Backpropagation (CBP): selective re-initialisation of low-utility hidden units.

Reference
---------
Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R. & Sutton, R. S. (2024).
"Loss of plasticity in deep continual learning". *Nature* 632, 768-774. Methods section
"Continual backpropagation" (Algorithm 1) and the public implementation ``lop/algos/gnt.py`` of
github.com/shibhansh/loss-of-plasticity, which fixes the few details the paper leaves open.

Formulation implemented (per hidden layer ``l`` with post-activation features ``h_l`` of shape (B, n_l))
---------------------------------------------------------------------------------------------------
After every optimiser step (``after_step``), for every hidden unit ``i`` of every hidden layer:

    a_i      <- a_i + 1                                                    (age, steps since (re)init)
    f_i      <- eta * f_i + (1 - eta) * mean_batch(h_i)                     (running mean of the activation)
    f^_i     =  f_i / (1 - eta^a_i)                                         (bias-corrected running mean)
    u_i      <- eta * u_i + (1 - eta) * mean_batch |h_i - f^_i| * sum_k |W^{l+1}_{k,i}| / sum_j |W^{l}_{i,j}|
    u^_i     =  u_i / (1 - eta^a_i)                                         (bias-corrected utility, used for ranking)

This is the *mean-corrected contribution utility with the adaptation term* of the Nature paper
("adaptable contribution" in the public code; ``utility: contribution`` and
``utility: adaptable_contribution`` both select it). ``W^{l}`` / ``W^{l+1}`` are the **effective**
weights of layer ``l`` (incoming, row ``i``) and of the next layer (outgoing, column ``i``;
``model.output`` for the last hidden layer), obtained through ``ReparamLinear.effective_weight()`` so
the method is defined identically for the standard, sin and tanh parametrisations. As in the public
implementation, the running mean is updated *before* the utility and enters it bias-corrected. The
public code uses the *mean* of |W| over the row / column; we use the *sum* as written in the paper -
the two differ by a layer-wide constant and give the same ranking.

Generate-and-test (per layer, every step): the eligible units are those with ``a_i > m``;
``n_l += rho * n_eligible`` accumulates a real-valued counter; when ``n_l >= 1`` the
``floor(n_l)`` eligible units with the *lowest* bias-corrected utility are replaced and
``n_l -= floor(n_l)``. Replacing unit ``i`` of layer ``l`` means
  * incoming weights re-drawn from the layer's own initialiser ``d_l`` (``ReparamLinear.reinit_output_units``:
    Kaiming-uniform effective weights written into Theta through the inverse map, bias re-drawn by the
    layer's ``bias_init`` rule, i.e. zero by default),
  * outgoing weights of the next layer set to zero (``zero_input_units``; Theta = 0 => W = 0 in every mode),
  * ``u_i, f_i, a_i <- 0``,
  * the optimiser state of the touched rows / columns (Adam moments, SGD momentum buffer) zeroed, as in the
    public implementation.

Simplifications / choices: (i) feed-forward MLP hidden layers only; (ii) the next-layer bias
compensation ``b_{l+1} += W^{l+1}_{:,i} f^_i`` performed by the public code when outgoing weights are
zeroed is **off** by default (``compensate_bias: false``) because Algorithm 1 of the paper does not
include it; it can be switched on; (iii) for sin / tanh layers a replaced unit restarts on the
principal branch of the inverse map (|Theta| <= pi/2 * A), exactly like a freshly initialised unit.

Hyper-parameters (``method:`` config section) and defaults = Online Permuted MNIST values of the paper:
``replacement_rate`` rho = 1e-4, ``maturity_threshold`` m = 100, ``decay_rate`` eta = 0.99,
``utility`` = 'contribution' (alias 'adaptable_contribution'), ``compensate_bias`` = false.

Complexity: everything is vectorised over units; a step costs O(sum_l n_l * (n_{l-1} + n_{l+1})) tensor
work (the |W| row / column sums) and no Python loop over units; replacement events (rare at rho=1e-4)
are the only place with indexed writes.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch

from ..models import MLP, ReparamLinear
from .base import Method

_UTILITY_TYPES = {"contribution", "adaptable_contribution"}
_KNOWN_KEYS = {"replacement_rate", "maturity_threshold", "decay_rate", "utility", "compensate_bias"}
_EPS = 1e-12


class ContinualBackprop(Method):
    """Continual backpropagation (Dohare et al., 2024): generate-and-test on hidden units."""

    name = "continual_backprop"
    requires_features = True

    def __init__(self, model: MLP, cfg: Dict[str, Any], optimizer_cfg: Dict[str, Any], generator: Optional[torch.Generator] = None):
        super().__init__(model, cfg, optimizer_cfg, generator)
        unknown = set(cfg) - _KNOWN_KEYS
        if unknown:
            raise ValueError(f"continual_backprop: unknown config keys {sorted(unknown)}; known: {sorted(_KNOWN_KEYS)}")
        self.replacement_rate = float(cfg.get("replacement_rate", 1e-4))
        self.maturity_threshold = int(cfg.get("maturity_threshold", 100))
        self.decay_rate = float(cfg.get("decay_rate", 0.99))
        self.utility_type = str(cfg.get("utility", "contribution"))
        self.compensate_bias = bool(cfg.get("compensate_bias", False))
        if not 0.0 <= self.replacement_rate <= 1.0:
            raise ValueError(f"replacement_rate must be in [0, 1], got {self.replacement_rate}")
        if self.maturity_threshold < 0:
            raise ValueError(f"maturity_threshold must be >= 0, got {self.maturity_threshold}")
        if not 0.0 <= self.decay_rate < 1.0:
            raise ValueError(f"decay_rate must be in [0, 1), got {self.decay_rate}")
        if self.utility_type not in _UTILITY_TYPES:
            raise ValueError(f"utility must be one of {sorted(_UTILITY_TYPES)}, got {self.utility_type!r}")

        self.layers: List[ReparamLinear] = model.linear_layers  # hidden layers followed by the output layer
        self.n_hidden = len(model.hidden)
        if self.n_hidden == 0:
            raise ValueError("continual_backprop needs at least one hidden layer")
        widths = [lin.out_features for lin in model.hidden]
        # per-unit buffers (one tensor per hidden layer)
        self.ages: List[torch.Tensor] = [torch.zeros(n, dtype=torch.long) for n in widths]
        self.mean_act: List[torch.Tensor] = [torch.zeros(n) for n in widths]
        self.utility: List[torch.Tensor] = [torch.zeros(n) for n in widths]
        # generate-and-test bookkeeping
        self.to_replace: List[float] = [0.0] * self.n_hidden  # accumulated (fractional) number of units to replace
        self.replaced: List[int] = [0] * self.n_hidden
        self.replaced_total = 0

    # --- helpers -----------------------------------------------------------------------------------
    def _bias_correction(self, l: int) -> torch.Tensor:
        """``1 - eta^age`` per unit (0 for age 0)."""
        return 1.0 - torch.pow(self.decay_rate, self.ages[l].to(torch.float32))

    def corrected_utility(self, l: int) -> torch.Tensor:
        """Bias-corrected utility trace ``u / (1 - eta^age)`` of layer ``l`` (0 for units of age 0)."""
        corr = self._bias_correction(l)
        return torch.where(self.ages[l] > 0, self.utility[l] / corr.clamp_min(_EPS), torch.zeros_like(self.utility[l]))

    def _reset_optimizer_state(self, param: Optional[torch.Tensor], rows: Optional[torch.Tensor] = None, cols: Optional[torch.Tensor] = None) -> None:
        if param is None or self.optimizer is None:
            return
        state = self.optimizer.state.get(param)
        if not state:
            return
        for val in state.values():  # exp_avg / exp_avg_sq / max_exp_avg_sq / momentum_buffer ... (scalars such as 'step' are skipped)
            if torch.is_tensor(val) and val.shape == param.shape:
                if rows is not None:
                    val[rows] = 0.0
                if cols is not None:
                    val[:, cols] = 0.0

    @torch.no_grad()
    def _replace_units(self, l: int, units: torch.Tensor) -> None:
        lin, nxt = self.layers[l], self.layers[l + 1]
        if self.compensate_bias and nxt.theta_bias is not None:
            # keep the next layer's pre-activation mean unchanged: b += sum_i W[:, i] * f^_i  (public implementation)
            f_hat = self.mean_act[l][units] / self._bias_correction(l)[units].clamp_min(_EPS)
            b = nxt.effective_bias() + (nxt.effective_weight()[:, units] * f_hat).sum(dim=1)
            nxt.set_effective_bias(b)
        lin.reinit_output_units(units, self.generator)  # incoming weights + bias from the initialiser
        nxt.zero_input_units(units)  # outgoing weights -> 0
        self.utility[l][units] = 0.0
        self.mean_act[l][units] = 0.0
        self.ages[l][units] = 0
        self._reset_optimizer_state(lin.theta, rows=units)
        self._reset_optimizer_state(lin.theta_bias, rows=units)
        self._reset_optimizer_state(nxt.theta, cols=units)
        k = int(units.numel())
        self.replaced[l] += k
        self.replaced_total += k

    # --- step hook ---------------------------------------------------------------------------------
    @torch.no_grad()
    def after_step(self, step: int, features: Optional[List[torch.Tensor]] = None) -> None:
        if features is None:
            raise ValueError("ContinualBackprop.after_step needs the hidden-layer features (requires_features=True)")
        if len(features) != self.n_hidden:
            raise ValueError(f"expected {self.n_hidden} feature tensors, got {len(features)}")
        eta, rho, m = self.decay_rate, self.replacement_rate, self.maturity_threshold
        # effective weights of all layers once (after the optimiser update, as in the reference implementation)
        W = [lin.effective_weight() for lin in self.layers]
        selected: List[Optional[torch.Tensor]] = [None] * self.n_hidden
        # ---- test phase: update the traces and pick the units to replace -------------------------
        for l in range(self.n_hidden):
            h = features[l].detach()
            if h.dim() == 1:
                h = h.unsqueeze(0)
            ages = self.ages[l]
            ages += 1
            corr = 1.0 - torch.pow(eta, ages.to(torch.float32))  # ages >= 1 here -> corr > 0
            f = self.mean_act[l]
            f.mul_(eta).add_(h.mean(dim=0), alpha=1.0 - eta)
            f_hat = f / corr
            out_mag = W[l + 1].abs().sum(dim=0)  # sum_k |W^{l+1}_{k,i}|   (outgoing)
            in_mag = W[l].abs().sum(dim=1)  # sum_j |W^{l}_{i,j}|     (incoming)
            new_u = (h - f_hat).abs().mean(dim=0) * out_mag / in_mag.clamp_min(_EPS)
            u = self.utility[l]
            u.mul_(eta).add_(new_u, alpha=1.0 - eta)
            if rho <= 0.0:
                continue
            eligible = ages > m
            n_eligible = int(eligible.sum())
            if n_eligible == 0:
                continue
            self.to_replace[l] += n_eligible * rho
            if self.to_replace[l] < 1.0:
                continue
            k = min(int(math.floor(self.to_replace[l])), n_eligible)
            self.to_replace[l] -= k
            idx = eligible.nonzero(as_tuple=False).squeeze(1)
            u_hat = u[idx] / corr[idx]
            selected[l] = idx[torch.topk(u_hat, k, largest=False).indices]
        # ---- generate phase: re-initialise the selected units (rare) ------------------------------
        for l, units in enumerate(selected):
            if units is not None:
                self._replace_units(l, units)

    # --- reporting ---------------------------------------------------------------------------------
    def state_summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"units_replaced_total": int(self.replaced_total)}
        utils = []
        for l in range(self.n_hidden):
            out[f"units_replaced/L{l}"] = int(self.replaced[l])
            u_hat = self.corrected_utility(l)
            out[f"mean_utility/L{l}"] = float(u_hat.mean())
            utils.append(u_hat)
        out["mean_utility"] = float(torch.cat(utils).mean())
        out["mean_age"] = float(torch.cat([a.to(torch.float32) for a in self.ages]).mean())
        return out

    def extra_repr(self) -> str:
        return (f"replacement_rate={self.replacement_rate}, maturity_threshold={self.maturity_threshold}, "
                f"decay_rate={self.decay_rate}, utility={self.utility_type}, compensate_bias={self.compensate_bias}")
