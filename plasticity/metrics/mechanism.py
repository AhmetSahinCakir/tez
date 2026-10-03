"""Mechanism-level diagnostics of loss of plasticity.

Weight / activation / rank / gradient statistics follow Dohare et al. (2024); Fisher-information
quantities follow Chen & Zhang (2026) **with the coordinate caveat of the thesis**: for a
reparametrised layer ``W = f(Θ)`` with Jacobian ``J = ∂W/∂Θ`` one has ``F_Θ = Jᵀ F_W J``, so
``Tr(F_Θ) ≠ Tr(F_W)`` and ``‖∇_Θ L‖ ≠ ‖∇_W L‖``. Every gradient / Fisher quantity is therefore
reported in *both* coordinate systems (``*_theta`` = free parameters, ``*_w`` = effective weights);
the effective-weight coordinates are the ones comparable across parametrisations.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..models import MLP, ReparamLinear


def _f(x) -> float:
    return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)


# ----------------------------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def weight_statistics(model: MLP) -> Dict[str, float]:
    """Magnitude / norm of effective weights, plus reparametrisation-specific saturation statistics.

    Keys (``L`` = layer index, ``all`` = pooled over all weight matrices):
      ``w_abs_mean``, ``w_fro_norm``, ``w_abs_max`` (effective weights W, biases excluded)
      ``w_over_A_mean``  mean |W|/A           (reparam layers only; 1 = at the bound)
      ``jac_mean``       mean |cos(angle)| (sin) / mean (1-tanh²(angle)) (tanh): normalised Jacobian
      ``jac_sq_mean``    mean cos²(angle): the state-dependent factor of the effective step (η_eff, H3)
      ``sat_frac``       fraction of weights with normalised Jacobian < 0.1 (nearly frozen at the bound)
      ``theta_abs_mean`` mean |Θ| (free parameters incl. biases)
    """
    out: Dict[str, float] = {}
    abs_sum, sq_sum, n_all, max_all = 0.0, 0.0, 0, 0.0
    th_abs, th_n = 0.0, 0
    rep_terms = {"w_over_A": [0.0, 0], "jac": [0.0, 0], "jac_sq": [0.0, 0], "sat": [0.0, 0]}
    for li, lin in enumerate(model.linear_layers):
        w = lin.effective_weight()
        a = w.abs()
        out[f"w_abs_mean/L{li}"] = _f(a.mean())
        out[f"w_fro_norm/L{li}"] = _f(w.norm())
        out[f"w_abs_max/L{li}"] = _f(a.max())
        abs_sum += _f(a.sum()); sq_sum += _f((w * w).sum()); n_all += w.numel(); max_all = max(max_all, _f(a.max()))
        th_abs += _f(lin.theta.abs().sum()); th_n += lin.theta.numel()
        if lin.theta_bias is not None:
            th_abs += _f(lin.theta_bias.abs().sum()); th_n += lin.theta_bias.numel()
        if lin.mode != "standard":
            A = float(lin.amplitude)
            jac = lin.normalized_jacobian()
            r = a / A
            out[f"w_over_A_mean/L{li}"] = _f(r.mean())
            out[f"jac_mean/L{li}"] = _f(jac.abs().mean())
            out[f"jac_sq_mean/L{li}"] = _f((jac * jac).mean())
            out[f"sat_frac/L{li}"] = _f((jac.abs() < 0.1).float().mean())
            rep_terms["w_over_A"][0] += _f(r.sum()); rep_terms["w_over_A"][1] += r.numel()
            rep_terms["jac"][0] += _f(jac.abs().sum()); rep_terms["jac"][1] += jac.numel()
            rep_terms["jac_sq"][0] += _f((jac * jac).sum()); rep_terms["jac_sq"][1] += jac.numel()
            rep_terms["sat"][0] += _f((jac.abs() < 0.1).float().sum()); rep_terms["sat"][1] += jac.numel()
    out["w_abs_mean/all"] = abs_sum / max(n_all, 1)
    out["w_fro_norm/all"] = math.sqrt(sq_sum)
    out["w_abs_max/all"] = max_all
    out["theta_abs_mean/all"] = th_abs / max(th_n, 1)
    for k, (s, n) in rep_terms.items():
        if n:
            out[f"{k}_mean/all" if k != "sat" else "sat_frac/all"] = s / n
    return out


# ----------------------------------------------------------------------------------------------
# Representations
# ----------------------------------------------------------------------------------------------
def _stable_rank(h: torch.Tensor) -> float:
    fro2 = float((h * h).sum())
    if fro2 <= 0:
        return 0.0
    s = torch.linalg.svdvals(h)
    return fro2 / float(s[0] ** 2) if s.numel() else 0.0


def _effective_rank(h: torch.Tensor) -> float:
    """Roy & Vetterli (2007): exp(entropy of normalised singular values)."""
    s = torch.linalg.svdvals(h)
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()))


@torch.no_grad()
def representation_statistics(model: MLP, x_probe: torch.Tensor, dead_tol: float = 0.0) -> Dict[str, float]:
    """Dead-unit fraction and (stable / effective) rank of each hidden representation on a probe batch.

    ``dead``: a unit is dead if its post-activation is ≤ ``dead_tol`` for *every* probe sample (ReLU
    semantics, Dohare et al. 2024). ``inactive``: unit whose activation standard deviation over the
    probe is < 1e-6 (constant output -> contributes no information; meaningful for all activations).
    """
    model.eval()
    _, feats = model(x_probe, return_features=True)
    model.train()
    out: Dict[str, float] = {}
    dead_tot, inact_tot, n_tot = 0.0, 0.0, 0
    for li, h in enumerate(feats):
        h = h.detach().float()
        dead = (h.max(dim=0).values <= dead_tol).float().mean()
        inactive = (h.std(dim=0) < 1e-6).float().mean()
        out[f"dead_frac/L{li}"] = _f(dead)
        out[f"inactive_frac/L{li}"] = _f(inactive)
        out[f"stable_rank/L{li}"] = _stable_rank(h)
        out[f"effective_rank/L{li}"] = _effective_rank(h)
        hc = h - h.mean(dim=0, keepdim=True)
        out[f"stable_rank_c/L{li}"] = _stable_rank(hc)
        out[f"effective_rank_c/L{li}"] = _effective_rank(hc)
        out[f"act_abs_mean/L{li}"] = _f(h.abs().mean())
        out[f"act_fro_norm/L{li}"] = _f(h.norm()) / math.sqrt(h.shape[0])
        dead_tot += _f(dead) * h.shape[1]; inact_tot += _f(inactive) * h.shape[1]; n_tot += h.shape[1]
    if n_tot:
        out["dead_frac/all"] = dead_tot / n_tot
        out["inactive_frac/all"] = inact_tot / n_tot
    if feats:
        out["stable_rank/last"] = out[f"stable_rank/L{len(feats)-1}"]
        out["effective_rank/last"] = out[f"effective_rank/L{len(feats)-1}"]
        out["stable_rank_c/last"] = out[f"stable_rank_c/L{len(feats)-1}"]
        out["effective_rank_c/last"] = out[f"effective_rank_c/L{len(feats)-1}"]
    return out


# ----------------------------------------------------------------------------------------------
# Gradients and Fisher information (both coordinate systems)
# ----------------------------------------------------------------------------------------------
def _flat_grads(model: MLP, coords: str) -> torch.Tensor:
    parts = []
    for lin in model.linear_layers:
        if coords == "theta":
            g = lin.theta.grad
            gb = lin.theta_bias.grad if lin.theta_bias is not None else None
        else:  # effective-weight coordinates
            if lin.mode == "standard":
                g = lin.theta.grad
            else:
                g = lin.last_weight.grad if lin.last_weight is not None else None
            if lin.theta_bias is None:
                gb = None
            elif lin.bias_mode == "standard":
                gb = lin.theta_bias.grad
            else:
                gb = lin.last_bias.grad if lin.last_bias is not None else None
        parts.append(torch.zeros_like(lin.theta).flatten() if g is None else g.detach().flatten())
        if lin.theta_bias is not None:
            parts.append(torch.zeros_like(lin.theta_bias).flatten() if gb is None else gb.detach().flatten())
    return torch.cat(parts)


def gradient_fisher_statistics(model: MLP, x_probe: torch.Tensor, y_probe: torch.Tensor, n_samples: int = 32,
                               generator: Optional[torch.Generator] = None) -> Dict[str, float]:
    """Per-sample gradient norms and Fisher-information statistics in Θ- and W-coordinates.

    * ``grad_norm_theta|w``        mean ‖∇ log p(y|x)‖ over probe samples with the *true* labels
    * ``fisher_trace_theta|w``     Tr(F) ≈ mean ‖∇ log p(ŷ|x)‖² with ŷ ~ p(·|x)  (Monte-Carlo Fisher)
    * ``fisher_erank_theta|w``     effective rank of the n×n Gram matrix of those gradients
                                   (= spectrum of the sampled Fisher restricted to the probe)
    * ``jac_fro_ratio``            ‖∇_Θ‖ / ‖∇_W‖ (pooled), the Jacobian scaling of the gradient
    """
    n = min(int(n_samples), int(x_probe.shape[0]))
    was_training = model.training
    model.eval()
    model.set_capture_effective(True)
    grads_true = {"theta": [], "w": []}
    grads_fish = {"theta": [], "w": []}
    for i in range(n):
        x = x_probe[i : i + 1]
        # true-label gradient
        model.zero_grad(set_to_none=True)
        logits = model(x)
        F.cross_entropy(logits, y_probe[i : i + 1]).backward()
        grads_true["theta"].append(_flat_grads(model, "theta"))
        grads_true["w"].append(_flat_grads(model, "w"))
        # sampled-label gradient (Fisher)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            y_s = torch.multinomial(probs, 1, generator=generator).squeeze(1)
        F.cross_entropy(logits, y_s).backward()
        grads_fish["theta"].append(_flat_grads(model, "theta"))
        grads_fish["w"].append(_flat_grads(model, "w"))
    model.zero_grad(set_to_none=True)
    model.set_capture_effective(False)
    model.train(was_training)

    out: Dict[str, float] = {}
    for coords in ("theta", "w"):
        Gt = torch.stack(grads_true[coords])
        Gf = torch.stack(grads_fish[coords])
        out[f"grad_norm_{coords}"] = _f(Gt.norm(dim=1).mean())
        out[f"fisher_trace_{coords}"] = _f((Gf * Gf).sum(dim=1).mean())
        gram = Gf @ Gf.T / n
        ev = torch.linalg.eigvalsh(gram).clamp_min(0)
        ev = ev[ev > 1e-12 * max(float(ev.max()), 1e-30)]
        if ev.numel():
            p = ev / ev.sum()
            out[f"fisher_erank_{coords}"] = float(torch.exp(-(p * torch.log(p)).sum()))
        else:
            out[f"fisher_erank_{coords}"] = 0.0
    gw = out["grad_norm_w"]
    out["jac_fro_ratio"] = out["grad_norm_theta"] / gw if gw > 0 else float("nan")
    return out


# ----------------------------------------------------------------------------------------------
def compute_mechanism_metrics(model: MLP, x_probe: torch.Tensor, y_probe: torch.Tensor, cfg: Dict[str, Any],
                              generator: Optional[torch.Generator] = None) -> Dict[str, float]:
    """All end-of-task diagnostics selected by the ``metrics`` config section."""
    out: Dict[str, float] = {}
    if cfg.get("weights", True):
        out.update(weight_statistics(model))
    if cfg.get("representation", True):
        out.update(representation_statistics(model, x_probe, dead_tol=float(cfg.get("dead_tol", 0.0))))
    if cfg.get("fisher", True):
        out.update(gradient_fisher_statistics(model, x_probe, y_probe, n_samples=int(cfg.get("fisher_samples", 32)), generator=generator))
    return out
