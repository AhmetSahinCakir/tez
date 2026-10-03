"""Linear layer with an optional bounded (periodic or non-periodic) weight reparametrisation.

Effective weight of layer *l*::

    standard :  W = Θ                      (free, unbounded)
    sin      :  W = A · sin(Θ)             (bounded in [-A, A], periodic in Θ)        <- proposed
    tanh     :  W = A · tanh(Θ)            (bounded in [-A, A], non-periodic)          <- control (H2)

``Θ`` is the free parameter updated by the optimiser, ``A`` the (fixed, layer-specific) amplitude.

Matched initialisation (thesis, "Başlangıç Koşullarının Eşleştirilmesi"): a common ``W0`` is drawn
with the standard initialiser (Kaiming-uniform by default, He et al. 2015). With ``b`` the
theoretical bound of that uniform distribution, ``A = γ · b`` (γ > 1) and

    Θ0 = arcsin(W0 / A)   (sin)      Θ0 = atanh(W0 / A)   (tanh)      Θ0 = W0   (standard)

so that all three models start from *exactly* the same effective weights. Bias terms are treated
the same way when ``compact_bias=True`` ("tam-kompakt" configuration, the main analysis setting).

The Jacobian of the reparametrisation, ``∂W/∂Θ = A cos Θ`` (sin) or ``A (1 - tanh²Θ)`` (tanh), is
exposed through :meth:`jacobian` for the mechanism analyses (η_eff ∝ η A² cos²Θ, H3).

Scaling of the free parameter (``theta_scale``)
----------------------------------------------
``'unit'``       : the literal formulation ``W = A sin(Θ)``. Under SGD ``ΔW ≈ -η A² cos²Θ ∂L/∂W``: the
                   effective learning rate carries a *layer-dependent* constant ``A_l²`` (≈0.017 for a
                   784-input layer vs ≈0.135 for a 100-input layer with γ=1.5), i.e. a different
                   per-layer learning rate than the standard network, unrelated to boundedness.
``'amplitude'``  : the free parameter is ``Φ = A Θ`` so that ``W = A sin(Φ / A)`` and
                   ``ΔW ≈ -η cos²(Φ/A) ∂L/∂W``. This is exactly SGD on Θ with a per-layer learning rate
                   ``η / A_l²`` ("amplitude-matched learning rate"): at initialisation every layer has the
                   same effective step as the standard network, so the comparison isolates boundedness /
                   periodicity from a trivial per-layer learning-rate rescaling. Used in the main
                   experiments; ``'unit'`` is kept as a sensitivity analysis.
In both cases :meth:`angle` returns the argument of sin/tanh and :meth:`normalized_jacobian` the factor
``cos(angle)`` (sin) or ``1 - tanh²(angle)`` (tanh) that multiplies the effective step (H3).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

REPARAM_MODES = ("standard", "sin", "tanh")
INIT_SCHEMES = ("kaiming_uniform", "torch_default", "kaiming_normal")


def init_bound(fan_in: int, scheme: str, gain: float = math.sqrt(2.0)) -> float:
    """Theoretical bound ``b`` of the weight initialiser (uniform schemes) or 3σ (normal)."""
    if scheme == "kaiming_uniform":  # U(-b, b), b = gain * sqrt(3 / fan_in)   (He et al., 2015)
        return gain * math.sqrt(3.0 / fan_in)
    if scheme == "torch_default":  # nn.Linear default: kaiming_uniform(a=sqrt(5)) -> U(-1/sqrt(fan_in), ..)
        return 1.0 / math.sqrt(fan_in)
    if scheme == "kaiming_normal":  # N(0, gain²/fan_in); use 3σ as the nominal bound
        return 3.0 * gain / math.sqrt(fan_in)
    raise ValueError(f"unknown init scheme {scheme!r}")


def draw_init(shape, fan_in: int, scheme: str, gain: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    b = init_bound(fan_in, scheme, gain)
    if scheme in ("kaiming_uniform", "torch_default"):
        return torch.empty(shape).uniform_(-b, b, generator=generator)
    w = torch.empty(shape).normal_(0.0, gain / math.sqrt(fan_in), generator=generator)
    return w.clamp_(-b, b)  # keep |W0| <= b so the matched inverse map is well defined


class _BoundedMap(torch.autograd.Function):
    """Fused ``W = A·sin(ang)`` / ``A·tanh(ang)`` with ``ang = Θ/A`` (amplitude scale) or ``Θ`` (unit scale).

    Mathematically identical to composing the elementary ops, but saves the derivative factor in the forward
    pass so the backward is a single multiply (the batch-size-1 online loop is dominated by per-op overhead).
    Gradients w.r.t. both ``Θ`` and ``A`` (learnable amplitude) are provided.
    """

    @staticmethod
    def forward(ctx, theta: torch.Tensor, amplitude: torch.Tensor, is_sin: bool, scaled: bool):
        ang = theta / amplitude if scaled else theta
        if is_sin:
            f = torch.sin(ang)
            d = torch.cos(ang)  # f'(ang)
        else:
            f = torch.tanh(ang)
            d = 1.0 - f * f
        ctx.save_for_backward(ang, f, d, amplitude)
        ctx.scaled = scaled
        return amplitude * f

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ang, f, d, amplitude = ctx.saved_tensors
        scaled = ctx.scaled
        g_theta = g_amp = None
        if ctx.needs_input_grad[0]:
            g_theta = grad * d if scaled else grad * (amplitude * d)  # dW/dΘ = f'(ang)·d(ang)/dΘ·A
        if ctx.needs_input_grad[1]:
            # W = A f(ang): dW/dA = f(ang) + A f'(ang) d(ang)/dA, with d(ang)/dA = -Θ/A² = -ang/A (scaled) or 0
            dWdA = f - ang * d if scaled else f
            g_amp = (grad * dWdA).sum().reshape(amplitude.shape)
        return g_theta, g_amp, None, None


class ReparamLinear(nn.Module):
    """``y = x Wᵀ + b`` with ``W = f(Θ)``, ``b = f(θ_b)`` (see module docstring)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mode: str = "standard",
        bias: bool = True,
        compact_bias: bool = True,
        gamma: float = 1.5,
        init: str = "kaiming_uniform",
        gain: float = math.sqrt(2.0),
        bias_init: str = "zeros",
        learn_amplitude: bool = False,
        theta_scale: str = "amplitude",
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()
        if theta_scale not in ("unit", "amplitude"):
            raise ValueError("theta_scale must be 'unit' or 'amplitude'")
        self.theta_scale = theta_scale
        if mode not in REPARAM_MODES:
            raise ValueError(f"mode must be one of {REPARAM_MODES}, got {mode!r}")
        if mode != "standard" and gamma <= 1.0:
            raise ValueError("gamma must be > 1 so that |W0|/A < 1 and the inverse map is defined")
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.mode = mode
        self.gamma = float(gamma)
        self.init_scheme, self.gain, self.bias_init = init, float(gain), bias_init
        self.compact_bias = bool(compact_bias) and mode != "standard"
        self.bias_mode = mode if self.compact_bias else "standard"

        # --- common W0 (matched initialisation) -------------------------------------------------
        w0 = draw_init((out_features, in_features), in_features, init, gain, generator)
        self.weight_init_bound = init_bound(in_features, init, gain)
        self.bias_init_bound = 1.0 / math.sqrt(in_features)  # nn.Linear bias bound
        if bias:
            if bias_init == "zeros":
                b0 = torch.zeros(out_features)
            elif bias_init == "uniform":
                b0 = torch.empty(out_features).uniform_(-self.bias_init_bound, self.bias_init_bound, generator=generator)
            else:
                raise ValueError(f"unknown bias_init {bias_init!r}")
        else:
            b0 = None

        amp = self.gamma * self.weight_init_bound if mode != "standard" else 1.0
        amp_b = self.gamma * self.bias_init_bound if self.compact_bias else 1.0
        if learn_amplitude and mode != "standard":
            self.amplitude = nn.Parameter(torch.tensor(float(amp)))
            if bias and self.compact_bias:
                self.bias_amplitude = nn.Parameter(torch.tensor(float(amp_b)))
            else:  # bias not reparametrised: keep a constant (unused) amplitude so every code path can read it
                self.register_buffer("bias_amplitude", torch.tensor(float(amp_b)))
        else:
            self.register_buffer("amplitude", torch.tensor(float(amp)))
            self.register_buffer("bias_amplitude", torch.tensor(float(amp_b)))
        self.learn_amplitude = bool(learn_amplitude and mode != "standard")

        self.theta = nn.Parameter(self.inverse(w0, mode, amp, theta_scale))
        if bias:
            self.theta_bias = nn.Parameter(self.inverse(b0, self.bias_mode, amp_b, theta_scale))
        else:
            self.register_parameter("theta_bias", None)
        self.capture_effective = False  # when True, keep the effective W (with grad) after forward
        self.last_weight: Optional[torch.Tensor] = None
        self.last_bias: Optional[torch.Tensor] = None

    # --- maps ------------------------------------------------------------------------------------
    @staticmethod
    def _angle(theta: torch.Tensor, amplitude, theta_scale: str) -> torch.Tensor:
        return theta / amplitude if theta_scale == "amplitude" else theta

    @staticmethod
    def forward_map(theta: torch.Tensor, mode: str, amplitude, theta_scale: str = "unit") -> torch.Tensor:
        if mode == "standard":
            return theta
        if mode not in ("sin", "tanh"):
            raise ValueError(mode)
        amp = amplitude if torch.is_tensor(amplitude) else torch.tensor(float(amplitude), dtype=theta.dtype)
        return _BoundedMap.apply(theta, amp, mode == "sin", theta_scale == "amplitude")

    @staticmethod
    def inverse(w: torch.Tensor, mode: str, amplitude: float, theta_scale: str = "unit") -> torch.Tensor:
        """Θ = f⁻¹(W / A) (times A for ``theta_scale='amplitude'``); principal branch arcsin ∈ [-π/2, π/2]."""
        if mode == "standard":
            return w.clone()
        r = (w / float(amplitude)).clamp(-1 + 1e-6, 1 - 1e-6)
        if mode == "sin":
            ang = torch.asin(r)
        elif mode == "tanh":
            ang = torch.atanh(r)
        else:
            raise ValueError(mode)
        return ang * float(amplitude) if theta_scale == "amplitude" else ang

    @staticmethod
    def jacobian_map(theta: torch.Tensor, mode: str, amplitude, theta_scale: str = "unit") -> torch.Tensor:
        """``∂W/∂Θ`` element-wise (``A cos Θ`` for unit scale, ``cos(Φ/A)`` for amplitude scale)."""
        if mode == "standard":
            return torch.ones_like(theta)
        nj = ReparamLinear.normalized_jacobian_map(theta, mode, amplitude, theta_scale)
        return nj if theta_scale == "amplitude" else amplitude * nj

    @staticmethod
    def normalized_jacobian_map(theta: torch.Tensor, mode: str, amplitude, theta_scale: str = "unit") -> torch.Tensor:
        """``cos(angle)`` (sin) / ``1 - tanh²(angle)`` (tanh): the state-dependent factor of the effective step."""
        if mode == "standard":
            return torch.ones_like(theta)
        ang = ReparamLinear._angle(theta, amplitude, theta_scale)
        if mode == "sin":
            return torch.cos(ang)
        if mode == "tanh":
            return 1.0 - torch.tanh(ang) ** 2
        raise ValueError(mode)

    # --- effective quantities ----------------------------------------------------------------------
    def effective_weight(self) -> torch.Tensor:
        return self.forward_map(self.theta, self.mode, self.amplitude, self.theta_scale)

    def effective_bias(self) -> Optional[torch.Tensor]:
        if self.theta_bias is None:
            return None
        return self.forward_map(self.theta_bias, self.bias_mode, self.bias_amplitude, self.theta_scale)

    def angle(self) -> torch.Tensor:
        """Argument of sin / tanh (Θ for unit scale, Φ/A for amplitude scale)."""
        return self._angle(self.theta, self.amplitude, self.theta_scale)

    def jacobian(self) -> torch.Tensor:
        return self.jacobian_map(self.theta, self.mode, self.amplitude, self.theta_scale)

    def normalized_jacobian(self) -> torch.Tensor:
        return self.normalized_jacobian_map(self.theta, self.mode, self.amplitude, self.theta_scale)

    def bias_jacobian(self) -> Optional[torch.Tensor]:
        if self.theta_bias is None:
            return None
        return self.jacobian_map(self.theta_bias, self.bias_mode, self.bias_amplitude, self.theta_scale)

    def bias_normalized_jacobian(self) -> Optional[torch.Tensor]:
        if self.theta_bias is None:
            return None
        return self.normalized_jacobian_map(self.theta_bias, self.bias_mode, self.bias_amplitude, self.theta_scale)

    @torch.no_grad()
    def set_effective_weight(self, w: torch.Tensor, rows: Optional[torch.Tensor] = None, cols: Optional[torch.Tensor] = None) -> None:
        """Write effective weights ``w`` (full matrix, or a row / column block) into Θ via the inverse map."""
        th = self.inverse(w, self.mode, float(self.amplitude), self.theta_scale)
        if rows is not None:
            self.theta[rows] = th
        elif cols is not None:
            self.theta[:, cols] = th
        else:
            self.theta.copy_(th)

    @torch.no_grad()
    def set_effective_bias(self, b: torch.Tensor, rows: Optional[torch.Tensor] = None) -> None:
        if self.theta_bias is None:
            return
        th = self.inverse(b, self.bias_mode, float(self.bias_amplitude), self.theta_scale)
        if rows is not None:
            self.theta_bias[rows] = th
        else:
            self.theta_bias.copy_(th)

    # --- (re)initialisation helpers used by Continual Backprop -------------------------------------
    @torch.no_grad()
    def reinit_output_units(self, units: torch.Tensor, generator: Optional[torch.Generator] = None) -> None:
        """Re-draw the *incoming* weights (rows) and biases of the given output units from the initialiser."""
        if units.numel() == 0:
            return
        w = draw_init((units.numel(), self.in_features), self.in_features, self.init_scheme, self.gain, generator)
        self.set_effective_weight(w, rows=units)
        if self.theta_bias is not None:
            if self.bias_init == "zeros":
                b = torch.zeros(units.numel())
            else:
                b = torch.empty(units.numel()).uniform_(-self.bias_init_bound, self.bias_init_bound, generator=generator)
            self.set_effective_bias(b, rows=units)

    @torch.no_grad()
    def zero_input_units(self, units: torch.Tensor) -> None:
        """Zero the *outgoing* weights (columns) of the given input units (Θ = 0 ⇒ W = 0 in every mode)."""
        if units.numel() == 0:
            return
        self.theta[:, units] = 0.0

    # --- forward -------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight()
        b = self.effective_bias()
        if self.capture_effective:
            if self.mode != "standard" and w.requires_grad:
                w.retain_grad()
            if b is not None and self.bias_mode != "standard" and b.requires_grad:
                b.retain_grad()
            self.last_weight, self.last_bias = w, b
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, mode={self.mode}, compact_bias={self.compact_bias}, "
            f"gamma={self.gamma}, A={float(self.amplitude):.4f}, theta_scale={self.theta_scale}, init={self.init_scheme}"
        )
