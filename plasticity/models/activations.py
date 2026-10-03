"""Activation functions used in the thesis experiments.

* ``relu``          -- standard (Dohare et al., 2024 baseline networks)
* ``leaky_relu``    -- slope 0.01
* ``tanh``
* ``sin``           -- Sin-MLP control of Chen & Zhang (2026): periodic *activation*, unbounded weights
* ``smooth_leaky``  -- smooth leaky unit inspired by Lillo & Cheney (2026):
                       f(x) = a*x + (1-a)*softplus(x); f'(x) = a + (1-a)*sigmoid(x) in (a, 1), so no
                       unit can become exactly dead. (Our formulation; see docs/RAPOR.md.)
* ``rand_smooth_leaky`` -- same with a per-unit slope a ~ U(a_min, a_max) drawn at construction.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sin(nn.Module):
    def __init__(self, omega: float = 1.0):
        super().__init__()
        self.omega = float(omega)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)

    def extra_repr(self) -> str:
        return f"omega={self.omega}"


class SmoothLeaky(nn.Module):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + (1.0 - self.alpha) * F.softplus(x)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class RandomizedSmoothLeaky(nn.Module):
    """Per-unit slopes drawn once at construction (fixed afterwards)."""

    def __init__(self, width: int, alpha_min: float = 0.01, alpha_max: float = 0.3):
        super().__init__()
        alphas = torch.empty(width).uniform_(alpha_min, alpha_max)
        self.register_buffer("alpha", alphas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        return a * x + (1.0 - a) * F.softplus(x)


def get_activation(name: str, width: int | None = None, **kw) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(kw.get("negative_slope", 0.01))
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "sin":
        return Sin(kw.get("omega", 1.0))
    if name == "smooth_leaky":
        return SmoothLeaky(kw.get("alpha", 0.1))
    if name == "rand_smooth_leaky":
        assert width is not None, "rand_smooth_leaky needs the layer width"
        return RandomizedSmoothLeaky(width, kw.get("alpha_min", 0.01), kw.get("alpha_max", 0.3))
    raise ValueError(f"unknown activation {name!r}")


def activation_gain(name: str) -> float:
    """Gain used by Kaiming initialisation for the given nonlinearity."""
    name = name.lower()
    if name == "relu":
        return math.sqrt(2.0)
    if name == "leaky_relu":
        return math.sqrt(2.0 / (1 + 0.01**2))
    if name in ("tanh",):
        return 5.0 / 3.0
    return 1.0
