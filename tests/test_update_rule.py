"""First-order weight update of the bounded maps (advisor review of 16 November 2026, item 2.2).

One SGD step on the free parameter Θ changes the effective weight by  ΔW ≈ -η · J · J~ · g  with the true
derivative J = f'(Θ/A), the derivative used in the backward pass J~ and g = ∂L/∂W:
    sine (ε = 0):           J·J~ = cos²(Θ/A)
    reflected sine (ε = 1): J·J~ = |cos(Θ/A)|        (J~ = sign(cos) is a surrogate, not the true derivative)
    triangle wave:          J·J~ = 1                 (away from the corner points)
The tests take one real optimiser step and compare the realised ΔW with these expressions.
"""
import math

import pytest
import torch

from plasticity.models.reparam import ReparamLinear

A, ETA = 0.3, 1e-4


def one_step(mode, floor, theta, g):
    theta = theta.clone().requires_grad_(True)
    w0 = ReparamLinear.forward_map(theta, mode, A, "amplitude", floor)
    (w0 * g).sum().backward()                      # dL/dW = g exactly
    with torch.no_grad():
        theta1 = theta - ETA * theta.grad
        w1 = ReparamLinear.forward_map(theta1, mode, A, "amplitude", floor)
    return (w1 - w0).detach(), theta.grad.detach()


def setup(lo, hi, n=20001, seed=0):
    gen = torch.Generator().manual_seed(seed)
    ang = torch.linspace(lo, hi, n, dtype=torch.float64)
    g = torch.randn(n, generator=gen, dtype=torch.float64)
    return ang, ang * A, g


def test_sine_step_is_cos_squared():
    ang, theta, g = setup(-1.5, 1.5)
    dw, grad = one_step("sin", 0.0, theta, g)
    assert torch.allclose(grad, torch.cos(ang) * g, atol=1e-12)            # backward pass = true derivative
    expected = -ETA * torch.cos(ang) ** 2 * g
    assert (dw - expected).abs().max() < 5e-3 * expected.abs().max()       # first order in η


def test_reflected_sine_step_is_abs_cos_not_one():
    ang, theta, g = setup(-4.5, 4.5)                                       # includes both sides of two crests
    dw, grad = one_step("sin", 1.0, theta, g)
    assert torch.allclose(grad, torch.sign(torch.cos(ang)) * g, atol=1e-12)  # surrogate: |J~| = 1
    expected = -ETA * torch.cos(ang).abs() * g
    undamped = -ETA * g
    assert (dw - expected).abs().max() < 5e-3 * undamped.abs().max()
    near_crest = (ang.abs() - math.pi / 2).abs() < 0.05
    assert dw[near_crest].abs().max() < 0.06 * undamped[near_crest].abs().max()   # W-step still vanishes at the crest
    assert torch.allclose(grad.abs(), g.abs(), atol=1e-12)                 # ...but the Θ-step is not attenuated


@pytest.mark.parametrize("lo,hi", [(-0.9, 0.9), (1.1, 2.9), (-2.9, -1.1)])
def test_triangle_step_is_undamped_away_from_corners(lo, hi):
    ang, theta, g = setup(lo, hi)
    dw, grad = one_step("tri", 0.0, theta, g)
    assert torch.allclose(grad.abs(), g.abs(), atol=1e-12)
    assert torch.allclose(dw, -ETA * g, atol=1e-12)                        # exact, not only first order
