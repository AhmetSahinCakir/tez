"""Non-damped bounded maps: triangle wave (reflection) and the Jacobian floor (straight-through sin)."""
import math

import pytest
import torch

from plasticity.methods import build_method
from plasticity.models import MLP, ReparamLinear


def test_tri_values_and_period():
    th = torch.tensor([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5], dtype=torch.double)
    w = ReparamLinear.forward_map(th, "tri", torch.tensor(1.0, dtype=torch.double), "unit")
    assert torch.allclose(w, torch.tensor([1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0, -1.0, 0.5], dtype=torch.double), atol=1e-12)


@pytest.mark.parametrize("mode", ("tri", "sin", "tanh"))
def test_gradcheck_away_from_kinks(mode):
    t = (torch.rand(5, 4, dtype=torch.double) * 1.6 - 0.8).requires_grad_(True)
    A = torch.tensor(1.3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, a: ReparamLinear.forward_map(x, mode, a, "amplitude"), (t, A))


@pytest.mark.parametrize("eps", (0.0, 0.3, 1.0))
def test_jacobian_floor_backward(eps):
    t = (torch.rand(6, 5, dtype=torch.double) * 4 - 2).requires_grad_(True)
    A = torch.tensor(1.0, dtype=torch.double)
    w = ReparamLinear.forward_map(t, "sin", A, "amplitude", eps)
    g = torch.randn_like(w)
    (gt,) = torch.autograd.grad((w * g).sum(), t)
    c = torch.cos(t.detach())
    d = c if eps == 0 else torch.sign(c) * c.abs().clamp_min(eps)
    assert torch.allclose(gt, g * d)
    # the forward pass (and hence the reported Jacobian statistics) is unchanged by the floor
    assert torch.allclose(w.detach(), torch.sin(t.detach()))


def test_reflection_vs_freeze():
    """Push W up for 20 steps (all maps reach the bound), then push it down for 5 steps: the triangle map and
    the floored sin respond immediately (reflection), the plain sin is frozen by its vanished Jacobian."""
    drop = {}
    for mode, fl in [("tri", 0.0), ("sin", 0.0), ("sin", 1.0)]:
        lin = ReparamLinear(1, 1, mode=mode, gamma=1.0, jacobian_floor=fl)
        with torch.no_grad():
            lin.theta.fill_(0.0)
            lin.theta_bias.fill_(0.0)
        opt = torch.optim.SGD(lin.parameters(), lr=0.2)
        for sign in [-1.0] * 20 + [1.0] * 5:  # loss = sign * W  ->  -1 pushes W up, +1 pushes it down
            opt.zero_grad()
            (sign * lin.effective_weight().sum()).backward()
            opt.step()
            assert float(lin.effective_weight().detach().abs()) <= float(lin.amplitude) + 1e-6
            if sign < 0:
                w_top = float(lin.effective_weight().detach())
        drop[(mode, fl)] = w_top - float(lin.effective_weight().detach())
        assert w_top > 0.9 * float(lin.amplitude)  # every map got (close) to the bound; plain sin creeps
    assert drop[("tri", 0.0)] > 0.9  # 5 full-size steps back (0.2 each)
    assert drop[("sin", 0.0)] < 0.3 * drop[("tri", 0.0)]  # damped: cos² small near the bound
    assert drop[("tri", 0.0)] > drop[("sin", 1.0)] > drop[("sin", 0.0)]


def test_tri_matched_init_and_training():
    g = torch.Generator().manual_seed(3)
    m0 = MLP(784, (50,), 10, generator=torch.Generator().manual_seed(3))
    m1 = MLP(784, (50,), 10, reparam={"hidden": "tri", "output": "tri"}, gamma=1.0, generator=torch.Generator().manual_seed(3))
    x = torch.randn(4, 784, generator=g)
    assert torch.allclose(m0(x), m1(x), atol=1e-5)
    method = build_method({"name": "continual_backprop", "replacement_rate": 1.0, "maturity_threshold": 0}, m1, {"name": "sgd", "lr": 0.01})
    opt = method.build_optimizer()
    y = torch.randint(0, 10, (4,), generator=g)
    for step in range(5):
        logits, feats = m1(x, return_features=True)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        method.after_step(step, feats)
    for lin in m1.linear_layers:
        assert float(lin.effective_weight().detach().abs().max()) <= float(lin.amplitude) + 1e-6


def test_tri_closed_form_matches_reference_bitwise():
    """The trimmed triangle-wave forward (7 ops) must equal the reference formulation bit for bit, including
    exactly at the flank boundaries ``(ang + 1) mod 4 == 2`` (falling flank, d = -1) and ``== 0``."""
    from plasticity.models.reparam import ReparamLinear, _BoundedMap

    gen = torch.Generator().manual_seed(0)
    ang = torch.cat([torch.randn(20000, generator=gen) * 5.0,
                     torch.tensor([1.0, -3.0, 5.0, 9.0, -7.0, 3.0, -1.0, 7.0, 0.5, 2.5, -2.5])])
    for scaled, amp in ((False, 1.0), (True, 0.37)):
        theta = (ang * amp if scaled else ang).clone().requires_grad_(True)
        a = torch.tensor(amp)
        ang_eff = (theta / a if scaled else theta).detach()  # exactly what the map sees ((ang*amp)/amp != ang)
        y = torch.remainder(ang_eff + 1.0, 4.0)
        f_ref = 1.0 - (y - 2.0).abs()
        d_ref = torch.where(y < 2.0, torch.ones_like(ang), -torch.ones_like(ang))
        w = _BoundedMap.apply(theta, a, "tri", scaled, 0.0)
        assert torch.equal(w, a * f_ref)
        grad = torch.randn(ang.shape, generator=gen)
        w.backward(grad)
        assert torch.equal(theta.grad, grad * d_ref if scaled else grad * (a * d_ref))
        nj = ReparamLinear.normalized_jacobian_map(theta.detach(), "tri", a, "amplitude" if scaled else "unit")
        assert torch.equal(nj, d_ref)


def test_tri_fused_kernel_matches_eager_bitwise():
    """The single-pass C++ kernel (plasticity.models.tri_fused) must reproduce the eager triangle-wave map and
    its gradient bit for bit; the eager path is forced by raising the size threshold."""
    from plasticity.models import tri_fused
    from plasticity.models.reparam import _BoundedMap

    if tri_fused.get_extension() is None:
        pytest.skip("fused kernel not available on this machine (no C++ toolchain)")
    gen = torch.Generator().manual_seed(5)
    theta = torch.cat([torch.randn(300000, generator=gen) * s for s in (0.3, 3.0, 300.0)]
                      + [torch.tensor([-0.0, 0.0, 1.0, -1.0, 3.0, -3.0, 5.0, 1e-30, -1e-30, 2.9999998, 3.0000002])])
    grad = torch.randn(theta.shape, generator=gen)
    saved = tri_fused.MIN_NUMEL
    try:
        for scaled, amp in ((False, 1.0), (True, 0.37)):
            a = torch.tensor(amp)
            outs = []
            for min_numel in (0, 10 ** 12):  # fused, then eager
                tri_fused.MIN_NUMEL = min_numel
                th = theta.clone().requires_grad_(True)
                w = _BoundedMap.apply(th, a, "tri", scaled, 0.0)
                w.backward(grad)
                outs.append((w.detach(), th.grad.clone()))
            assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    finally:
        tri_fused.MIN_NUMEL = saved
