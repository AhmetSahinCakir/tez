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
