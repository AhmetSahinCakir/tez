"""Tests for the bounded-periodic weight reparametrisation (plasticity/models/reparam.py).

Every property is checked for all modes ``{standard, sin, tanh}`` and both free-parameter scalings
``theta_scale in {unit, amplitude}``: matched initialisation across modes, the bound |W| <= A, the
inverse / forward round trip, the analytic Jacobian against autograd, the normalised Jacobian,
bias options, the (re)initialisation helpers used by Continual Backprop and the effective-weight
capture used by the gradient / Fisher diagnostics.
"""
from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.nn as nn

from plasticity.models import MLP, ReparamLinear
from plasticity.models.reparam import REPARAM_MODES, draw_init, init_bound

IN, OUT = 16, 8
SCALES = ("unit", "amplitude")
MODE_SCALE = list(itertools.product(REPARAM_MODES, SCALES))
BOUNDED = [(m, s) for m, s in MODE_SCALE if m != "standard"]


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def make_layer(mode: str, theta_scale: str = "amplitude", seed: int = 0, **kw) -> ReparamLinear:
    kw.setdefault("bias_init", "uniform")  # non-trivial biases so the bias path is exercised
    return ReparamLinear(IN, OUT, mode=mode, theta_scale=theta_scale, generator=gen(seed), **kw)


def big_random_update(layer: ReparamLinear, scale: float = 50.0, seed: int = 1) -> None:
    with torch.no_grad():
        layer.theta.add_(scale * torch.randn(layer.theta.shape, generator=gen(seed)))
        if layer.theta_bias is not None:
            layer.theta_bias.add_(scale * torch.randn(layer.theta_bias.shape, generator=gen(seed + 1)))


# ------------------------------------------------------------------------------- initialisation
@pytest.mark.parametrize("theta_scale", SCALES)
def test_matched_initialisation_across_modes(theta_scale):
    layers = {m: make_layer(m, theta_scale, seed=3) for m in REPARAM_MODES}
    ref = layers["standard"]
    x = 0.5 * torch.randn(6, IN, generator=gen(9))
    with torch.no_grad():
        y_ref = ref(x)
        for m in ("sin", "tanh"):
            lin = layers[m]
            assert torch.allclose(lin.effective_weight(), ref.effective_weight(), atol=1e-5)
            assert torch.allclose(lin.effective_bias(), ref.effective_bias(), atol=1e-5)
            assert torch.allclose(lin(x), y_ref, atol=1e-5)
            # ... but the free parameters differ from the effective weights (they are angles)
            assert not torch.allclose(lin.theta, lin.effective_weight())
    assert torch.equal(ref.theta, ref.effective_weight())  # standard: W is Theta


@pytest.mark.parametrize("mode,theta_scale", MODE_SCALE)
def test_init_bounds_and_amplitude(mode, theta_scale):
    lin = make_layer(mode, theta_scale, gamma=1.5)
    b = init_bound(IN, "kaiming_uniform", math.sqrt(2.0))
    assert lin.weight_init_bound == pytest.approx(b)
    assert lin.bias_init_bound == pytest.approx(1 / math.sqrt(IN))
    w = lin.effective_weight().detach()
    assert float(w.abs().max()) <= b + 1e-6
    if mode == "standard":
        assert float(lin.amplitude) == 1.0 and not lin.compact_bias and lin.bias_mode == "standard"
    else:
        assert float(lin.amplitude) == pytest.approx(1.5 * b)
        assert float(lin.bias_amplitude) == pytest.approx(1.5 / math.sqrt(IN))
        assert lin.compact_bias and lin.bias_mode == mode
        # the principal-branch inverse keeps |angle| < pi/2 at initialisation (sin) / finite (tanh)
        assert float(lin.angle().detach().abs().max()) < math.pi / 2
    assert isinstance(lin.amplitude, torch.Tensor) and not isinstance(lin.amplitude, nn.Parameter)


def test_same_generator_seed_reproduces_layer():
    a, b = make_layer("sin", seed=11), make_layer("sin", seed=11)
    assert torch.equal(a.theta, b.theta) and torch.equal(a.theta_bias, b.theta_bias)
    c = make_layer("sin", seed=12)
    assert not torch.equal(a.theta, c.theta)


# ------------------------------------------------------------------------------- boundedness
@pytest.mark.parametrize("mode,theta_scale", BOUNDED)
def test_effective_weights_are_bounded_always(mode, theta_scale):
    lin = make_layer(mode, theta_scale)
    A, Ab = float(lin.amplitude), float(lin.bias_amplitude)
    for scale in (0.0, 1.0, 50.0, 1e4):
        if scale:
            big_random_update(lin, scale)
        assert float(lin.effective_weight().detach().abs().max()) <= A + 1e-7
        assert float(lin.effective_bias().detach().abs().max()) <= Ab + 1e-7
        assert torch.isfinite(lin.effective_weight()).all()
    # after huge updates the weights actually use the bound (sin is surjective onto [-A, A])
    assert float(lin.effective_weight().detach().abs().max()) > 0.9 * A


def test_standard_mode_is_unbounded():
    lin = make_layer("standard")
    big_random_update(lin, 50.0)
    assert float(lin.effective_weight().detach().abs().max()) > 10.0


# ------------------------------------------------------------------------------- inverse / forward
@pytest.mark.parametrize("mode,theta_scale", MODE_SCALE)
def test_inverse_forward_round_trip(mode, theta_scale):
    A = 0.4
    w = 0.95 * A * (2 * torch.rand(OUT, IN, generator=gen(5)) - 1)
    th = ReparamLinear.inverse(w, mode, A, theta_scale)
    back = ReparamLinear.forward_map(th, mode, A, theta_scale)
    assert torch.allclose(back, w, atol=1e-6)
    if mode != "standard":
        # forward -> inverse on the principal branch
        ang = 0.5 * math.pi * 0.95 * (2 * torch.rand(OUT, IN, generator=gen(6)) - 1)
        th0 = ang * A if theta_scale == "amplitude" else ang
        w0 = ReparamLinear.forward_map(th0, mode, A, theta_scale)
        assert torch.allclose(ReparamLinear.inverse(w0, mode, A, theta_scale), th0, atol=1e-5)
        # values beyond the bound are clamped to the open interval, never NaN
        assert torch.isfinite(ReparamLinear.inverse(torch.tensor([2 * A, -2 * A]), mode, A, theta_scale)).all()


@pytest.mark.parametrize("mode,theta_scale", MODE_SCALE)
def test_set_effective_weight_round_trip(mode, theta_scale):
    lin = make_layer(mode, theta_scale)
    A = float(lin.amplitude)
    w = 0.95 * A * (2 * torch.rand(OUT, IN, generator=gen(7)) - 1)
    lin.set_effective_weight(w)
    assert torch.allclose(lin.effective_weight(), w, atol=1e-6)
    # row block
    rows = torch.tensor([1, 6])
    wr = 0.9 * A * (2 * torch.rand(2, IN, generator=gen(8)) - 1)
    before = lin.effective_weight().clone()
    lin.set_effective_weight(wr, rows=rows)
    after = lin.effective_weight()
    assert torch.allclose(after[rows], wr, atol=1e-6)
    keep = torch.ones(OUT, dtype=torch.bool); keep[rows] = False
    assert torch.equal(after[keep], before[keep])
    # column block
    cols = torch.tensor([0, 3, 15])
    wc = 0.9 * A * (2 * torch.rand(OUT, 3, generator=gen(9)) - 1)
    before = lin.effective_weight().clone()
    lin.set_effective_weight(wc, cols=cols)
    after = lin.effective_weight()
    assert torch.allclose(after[:, cols], wc, atol=1e-6)
    keepc = torch.ones(IN, dtype=torch.bool); keepc[cols] = False
    assert torch.equal(after[:, keepc], before[:, keepc])
    # bias
    Ab = float(lin.bias_amplitude)
    b = 0.9 * Ab * (2 * torch.rand(OUT, generator=gen(10)) - 1)
    lin.set_effective_bias(b)
    assert torch.allclose(lin.effective_bias(), b, atol=1e-6)


# ------------------------------------------------------------------------------- Jacobians
@pytest.mark.parametrize("mode,theta_scale", MODE_SCALE)
def test_jacobian_matches_autograd(mode, theta_scale):
    lin = make_layer(mode, theta_scale)
    big_random_update(lin, 2.0)  # move away from the init so cos / 1-tanh^2 are non-trivial
    R = torch.randn(OUT, IN, generator=gen(21))
    (g,) = torch.autograd.grad((lin.effective_weight() * R).sum(), lin.theta)
    assert torch.allclose(g, R * lin.jacobian(), atol=1e-6)
    Rb = torch.randn(OUT, generator=gen(22))
    (gb,) = torch.autograd.grad((lin.effective_bias() * Rb).sum(), lin.theta_bias)
    assert torch.allclose(gb, Rb * lin.bias_jacobian(), atol=1e-6)


@pytest.mark.parametrize("mode,theta_scale", MODE_SCALE)
def test_normalized_jacobian_range_and_relation(mode, theta_scale):
    lin = make_layer(mode, theta_scale)
    for scale in (0.0, 3.0):
        if scale:
            big_random_update(lin, scale)
        nj = lin.normalized_jacobian().detach()
        assert float(nj.min()) >= -1.0 - 1e-7 and float(nj.max()) <= 1.0 + 1e-7
        jac = lin.jacobian().detach()
        if theta_scale == "amplitude" or mode == "standard":
            assert torch.allclose(nj, jac, atol=1e-7)
        else:
            assert torch.allclose(nj, jac / lin.amplitude, atol=1e-6)
        njb = lin.bias_normalized_jacobian().detach()
        assert float(njb.abs().max()) <= 1.0 + 1e-7
        if mode == "standard":
            assert torch.equal(nj, torch.ones_like(nj)) and torch.equal(jac, torch.ones_like(jac))
        elif mode == "tanh":
            assert float(nj.min()) >= 0.0  # 1 - tanh^2 is non-negative
    if mode != "standard":
        # at the bound the (normalised) Jacobian vanishes: the weight is frozen
        with torch.no_grad():
            A = float(lin.amplitude)
            lin.theta.fill_((math.pi / 2) * (A if theta_scale == "amplitude" else 1.0) if mode == "sin" else 40.0 * (A if theta_scale == "amplitude" else 1.0))
        assert float(lin.normalized_jacobian().detach().abs().max()) < 1e-5
        assert float(lin.effective_weight().detach().abs().min()) > 0.999 * A


# ------------------------------------------------------------------------------- bias options
@pytest.mark.parametrize("mode", ("sin", "tanh"))
def test_compact_bias_false_leaves_bias_standard(mode):
    lin = make_layer(mode, compact_bias=False)
    assert not lin.compact_bias and lin.bias_mode == "standard" and lin.mode == mode
    assert torch.equal(lin.effective_bias(), lin.theta_bias)
    assert float(lin.bias_amplitude) == 1.0
    assert torch.equal(lin.bias_jacobian(), torch.ones(OUT))
    big_random_update(lin, 50.0)
    assert float(lin.effective_bias().detach().abs().max()) > 10.0  # unbounded bias ...
    assert float(lin.effective_weight().detach().abs().max()) <= float(lin.amplitude) + 1e-7  # ... bounded weights


def test_bias_false_and_zero_bias_init():
    lin = ReparamLinear(IN, OUT, mode="sin", bias=False, generator=gen(0))
    assert lin.theta_bias is None and lin.effective_bias() is None and lin.bias_jacobian() is None
    lin.set_effective_bias(torch.zeros(OUT))  # no-op, must not raise
    y = lin(torch.randn(3, IN))
    assert y.shape == (3, OUT)
    z = ReparamLinear(IN, OUT, mode="sin", bias_init="zeros", generator=gen(0))
    assert torch.equal(z.theta_bias, torch.zeros(OUT)) and torch.equal(z.effective_bias(), torch.zeros(OUT))


@pytest.mark.parametrize("mode", ("sin", "tanh"))
def test_learn_amplitude_makes_amplitude_a_parameter(mode):
    lin = make_layer(mode, learn_amplitude=True)
    assert lin.learn_amplitude
    assert isinstance(lin.amplitude, nn.Parameter) and isinstance(lin.bias_amplitude, nn.Parameter)
    names = {n for n, _ in lin.named_parameters()}
    assert names == {"theta", "theta_bias", "amplitude", "bias_amplitude"}
    lin(torch.randn(4, IN, generator=gen(1))).sum().backward()
    assert lin.amplitude.grad is not None and lin.bias_amplitude.grad is not None
    fixed = make_layer(mode, learn_amplitude=False)
    assert {n for n, _ in fixed.named_parameters()} == {"theta", "theta_bias"}
    std = make_layer("standard", learn_amplitude=True)  # ignored for the standard parametrisation
    assert not std.learn_amplitude and not isinstance(std.amplitude, nn.Parameter)


# ------------------------------------------------------------------------------- CBP helpers
@pytest.mark.parametrize("mode,theta_scale", MODE_SCALE)
def test_reinit_output_units_changes_only_chosen_rows(mode, theta_scale):
    lin = make_layer(mode, theta_scale)
    big_random_update(lin, 3.0)
    units = torch.tensor([1, 5])
    keep = torch.ones(OUT, dtype=torch.bool); keep[units] = False
    w_before, b_before = lin.effective_weight().clone(), lin.effective_bias().clone()
    lin.reinit_output_units(units, generator=gen(42))
    w_after, b_after = lin.effective_weight().detach(), lin.effective_bias().detach()
    assert torch.equal(w_after[keep], w_before[keep]) and torch.equal(b_after[keep], b_before[keep])
    assert not torch.allclose(w_after[units], w_before[units])
    assert float(w_after[units].abs().max()) <= lin.weight_init_bound + 1e-6
    assert float(b_after[units].abs().max()) <= lin.bias_init_bound + 1e-6
    # the new rows are exactly a fresh draw of the initialiser
    ref = draw_init((2, IN), IN, lin.init_scheme, lin.gain, gen(42))
    assert torch.allclose(w_after[units], ref, atol=1e-6)
    # reproducible given the generator seed; empty selection is a no-op
    other = make_layer(mode, theta_scale); big_random_update(other, 3.0)
    other.reinit_output_units(units, generator=gen(42))
    assert torch.allclose(other.effective_weight()[units], w_after[units], atol=1e-6)
    lin.reinit_output_units(torch.tensor([], dtype=torch.long), generator=gen(0))
    assert torch.equal(lin.effective_weight(), w_after)


def test_reinit_with_zero_bias_init():
    lin = ReparamLinear(IN, OUT, mode="sin", bias_init="zeros", generator=gen(0))
    with torch.no_grad():
        lin.theta_bias.fill_(0.3)
    lin.reinit_output_units(torch.tensor([2]), generator=gen(1))
    b = lin.effective_bias().detach()
    assert float(b[2]) == 0.0 and float(b[0]) != 0.0


@pytest.mark.parametrize("mode,theta_scale", MODE_SCALE)
def test_zero_input_units_zeroes_columns(mode, theta_scale):
    lin = make_layer(mode, theta_scale)
    big_random_update(lin, 3.0)
    cols = torch.tensor([0, 3, 15])
    keep = torch.ones(IN, dtype=torch.bool); keep[cols] = False
    before = lin.effective_weight().clone()
    lin.zero_input_units(cols)
    after = lin.effective_weight()
    assert torch.equal(after[:, cols], torch.zeros(OUT, 3))
    assert torch.equal(after[:, keep], before[:, keep])
    assert torch.equal(lin.theta[:, cols], torch.zeros(OUT, 3))
    lin.zero_input_units(torch.tensor([], dtype=torch.long))
    assert torch.equal(lin.effective_weight(), after)


# ------------------------------------------------------------------------------- validation
@pytest.mark.parametrize("mode", ("sin", "tanh"))
@pytest.mark.parametrize("gamma", (1.0, 0.5, 0.0))
def test_gamma_at_most_one_raises(mode, gamma):
    with pytest.raises(ValueError, match="gamma"):
        ReparamLinear(IN, OUT, mode=mode, gamma=gamma)


def test_standard_mode_ignores_gamma_and_bad_arguments_raise():
    ReparamLinear(IN, OUT, mode="standard", gamma=1.0)  # fine: no bound for the standard map
    with pytest.raises(ValueError):
        ReparamLinear(IN, OUT, mode="cos")
    with pytest.raises(ValueError):
        ReparamLinear(IN, OUT, mode="sin", theta_scale="radians")
    with pytest.raises(ValueError):
        ReparamLinear(IN, OUT, mode="sin", init="orthogonal")
    with pytest.raises(ValueError):
        ReparamLinear(IN, OUT, mode="sin", bias_init="ones")
    with pytest.raises(ValueError):
        ReparamLinear.forward_map(torch.zeros(2), "cos", 1.0)


# ------------------------------------------------------------------------------- capture + chain rule
@pytest.mark.parametrize("mode,theta_scale", MODE_SCALE)
def test_capture_effective_and_chain_rule(mode, theta_scale):
    lin = make_layer(mode, theta_scale)
    big_random_update(lin, 2.0)
    assert lin.last_weight is None and lin.last_bias is None
    lin.capture_effective = True
    x = 0.5 * torch.randn(4, IN, generator=gen(31))
    R = 0.5 * torch.randn(4, OUT, generator=gen(32))
    (lin(x) * R).sum().backward()
    assert lin.last_weight is not None and lin.last_bias is not None
    gW, gb = lin.last_weight.grad, lin.last_bias.grad
    assert gW is not None and gb is not None and gW.shape == (OUT, IN) and gb.shape == (OUT,)
    # dL/dW = R^T x, dL/db = sum_i R_i
    assert torch.allclose(gW, R.T @ x, atol=1e-6) and torch.allclose(gb, R.sum(0), atol=1e-6)
    # chain rule: grad_theta = grad_W * dW/dtheta (element-wise)
    assert torch.allclose(lin.theta.grad, gW * lin.jacobian(), atol=1e-6)
    assert torch.allclose(lin.theta_bias.grad, gb * lin.bias_jacobian(), atol=1e-6)
    if mode == "standard":
        assert lin.last_weight is lin.theta  # W *is* the parameter
    else:
        assert lin.last_weight is not lin.theta
    lin.capture_effective = False
    lin.zero_grad()
    lin(x).sum().backward()  # still works when capture is off; stale capture not refreshed
    assert lin.theta.grad is not None


def test_mlp_integration_capture_and_features():
    g = gen(0)
    model = MLP(IN, hidden_sizes=(12, 10), n_classes=3, reparam={"hidden": "sin", "output": "tanh"}, generator=g)
    assert [lin.mode for lin in model.linear_layers] == ["sin", "sin", "tanh"]
    assert len(model.hidden) == 2 and model.n_hidden_layers == 2
    x = torch.randn(5, IN, generator=gen(1))
    logits, feats = model(x, return_features=True)
    assert logits.shape == (5, 3) and len(feats) == 2 and feats[0].shape == (5, 12) and feats[1].shape == (5, 10)
    assert (feats[0] >= 0).all()  # ReLU features
    model.set_capture_effective(True)
    model(x).sum().backward()
    for lin in model.linear_layers:
        assert lin.last_weight is not None and lin.last_weight.grad is not None
        assert torch.allclose(lin.theta.grad, lin.last_weight.grad * lin.jacobian(), atol=1e-6)
    model.set_capture_effective(False)
    assert all(lin.last_weight is None and lin.last_bias is None for lin in model.linear_layers)
    # matched initialisation holds at the network level too
    std = MLP(IN, hidden_sizes=(12, 10), n_classes=3, generator=gen(0))
    with torch.no_grad():
        assert torch.allclose(std(x), model(x), atol=1e-5)
