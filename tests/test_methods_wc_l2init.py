"""Tests for Weight Clipping (Elsayed et al. 2024) and L2 Init (Kumar et al. 2025).

Fast, synthetic-data tests (torch.randn inputs); no dataset access.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from plasticity.data import Task
from plasticity.methods import build_method
from plasticity.methods.l2_init import L2Init
from plasticity.methods.weight_clipping import WeightClipping
from plasticity.models import MLP
from plasticity.training.online import _train_on_task

IN, HID, NC = 20, (16, 12), 5


def make_model(mode="standard", seed=0, layer_norm=False):
    g = torch.Generator().manual_seed(seed)
    return MLP(IN, HID, NC, reparam={"hidden": mode, "output": mode}, layer_norm=layer_norm, generator=g)


def synthetic_batch(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN, generator=g)
    y = torch.randint(0, NC, (n,), generator=g)
    return x, y


def run_steps(model, method, optimizer, n_steps, seed=1, batch=1):
    x, y = synthetic_batch(n_steps * batch, seed)
    for s in range(n_steps):
        xb, yb = x[s * batch:(s + 1) * batch], y[s * batch:(s + 1) * batch]
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        reg = method.regularizer()
        if reg is not None:
            loss = loss + reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        method.before_step(s)
        optimizer.step()
        method.after_step(s, None)


def assert_within_bounds(model, kappa, clip_bias=True, tol=1e-6):
    for lin in model.linear_layers:
        bw = kappa * lin.weight_init_bound
        assert float(lin.theta.detach().abs().max()) <= bw + tol
        if clip_bias and lin.theta_bias is not None:
            bb = kappa * lin.bias_init_bound
            assert float(lin.theta_bias.detach().abs().max()) <= bb + tol


# ------------------------------------------------------------------------------------------------
# Weight Clipping
# ------------------------------------------------------------------------------------------------
class TestWeightClipping:
    def test_registry_and_defaults(self):
        m = make_model()
        wc = build_method({"name": "weight_clipping"}, m, {"name": "sgd", "lr": 0.01})
        assert isinstance(wc, WeightClipping)
        assert wc.kappa == 2.0 and wc.clip_bias is True
        assert wc.requires_features is False
        wc2 = build_method({"name": "weight_clipping", "kappa": 3.0, "clip_bias": False}, m, {"name": "sgd", "lr": 0.01})
        assert wc2.kappa == 3.0 and wc2.clip_bias is False and wc2._biases == []

    def test_noop_at_init_for_kappa_ge_1(self):
        m = make_model()
        before = [p.detach().clone() for p in m.parameters()]
        WeightClipping(m, {"kappa": 1.0}, {"name": "sgd", "lr": 0.01})
        for p, q in zip(m.parameters(), before):
            assert torch.equal(p.detach(), q)

    @pytest.mark.parametrize("kappa", [1.0, 2.0])
    @pytest.mark.parametrize("opt", ["sgd", "adam"])
    def test_bounds_hold_after_large_lr_steps(self, kappa, opt):
        m = make_model(seed=3)
        wc = WeightClipping(m, {"kappa": kappa}, {"name": opt, "lr": 5.0})
        optimizer = wc.build_optimizer()
        x, y = synthetic_batch(30, seed=7)
        for s in range(30):
            loss = F.cross_entropy(m(x[s:s + 1]), y[s:s + 1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            wc.before_step(s)
            optimizer.step()
            # with lr=5 the raw step leaves the box (otherwise the test would be vacuous)
            if s == 0:
                left_box = any(float(l.theta.detach().abs().max()) > kappa * l.weight_init_bound for l in m.linear_layers)
                assert left_box
            wc.after_step(s, None)
            assert_within_bounds(m, kappa, clip_bias=True)
        assert wc.n_clipped_steps == 30
        s = wc.state_summary()
        assert 0.0 <= s["clip_frac"] <= 1.0 and 0.0 <= s["clip_frac_bias"] <= 1.0
        assert s["clip_frac"] > 0.0  # large steps must have pushed some entries onto the bound
        assert all(math.isfinite(float(p.detach().abs().max())) for p in m.parameters())

    def test_clip_bias_false_leaves_bias_unbounded(self):
        m = make_model(seed=5)
        wc = WeightClipping(m, {"kappa": 1.0, "clip_bias": False}, {"name": "sgd", "lr": 5.0})
        optimizer = wc.build_optimizer()
        run_steps(m, wc, optimizer, 20, seed=11)
        assert_within_bounds(m, 1.0, clip_bias=False)
        assert "clip_frac_bias" not in wc.state_summary()
        bias_outside = any(float(l.theta_bias.detach().abs().max()) > 1.0 * l.bias_init_bound for l in m.linear_layers)
        assert bias_outside

    def test_clip_frac_exact_on_manual_weights(self):
        m = make_model(seed=2)
        wc = WeightClipping(m, {"kappa": 2.0}, {"name": "sgd", "lr": 0.01})
        with torch.no_grad():
            for l in m.linear_layers:
                l.theta.zero_()
            lin0 = m.linear_layers[0]
            lin0.theta[0, :] = 2.0 * lin0.weight_init_bound          # one full row at +bound
            lin0.theta[1, :] = -2.0 * lin0.weight_init_bound         # one full row at -bound
        n_total = sum(l.theta.numel() for l in m.linear_layers)
        expected = 2 * lin0.in_features / n_total
        assert wc.state_summary()["clip_frac"] == pytest.approx(expected)
        assert wc.state_summary()["clip_frac_bias"] == 0.0

    def test_projection_at_construction_for_kappa_lt_1(self):
        m = make_model(seed=4)
        WeightClipping(m, {"kappa": 0.5}, {"name": "sgd", "lr": 0.01})
        assert_within_bounds(m, 0.5)

    @pytest.mark.parametrize("mode", ["sin", "tanh"])
    def test_raises_for_reparametrised_layers(self, mode):
        m = make_model(mode)
        with pytest.raises(ValueError, match="standard parametrisation"):
            WeightClipping(m, {}, {"name": "sgd", "lr": 0.01})
        # mixed: only output layer reparametrised still raises
        g = torch.Generator().manual_seed(0)
        m2 = MLP(IN, HID, NC, reparam={"hidden": "standard", "output": mode}, generator=g)
        with pytest.raises(ValueError):
            WeightClipping(m2, {}, {"name": "sgd", "lr": 0.01})

    def test_invalid_kappa(self):
        with pytest.raises(ValueError):
            WeightClipping(make_model(), {"kappa": 0.0}, {"name": "sgd", "lr": 0.01})

    def test_runs_inside_online_trainer(self):
        m = make_model(seed=8)
        wc = build_method({"name": "weight_clipping", "kappa": 1.0}, m, {"name": "sgd", "lr": 2.0})
        opt = wc.build_optimizer()
        x, y = synthetic_batch(64, seed=9)
        task = Task(index=0, x_train=x.numpy().astype(np.float32), y_train=y.numpy().astype(np.int64))
        res = _train_on_task(m, wc, opt, task, {"batch_size": 1, "epochs_per_task": 1}, 0)
        assert res["global_step"] == 64 and wc.n_clipped_steps == 64
        assert_within_bounds(m, 1.0)


# ------------------------------------------------------------------------------------------------
# L2 Init
# ------------------------------------------------------------------------------------------------
def perturb_(model, scale=0.3, seed=21):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))


def autograd_penalty_grads(model, init_params, lam, names):
    """Reference: autograd gradient of lam * sum((p - p0)**2) over the regularised params."""
    params = dict(model.named_parameters())
    model.zero_grad(set_to_none=True)
    loss = sum(lam * ((params[n] - p0) ** 2).sum() for n, p0 in zip(names, init_params))
    loss.backward()
    return {n: params[n].grad.detach().clone() for n in names}


class TestL2Init:
    def test_registry_and_defaults(self):
        m = make_model()
        l2 = build_method({"name": "l2_init"}, m, {"name": "sgd", "lr": 0.01})
        assert isinstance(l2, L2Init)
        assert l2.lam == 1e-2 and l2.include_bias is True
        assert l2.regularizer() is None
        assert len(l2.params) == len(list(m.parameters()))
        assert l2.state_summary()["dist_to_init"] == 0.0

    def test_theta0_is_detached_clone(self):
        m = make_model()
        l2 = L2Init(m, {"lam": 0.1}, {"name": "sgd", "lr": 0.01})
        for p0, p in zip(l2.init_params, l2.params):
            assert not p0.requires_grad and p0.data_ptr() != p.data_ptr()
            assert torch.equal(p0, p.detach())
        perturb_(m)
        for p0, p in zip(l2.init_params, l2.params):
            assert not torch.equal(p0, p.detach())  # θ0 did not follow the parameters

    @pytest.mark.parametrize("mode", ["standard", "sin", "tanh"])
    @pytest.mark.parametrize("layer_norm", [False, True])
    def test_gradient_matches_autograd(self, mode, layer_norm):
        m = make_model(mode, seed=13, layer_norm=layer_norm)
        lam = 0.37
        l2 = L2Init(m, {"lam": lam}, {"name": "sgd", "lr": 0.01})
        perturb_(m, seed=5)
        ref = autograd_penalty_grads(m, l2.init_params, lam, l2.param_names)
        # method path: start from zero grads (pure penalty gradient)
        m.zero_grad(set_to_none=True)
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        l2.before_step(0)
        for n, p in m.named_parameters():
            assert torch.allclose(p.grad, ref[n], atol=1e-6, rtol=1e-5), n
            assert torch.allclose(p.grad, 2 * lam * (p.detach() - dict(zip(l2.param_names, l2.init_params))[n]), atol=1e-6)

    def test_gradient_is_additive_to_data_gradient(self):
        m = make_model(seed=17)
        lam = 0.05
        l2 = L2Init(m, {"lam": lam}, {"name": "sgd", "lr": 0.01})
        perturb_(m, seed=6)
        x, y = synthetic_batch(4, seed=3)
        # data gradient alone
        m.zero_grad(set_to_none=True)
        F.cross_entropy(m(x), y).backward()
        data_g = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
        # combined via the loss (reference)
        m.zero_grad(set_to_none=True)
        params = dict(m.named_parameters())
        loss = F.cross_entropy(m(x), y) + sum(lam * ((params[n] - p0) ** 2).sum() for n, p0 in zip(l2.param_names, l2.init_params))
        loss.backward()
        ref = {n: p.grad.detach().clone() for n, p in m.named_parameters()}
        # combined via the method
        m.zero_grad(set_to_none=True)
        F.cross_entropy(m(x), y).backward()
        l2.before_step(0)
        for n, p in m.named_parameters():
            assert torch.allclose(p.grad, ref[n], atol=1e-6, rtol=1e-5), n
            assert not torch.allclose(p.grad, data_g[n])  # the penalty actually changed the gradient

    def test_none_grad_gets_penalty_gradient(self):
        m = make_model(seed=19)
        l2 = L2Init(m, {"lam": 0.2}, {"name": "sgd", "lr": 0.01})
        perturb_(m, seed=8)
        m.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in m.parameters())
        l2.before_step(0)
        for p, p0 in zip(l2.params, l2.init_params):
            assert p.grad is not None
            assert torch.allclose(p.grad, 0.4 * (p.detach() - p0), atol=1e-7)

    @pytest.mark.parametrize("mode", ["standard", "sin"])
    @pytest.mark.parametrize("opt", ["sgd", "adam"])
    def test_zero_data_gradient_moves_toward_init(self, mode, opt):
        m = make_model(mode, seed=23)
        l2 = L2Init(m, {"lam": 0.1}, {"name": opt, "lr": 0.05})
        optimizer = l2.build_optimizer()
        perturb_(m, seed=9)
        d_prev = l2.state_summary()["dist_to_init"]
        assert d_prev > 0
        per_param_prev = [float((p.detach() - p0).norm()) for p, p0 in zip(l2.params, l2.init_params)]
        for s in range(5):
            optimizer.zero_grad(set_to_none=True)
            for p in m.parameters():
                p.grad = torch.zeros_like(p)  # zero data gradient
            l2.before_step(s)
            optimizer.step()
            l2.after_step(s, None)
            d = l2.state_summary()["dist_to_init"]
            assert d < d_prev
            d_prev = d
        per_param = [float((p.detach() - p0).norm()) for p, p0 in zip(l2.params, l2.init_params)]
        assert all(a < b for a, b in zip(per_param, per_param_prev))

    def test_sgd_step_closed_form(self):
        """SGD with zero data gradient: p <- p - lr*2*lam*(p-p0), i.e. (p-p0) shrinks by (1 - 2*lr*lam)."""
        m = make_model(seed=29)
        lam, lr = 0.25, 0.1
        l2 = L2Init(m, {"lam": lam}, {"name": "sgd", "lr": lr})
        optimizer = l2.build_optimizer()
        perturb_(m, seed=10)
        diffs0 = [p.detach().clone() - p0 for p, p0 in zip(l2.params, l2.init_params)]
        optimizer.zero_grad(set_to_none=True)
        l2.before_step(0)
        optimizer.step()
        for p, p0, d0 in zip(l2.params, l2.init_params, diffs0):
            assert torch.allclose(p.detach() - p0, (1 - 2 * lr * lam) * d0, atol=1e-6)

    def test_lam_zero_is_noop(self):
        m = make_model(seed=31)
        l2 = L2Init(m, {"lam": 0.0}, {"name": "sgd", "lr": 0.01})
        perturb_(m)
        m.zero_grad(set_to_none=True)
        l2.before_step(0)
        assert all(p.grad is None for p in m.parameters())

    def test_include_bias_false(self):
        m = make_model(seed=37, layer_norm=True)
        l2 = L2Init(m, {"lam": 0.1, "include_bias": False}, {"name": "sgd", "lr": 0.01})
        assert all(not n.endswith("bias") for n in l2.param_names)
        assert any(n.endswith("theta") for n in l2.param_names)
        assert any("norms" in n and n.endswith("weight") for n in l2.param_names)  # LayerNorm scale kept
        perturb_(m)
        m.zero_grad(set_to_none=True)
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        l2.before_step(0)
        for n, p in m.named_parameters():
            if n.endswith("bias"):
                assert float(p.grad.abs().max()) == 0.0
            else:
                assert float(p.grad.abs().max()) > 0.0

    def test_dist_to_init_value(self):
        m = make_model(seed=41)
        l2 = L2Init(m, {"lam": 0.5}, {"name": "sgd", "lr": 0.01})
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.1)
        n = sum(p.numel() for p in m.parameters())
        s = l2.state_summary()
        assert s["dist_to_init"] == pytest.approx(0.1 * math.sqrt(n), rel=1e-5)
        assert s["reg_loss"] == pytest.approx(0.5 * 0.01 * n, rel=1e-5)
        assert l2.penalty() == pytest.approx(s["reg_loss"], rel=1e-6)

    def test_reset_init(self):
        m = make_model(seed=43)
        l2 = L2Init(m, {"lam": 0.5}, {"name": "sgd", "lr": 0.01})
        perturb_(m)
        assert l2.dist_to_init() > 0
        l2.reset_init()
        assert l2.dist_to_init() == 0.0

    @pytest.mark.parametrize("mode", ["standard", "sin", "tanh"])
    def test_runs_inside_online_trainer(self, mode):
        m = make_model(mode, seed=47)
        l2 = build_method({"name": "l2_init", "lam": 0.01}, m, {"name": "sgd", "lr": 0.01})
        opt = l2.build_optimizer()
        x, y = synthetic_batch(64, seed=12)
        task = Task(index=0, x_train=x.numpy().astype(np.float32), y_train=y.numpy().astype(np.int64))
        res = _train_on_task(m, l2, opt, task, {"batch_size": 1, "epochs_per_task": 1}, 0)
        assert res["global_step"] == 64
        assert 0.0 < l2.state_summary()["dist_to_init"] < 10.0
