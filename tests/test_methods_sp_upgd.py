"""Tests for Shrink & Perturb (Ash & Adams 2020 / Dohare et al. 2024) and UPGD (Elsayed & Mahmood 2024).

Fast, synthetic-data tests (torch.randn inputs, manually set gradients); no dataset access.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from plasticity.data import Task
from plasticity.methods import build_method
from plasticity.methods.shrink_perturb import ShrinkPerturb
from plasticity.methods.upgd import UPGD
from plasticity.models import MLP
from plasticity.training.online import _train_on_task

IN, HID, NC = 20, (16, 12), 5
SGD = {"name": "sgd", "lr": 0.01}


def make_model(mode="standard", seed=0, layer_norm=False, inp=IN, hid=HID, nc=NC):
    g = torch.Generator().manual_seed(seed)
    return MLP(inp, hid, nc, reparam={"hidden": mode, "output": mode}, layer_norm=layer_norm, generator=g)


def big_model(seed=0):
    """~9k parameters: enough elements for tight statistical checks on noise."""
    return make_model(seed=seed, inp=64, hid=(64, 64), nc=10)


def synthetic_batch(n, seed=1, inp=IN, nc=NC):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, inp, generator=g), torch.randint(0, nc, (n,), generator=g)


def snapshot(model):
    return [p.detach().clone() for p in model.parameters()]


def set_grads(model, seed=3, scale=1.0, params=None):
    """Deterministic random gradients on every parameter (or the given list)."""
    g = torch.Generator().manual_seed(seed)
    for p in (params if params is not None else model.parameters()):
        p.grad = scale * torch.randn(p.shape, generator=g)


def run_steps(model, method, optimizer, n_steps, seed=1, batch=1):
    x, y = synthetic_batch(n_steps * batch, seed)
    for s in range(n_steps):
        xb, yb = x[s * batch:(s + 1) * batch], y[s * batch:(s + 1) * batch]
        loss = F.cross_entropy(model(xb), yb)
        reg = method.regularizer()
        if reg is not None:
            loss = loss + reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        method.before_step(s)
        optimizer.step()
        method.after_step(s, None)


# ------------------------------------------------------------------------------------------------
# Shrink & Perturb
# ------------------------------------------------------------------------------------------------
class TestShrinkPerturb:
    def test_registry_and_defaults(self):
        m = make_model()
        sp = build_method({"name": "shrink_perturb"}, m, SGD, torch.Generator().manual_seed(1))
        assert isinstance(sp, ShrinkPerturb)
        assert sp.shrink == 1e-4 and sp.noise_std == 1e-3 and sp.every == "step" and sp.perturb_bias is True
        assert sp.requires_features is False and sp.regularizer() is None
        assert sp.state_summary() == {"n_applications": 0}
        assert len(sp.params) == 2 * len(m.linear_layers)  # weights + biases
        sp2 = build_method({"name": "shrink_perturb", "shrink": 0.01, "noise_std": 0.0, "every": "task", "perturb_bias": False}, m, SGD)
        assert sp2.shrink == 0.01 and sp2.noise_std == 0.0 and sp2.every == "task" and sp2.perturb_bias is False
        assert len(sp2.params) == len(m.linear_layers)

    def test_shrink_half_without_noise_halves_parameters(self):
        m = make_model(seed=2, layer_norm=True)
        sp = ShrinkPerturb(m, {"shrink": 0.5, "noise_std": 0.0}, SGD, torch.Generator().manual_seed(0))
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        sp.after_step(0, None)
        for n, p in m.named_parameters():
            if "norms" in n:  # LayerNorm affine parameters are not touched
                assert torch.equal(p.detach(), before[n]), n
            else:
                assert torch.allclose(p.detach(), 0.5 * before[n], atol=0, rtol=0), n
        assert sp.state_summary()["n_applications"] == 1
        sp.after_step(1, None)
        for n, p in m.named_parameters():
            if "norms" not in n:
                assert torch.allclose(p.detach(), 0.25 * before[n], atol=1e-8)
        assert sp.n_applications == 2

    def test_perturb_bias_false_leaves_biases_untouched(self):
        m = make_model(seed=4)
        with torch.no_grad():
            for lin in m.linear_layers:
                lin.theta_bias.fill_(0.7)
        sp = ShrinkPerturb(m, {"shrink": 0.5, "noise_std": 0.1, "perturb_bias": False}, SGD, torch.Generator().manual_seed(0))
        before = snapshot(m)
        sp.after_step(0, None)
        for lin in m.linear_layers:
            assert torch.all(lin.theta_bias.detach() == 0.7)
        changed = [not torch.equal(p.detach(), q) for p, q in zip(m.parameters(), before)]
        assert sum(changed) == len(m.linear_layers)  # exactly the weight matrices changed

    def test_noise_has_expected_std_when_shrink_is_zero(self):
        m = big_model(seed=5)
        noise_std = 0.01
        sp = ShrinkPerturb(m, {"shrink": 0.0, "noise_std": noise_std}, SGD, torch.Generator().manual_seed(123))
        before = snapshot(m)
        sp.after_step(0, None)
        diffs = torch.cat([(p.detach() - q).flatten() for p, q in zip(m.parameters(), before)])
        n = diffs.numel()
        assert n > 8000
        # std of n i.i.d. N(0, noise_std²) samples: relative error ~ 1/sqrt(2n) ≈ 0.75 %  -> 5 % tolerance
        assert float(diffs.std()) == pytest.approx(noise_std, rel=0.05)
        # mean ≈ 0 within 5 standard errors
        assert abs(float(diffs.mean())) < 5 * noise_std / math.sqrt(n)
        # per-tensor: every tensor received its own (non-constant) noise
        for p, q in zip(m.parameters(), before):
            d = p.detach() - q
            assert float(d.std()) > 0.5 * noise_std

    def test_combined_update_formula(self):
        """θ' = (1-shrink) θ + noise_std ε  <=>  (θ' - (1-shrink) θ) / noise_std has unit std."""
        m = big_model(seed=6)
        sp = ShrinkPerturb(m, {"shrink": 0.3, "noise_std": 0.02}, SGD, torch.Generator().manual_seed(9))
        before = snapshot(m)
        sp.apply()
        eps = torch.cat([((p.detach() - 0.7 * q) / 0.02).flatten() for p, q in zip(m.parameters(), before)])
        assert float(eps.std()) == pytest.approx(1.0, rel=0.05)
        assert abs(float(eps.mean())) < 5 / math.sqrt(eps.numel())

    def test_every_task_applies_only_in_on_task_end(self):
        m = make_model(seed=7)
        sp = ShrinkPerturb(m, {"shrink": 0.5, "noise_std": 0.1, "every": "task"}, SGD, torch.Generator().manual_seed(0))
        before = snapshot(m)
        for s in range(5):
            sp.after_step(s, None)
        for p, q in zip(m.parameters(), before):
            assert torch.equal(p.detach(), q)
        assert sp.n_applications == 0
        sp.on_task_start(0)
        assert sp.n_applications == 0
        sp.on_task_end(0)
        assert sp.n_applications == 1
        assert all(not torch.equal(p.detach(), q) for p, q in zip(m.parameters(), before))
        # every='step' never applies at task end
        m2 = make_model(seed=7)
        sp2 = ShrinkPerturb(m2, {"shrink": 0.5, "noise_std": 0.1, "every": "step"}, SGD, torch.Generator().manual_seed(0))
        b2 = snapshot(m2)
        sp2.on_task_end(0)
        assert sp2.n_applications == 0 and all(torch.equal(p.detach(), q) for p, q in zip(m2.parameters(), b2))

    def test_reproducible_with_generator(self):
        outs = []
        for _ in range(2):
            m = make_model(seed=8)
            sp = ShrinkPerturb(m, {"shrink": 0.1, "noise_std": 0.05}, SGD, torch.Generator().manual_seed(77))
            sp.apply()
            sp.apply()
            outs.append(snapshot(m))
        for a, b in zip(*outs):
            assert torch.equal(a, b)
        m = make_model(seed=8)
        sp = ShrinkPerturb(m, {"shrink": 0.1, "noise_std": 0.05}, SGD, torch.Generator().manual_seed(78))
        sp.apply()
        sp.apply()
        assert any(not torch.equal(a, b) for a, b in zip(snapshot(m), outs[0]))

    @pytest.mark.parametrize("mode", ["sin", "tanh"])
    def test_works_on_free_parameters_of_reparam_models(self, mode):
        m = make_model(mode, seed=9)
        sp = ShrinkPerturb(m, {"shrink": 0.5, "noise_std": 0.0}, SGD, torch.Generator().manual_seed(0))  # no ValueError
        before = snapshot(m)
        sp.apply()
        for p, q in zip(m.parameters(), before):
            assert torch.allclose(p.detach(), 0.5 * q)
        w = m.linear_layers[0].effective_weight()
        w = w.detach()
        assert torch.isfinite(w).all() and float(w.abs().max()) <= float(m.linear_layers[0].amplitude) + 1e-6

    def test_invalid_config(self):
        m = make_model()
        for bad in ({"shrink": 1.5}, {"shrink": -0.1}, {"noise_std": -1.0}, {"every": "epoch"}, {"bogus": 1}):
            with pytest.raises(ValueError):
                ShrinkPerturb(m, bad, SGD)

    @pytest.mark.parametrize("every", ["step", "task"])
    def test_runs_inside_online_trainer(self, every):
        m = make_model(seed=10)
        sp = build_method({"name": "shrink_perturb", "every": every}, m, SGD, torch.Generator().manual_seed(3))
        opt = sp.build_optimizer()
        x, y = synthetic_batch(64, seed=11)
        task = Task(index=0, x_train=x.numpy().astype(np.float32), y_train=y.numpy().astype(np.int64))
        sp.on_task_start(0)
        res = _train_on_task(m, sp, opt, task, {"batch_size": 1, "epochs_per_task": 1}, 0)
        sp.on_task_end(0)
        assert res["global_step"] == 64
        assert sp.state_summary()["n_applications"] == (64 if every == "step" else 1)
        assert all(torch.isfinite(p).all() for p in m.parameters())


# ------------------------------------------------------------------------------------------------
# UPGD
# ------------------------------------------------------------------------------------------------
SIG1 = 1.0 / (1.0 + math.exp(-1.0))  # sigmoid(1) ≈ 0.7311


class TestUPGD:
    def test_registry_and_defaults(self):
        m = make_model()
        u = build_method({"name": "upgd"}, m, SGD, torch.Generator().manual_seed(1))
        assert isinstance(u, UPGD)
        assert u.beta == 0.999 and u.sigma == 1e-3 and u.eps == 1e-8 and u.include_bias is True
        assert u.gate_scale == 1.0 and u.protect_ratio == 0.5
        assert u.requires_features is False and u.regularizer() is None
        assert len(u.params) == len(list(m.parameters()))
        assert all(float(t.abs().sum()) == 0.0 for t in u.utility_trace)
        assert u.state_summary() == {"mean_scaled_utility": 0.5, "frac_protected": 0.0}
        u2 = build_method({"name": "upgd", "beta": 0.9, "sigma": 0.0, "eps": 1e-6, "include_bias": False}, m, SGD)
        assert u2.beta == 0.9 and u2.sigma == 0.0 and u2.eps == 1e-6 and u2.include_bias is False

    def test_zero_utility_gives_half_sgd_step(self):
        """θ = 0 => U = -gθ = 0 everywhere => s = 0.5 => replaced grad = 0.5 g and the SGD step is halved."""
        lr = 0.1
        m_sgd = make_model(seed=12)
        with torch.no_grad():
            for p in m_sgd.parameters():
                p.zero_()
        m_upgd = copy.deepcopy(m_sgd)
        u = UPGD(m_upgd, {"sigma": 0.0}, {"name": "sgd", "lr": lr}, torch.Generator().manual_seed(0))
        opt_u = u.build_optimizer()
        opt_s = torch.optim.SGD(m_sgd.parameters(), lr=lr)
        set_grads(m_sgd, seed=5)
        set_grads(m_upgd, seed=5)
        g_orig = [p.grad.clone() for p in m_upgd.parameters()]
        u.before_step(0)
        for p, g in zip(m_upgd.parameters(), g_orig):
            assert torch.allclose(p.grad, 0.5 * g, atol=1e-7, rtol=1e-6)
        opt_u.step()
        opt_s.step()
        for p_u, p_s in zip(m_upgd.parameters(), m_sgd.parameters()):
            assert torch.allclose(p_u.detach(), 0.5 * p_s.detach(), atol=1e-7, rtol=1e-6)
            assert float(p_s.detach().abs().max()) > 0  # the SGD step was not trivial
        s = u.state_summary()
        assert s["mean_scaled_utility"] == pytest.approx(0.5, abs=1e-6) and s["frac_protected"] == 0.0

    @pytest.mark.parametrize("sign", [+1.0, -1.0])
    def test_uniform_nonzero_utility_closed_form(self, sign):
        """θ = c, g = sign => U = -sign*c for every element => û/max|û| = -sign => gate = 1 - sigmoid(-sign)."""
        m = make_model(seed=13)
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(0.3)
        u = UPGD(m, {"sigma": 0.0, "beta": 0.9}, SGD, torch.Generator().manual_seed(0))
        for p in m.parameters():
            p.grad = torch.full_like(p, sign)
        u.before_step(0)
        s = 1.0 / (1.0 + math.exp(sign))        # sigmoid(-sign)
        expected_gate = 1.0 - s
        for p in m.parameters():
            assert torch.allclose(p.grad, torch.full_like(p, sign * expected_gate), atol=1e-6)
        summ = u.state_summary()
        assert summ["mean_scaled_utility"] == pytest.approx(s, abs=1e-6)
        # positive utility (sign = -1): every element has û = max|û| > 0.5*max  -> all protected
        assert summ["frac_protected"] == (1.0 if sign < 0 else 0.0)

    def test_high_utility_elements_are_updated_less(self):
        m = make_model(seed=14)
        lin0 = m.linear_layers[0]
        with torch.no_grad():
            lin0.theta.fill_(1.0)
        u = UPGD(m, {"sigma": 0.0}, SGD, torch.Generator().manual_seed(0))
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        g0 = torch.linspace(-1.0, 1.0, lin0.theta.numel()).view_as(lin0.theta)
        g0[g0 == 0] = 1e-3
        lin0.theta.grad = g0.clone()
        u.before_step(0)
        utility = (-g0 * lin0.theta.detach()).flatten()          # U = -gθ, decreasing in g
        gate = (lin0.theta.grad / g0).flatten()                   # (1 - s): the per-element step multiplier
        order = torch.argsort(utility)
        gate_sorted = gate[order]
        assert torch.all(gate_sorted[1:] <= gate_sorted[:-1] + 1e-7)   # higher utility -> smaller multiplier
        assert float(gate[utility.argmax()]) == pytest.approx(1.0 - SIG1, abs=1e-5)   # most useful: 0.269
        assert float(gate[utility.argmin()]) == pytest.approx(SIG1, abs=1e-5)         # least useful: 0.731
        assert float(gate[utility.argmax()]) < 0.5 < float(gate[utility.argmin()])
        # magnitude of the actual parameter change follows the same ordering
        before = lin0.theta.detach().clone()
        u.build_optimizer().step()
        step = (lin0.theta.detach() - before).abs().flatten() / g0.abs().flatten()
        assert float(step[utility.argmax()]) < float(step[utility.argmin()])
        # elements with s > 0.75 cannot exist (s <= sigmoid(1)); protected = û > 0.5 * max|û|
        summ = u.state_summary()
        n_all = sum(p.numel() for p in m.parameters())
        n_prot = int((utility > 0.5 * utility.abs().max()).sum())
        assert summ["frac_protected"] == pytest.approx(n_prot / n_all)
        assert 0 < summ["frac_protected"] < 1

    def test_trace_is_bias_corrected(self):
        beta = 0.9
        m = make_model(seed=15)
        u = UPGD(m, {"beta": beta, "sigma": 0.0}, SGD, torch.Generator().manual_seed(0))
        set_grads(m, seed=21)
        U1 = [(-p.grad * p.detach()).clone() for p in m.parameters()]
        u.before_step(0)
        assert u.step_count == 1 and u.bias_correction() == pytest.approx(1 - beta)
        for raw, uh, U in zip(u.utility_trace, u.corrected_utility(), U1):
            assert torch.allclose(raw, (1 - beta) * U, atol=1e-7)
            assert torch.allclose(uh, U, atol=1e-6, rtol=1e-5)          # after one step û == U exactly
        set_grads(m, seed=22)
        U2 = [(-p.grad * p.detach()).clone() for p in m.parameters()]
        u.before_step(1)
        for uh, a, b in zip(u.corrected_utility(), U1, U2):
            expected = (beta * (1 - beta) * a + (1 - beta) * b) / (1 - beta ** 2)
            assert torch.allclose(uh, expected, atol=1e-6, rtol=1e-5)

    def test_noise_statistics_and_reproducibility(self):
        """θ = 0, g = 0 => gate = 0.5 => replaced grad = 0.5 * sigma * ξ."""
        sigma = 0.02
        grads = []
        for seed in (99, 99, 100):
            m = big_model(seed=16)
            with torch.no_grad():
                for p in m.parameters():
                    p.zero_()
            u = UPGD(m, {"sigma": sigma}, SGD, torch.Generator().manual_seed(seed))
            for p in m.parameters():
                p.grad = torch.zeros_like(p)
            u.before_step(0)
            grads.append(torch.cat([p.grad.flatten() for p in m.parameters()]))
        d = grads[0]
        assert d.numel() > 8000
        assert float(d.std()) == pytest.approx(0.5 * sigma, rel=0.05)
        assert abs(float(d.mean())) < 5 * 0.5 * sigma / math.sqrt(d.numel())
        assert torch.equal(grads[0], grads[1])          # same generator seed -> identical noise
        assert not torch.equal(grads[0], grads[2])      # different seed -> different noise

    def test_sigma_zero_is_deterministic_and_noise_free(self):
        m = make_model(seed=17)
        u = UPGD(m, {"sigma": 0.0}, SGD, torch.Generator().manual_seed(0))
        set_grads(m, seed=4)
        g = [p.grad.clone() for p in m.parameters()]
        u.before_step(0)
        for p, g0 in zip(m.parameters(), g):
            assert torch.all(torch.sign(p.grad) == torch.sign(g0))       # pure rescaling
            ratio = (p.grad / g0)[g0 != 0]
            assert float(ratio.min()) >= 1 - SIG1 - 1e-6 and float(ratio.max()) <= SIG1 + 1e-6

    @pytest.mark.parametrize("mode", ["sin", "tanh"])
    def test_raises_for_reparametrised_layers(self, mode):
        with pytest.raises(ValueError, match="standard parametrisation"):
            UPGD(make_model(mode), {}, SGD)
        g = torch.Generator().manual_seed(0)
        m2 = MLP(IN, HID, NC, reparam={"hidden": "standard", "output": mode}, generator=g)
        with pytest.raises(ValueError):
            UPGD(m2, {}, SGD)

    def test_include_bias_false(self):
        m = make_model(seed=18, layer_norm=True)
        u = UPGD(m, {"include_bias": False, "sigma": 0.0}, SGD, torch.Generator().manual_seed(0))
        assert all(not n.endswith("bias") for n in u.param_names)
        assert any(n.endswith("theta") for n in u.param_names)
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(0.5)
        set_grads(m, seed=6)
        g = {n: p.grad.clone() for n, p in m.named_parameters()}
        u.before_step(0)
        for n, p in m.named_parameters():
            if n.endswith("bias"):
                assert torch.equal(p.grad, g[n]), n          # untouched: plain gradient step
            else:
                assert not torch.equal(p.grad, g[n]), n

    def test_none_grad_is_treated_as_zero(self):
        m = make_model(seed=19)
        u = UPGD(m, {"beta": 0.5, "sigma": 0.0}, SGD, torch.Generator().manual_seed(0))
        set_grads(m, seed=7)
        u.before_step(0)
        tr0 = [t.clone() for t in u.utility_trace]
        set_grads(m, seed=8)
        m.output.theta.grad = None
        u.before_step(1)
        idx = u.param_names.index("output.theta")
        assert torch.allclose(u.utility_trace[idx], 0.5 * tr0[idx])      # decayed only
        assert m.output.theta.grad is None                               # optimiser will skip it
        assert all(p.grad is not None for n, p in m.named_parameters() if n != "output.theta")

    def test_state_summary_matches_scaled_utility(self):
        m = big_model(seed=20)
        u = UPGD(m, {"sigma": 0.0}, SGD, torch.Generator().manual_seed(0))
        set_grads(m, seed=9)
        u.before_step(0)
        s = torch.cat([t.flatten() for t in u.scaled_utility()])
        summ = u.state_summary()
        assert summ["mean_scaled_utility"] == pytest.approx(float(s.mean()), abs=1e-6)
        assert summ["frac_protected"] == pytest.approx(float((s > u._protect_s).float().mean()), abs=1e-6)
        assert float(s.min()) >= 1 - SIG1 - 1e-6 and float(s.max()) <= SIG1 + 1e-6
        assert 0.0 < summ["frac_protected"] < 1.0

    def test_gate_scale_two_reproduces_public_convention(self):
        """gate_scale=2 (public code: alpha = -2 lr): a neutral weight takes the full SGD step."""
        m = make_model(seed=21)
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
        u = UPGD(m, {"sigma": 0.0, "gate_scale": 2.0}, SGD, torch.Generator().manual_seed(0))
        set_grads(m, seed=10)
        g = [p.grad.clone() for p in m.parameters()]
        u.before_step(0)
        for p, g0 in zip(m.parameters(), g):
            assert torch.allclose(p.grad, g0, atol=1e-7)
        assert u.state_summary()["mean_scaled_utility"] == pytest.approx(0.5, abs=1e-6)

    def test_invalid_config(self):
        m = make_model()
        for bad in ({"beta": 1.0}, {"beta": -0.1}, {"sigma": -1.0}, {"eps": 0.0}, {"gate_scale": 0.0}, {"bogus": 1}):
            with pytest.raises(ValueError):
                UPGD(m, bad, SGD)

    @pytest.mark.parametrize("opt", ["sgd", "adam"])
    def test_runs_inside_online_trainer(self, opt):
        m = make_model(seed=22)
        u = build_method({"name": "upgd"}, m, {"name": opt, "lr": 0.01}, torch.Generator().manual_seed(2))
        optimizer = u.build_optimizer()
        x, y = synthetic_batch(64, seed=13)
        task = Task(index=0, x_train=x.numpy().astype(np.float32), y_train=y.numpy().astype(np.int64))
        res = _train_on_task(m, u, optimizer, task, {"batch_size": 1, "epochs_per_task": 1}, 0)
        assert res["global_step"] == 64 and u.step_count == 64
        summ = u.state_summary()
        assert 1 - SIG1 - 1e-6 <= summ["mean_scaled_utility"] <= SIG1 + 1e-6
        assert 0.0 <= summ["frac_protected"] <= 1.0
        assert all(torch.isfinite(p).all() for p in m.parameters())
        assert all(torch.isfinite(t).all() for t in u.utility_trace)

    def test_learns_on_synthetic_task(self):
        """Sanity: with UPGD the loss on a fixed small batch decreases over repeated steps."""
        m = make_model(seed=23)
        u = UPGD(m, {"sigma": 1e-3}, {"name": "sgd", "lr": 0.05}, torch.Generator().manual_seed(0))
        opt = u.build_optimizer()
        x, y = synthetic_batch(8, seed=14)
        losses = []
        for s in range(40):
            loss = F.cross_entropy(m(x), y)
            losses.append(float(loss.detach()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            u.before_step(s)
            opt.step()
            u.after_step(s, None)
        assert losses[-1] < 0.7 * losses[0]
