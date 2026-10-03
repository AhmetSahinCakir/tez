"""Tests for Parseval regularisation (Chung et al. 2024) and the scale-corrected update (thesis variant).

Fast, synthetic-data tests (torch.randn inputs); no dataset access.
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
from plasticity.methods.parseval import Parseval
from plasticity.methods.scale_corrected import ScaleCorrectedSin
from plasticity.models import MLP
from plasticity.training.online import _train_on_task

torch.set_num_threads(1)

IN, HID, NC = 20, (16, 12), 5
SGD = {"name": "sgd", "lr": 0.01}


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def make_model(mode="standard", seed=0, inp=IN, hid=HID, nc=NC, output_mode=None, **kw):
    g = torch.Generator().manual_seed(seed)
    out_mode = mode if output_mode is None else output_mode
    return MLP(inp, hid, nc, reparam={"hidden": mode, "output": out_mode}, generator=g, **kw)


def synthetic_batch(n, seed=1, inp=IN, nc=NC, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, inp, generator=g, dtype=dtype), torch.randint(0, nc, (n,), generator=g)


def orthonormal(out, inp, seed=0):
    """(out, inp) matrix with orthonormal rows (out <= inp) or orthonormal columns (out > inp)."""
    g = torch.Generator().manual_seed(seed)
    if out <= inp:
        q, _ = torch.linalg.qr(torch.randn(inp, out, generator=g))
        return q.T.contiguous()
    q, _ = torch.linalg.qr(torch.randn(out, inp, generator=g))
    return q.contiguous()


def gram_residual(w):
    """Reference ``G - I`` with the smaller Gram matrix."""
    w = w.detach()
    if w.shape[0] <= w.shape[1]:
        return w @ w.T - torch.eye(w.shape[0], dtype=w.dtype)
    return w.T @ w - torch.eye(w.shape[1], dtype=w.dtype)


def reference_penalty(layers, beta):
    return beta * sum(float((gram_residual(l.effective_weight()) ** 2).sum()) for l in layers)


def val(t):
    """Scalar value of a (possibly autograd-attached) tensor."""
    return float(t.detach())


def near_bound_model(mode="sin", theta_scale="amplitude", seed=0, frac_near=0.3, dtype=torch.float64, **kw):
    """Reparametrised model with ~30 % of the weights pushed close to the bound (|W|/A in [0.96, 0.9999])."""
    m = make_model(mode, seed=seed, theta_scale=theta_scale, **kw).to(dtype)
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for lin in m.linear_layers:
            if lin.mode == "standard":
                continue
            shape = lin.theta.shape
            r = torch.empty(shape, dtype=dtype).uniform_(-0.7, 0.7, generator=g)
            near = torch.rand(shape, generator=g, dtype=dtype) < frac_near
            sign = torch.where(torch.rand(shape, generator=g, dtype=dtype) < 0.5, -1.0, 1.0).to(dtype)
            r = torch.where(near, sign * torch.empty(shape, dtype=dtype).uniform_(0.96, 0.9999, generator=g), r)
            lin.set_effective_weight(float(lin.amplitude) * r)
            if lin.theta_bias is not None and lin.bias_mode != "standard":
                rb = torch.empty(lin.theta_bias.shape, dtype=dtype).uniform_(-0.999, 0.999, generator=g)
                lin.set_effective_bias(float(lin.bias_amplitude) * rb)
    return m


# ------------------------------------------------------------------------------------------------
# Parseval
# ------------------------------------------------------------------------------------------------
class TestParseval:
    @pytest.mark.parametrize("mode", ["standard", "sin"])
    def test_zero_for_orthonormal_rows(self, mode):
        m = make_model(mode)
        for i, lin in enumerate(m.hidden):
            lin.set_effective_weight(orthonormal(lin.out_features, lin.in_features, seed=i))
        p = Parseval(m, {"beta": 0.5}, SGD)
        reg = p.regularizer()
        assert isinstance(reg, torch.Tensor) and reg.ndim == 0 and reg.requires_grad
        assert val(reg) < 1e-9
        s = p.state_summary()
        assert s["orth_residual"] < 1e-4 and s["reg_loss"] < 1e-9
        # the (non-orthonormal) output layer is outside scope='hidden'
        assert float((gram_residual(m.output.effective_weight()) ** 2).sum()) > 1e-3

    def test_tall_layer_uses_column_gram(self):
        m = make_model("standard", hid=(24,))  # hidden layer 20 -> 24: out > in
        lin = m.hidden[0]
        lin.set_effective_weight(orthonormal(24, 20, seed=3))
        p = Parseval(m, {"beta": 1.0}, SGD)
        assert p._eyes[0].shape == (20, 20)
        assert val(p.regularizer()) < 1e-9
        w = lin.effective_weight().detach()
        assert float((w @ w.T - torch.eye(24)).norm()) > 1.0  # the 24x24 Gram is rank 20: never the identity

    def test_scope_hidden_vs_all(self):
        m = make_model("standard")
        for i, lin in enumerate(m.linear_layers):
            lin.set_effective_weight(orthonormal(lin.out_features, lin.in_features, seed=10 + i))
        p_all = Parseval(m, {"beta": 1.0, "scope": "all"}, SGD)
        p_hid = Parseval(m, {"beta": 1.0}, SGD)
        assert p_all.layers == m.linear_layers and p_all.layer_indices == [0, 1, 2]
        assert p_hid.layers == list(m.hidden) and p_hid.layer_indices == [0, 1]
        assert val(p_all.regularizer()) < 1e-9 and val(p_hid.regularizer()) < 1e-9
        with torch.no_grad():
            m.output.theta.add_(0.1)
        assert val(p_all.regularizer()) > 1e-3
        assert val(p_hid.regularizer()) < 1e-9
        assert set(p_all.state_summary()) == {"orth_residual", "orth_residual/L0", "orth_residual/L1", "orth_residual/L2", "reg_loss"}

    @pytest.mark.parametrize("mode", ["standard", "sin", "tanh"])
    def test_value_matches_reference(self, mode):
        m = make_model(mode, seed=3)
        for scope in ("hidden", "all"):
            p = Parseval(m, {"beta": 0.37, "scope": scope}, SGD)
            ref = reference_penalty(p.layers, 0.37)
            assert ref > 0
            assert math.isclose(val(p.regularizer()), ref, rel_tol=1e-5)
            s = p.state_summary()
            assert math.isclose(s["reg_loss"], ref, rel_tol=1e-5)
            norms = [float(gram_residual(l.effective_weight()).norm()) for l in p.layers]
            assert math.isclose(s["orth_residual"], sum(norms) / len(norms), rel_tol=1e-5)
            for li, n in zip(p.layer_indices, norms):
                assert math.isclose(s[f"orth_residual/L{li}"], n, rel_tol=1e-5)

    @pytest.mark.parametrize("mode", ["standard", "sin"])
    def test_gradient_is_4beta_RW_through_jacobian(self, mode):
        beta = 0.25
        m = make_model(mode, seed=5)
        p = Parseval(m, {"beta": beta}, SGD)
        m.zero_grad(set_to_none=True)
        p.regularizer().backward()
        for lin in m.hidden:
            w = lin.effective_weight().detach()
            grad_w = 4 * beta * gram_residual(w) @ w  # d/dW beta ||W W^T - I||_F^2  (out <= in)
            expected = lin.jacobian().detach() * grad_w  # chain rule through W = f(Theta)
            assert torch.allclose(lin.theta.grad, expected, rtol=1e-4, atol=1e-6)
            assert lin.theta_bias.grad is None  # biases are never regularised
        assert m.output.theta.grad is None  # out of scope

    @pytest.mark.parametrize("mode", ["standard", "sin", "tanh"])
    def test_steps_on_regularizer_alone_reduce_residual(self, mode):
        m = make_model(mode, seed=7)
        p = Parseval(m, {"beta": 1.0}, {"name": "sgd", "lr": 0.05})
        opt = p.build_optimizer()
        r0 = p.state_summary()["orth_residual"]
        assert r0 > 1.0  # Kaiming-uniform init: W W^T ~ gain^2 I = 2 I
        hist = []
        for s in range(40):
            opt.zero_grad(set_to_none=True)
            p.regularizer().backward()
            p.before_step(s)
            opt.step()
            p.after_step(s, None)
            hist.append(p.state_summary()["orth_residual"])
        assert hist[9] < 0.2 * r0 and hist[-1] < 1e-2 * r0
        for lin in m.hidden:
            assert torch.allclose(Parseval.gram(lin.effective_weight().detach()), torch.eye(lin.out_features), atol=1e-2)

    def test_config_validation_and_defaults(self):
        m = make_model()
        with pytest.raises(ValueError, match="scope"):
            Parseval(m, {"scope": "output"}, SGD)
        with pytest.raises(ValueError, match="beta"):
            Parseval(m, {"beta": -1.0}, SGD)
        with pytest.raises(ValueError, match="unknown"):
            Parseval(m, {"gamma": 1.0}, SGD)
        assert Parseval(m, {"beta": 0.0}, SGD).regularizer() is None
        p = Parseval(m, {}, SGD)
        assert p.beta == 1e-3 and p.scope == "hidden" and p.requires_features is False

    def test_dtype_cast_after_construction(self):
        m = make_model("sin", seed=2)
        p = Parseval(m, {"beta": 0.2}, SGD)
        m.double()
        reg = p.regularizer()
        assert reg.dtype == torch.float64
        assert math.isclose(val(reg), reference_penalty(p.layers, 0.2), rel_tol=1e-9)

    def test_registry_and_online_trainer(self):
        m = make_model("sin", seed=8)
        p = build_method({"name": "parseval", "beta": 1e-2}, m, SGD)
        assert isinstance(p, Parseval) and p.beta == 1e-2
        opt = p.build_optimizer()
        x, y = synthetic_batch(48, seed=9)
        task = Task(index=0, x_train=x.numpy().astype(np.float32), y_train=y.numpy().astype(np.int64))
        res = _train_on_task(m, p, opt, task, {"batch_size": 1, "epochs_per_task": 1}, 0)
        assert res["global_step"] == 48
        s = p.state_summary()
        assert set(s) == {"orth_residual", "orth_residual/L0", "orth_residual/L1", "reg_loss"}
        assert all(math.isfinite(v) for v in s.values())
        assert all(bool(torch.isfinite(q).all()) for q in m.parameters())


# ------------------------------------------------------------------------------------------------
# Scale-corrected update
# ------------------------------------------------------------------------------------------------
class TestScaleCorrectedSin:
    @pytest.mark.parametrize("mode,theta_scale", [("sin", "amplitude"), ("sin", "unit"), ("tanh", "amplitude")])
    def test_effective_step_matches_formula(self, mode, theta_scale):
        """One SGD step: dW = -lr * [A^2] * min(1, M c^2) * dL/dW element-wise (A^2 only for theta_scale='unit')."""
        M, lr = 10.0, 1e-5
        m = near_bound_model(mode, theta_scale)
        sc = ScaleCorrectedSin(m, {"max_scale": M}, {"name": "sgd", "lr": lr})
        opt = sc.build_optimizer()
        m.set_capture_effective(True)
        layers = m.linear_layers
        x, y = synthetic_batch(4, seed=2, dtype=torch.float64)
        w0 = [l.effective_weight().detach().clone() for l in layers]
        b0 = [l.effective_bias().detach().clone() for l in layers]
        th0 = [l.theta.detach().clone() for l in layers]
        loss = F.cross_entropy(m(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_w = [l.last_weight.grad.clone() for l in layers]  # dL/dW (effective weights)
        g_b = [l.last_bias.grad.clone() for l in layers]
        g_th = [l.theta.grad.clone() for l in layers]  # dL/dTheta before the correction
        g_thb = [l.theta_bias.grad.clone() for l in layers]
        c = [l.normalized_jacobian().detach().clone() for l in layers]
        cb = [l.bias_normalized_jacobian().detach().clone() for l in layers]
        # premise of the derivation: dL/dTheta = (dW/dTheta) dL/dW
        for i, l in enumerate(layers):
            assert torch.allclose(g_th[i], l.jacobian().detach() * g_w[i], rtol=1e-9, atol=1e-14)

        sc.before_step(0)
        # (a) exact Theta-space rescaling by s = min(1/c^2, M)
        n_cap = n_unc = 0
        for i, l in enumerate(layers):
            s = torch.clamp(1.0 / c[i] ** 2, max=M)
            sb = torch.clamp(1.0 / cb[i] ** 2, max=M)
            assert torch.allclose(l.theta.grad, s * g_th[i], rtol=1e-12, atol=0)
            assert torch.allclose(l.theta_bias.grad, sb * g_thb[i], rtol=1e-12, atol=0)
            capped = c[i] ** 2 < 1.0 / M
            assert bool(torch.all(s[capped] == M)) and bool(torch.all(s[~capped] < M))
            n_cap += int(capped.sum()) + int((cb[i] ** 2 < 1.0 / M).sum())
            n_unc += int((~capped).sum()) + int((cb[i] ** 2 >= 1.0 / M).sum())
        assert n_cap > 0 and n_unc > 0

        opt.step()
        # (b) effective-weight step, compared against the manual computation from last_weight.grad
        for i, l in enumerate(layers):
            A = float(l.amplitude)
            A2 = A * A if theta_scale == "unit" else 1.0
            dW = l.effective_weight().detach() - w0[i]
            dTh = l.theta.detach() - th0[i]
            pred = -lr * A2 * torch.clamp(M * c[i] ** 2, max=1.0) * g_w[i]
            assert torch.allclose(dW, pred, rtol=1e-3, atol=1e-12)
            # rigorous Taylor-remainder bound  |dW - c dTheta| <= max|f''| dTheta^2 / 2
            curv = A if theta_scale == "unit" else 1.0 / A
            assert bool(torch.all((dW - pred).abs() <= 1.01 * curv * dTh ** 2 / 2 + 1e-12))
            # capped elements (c^2 < 1/M): the step is M c^2 times the standard step, not the full one
            capped = c[i] ** 2 < 1.0 / M
            assert torch.allclose(dW[capped], (-lr * A2 * M * c[i] ** 2 * g_w[i])[capped], rtol=1e-3, atol=1e-12)
            assert bool(torch.all(dW[capped].abs() < (lr * A2 * g_w[i].abs())[capped] + 1e-12))
            # reparametrised bias: same formula with the bias Jacobian
            Ab = float(l.bias_amplitude)
            Ab2 = Ab * Ab if theta_scale == "unit" else 1.0
            dB = l.effective_bias().detach() - b0[i]
            pred_b = -lr * Ab2 * torch.clamp(M * cb[i] ** 2, max=1.0) * g_b[i]
            assert torch.allclose(dB, pred_b, rtol=1e-3, atol=1e-12)
        # diagnostics of the last step
        s_all = [torch.clamp(1.0 / t ** 2, max=M) for t in c + cb]
        n_all = sum(t.numel() for t in s_all)
        summ = sc.state_summary()
        assert math.isclose(summ["frac_capped"], n_cap / n_all, rel_tol=1e-12)
        assert math.isclose(summ["mean_scale"], sum(float(t.sum()) for t in s_all) / n_all, rel_tol=1e-9)
        assert 1.0 < summ["mean_scale"] < M

    def test_amplifies_step_relative_to_plain_sgd(self):
        """Element-wise ratio dW_corrected / dW_plain ~ min(1/c^2, M) (first order)."""
        M, lr = 10.0, 1e-5
        m_base = near_bound_model("sin", "amplitude", seed=4)
        m_sc = copy.deepcopy(m_base)
        base = build_method({"name": "baseline"}, m_base, {"name": "sgd", "lr": lr})
        sc = build_method({"name": "scale_corrected", "max_scale": M}, m_sc, {"name": "sgd", "lr": lr})
        x, y = synthetic_batch(4, seed=6, dtype=torch.float64)
        w0 = [l.effective_weight().detach().clone() for l in m_base.linear_layers]
        c = [l.normalized_jacobian().detach().clone() for l in m_base.linear_layers]
        for model, method in ((m_base, base), (m_sc, sc)):
            opt = method.build_optimizer()
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            method.before_step(0)
            opt.step()
        for i in range(len(w0)):
            d_base = m_base.linear_layers[i].effective_weight().detach() - w0[i]
            d_sc = m_sc.linear_layers[i].effective_weight().detach() - w0[i]
            mask = d_base.abs() > 1e-9
            assert int(mask.sum()) > 10
            ratio = d_sc[mask] / d_base[mask]
            assert torch.allclose(ratio, torch.clamp(1.0 / c[i] ** 2, max=M)[mask], rtol=1e-2)

    def test_standard_layers_and_standard_biases_untouched(self):
        m = near_bound_model("sin", output_mode="standard", compact_bias=False)
        sc = ScaleCorrectedSin(m, {"max_scale": 5.0}, SGD)
        assert sc.layer_indices == [0, 1] and all(l.bias_mode == "standard" for l in m.linear_layers)
        x, y = synthetic_batch(2, seed=4, dtype=torch.float64)
        F.cross_entropy(m(x), y).backward()
        g_out = m.output.theta.grad.clone()
        g_b = [l.theta_bias.grad.clone() for l in m.linear_layers]
        g_h = [l.theta.grad.clone() for l in m.hidden]
        sc.before_step(0)
        assert torch.equal(m.output.theta.grad, g_out)
        for l, g in zip(m.linear_layers, g_b):
            assert torch.equal(l.theta_bias.grad, g)
        for l, g in zip(m.hidden, g_h):
            assert not torch.equal(l.theta.grad, g)
            assert torch.allclose(l.theta.grad, torch.clamp(1.0 / l.normalized_jacobian() ** 2, max=5.0) * g)
        assert len(sc._last_scales) == 2  # only the two hidden weight matrices were treated

    def test_scale_helper_handles_exact_bound(self):
        sc = ScaleCorrectedSin(make_model("sin"), {"max_scale": 10.0}, SGD)
        c = torch.tensor([0.0, 0.1, 0.5, 1.0, -0.5])
        assert torch.allclose(sc.scale(c), torch.tensor([10.0, 10.0, 4.0, 1.0, 4.0]))
        assert bool(torch.isfinite(sc.scale(torch.zeros(3))).all())

    def test_max_scale_one_is_a_no_op(self):
        m = near_bound_model("sin")
        sc = ScaleCorrectedSin(m, {"max_scale": 1.0}, SGD)
        x, y = synthetic_batch(2, seed=5, dtype=torch.float64)
        F.cross_entropy(m(x), y).backward()
        g = [p.grad.clone() for p in m.parameters()]
        sc.before_step(0)
        for p, g0 in zip(m.parameters(), g):
            assert torch.equal(p.grad, g0)
        s = sc.state_summary()
        assert s["mean_scale"] == 1.0 and s["frac_capped"] == 1.0  # s == max_scale == 1 everywhere

    def test_validation_registry_and_summary_before_first_step(self):
        with pytest.raises(ValueError, match="bounded reparametrisations"):
            ScaleCorrectedSin(make_model("standard"), {}, SGD)
        with pytest.raises(ValueError, match="bounded reparametrisations"):
            build_method({"name": "scale_corrected"}, make_model("standard"), SGD)
        m = make_model("sin")
        for bad in (0.5, 0.0, -1.0, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="max_scale"):
                ScaleCorrectedSin(m, {"max_scale": bad}, SGD)
        with pytest.raises(ValueError, match="unknown"):
            ScaleCorrectedSin(m, {"scale": 3.0}, SGD)
        sc = build_method({"name": "scale_corrected"}, m, SGD)
        assert isinstance(sc, ScaleCorrectedSin) and sc.max_scale == 10.0 and sc.layer_indices == [0, 1, 2]
        assert sc.requires_features is False and sc.regularizer() is None
        assert ScaleCorrectedSin(make_model("tanh", output_mode="standard"), {}, SGD).layer_indices == [0, 1]
        assert sc.state_summary() == {"mean_scale": 1.0, "frac_capped": 0.0}
        sc.before_step(0)  # no gradients yet: nothing to do, no error
        assert sc.state_summary() == {"mean_scale": 1.0, "frac_capped": 0.0}

    def test_runs_inside_online_trainer(self):
        m = near_bound_model("sin", dtype=torch.float32)
        sc = build_method({"name": "scale_corrected", "max_scale": 10.0}, m, {"name": "sgd", "lr": 0.01})
        opt = sc.build_optimizer()
        x, y = synthetic_batch(64, seed=11)
        task = Task(index=0, x_train=x.numpy().astype(np.float32), y_train=y.numpy().astype(np.int64))
        res = _train_on_task(m, sc, opt, task, {"batch_size": 1, "epochs_per_task": 1}, 0)
        assert res["global_step"] == 64 and sc.n_steps == 64
        s = sc.state_summary()
        assert 1.0 <= s["mean_scale"] <= 10.0 and 0.0 < s["frac_capped"] < 1.0
        assert all(bool(torch.isfinite(p).all()) for p in m.parameters())
