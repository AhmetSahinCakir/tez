"""Tests for Normalize-and-Project (Lyle et al. 2024) and LayerNorm + weight decay (Lyle et al. 2025).

Fast, synthetic-data tests (torch.randn inputs); no dataset access.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from plasticity.data import Task
from plasticity.methods import build_method
from plasticity.methods.ln_wd import LayerNormWeightDecay
from plasticity.methods.nap import NormalizeAndProject
from plasticity.models import MLP
from plasticity.training.online import _train_on_task

torch.set_num_threads(1)  # as the trainer does (threads: 1); affine-LayerNorm backward is ~30x slower per tiny step on a thread pool

IN, HID, NC = 20, (16, 12), 5
SGD = {"name": "sgd", "lr": 0.05}
ADAM = {"name": "adam", "lr": 1e-2}
ADAMW = {"name": "adamw", "lr": 1e-2}


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def make_model(mode="standard", seed=0, layer_norm=True, ln_affine=False, **kw):
    g = torch.Generator().manual_seed(seed)
    return MLP(IN, HID, NC, reparam={"hidden": mode, "output": mode}, layer_norm=layer_norm, ln_affine=ln_affine, generator=g, **kw)


def synthetic_batch(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, IN, generator=g), torch.randint(0, NC, (n,), generator=g)


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


def fro(t):
    return float(t.detach().norm())


def nap(model, opt_cfg=SGD, **cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = NormalizeAndProject(model, cfg, opt_cfg, torch.Generator().manual_seed(1))
    opt = m.build_optimizer()
    return m, opt


def lnwd(model, opt_cfg=SGD, **cfg):
    m = LayerNormWeightDecay(model, cfg, opt_cfg, torch.Generator().manual_seed(1))
    opt = m.build_optimizer()
    return m, opt


# ================================================================================================
# NaP
# ================================================================================================
@pytest.mark.parametrize("opt_cfg", [SGD, ADAM])
def test_nap_keeps_hidden_matrix_norms_and_leaves_output_free(opt_cfg):
    model = make_model()
    init = [fro(lin.theta) for lin in model.hidden]
    init_out = fro(model.output.theta)
    th0 = [lin.theta.detach().clone() for lin in model.hidden]
    method, opt = nap(model, opt_cfg)
    run_steps(model, method, opt, 150)
    for lin, r in zip(model.hidden, init):
        assert fro(lin.theta) == pytest.approx(r, abs=1e-5)
    assert abs(fro(model.output.theta) - init_out) > 1e-4  # output layer is not projected by default
    for lin, t0 in zip(model.hidden, th0):  # the weights did move (projection is onto a sphere, not a point)
        assert fro(lin.theta - t0) > 1e-3
    assert method.n_projections == 150


def test_nap_norm_changes_without_projection_in_the_same_run():
    """Control: the same steps with the baseline change the hidden norms, so the equality above is NaP's doing."""
    model = make_model()
    init = [fro(lin.theta) for lin in model.hidden]
    method = build_method({"name": "baseline"}, model, SGD)
    run_steps(model, method, method.build_optimizer(), 150)
    assert any(abs(fro(lin.theta) - r) > 1e-4 for lin, r in zip(model.hidden, init))


def test_nap_raises_without_layer_norm():
    with pytest.raises(ValueError, match="layer_norm"):
        NormalizeAndProject(make_model(layer_norm=False), {}, SGD)


@pytest.mark.parametrize("mode", ["sin", "tanh"])
def test_nap_raises_for_non_standard_parametrisation(mode):
    with pytest.raises(ValueError, match="standard parametrisation"):
        NormalizeAndProject(make_model(mode=mode), {}, SGD)


def test_nap_rejects_unknown_scope():
    with pytest.raises(ValueError, match="scope"):
        NormalizeAndProject(make_model(), {"scope": "column"}, SGD)


def test_nap_warns_about_affine_layernorm():
    with pytest.warns(UserWarning, match="ln_affine"):
        NormalizeAndProject(make_model(ln_affine=True), {}, SGD)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NormalizeAndProject(make_model(ln_affine=False), {}, SGD)  # no warning


def test_nap_row_scope_keeps_every_row_norm():
    model = make_model()
    init_rows = [lin.theta.detach().norm(dim=1).clone() for lin in model.hidden]
    method, opt = nap(model, scope="row")
    run_steps(model, method, opt, 100)
    for lin, r in zip(model.hidden, init_rows):
        assert torch.allclose(lin.theta.detach().norm(dim=1), r, atol=1e-5)
    s = method.state_summary()
    assert np.isfinite(s["norm_growth"]) and s["norm_growth"] > 0


def test_nap_project_bias_keeps_joint_norm_and_lets_bias_move():
    model = make_model(bias_init="uniform")
    init_joint = [math.sqrt(fro(lin.theta) ** 2 + fro(lin.theta_bias) ** 2) for lin in model.hidden]
    init_w = [fro(lin.theta) for lin in model.hidden]
    method, opt = nap(model, project_bias=True)
    run_steps(model, method, opt, 100)
    for lin, r, rw in zip(model.hidden, init_joint, init_w):
        joint = math.sqrt(fro(lin.theta) ** 2 + fro(lin.theta_bias) ** 2)
        assert joint == pytest.approx(r, abs=1e-5)
        assert abs(fro(lin.theta) - rw) > 1e-6  # the matrix alone is not pinned: the bias takes part of the budget
    # row scope + bias: per-row joint norms preserved
    model = make_model(bias_init="uniform", seed=3)
    init_rows = [torch.sqrt(lin.theta.detach().pow(2).sum(1) + lin.theta_bias.detach() ** 2) for lin in model.hidden]
    method, opt = nap(model, scope="row", project_bias=True)
    run_steps(model, method, opt, 60)
    for lin, r in zip(model.hidden, init_rows):
        cur = torch.sqrt(lin.theta.detach().pow(2).sum(1) + lin.theta_bias.detach() ** 2)
        assert torch.allclose(cur, r, atol=1e-5)


def test_nap_include_output_projects_output_layer_too():
    model = make_model()
    r_out = fro(model.output.theta)
    method, opt = nap(model, include_output=True)
    assert method.layer_names[-1] == "output" and len(method.layers) == len(HID) + 1
    run_steps(model, method, opt, 80)
    assert fro(model.output.theta) == pytest.approx(r_out, abs=1e-5)
    assert "norm_growth/L2" in method.state_summary()


def test_nap_projection_is_a_symmetry_of_the_layernorm_network():
    """Scaling [W, b] jointly leaves LN(Wx+b) unchanged, so the projection does not change the function."""
    model = make_model(ln_affine=False, bias_init="uniform")
    method, opt = nap(model, project_bias=True)
    x, _ = synthetic_batch(16, seed=7)
    with torch.no_grad():
        for lin in model.hidden:  # inflate the parameters by different factors to make the projection non-trivial
            lin.theta.mul_(1.7)
            lin.theta_bias.mul_(1.7)
        before = model(x)
        method.project()
        after = model(x)
    assert torch.allclose(before, after, atol=1e-4, rtol=1e-4)
    assert method.state_summary()["norm_growth"] == pytest.approx(1.7, abs=1e-5)


def test_nap_norm_growth_matches_norm_before_projection():
    model = make_model()
    method, opt = nap(model, ADAM)
    x, y = synthetic_batch(1, seed=11)
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    pre = [fro(lin.theta) for lin in model.hidden]  # norms after the update, before the projection
    method.after_step(0, None)
    s = method.state_summary()
    expected = [p / float(r) for p, r in zip(pre, method.init_norms)]
    for i, e in enumerate(expected):
        assert s[f"norm_growth/L{i}"] == pytest.approx(e, rel=1e-6)
    assert s["norm_growth"] == pytest.approx(float(np.mean(expected)), rel=1e-6)
    assert s["n_projections"] == 1


def test_nap_leaves_layernorm_affine_params_and_optimizer_state_alone():
    model = make_model(ln_affine=True)
    method, opt = nap(model, ADAM)
    ln0 = [p.detach().clone() for n in model.norms for p in n.parameters()]
    run_steps(model, method, opt, 20)
    ln_now = [p.detach() for n in model.norms for p in n.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(ln0, ln_now))  # LN params are trained ...
    # ... but NaP never rescales them: a run with the projection disabled gives the same LN params up to the
    # (tiny) effect of the rescaled weights on the gradient, so just check they are finite and untouched by project()
    snap = [p.detach().clone() for n in model.norms for p in n.parameters()]
    method.project()
    assert all(torch.equal(a, b) for a, b in zip(snap, [p.detach() for n in model.norms for p in n.parameters()]))
    assert all(torch.isfinite(v).all() for st in opt.state.values() for v in st.values() if torch.is_tensor(v))


def test_nap_registry_and_train_on_task_integration():
    model = make_model()
    init = [fro(lin.theta) for lin in model.hidden]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        method = build_method({"name": "nap", "scope": "matrix"}, model, SGD, torch.Generator().manual_seed(0))
    assert isinstance(method, NormalizeAndProject) and method.requires_features is False
    opt = method.build_optimizer()
    rng = np.random.default_rng(0)
    task = Task(index=0, x_train=rng.standard_normal((120, IN)).astype(np.float32), y_train=rng.integers(0, NC, 120).astype(np.int64))
    res = _train_on_task(model, method, opt, task, {"batch_size": 1, "epochs_per_task": 1}, 0)
    assert res["global_step"] == 120 and 0.0 <= res["online_accuracy"] <= 1.0
    for lin, r in zip(model.hidden, init):
        assert fro(lin.theta) == pytest.approx(r, abs=1e-5)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert "nap" in repr(method).lower() or "scope=" in repr(method)


# ================================================================================================
# LN + WD
# ================================================================================================
def test_lnwd_raises_without_layer_norm():
    with pytest.raises(ValueError, match="layer_norm"):
        LayerNormWeightDecay(make_model(layer_norm=False), {}, SGD)


@pytest.mark.parametrize("mode", ["sin", "tanh"])
def test_lnwd_raises_for_non_standard_parametrisation(mode):
    with pytest.raises(ValueError, match="standard parametrisation"):
        LayerNormWeightDecay(make_model(mode=mode), {}, SGD)


def test_lnwd_rejects_conflicting_optimizer_weight_decay():
    with pytest.raises(ValueError, match="conflicts"):
        LayerNormWeightDecay(make_model(), {"weight_decay": 1e-3}, {**SGD, "weight_decay": 0.1})
    # equal values are accepted, zero is accepted
    LayerNormWeightDecay(make_model(), {"weight_decay": 0.1}, {**SGD, "weight_decay": 0.1})
    LayerNormWeightDecay(make_model(), {"weight_decay": 0.1}, {**SGD, "weight_decay": 0.0})
    with pytest.raises(ValueError, match="weight_decay"):
        LayerNormWeightDecay(make_model(), {"weight_decay": -1.0}, SGD)


@pytest.mark.parametrize("opt_cfg", [SGD, ADAMW])
def test_lnwd_shrinks_weight_norm_relative_to_no_decay(opt_cfg):
    model_wd = make_model(ln_affine=True, seed=5)
    model_nd = make_model(ln_affine=True, seed=5)
    assert all(torch.equal(a, b) for a, b in zip(model_wd.parameters(), model_nd.parameters()))
    m_wd, o_wd = lnwd(model_wd, opt_cfg, weight_decay=0.1)
    m_nd, o_nd = lnwd(model_nd, opt_cfg, weight_decay=0.0)
    run_steps(model_wd, m_wd, o_wd, 200, seed=2)
    run_steps(model_nd, m_nd, o_nd, 200, seed=2)
    assert m_wd.state_summary()["w_fro_norm"] < 0.95 * m_nd.state_summary()["w_fro_norm"]
    for a, b in zip(model_wd.linear_layers, model_nd.linear_layers):
        assert fro(a.theta) < fro(b.theta)
    assert m_wd.state_summary()["decayed_fro_norm"] >= m_wd.state_summary()["w_fro_norm"]


def test_lnwd_no_decay_matches_baseline_exactly():
    model_a, model_b = make_model(seed=9), make_model(seed=9)
    m_a, o_a = lnwd(model_a, SGD, weight_decay=0.0)
    m_b = build_method({"name": "baseline"}, model_b, SGD)
    o_b = m_b.build_optimizer()
    run_steps(model_a, m_a, o_a, 50)
    run_steps(model_b, m_b, o_b, 50)
    assert all(torch.allclose(a, b) for a, b in zip(model_a.parameters(), model_b.parameters()))


def test_lnwd_layernorm_params_excluded_from_decay_by_default():
    model = make_model(ln_affine=True)
    method, opt = lnwd(model, SGD, weight_decay=1e-3)
    ln_ids = {id(p) for n in model.norms for p in n.parameters()}
    assert len(ln_ids) == 2 * len(HID) and method.ln_params
    groups = {g["weight_decay"]: {id(p) for p in g["params"]} for g in opt.param_groups}
    assert set(groups) == {1e-3, 0.0}
    assert ln_ids <= groups[0.0] and not (ln_ids & groups[1e-3])
    assert {id(lin.theta) for lin in model.linear_layers} <= groups[1e-3]
    assert {id(lin.theta_bias) for lin in model.linear_layers} <= groups[1e-3]  # decay_bias=True by default
    # every trainable parameter is in exactly one group
    all_ids = {id(p) for p in model.parameters()}
    assert groups[1e-3] | groups[0.0] == all_ids and not (groups[1e-3] & groups[0.0])
    # functional check: LN gains are not pulled towards zero when the data gradient is absent
    x = torch.zeros(1, IN)
    with torch.no_grad():
        model.output.theta.zero_()  # logits independent of the hidden units -> zero gradient on LN params
    loss = F.cross_entropy(model(x), torch.tensor([0]))
    opt.zero_grad()
    loss.backward()
    gains_before = [n.weight.detach().clone() for n in model.norms]
    opt.step()
    assert all(torch.equal(g, n.weight.detach()) for g, n in zip(gains_before, model.norms))


def test_lnwd_decay_ln_and_decay_bias_flags():
    model = make_model(ln_affine=True)
    method, opt = lnwd(model, SGD, weight_decay=1e-3, decay_ln=True, decay_bias=False)
    groups = {g["weight_decay"]: {id(p) for p in g["params"]} for g in opt.param_groups}
    ln_ids = {id(p) for n in model.norms for p in n.parameters()}
    assert ln_ids <= groups[1e-3]
    assert {id(lin.theta_bias) for lin in model.linear_layers} <= groups[0.0]
    # ln_affine=False, decay everything: a single param group
    model = make_model(ln_affine=False)
    method, opt = lnwd(model, SGD, weight_decay=1e-3)
    assert len(opt.param_groups) == 1 and opt.param_groups[0]["weight_decay"] == 1e-3 and not method.non_decayed


def test_lnwd_sgd_is_coupled_l2_and_adamw_is_decoupled():
    lam, lr = 0.2, 0.1
    # SGD: theta' = theta - lr * (g + lam * theta)
    model = make_model(ln_affine=False)
    method, opt = lnwd(model, {"name": "sgd", "lr": lr}, weight_decay=lam)
    x, y = synthetic_batch(4, seed=3)
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    th, g = model.hidden[0].theta.detach().clone(), model.hidden[0].theta.grad.detach().clone()
    opt.step()
    assert torch.allclose(model.hidden[0].theta.detach(), th - lr * (g + lam * th), atol=1e-6)
    # AdamW: theta' = theta * (1 - lr * lam) - lr * adam_step, with |adam_step| = 1 element-wise at the first step
    model = make_model(ln_affine=False)
    method, opt = lnwd(model, {"name": "adamw", "lr": lr, "eps": 0.0}, weight_decay=lam)
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    th, g = model.hidden[0].theta.detach().clone(), model.hidden[0].theta.grad.detach().clone()
    opt.step()
    expected = th * (1 - lr * lam) - lr * torch.sign(g)
    assert torch.allclose(model.hidden[0].theta.detach(), expected, atol=1e-5)
    assert method.regularizer() is None


def test_lnwd_registry_defaults_and_train_on_task_integration():
    model = make_model(ln_affine=True)
    method = build_method({"name": "ln_wd"}, model, SGD, torch.Generator().manual_seed(0))
    assert isinstance(method, LayerNormWeightDecay)
    assert (method.weight_decay, method.decay_bias, method.decay_ln) == (1e-3, True, False)
    opt = method.build_optimizer()
    rng = np.random.default_rng(1)
    task = Task(index=0, x_train=rng.standard_normal((120, IN)).astype(np.float32), y_train=rng.integers(0, NC, 120).astype(np.int64))
    res = _train_on_task(model, method, opt, task, {"batch_size": 1, "epochs_per_task": 1}, 0)
    assert res["global_step"] == 120
    s = method.state_summary()
    assert set(s) == {"w_fro_norm", "decayed_fro_norm"} and all(np.isfinite(v) and v > 0 for v in s.values())
    assert "weight_decay=0.001" in repr(method)
