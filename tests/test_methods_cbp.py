"""Tests for Continual Backpropagation (plasticity/methods/continual_backprop.py)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from plasticity.data import Task
from plasticity.methods import build_method
from plasticity.methods.continual_backprop import ContinualBackprop
from plasticity.models import MLP
from plasticity.training.online import _train_on_task

D, HIDDEN, C = 8, (6, 5), 3


def make_model(hidden_mode="standard", output_mode="standard", hidden=HIDDEN, seed=0, **kw) -> MLP:
    g = torch.Generator().manual_seed(seed)
    return MLP(D, hidden, C, reparam={"hidden": hidden_mode, "output": output_mode}, generator=g, **kw)


def make_method(model, opt_cfg=None, seed=1, **cfg) -> ContinualBackprop:
    m = ContinualBackprop(model, cfg, opt_cfg or {"name": "sgd", "lr": 0.01}, torch.Generator().manual_seed(seed))
    m.build_optimizer()
    return m


def features(model, B=1, seed=0):
    x = torch.randn(B, model.input_dim, generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        _, feats = model(x, return_features=True)
    return feats


def params(model):
    return [p.detach().clone() for p in model.parameters()]


# ------------------------------------------------------------------------------------------------
def test_full_replacement_every_step_with_rho_one():
    model = make_model()
    method = make_method(model, replacement_rate=1.0, maturity_threshold=0)
    th_before = [lin.theta.detach().clone() for lin in model.hidden]
    method.after_step(0, features(model))
    s = method.state_summary()
    assert (s["units_replaced/L0"], s["units_replaced/L1"], s["units_replaced_total"]) == (6, 5, 11)
    for l in range(2):  # every incoming row re-drawn
        assert bool((model.hidden[l].theta != th_before[l]).any(dim=1).all())
        assert torch.all(method.ages[l] == 0)
        assert torch.all(method.utility[l] == 0) and torch.all(method.mean_act[l] == 0)
    assert torch.all(model.output.theta == 0)  # outgoing columns of the last hidden layer
    assert all(c < 1.0 for c in method.to_replace)
    method.after_step(1, features(model, seed=1))
    assert method.state_summary()["units_replaced_total"] == 22


def test_outgoing_columns_zeroed_and_other_layers_untouched():
    model = make_model()
    method = make_method(model, replacement_rate=1.0, maturity_threshold=1000)
    method.ages[0][:] = 5000  # only layer-0 units are mature
    th0, th1, b1, out = (model.hidden[0].theta.clone(), model.hidden[1].theta.clone(),
                         model.hidden[1].theta_bias.clone(), model.output.theta.clone())
    method.after_step(0, features(model))
    assert torch.all(model.hidden[1].theta == 0)  # all 6 input columns of layer 1 zeroed
    assert torch.equal(model.hidden[1].theta_bias, b1) and torch.equal(model.output.theta, out)
    assert bool((model.hidden[0].theta != th0).any(dim=1).all())
    assert torch.all(method.ages[0] == 0) and torch.all(method.ages[1] == 1)
    s = method.state_summary()
    assert s["units_replaced/L0"] == 6 and s["units_replaced/L1"] == 0


def test_partial_replacement_picks_lowest_utility_and_resets_ages():
    model = make_model()
    method = make_method(model, replacement_rate=0.5, maturity_threshold=0, decay_rate=0.99)
    method.ages[0][:] = 10
    method.ages[1][:] = 10
    method.utility[0] = torch.tensor([5.0, 1.0, 4.0, 0.0, 3.0, 2.0]) * 100  # ranking dominated by the prior trace
    method.after_step(0, features(model))
    replaced = sorted((method.ages[0] == 0).nonzero().flatten().tolist())
    assert replaced == [1, 3, 5]  # 6 * 0.5 = 3 lowest-utility units
    assert torch.all(method.ages[0][[0, 2, 4]] == 11)
    assert torch.all(method.utility[0][[1, 3, 5]] == 0) and torch.all(method.mean_act[0][[1, 3, 5]] == 0)
    kept_rows = method.ages[1] != 0  # rows of layer 1 that were not re-drawn in the same step
    assert torch.all(model.hidden[1].theta[kept_rows][:, [1, 3, 5]] == 0)
    assert int((~kept_rows).sum()) == 2  # 5 * 0.5 -> floor(2.5) = 2


def test_replacement_counter_accumulates_fractions():
    model = make_model()
    method = make_method(model, replacement_rate=0.25, maturity_threshold=0)
    expect = {0: [1, 3, 4, 6, 7, 9], 1: [1, 2, 3, 5, 6, 7]}  # 6*0.25=1.5 and 5*0.25=1.25 per step
    for t in range(6):
        method.after_step(t, features(model, seed=t))
        s = method.state_summary()
        assert (s["units_replaced/L0"], s["units_replaced/L1"]) == (expect[0][t], expect[1][t])
        assert all(0.0 <= c < 1.0 for c in method.to_replace)


def test_zero_replacement_rate_changes_nothing():
    model = make_model()
    method = make_method(model, replacement_rate=0.0, maturity_threshold=0)
    before = params(model)
    for t in range(20):
        method.after_step(t, features(model, B=2, seed=t))
    for p, q in zip(model.parameters(), before):
        assert torch.equal(p.detach(), q)
    s = method.state_summary()
    assert s["units_replaced_total"] == 0 and all(torch.all(a == 20) for a in method.ages)
    assert np.isfinite(s["mean_utility"]) and s["mean_utility"] >= 0


@pytest.mark.parametrize("mode", ["standard", "sin"])
def test_utility_matches_reference_formula(mode):
    eta = 0.9
    model = make_model(mode, mode)
    method = make_method(model, replacement_rate=0.0, decay_rate=eta)
    W = [lin.effective_weight().detach() for lin in model.linear_layers]
    f_ref = [torch.zeros(n) for n in HIDDEN]
    u_ref = [torch.zeros(n) for n in HIDDEN]
    for t in range(3):
        feats = features(model, B=3, seed=t)
        for l in range(2):
            h = feats[l]
            f_ref[l] = eta * f_ref[l] + (1 - eta) * h.mean(0)
            f_hat = f_ref[l] / (1 - eta ** (t + 1))
            new_u = (h - f_hat).abs().mean(0) * W[l + 1].abs().sum(0) / W[l].abs().sum(1)
            u_ref[l] = eta * u_ref[l] + (1 - eta) * new_u
        method.after_step(t, feats)
    for l in range(2):
        assert torch.allclose(method.utility[l], u_ref[l], atol=1e-6)
        assert torch.allclose(method.corrected_utility(l), u_ref[l] / (1 - eta ** 3), atol=1e-6)
        assert torch.all(torch.isfinite(method.utility[l])) and torch.all(method.utility[l] >= 0)


def test_utility_finite_nonnegative_with_dead_units_and_zero_rows():
    model = make_model()
    with torch.no_grad():
        model.hidden[0].theta[:2] = 0.0  # zero incoming rows -> guarded division
    method = make_method(model, replacement_rate=0.0)
    feats = features(model, B=4)
    feats[1] = torch.zeros_like(feats[1])  # dead layer
    for t in range(5):
        method.after_step(t, feats)
    for l in range(2):
        assert torch.all(torch.isfinite(method.utility[l])) and torch.all(method.utility[l] >= 0)
        assert torch.all(torch.isfinite(method.corrected_utility(l)))
    assert torch.all(method.utility[1] == 0)


def test_sin_reparam_effective_weight_path():
    model = make_model("sin", "sin")
    method = make_method(model, replacement_rate=1.0, maturity_threshold=0)
    w_before = [lin.effective_weight().detach().clone() for lin in model.hidden]
    method.after_step(0, features(model))
    for l, lin in enumerate(model.hidden):
        w = lin.effective_weight().detach()
        assert torch.all(torch.isfinite(lin.theta)) and torch.all(torch.isfinite(lin.theta_bias))
        assert bool((w != w_before[l]).any(dim=1).all())
        assert torch.all(w.abs() <= lin.weight_init_bound + 1e-6)  # re-drawn from the Kaiming-uniform d_l
        assert torch.all(lin.angle().abs() <= torch.pi / 2 + 1e-6)  # principal branch
        assert torch.allclose(lin.effective_bias(), torch.zeros(lin.out_features))  # bias_init = zeros
    assert torch.all(model.output.effective_weight() == 0)
    assert method.state_summary()["units_replaced_total"] == 11


def test_adam_state_rows_and_columns_reset():
    model = make_model()
    method = make_method(model, {"name": "adam", "lr": 1e-3}, replacement_rate=1.0, maturity_threshold=1000)
    opt = method.optimizer
    x = torch.randn(4, D, generator=torch.Generator().manual_seed(3))
    y = torch.tensor([0, 1, 2, 1])
    logits, feats = model(x, return_features=True)
    opt.zero_grad()
    F.cross_entropy(logits, y).backward()
    opt.step()
    st0 = opt.state[model.hidden[0].theta]
    assert float(st0["exp_avg"].abs().sum()) > 0
    method.ages[0][:] = 5000  # replace every unit of layer 0 only
    method.after_step(0, feats)
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.all(st0[key] == 0)
        assert torch.all(opt.state[model.hidden[0].theta_bias][key] == 0)
        assert torch.all(opt.state[model.hidden[1].theta][key] == 0)  # all 6 input columns
        assert float(opt.state[model.output.theta][key].abs().sum()) > 0  # untouched
    assert int(st0["step"]) == 1


def test_bias_compensation_option():
    model = make_model()
    method = make_method(model, replacement_rate=1.0, maturity_threshold=1000, compensate_bias=True)
    method.ages[0][:] = 5000
    f0 = torch.linspace(0.1, 0.6, 6)
    method.mean_act[0] = f0.clone()
    b1 = model.hidden[1].effective_bias().detach().clone()
    W1 = model.hidden[1].effective_weight().detach().clone()
    feats = features(model)
    method.after_step(0, feats)
    f_hat = (0.99 * f0 + 0.01 * feats[0].mean(0)) / (1 - 0.99 ** 5001)
    assert torch.allclose(model.hidden[1].effective_bias(), b1 + (W1 * f_hat).sum(1), atol=1e-6)


def test_reproducible_with_generator():
    feats = features(make_model())
    outs = []
    for seed in (7, 7, 8):
        model = make_model()
        make_method(model, seed=seed, replacement_rate=1.0, maturity_threshold=0).after_step(0, feats)
        outs.append(params(model))
    assert all(torch.equal(p, q) for p, q in zip(outs[0], outs[1]))
    assert not all(torch.equal(p, q) for p, q in zip(outs[0], outs[2]))


def test_runs_inside_training_loop_via_registry():
    model = make_model(hidden=(10, 10))
    method = build_method({"name": "continual_backprop", "replacement_rate": 0.1, "maturity_threshold": 5},
                          model, {"name": "sgd", "lr": 0.01}, torch.Generator().manual_seed(0))
    assert isinstance(method, ContinualBackprop) and method.requires_features
    opt = method.build_optimizer()
    rng = np.random.default_rng(0)
    task = Task(index=0, x_train=rng.standard_normal((200, D)).astype(np.float32), y_train=rng.integers(0, C, 200).astype(np.int64))
    res = _train_on_task(model, method, opt, task, {"batch_size": 1, "epochs_per_task": 1}, 0)
    assert res["global_step"] == 200
    s = method.state_summary()
    assert s["units_replaced_total"] > 0 and np.isfinite(s["mean_utility"])
    assert all(torch.all(torch.isfinite(p)) for p in model.parameters())


def test_config_validation():
    model = make_model()
    with pytest.raises(ValueError):
        make_method(model, utility="random")
    with pytest.raises(ValueError):
        make_method(model, replacement_rate=2.0)
    with pytest.raises(ValueError):
        make_method(model, decay_rate=1.0)
    with pytest.raises(ValueError):
        make_method(model, bogus=1)
    with pytest.raises(ValueError):
        make_method(model).after_step(0, None)
    assert make_method(model, utility="adaptable_contribution").utility_type == "adaptable_contribution"
