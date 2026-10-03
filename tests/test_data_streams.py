"""Tests for the task streams (plasticity/data/streams.py).

Covers the determinism contract (streams are pure functions of ``(seed, task_index)``), the
permutation / label bookkeeping of Online Permuted MNIST, the class-pair / balance bookkeeping of
the CIFAR-100 binary stream, the stationary control dataset and the ``build_stream`` factory.
"""
from __future__ import annotations

import numpy as np
import pytest

from plasticity.data import (
    CIFAR100BinaryStream,
    PermutedMNISTStream,
    StationaryDataset,
    Task,
    build_stream,
    load_mnist,
)
from plasticity.data.cifar100 import to_grayscale_vectors
from plasticity.data.streams import _task_rng

N_TASKS = 3
SPT = 500  # samples per task for the pilot-sized streams used here


# ------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def pm0(mnist_cache) -> PermutedMNISTStream:
    return PermutedMNISTStream(n_tasks=N_TASKS, samples_per_task=SPT, seed=0)


@pytest.fixture(scope="module")
def pm0_again(mnist_cache) -> PermutedMNISTStream:
    """A second, independently constructed instance with the same seed."""
    return PermutedMNISTStream(n_tasks=N_TASKS, samples_per_task=SPT, seed=0)


@pytest.fixture(scope="module")
def pm1(mnist_cache) -> PermutedMNISTStream:
    return PermutedMNISTStream(n_tasks=N_TASKS, samples_per_task=SPT, seed=1)


@pytest.fixture(scope="module")
def raw_mnist(mnist_cache):
    return load_mnist()


# ------------------------------------------------------------------------------- Permuted MNIST
def test_pmnist_metadata(pm0):
    assert pm0.name == "pmnist" and pm0.metric == "online_accuracy"
    assert pm0.input_dim == 784 and pm0.n_classes == 10
    assert len(pm0) == N_TASKS
    tasks = list(pm0)
    assert [t.index for t in tasks] == list(range(N_TASKS))
    assert all(isinstance(t, Task) for t in tasks)


def test_pmnist_shapes_dtypes_and_range(pm0):
    t = pm0.task(0)
    assert t.x_train.shape == (SPT, 784) and t.x_train.dtype == np.float32
    assert t.y_train.shape == (SPT,) and t.y_train.dtype == np.int64
    assert t.n_train == SPT
    assert t.x_test is None and t.y_test is None  # online protocol: no held-out set
    assert float(t.x_train.min()) >= 0.0 and float(t.x_train.max()) <= 1.0
    assert float(t.x_train.max()) == 1.0  # some pixel is fully on -> scaling by 255 happened
    assert set(np.unique(t.y_train)) <= set(range(10))
    assert len(np.unique(t.y_train)) == 10  # 500 samples cover every digit
    assert t.info["permutation_seed"] == 0


def test_same_seed_gives_identical_permutation_and_sample_order(pm0, pm0_again):
    for k in range(N_TASKS):
        assert np.array_equal(pm0.permutation(k), pm0_again.permutation(k))
        a, b = pm0.task(k), pm0_again.task(k)
        assert np.array_equal(a.x_train, b.x_train)
        assert np.array_equal(a.y_train, b.y_train)
    # calling task() twice on the same instance is also stable (no hidden RNG state)
    a, b = pm0.task(1), pm0.task(1)
    assert np.array_equal(a.x_train, b.x_train) and np.array_equal(a.y_train, b.y_train)


def test_permutation_is_a_permutation(pm0):
    for k in range(N_TASKS):
        p = pm0.permutation(k)
        assert p.shape == (784,)
        assert np.array_equal(np.sort(p), np.arange(784))


def test_different_tasks_have_different_permutations_and_orders(pm0):
    p0, p1, p2 = (pm0.permutation(k) for k in range(3))
    assert not np.array_equal(p0, p1) and not np.array_equal(p1, p2) and not np.array_equal(p0, p2)
    # a random permutation of 784 pixels leaves almost no pixel in place
    assert (p0 == np.arange(784)).mean() < 0.05
    t0, t1 = pm0.task(0), pm0.task(1)
    assert not np.array_equal(t0.y_train, t1.y_train)  # different sample order


def test_seed_changes_task0(pm0, pm1):
    assert not np.array_equal(pm0.permutation(0), pm1.permutation(0))
    a, b = pm0.task(0), pm1.task(0)
    assert not np.array_equal(a.x_train, b.x_train)
    assert not np.array_equal(a.y_train, b.y_train)


def test_labels_consistent_with_permutation_round_trip(pm0, raw_mnist):
    """Un-permuting ``x_train`` must reproduce the cached raw uint8 images, with their labels."""
    for k in range(N_TASKS):
        t = pm0.task(k)
        inv = np.argsort(pm0.permutation(k))
        x_unperm = t.x_train[:, inv]
        assert np.array_equal(x_unperm[:, pm0.permutation(k)], t.x_train)  # inverse really inverts
        raw8 = np.rint(x_unperm * 255.0).astype(np.uint8)
        # (i) the exact sample order is the documented function of (seed, task)
        order = _task_rng(pm0.seed, k, salt=2).permutation(60000)[:SPT]
        assert np.array_equal(raw8, raw_mnist["x_train"][order])
        assert np.array_equal(t.y_train, raw_mnist["y_train"][order])
    # (ii) API-only check: every un-permuted image is an MNIST training image carrying that label
    lookup = {}
    for row, lab in zip(raw_mnist["x_train"], raw_mnist["y_train"]):
        lookup.setdefault(row.tobytes(), set()).add(int(lab))
    t = pm0.task(0)
    inv = np.argsort(pm0.permutation(0))
    raw8 = np.rint(t.x_train[:, inv] * 255.0).astype(np.uint8)
    for row, lab in zip(raw8, t.y_train):
        assert int(lab) in lookup[row.tobytes()]


def test_pmnist_standard_normalisation(pm0, mnist_cache):
    s = PermutedMNISTStream(n_tasks=1, samples_per_task=SPT, seed=0, normalize="standard")
    a, b = pm0.task(0), s.task(0)
    np.testing.assert_allclose(b.x_train, (a.x_train - 0.1307) / 0.3081, atol=1e-5)
    assert np.array_equal(a.y_train, b.y_train)


def test_pmnist_argument_validation(pm0, mnist_cache):
    with pytest.raises(ValueError):
        PermutedMNISTStream(n_tasks=1, samples_per_task=10, seed=0, normalize="bogus")
    with pytest.raises(ValueError):
        PermutedMNISTStream(n_tasks=1, samples_per_task=60001, seed=0)
    with pytest.raises(IndexError):
        pm0.task(N_TASKS)
    with pytest.raises(IndexError):
        pm0.task(-1)


# ------------------------------------------------------------------------------- CIFAR-100 binary
TR, TE = 50, 20  # per-class subsets keep the stream cheap


@pytest.fixture(scope="module")
def cf0(cifar_cache) -> CIFAR100BinaryStream:
    return CIFAR100BinaryStream(n_tasks=5, seed=0, train_per_class=TR, test_per_class=TE)


def test_cifar_metadata_and_shapes(cf0):
    assert cf0.name == "cifar100_binary" and cf0.metric == "test_accuracy"
    assert cf0.input_dim == 1024 and cf0.n_classes == 2 and len(cf0) == 5
    t = cf0.task(0)
    assert t.x_train.shape == (2 * TR, 1024) and t.x_train.dtype == np.float32
    assert t.y_train.shape == (2 * TR,) and t.y_train.dtype == np.int64
    assert t.x_test.shape == (2 * TE, 1024) and t.y_test.shape == (2 * TE,)
    assert np.isfinite(t.x_train).all() and np.isfinite(t.x_test).all()
    with pytest.raises(IndexError):
        cf0.task(5)


def test_cifar_class_pairs_distinct_and_vary(cf0):
    pairs = [tuple(int(c) for c in cf0.class_pair(k)) for k in range(5)]
    for c0, c1 in pairs:
        assert c0 != c1 and 0 <= c0 < 100 and 0 <= c1 < 100
    assert len(set(pairs)) > 1
    assert cf0.task(0).info["classes"] == list(pairs[0])
    # the class pair is a pure function of (seed, task): another seed gives other pairs
    other = CIFAR100BinaryStream.__new__(CIFAR100BinaryStream)  # no data loading needed for class_pair
    other.seed = 1
    assert any(tuple(int(c) for c in other.class_pair(k)) != pairs[k] for k in range(5))


def test_cifar_labels_balanced_and_consistent(cf0):
    for k in range(3):
        t = cf0.task(k)
        assert int(t.y_train.sum()) == TR and set(np.unique(t.y_train)) == {0, 1}
        assert int(t.y_test.sum()) == TE and set(np.unique(t.y_test)) == {0, 1}
        c0, c1 = t.info["classes"]
        # every training row labelled 0 (resp. 1) is an image of class c0 (resp. c1)
        for lab, c in ((0, c0), (1, c1)):
            pool = cf0._xtr[cf0._tr_idx[c]]
            rows = t.x_train[t.y_train == lab]
            assert (rows[:, None, :] == pool[None, :, :]).all(-1).any(-1).all()
        # the test rows are the first TE images of each class, in order
        assert np.array_equal(t.x_test[:TE], cf0._xte[cf0._te_idx[c0][:TE]])
        assert np.array_equal(t.x_test[TE:], cf0._xte[cf0._te_idx[c1][:TE]])
        # training order is shuffled (labels are not sorted)
        assert not np.array_equal(t.y_train, np.sort(t.y_train))


def test_cifar_standardisation_stats_sane(cf0):
    assert abs(float(cf0._xtr.mean())) < 1e-3 and abs(float(cf0._xtr.std()) - 1.0) < 1e-3
    assert abs(float(cf0._xte.mean())) < 0.1 and abs(float(cf0._xte.std()) - 1.0) < 0.1
    t = cf0.task(0)
    assert float(np.abs(t.x_train).max()) < 10.0


def test_grayscale_vectors_formula():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(7, 32, 32, 3), dtype=np.uint8)
    g = to_grayscale_vectors(x)
    assert g.shape == (7, 1024) and g.dtype == np.float32
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0
    xf = x.astype(np.float32) / 255.0
    ref = (0.299 * xf[..., 0] + 0.587 * xf[..., 1] + 0.114 * xf[..., 2]).reshape(7, -1)
    np.testing.assert_allclose(g, ref, atol=1e-6)
    white = np.full((1, 32, 32, 3), 255, np.uint8)
    np.testing.assert_allclose(to_grayscale_vectors(white), 1.0, atol=1e-6)


@pytest.mark.slow
def test_cifar_determinism_across_instances(cf0, cifar_cache):
    other = CIFAR100BinaryStream(n_tasks=5, seed=0, train_per_class=TR, test_per_class=TE)
    for k in (0, 4):
        a, b = cf0.task(k), other.task(k)
        assert a.info == b.info
        assert np.array_equal(a.x_train, b.x_train) and np.array_equal(a.y_train, b.y_train)
        assert np.array_equal(a.x_test, b.x_test) and np.array_equal(a.y_test, b.y_test)
    a, b = cf0.task(0), cf0.task(0)
    assert np.array_equal(a.x_train, b.x_train) and np.array_equal(a.y_train, b.y_train)


# ------------------------------------------------------------------------------- stationary control
def test_stationary_mnist_shapes(mnist_cache):
    ds = StationaryDataset(dataset="mnist", seed=0)
    assert ds.name == "mnist_stationary" and ds.metric == "test_accuracy"
    assert ds.input_dim == 784 and ds.n_classes == 10
    assert ds.x_train.shape == (60000, 784) and ds.x_train.dtype == np.float32
    assert ds.y_train.shape == (60000,) and ds.y_train.dtype == np.int64
    assert ds.x_test.shape == (10000, 784) and ds.y_test.shape == (10000,)
    assert float(ds.x_train.min()) >= 0.0 and float(ds.x_train.max()) <= 1.0
    assert set(np.unique(ds.y_test)) == set(range(10))
    o0, o1 = ds.epoch_order(0), ds.epoch_order(1)
    assert np.array_equal(np.sort(o0), np.arange(60000))
    assert not np.array_equal(o0, o1)
    assert np.array_equal(o0, StationaryDataset(dataset="mnist", seed=0).epoch_order(0))
    assert not np.array_equal(o0, StationaryDataset(dataset="mnist", seed=1).epoch_order(0))


def test_stationary_unknown_dataset_raises():
    with pytest.raises(ValueError):
        StationaryDataset(dataset="svhn")


# ------------------------------------------------------------------------------- factory
def test_build_stream_factory(mnist_cache):
    s = build_stream({"name": "pmnist", "n_tasks": 2, "samples_per_task": 10}, seed=3)
    assert isinstance(s, PermutedMNISTStream) and s.seed == 3 and len(s) == 2 and s.samples_per_task == 10
    d = build_stream({"name": "stationary", "dataset": "mnist"}, seed=4)
    assert isinstance(d, StationaryDataset) and d.seed == 4


def test_build_stream_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown stream"):
        build_stream({"name": "imagenet"}, seed=0)
    with pytest.raises(KeyError):
        build_stream({}, seed=0)
