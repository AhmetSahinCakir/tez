"""Task streams for the continual-learning protocols.

All streams are deterministic functions of ``(seed, task_index)``: the same seed yields the same
permutations / class pairs / sample orders, so different methods run on *matched* task streams
(paired experimental design). Dev streams (hyper-parameter search) and final test streams must use
disjoint seed ranges (see ``configs/``).

Protocols
---------
``PermutedMNISTStream``  -- Online Permuted MNIST (Dohare et al., 2024). Each task applies one random
    pixel permutation to all MNIST training images; samples are presented once, in random order,
    one at a time (batch size 1 in the canonical protocol). No task boundary is signalled.
``CIFAR100BinaryStream`` -- Sequential binary classification (Chen & Zhang, 2026). Each task picks two
    distinct CIFAR-100 classes; images are grey-scaled and flattened to 1024-d vectors. Training uses
    the training images of the two classes; performance is the test accuracy on the two classes.
``StationaryDataset``    -- plain i.i.d. MNIST / CIFAR-100 (100-way) for the capacity control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

import numpy as np

from .cifar100 import load_cifar100, to_grayscale_vectors
from .mnist import load_mnist


@dataclass
class Task:
    """One task of a stream: training samples (presented in order) and optional held-out test set."""

    index: int
    x_train: np.ndarray  # (n, d) float32
    y_train: np.ndarray  # (n,) int64
    x_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])


def _task_rng(seed: int, task_index: int, salt: int = 0) -> np.random.Generator:
    """Independent RNG for each (seed, task) pair."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(task_index), int(salt)]))


class PermutedMNISTStream:
    """Online Permuted MNIST stream (Dohare et al., 2024).

    Parameters
    ----------
    n_tasks : number of permutations (800 in the canonical protocol).
    samples_per_task : images per task (60000 = full MNIST train set in the canonical protocol;
        smaller values sub-sample a random subset *per task* for pilot / dev runs).
    seed : stream seed (permutations + sample orders).
    normalize : ``'unit'`` scales pixels to [0,1]; ``'standard'`` additionally standardises with the
        global MNIST mean/std.
    """

    name = "pmnist"
    input_dim = 784
    n_classes = 10
    metric = "online_accuracy"

    def __init__(
        self,
        n_tasks: int = 800,
        samples_per_task: int = 60000,
        seed: int = 0,
        normalize: str = "unit",
        data_dir: Optional[str] = None,
    ) -> None:
        self.n_tasks = int(n_tasks)
        self.samples_per_task = int(samples_per_task)
        self.seed = int(seed)
        d = load_mnist(data_dir)
        x = d["x_train"].astype(np.float32) / 255.0
        if normalize == "standard":
            x = (x - 0.1307) / 0.3081
        elif normalize != "unit":
            raise ValueError(f"unknown normalize={normalize!r}")
        self._x = np.ascontiguousarray(x)
        self._y = d["y_train"].astype(np.int64)
        self._n = self._x.shape[0]
        if self.samples_per_task > self._n:
            raise ValueError("samples_per_task cannot exceed 60000")

    def permutation(self, task_index: int) -> np.ndarray:
        return _task_rng(self.seed, task_index, salt=1).permutation(self.input_dim)

    def task(self, task_index: int) -> Task:
        if not 0 <= task_index < self.n_tasks:
            raise IndexError(task_index)
        perm = self.permutation(task_index)
        order_rng = _task_rng(self.seed, task_index, salt=2)
        order = order_rng.permutation(self._n)[: self.samples_per_task]
        x = self._x[order][:, perm]
        y = self._y[order]
        return Task(index=task_index, x_train=x, y_train=y, info={"permutation_seed": self.seed})

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[Task]:
        for t in range(self.n_tasks):
            yield self.task(t)


class CIFAR100BinaryStream:
    """Sequential CIFAR-100 binary classification stream (Chen & Zhang, 2026).

    Parameters
    ----------
    n_tasks : number of binary tasks (3000 canonical).
    seed : stream seed (class pairs + sample orders).
    train_per_class : training images per class (500 = all).
    test_per_class : test images per class (100 = all).
    normalize : ``'unit'`` -> grey-scale in [0,1]; ``'standard'`` -> standardised with train mean/std.
    """

    name = "cifar100_binary"
    input_dim = 1024
    n_classes = 2
    metric = "test_accuracy"

    def __init__(
        self,
        n_tasks: int = 3000,
        seed: int = 0,
        train_per_class: int = 500,
        test_per_class: int = 100,
        normalize: str = "standard",
        data_dir: Optional[str] = None,
    ) -> None:
        self.n_tasks = int(n_tasks)
        self.seed = int(seed)
        self.train_per_class = int(train_per_class)
        self.test_per_class = int(test_per_class)
        d = load_cifar100(data_dir)
        xtr = to_grayscale_vectors(d["x_train"])
        xte = to_grayscale_vectors(d["x_test"])
        if normalize == "standard":
            mu, sd = xtr.mean(), xtr.std()
            xtr = (xtr - mu) / sd
            xte = (xte - mu) / sd
        elif normalize != "unit":
            raise ValueError(f"unknown normalize={normalize!r}")
        self._xtr, self._ytr = np.ascontiguousarray(xtr), d["y_train"].astype(np.int64)
        self._xte, self._yte = np.ascontiguousarray(xte), d["y_test"].astype(np.int64)
        self._tr_idx = [np.flatnonzero(self._ytr == c) for c in range(100)]
        self._te_idx = [np.flatnonzero(self._yte == c) for c in range(100)]

    def class_pair(self, task_index: int) -> np.ndarray:
        return _task_rng(self.seed, task_index, salt=11).choice(100, size=2, replace=False)

    def task(self, task_index: int) -> Task:
        if not 0 <= task_index < self.n_tasks:
            raise IndexError(task_index)
        c0, c1 = self.class_pair(task_index)
        rng = _task_rng(self.seed, task_index, salt=12)
        tr = np.concatenate(
            [
                rng.permutation(self._tr_idx[c0])[: self.train_per_class],
                rng.permutation(self._tr_idx[c1])[: self.train_per_class],
            ]
        )
        ytr = np.concatenate([np.zeros(self.train_per_class, np.int64), np.ones(self.train_per_class, np.int64)])
        order = rng.permutation(tr.shape[0])
        te = np.concatenate([self._te_idx[c0][: self.test_per_class], self._te_idx[c1][: self.test_per_class]])
        yte = np.concatenate([np.zeros(self.test_per_class, np.int64), np.ones(self.test_per_class, np.int64)])
        return Task(
            index=task_index,
            x_train=self._xtr[tr[order]],
            y_train=ytr[order],
            x_test=self._xte[te],
            y_test=yte,
            info={"classes": [int(c0), int(c1)]},
        )

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[Task]:
        for t in range(self.n_tasks):
            yield self.task(t)


class StationaryDataset:
    """i.i.d. control: MNIST (10-way) or CIFAR-100 grey-scale (100-way)."""

    metric = "test_accuracy"

    def __init__(self, dataset: str = "mnist", seed: int = 0, data_dir: Optional[str] = None, normalize: str = "unit"):
        self.seed = int(seed)
        if dataset == "mnist":
            d = load_mnist(data_dir)
            xtr = d["x_train"].astype(np.float32) / 255.0
            xte = d["x_test"].astype(np.float32) / 255.0
            if normalize == "standard":
                xtr, xte = (xtr - 0.1307) / 0.3081, (xte - 0.1307) / 0.3081
            self.input_dim, self.n_classes, self.name = 784, 10, "mnist_stationary"
        elif dataset == "cifar100":
            d = load_cifar100(data_dir)
            xtr, xte = to_grayscale_vectors(d["x_train"]), to_grayscale_vectors(d["x_test"])
            if normalize == "standard":
                mu, sd = xtr.mean(), xtr.std()
                xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
            self.input_dim, self.n_classes, self.name = 1024, 100, "cifar100_stationary"
        else:
            raise ValueError(dataset)
        self.x_train, self.y_train = np.ascontiguousarray(xtr), d["y_train"].astype(np.int64)
        self.x_test, self.y_test = np.ascontiguousarray(xte), d["y_test"].astype(np.int64)

    def epoch_order(self, epoch: int) -> np.ndarray:
        return _task_rng(self.seed, epoch, salt=21).permutation(self.x_train.shape[0])


def build_stream(cfg: Dict[str, Any], seed: int):
    """Factory from the ``stream`` section of a config."""
    kind = cfg["name"]
    kw = {k: v for k, v in cfg.items() if k != "name"}
    if kind == "pmnist":
        return PermutedMNISTStream(seed=seed, **kw)
    if kind == "cifar100_binary":
        return CIFAR100BinaryStream(seed=seed, **kw)
    if kind == "stationary":
        return StationaryDataset(seed=seed, **kw)
    raise ValueError(f"unknown stream {kind!r}")
