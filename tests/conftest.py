"""Shared pytest fixtures for the core test-suite.

* ``torch`` is pinned to a single thread for the whole session (deterministic, CPU-friendly).
* ``mnist_cache`` / ``cifar_cache`` are session fixtures that return the path of the processed
  cache and *skip* the requesting test when the cache is missing, so the suite degrades gracefully
  on a machine without the datasets.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from plasticity.utils import PROJECT_ROOT

MNIST_CACHE = PROJECT_ROOT / "data" / "processed" / "mnist.npz"
CIFAR_CACHE = PROJECT_ROOT / "data" / "processed" / "cifar100.npz"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: test that takes more than a few seconds (CIFAR-100 loading, longer runs)")


@pytest.fixture(scope="session", autouse=True)
def _single_thread_torch():
    """All tests run with one torch thread (the experiments are configured the same way)."""
    torch.set_num_threads(1)
    yield


@pytest.fixture(scope="session")
def mnist_cache() -> Path:
    if not MNIST_CACHE.exists():
        pytest.skip(f"processed MNIST cache missing: {MNIST_CACHE}")
    return MNIST_CACHE


@pytest.fixture(scope="session")
def cifar_cache() -> Path:
    if not CIFAR_CACHE.exists():
        pytest.skip(f"processed CIFAR-100 cache missing: {CIFAR_CACHE}")
    return CIFAR_CACHE
