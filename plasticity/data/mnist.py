"""MNIST loader.

Reads the raw IDX files (downloaded from the PyTorch/ossci S3 mirror) and caches a compact
``data/processed/mnist.npz`` with uint8 arrays. No torchvision dependency so the loader also works
in analysis-only environments.
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Dict

import numpy as np

from ..utils import PROJECT_ROOT

RAW_FILES = {
    "x_train": "mnist_train-images-idx3-ubyte.gz",
    "y_train": "mnist_train-labels-idx1-ubyte.gz",
    "x_test": "mnist_t10k-images-idx3-ubyte.gz",
    "y_test": "mnist_t10k-labels-idx1-ubyte.gz",
}
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(shape)


def load_mnist(data_dir: str | Path | None = None, flatten: bool = True) -> Dict[str, np.ndarray]:
    """Return ``{'x_train': (60000, 784) uint8, 'y_train': (60000,) int64, 'x_test', 'y_test'}``."""
    data_dir = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data"
    cache = data_dir / "processed" / "mnist.npz"
    if cache.exists():
        d = np.load(cache)
        out = {k: d[k] for k in d.files}
    else:
        raw = data_dir / "raw"
        missing = [v for v in RAW_FILES.values() if not (raw / v).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing raw MNIST files {missing} in {raw}. Download them from {MIRROR} "
                "(see scripts/download_data.sh)."
            )
        out = {k: _read_idx(raw / v) for k, v in RAW_FILES.items()}
        out["x_train"] = out["x_train"].reshape(-1, 784)
        out["x_test"] = out["x_test"].reshape(-1, 784)
        out["y_train"] = out["y_train"].astype(np.int64)
        out["y_test"] = out["y_test"].astype(np.int64)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
    if not flatten:
        out["x_train"] = out["x_train"].reshape(-1, 28, 28)
        out["x_test"] = out["x_test"].reshape(-1, 28, 28)
    return out
