"""CIFAR-100 loader.

The canonical ``cifar-100-python.tar.gz`` host (cs.toronto.edu) is not reachable from every
environment, so this loader also accepts the fast.ai mirror ``cifar100.tgz`` (PNG files organised
as ``cifar100/{train,test}/<superclass>/<class>/*.png``). Either source is converted once into
``data/processed/cifar100.npz`` holding uint8 images ``(N, 32, 32, 3)`` and fine labels in
``[0, 100)`` (classes sorted alphabetically by fine-class name, see ``class_names``).
"""
from __future__ import annotations

import io
import pickle
import tarfile
from pathlib import Path
from typing import Dict

import numpy as np

from ..utils import PROJECT_ROOT

FASTAI_MIRROR = "https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz"


def _convert_fastai_tgz(path: Path) -> Dict[str, np.ndarray]:
    from PIL import Image

    imgs: Dict[str, list] = {"train": [], "test": []}
    labels: Dict[str, list] = {"train": [], "test": []}
    names: Dict[str, list] = {"train": [], "test": []}
    with tarfile.open(path, "r:gz") as tar:
        # stream in archive order: random access into a gzip tar re-decompresses from the start
        for m in tar:
            if not (m.isfile() and m.name.endswith(".png")):
                continue
            parts = m.name.split("/")
            # cifar100/<split>/<superclass>/<class>/<file>.png
            split, fine = parts[1], parts[3]
            f = tar.extractfile(m)
            assert f is not None
            img = np.asarray(Image.open(io.BytesIO(f.read())).convert("RGB"), dtype=np.uint8)
            imgs[split].append(img)
            names[split].append(fine)
    class_names = sorted(set(names["train"]))
    idx = {n: i for i, n in enumerate(class_names)}
    for split in ("train", "test"):
        labels[split] = np.array([idx[n] for n in names[split]], dtype=np.int64)
    return {
        "x_train": np.stack(imgs["train"]),
        "y_train": labels["train"],
        "x_test": np.stack(imgs["test"]),
        "y_test": labels["test"],
        "class_names": np.array(class_names),
    }


def _convert_python_tgz(path: Path) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with tarfile.open(path, "r:gz") as tar:
        for split in ("train", "test"):
            f = tar.extractfile(f"cifar-100-python/{split}")
            assert f is not None
            d = pickle.load(f, encoding="latin1")
            x = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
            out[f"x_{split}"] = x
            out[f"y_{split}"] = np.array(d["fine_labels"], dtype=np.int64)
        f = tar.extractfile("cifar-100-python/meta")
        assert f is not None
        meta = pickle.load(f, encoding="latin1")
        out["class_names"] = np.array(meta["fine_label_names"])
    return out


def load_cifar100(data_dir: str | Path | None = None) -> Dict[str, np.ndarray]:
    """Return ``{'x_train': (50000,32,32,3) uint8, 'y_train', 'x_test': (10000,32,32,3), 'y_test', 'class_names'}``."""
    data_dir = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data"
    cache = data_dir / "processed" / "cifar100.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}
    raw = data_dir / "raw"
    if (raw / "cifar-100-python.tar.gz").exists():
        out = _convert_python_tgz(raw / "cifar-100-python.tar.gz")
    elif (raw / "cifar100.tgz").exists():
        out = _convert_fastai_tgz(raw / "cifar100.tgz")
    else:
        raise FileNotFoundError(
            f"No CIFAR-100 archive in {raw}. Download cifar100.tgz from {FASTAI_MIRROR} "
            "or cifar-100-python.tar.gz from https://www.cs.toronto.edu/~kriz/ (see scripts/download_data.sh)."
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def to_grayscale_vectors(x: np.ndarray) -> np.ndarray:
    """``(N,32,32,3) uint8`` -> ``(N,1024) float32`` luminance in [0,1] (Chen & Zhang 2026 protocol)."""
    xf = x.astype(np.float32) / 255.0
    g = 0.299 * xf[..., 0] + 0.587 * xf[..., 1] + 0.114 * xf[..., 2]
    return g.reshape(x.shape[0], -1).astype(np.float32)
