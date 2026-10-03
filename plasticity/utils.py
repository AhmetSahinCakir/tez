"""Small utilities: seeding, config merging, dotted overrides, JSON helpers."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import yaml

try:  # torch is optional for the pure-numpy helpers (data conversion, analysis)
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def deep_update(base: Dict[str, Any], upd: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``upd`` into a *copy* of ``base``."""
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_scalar(text: str) -> Any:
    """Parse a CLI override value with YAML semantics (``1e-3`` -> float, ``true`` -> bool, ...)."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``cfg['a']['b']['c'] = value`` for ``dotted == 'a.b.c'`` creating dicts as needed."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply ``key.subkey=value`` overrides (values parsed as YAML scalars)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, val = item.split("=", 1)
        set_dotted(cfg, key.strip(), _parse_scalar(val.strip()))
    return cfg


def load_config(path: str | os.PathLike, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    """Load a YAML config. Supports a top-level ``base: other.yaml`` include (relative to the file)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        base_cfg = load_config(path.parent / base)
        cfg = deep_update(base_cfg, cfg)
    return apply_overrides(cfg, overrides)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if torch is not None and isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        return super().default(o)


def dump_json(obj: Any, path: str | os.PathLike, indent: int | None = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=indent)


def append_jsonl(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, cls=NumpyEncoder) + "\n")


def read_jsonl(path: str | os.PathLike) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def flatten_dict(d: Mapping[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten_dict(v, key, sep))
        else:
            out[key] = v
    return out
