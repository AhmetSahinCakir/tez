"""Method registry.

Add a method by writing ``plasticity/methods/<file>.py`` with a :class:`~plasticity.methods.base.Method`
subclass and registering it in ``_REGISTRY`` below (module path, class name).
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import torch

from .base import Method, build_optimizer

_REGISTRY: Dict[str, tuple] = {
    "baseline": (".baseline", "Baseline"),
    "weight_clipping": (".weight_clipping", "WeightClipping"),
    "l2_init": (".l2_init", "L2Init"),
    "continual_backprop": (".continual_backprop", "ContinualBackprop"),
    "nap": (".nap", "NormalizeAndProject"),
    "ln_wd": (".ln_wd", "LayerNormWeightDecay"),
    "shrink_perturb": (".shrink_perturb", "ShrinkPerturb"),
    "upgd": (".upgd", "UPGD"),
    "parseval": (".parseval", "Parseval"),
    "scale_corrected": (".scale_corrected", "ScaleCorrectedSin"),
}


def available_methods():
    return sorted(_REGISTRY)


def build_method(cfg: Dict[str, Any], model, optimizer_cfg: Dict[str, Any], generator: Optional[torch.Generator] = None) -> Method:
    name = cfg.get("name", "baseline")
    if name not in _REGISTRY:
        raise ValueError(f"unknown method {name!r}; available: {available_methods()}")
    mod_path, cls_name = _REGISTRY[name]
    cls = getattr(importlib.import_module(mod_path, __package__), cls_name)
    return cls(model, {k: v for k, v in cfg.items() if k != "name"}, optimizer_cfg, generator)


__all__ = ["Method", "build_method", "build_optimizer", "available_methods"]
