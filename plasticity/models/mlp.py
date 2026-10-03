"""Multi-layer perceptron built from :class:`ReparamLinear` layers.

Architecture: ``x -> [Linear -> (LayerNorm) -> act] * n_hidden -> Linear``.

The reparametrisation can be applied to the hidden layers only, to the output layer only, or to all
layers (component analysis), via ``reparam = {'hidden': mode, 'output': mode}``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .activations import activation_gain, get_activation
from .reparam import ReparamLinear


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (100, 100),
        n_classes: int = 10,
        activation: str = "relu",
        activation_kwargs: Optional[Dict[str, Any]] = None,
        layer_norm: bool = False,
        ln_affine: bool = True,
        reparam: Optional[Dict[str, str]] = None,
        gamma: float = 1.5,
        compact_bias: bool = True,
        init: str = "kaiming_uniform",
        init_gain: Optional[float] = None,
        bias_init: str = "zeros",
        learn_amplitude: bool = False,
        theta_scale: str = "amplitude",
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()
        reparam = dict(reparam or {})
        hidden_mode = reparam.get("hidden", "standard")
        output_mode = reparam.get("output", "standard")
        gain = activation_gain(activation) if init_gain is None else float(init_gain)
        self.input_dim, self.hidden_sizes, self.n_classes = int(input_dim), tuple(int(h) for h in hidden_sizes), int(n_classes)
        self.activation_name = activation
        self.layer_norm = bool(layer_norm)

        layers: List[ReparamLinear] = []
        norms: List[nn.Module] = []
        acts: List[nn.Module] = []
        d = self.input_dim
        for h in self.hidden_sizes:
            layers.append(
                ReparamLinear(d, h, mode=hidden_mode, compact_bias=compact_bias, gamma=gamma, init=init, gain=gain,
                              bias_init=bias_init, learn_amplitude=learn_amplitude, theta_scale=theta_scale, generator=generator)
            )
            norms.append(nn.LayerNorm(h, elementwise_affine=ln_affine) if layer_norm else nn.Identity())
            acts.append(get_activation(activation, width=h, **(activation_kwargs or {})))
            d = h
        # output layer: linear gain (no nonlinearity follows)
        self.output = ReparamLinear(d, self.n_classes, mode=output_mode, compact_bias=compact_bias, gamma=gamma, init=init,
                                    gain=1.0, bias_init=bias_init, learn_amplitude=learn_amplitude, theta_scale=theta_scale, generator=generator)
        self.hidden = nn.ModuleList(layers)
        self.norms = nn.ModuleList(norms)
        self.acts = nn.ModuleList(acts)

    # --- convenience -----------------------------------------------------------------------------
    @property
    def linear_layers(self) -> List[ReparamLinear]:
        return list(self.hidden) + [self.output]

    @property
    def n_hidden_layers(self) -> int:
        return len(self.hidden)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats: List[torch.Tensor] = []
        h = x
        for lin, norm, act in zip(self.hidden, self.norms, self.acts):
            h = act(norm(lin(h)))
            if return_features:
                feats.append(h)
        out = self.output(h)
        if return_features:
            return out, feats
        return out

    def set_capture_effective(self, flag: bool) -> None:
        for lin in self.linear_layers:
            lin.capture_effective = flag
            if not flag:
                lin.last_weight = lin.last_bias = None


def build_model(cfg: Dict[str, Any], input_dim: int, n_classes: int, generator: Optional[torch.Generator] = None) -> MLP:
    """Build from the ``model`` section of a config."""
    kw = {k: v for k, v in cfg.items() if k not in ("name",)}
    name = cfg.get("name", "mlp")
    if name != "mlp":
        raise ValueError(f"unknown model {name!r}")
    return MLP(input_dim=input_dim, n_classes=n_classes, generator=generator, **kw)
