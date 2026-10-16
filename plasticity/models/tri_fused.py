"""Fused CPU kernel for the triangle-wave map (one pass over the parameters instead of ~14).

The eager formulation in :mod:`plasticity.models.reparam` is a chain of elementwise ops; on a 12 M-parameter
network every op is a full pass over 48 MB and the map alone costs ~160 ms per online step (bandwidth
bound).  This kernel reads Θ once and writes f(ang) and f'(ang) once, using exactly the same float32
operations in the same order (division by A, +1, ×0.25, floor, ×4, −, −2, |·|, 1−, ×0.5, floor, ×2, 1−),
compiled with ``-ffp-contract=off`` so no fused multiply-add changes a rounding.  The results are therefore
bit-identical to the eager path (tests/test_reparam_nondamped.py checks this on millions of values).

Built lazily with ``torch.utils.cpp_extension.load_inline`` (needs a C++ compiler and ninja; ~20 s once,
cached under ``~/.cache/torch_extensions``).  If the build fails the caller falls back to the eager path.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import torch

#: tensors with fewer elements use the eager path (per-op overhead, not bandwidth, dominates there, and the
#: extension call has its own fixed cost).  Override with PLASTICITY_TRI_FUSED_MIN_NUMEL (0 = always fused).
MIN_NUMEL = int(os.environ.get("PLASTICITY_TRI_FUSED_MIN_NUMEL", 1_000_000))
#: PLASTICITY_TRI_FUSED=0 disables the kernel entirely.
ENABLED = os.environ.get("PLASTICITY_TRI_FUSED", "1") not in ("0", "false", "no")

_SRC = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <vector>

// f(x) = 1 - |((x + 1) mod 4) - 2|  with (x + 1) mod 4 = u - 4*floor(u/4), f'(x) = 1 - 2*floor(y/2)
std::vector<torch::Tensor> tri_forward(torch::Tensor theta, double amplitude, bool scaled) {
    TORCH_CHECK(theta.scalar_type() == torch::kFloat32, "tri_forward: float32 tensors only");
    TORCH_CHECK(theta.device().is_cpu(), "tri_forward: CPU tensors only");
    auto th = theta.contiguous();
    auto f = torch::empty_like(th);
    auto d = torch::empty_like(th);
    const float a = static_cast<float>(amplitude);
    const float* x = th.data_ptr<float>();
    float* pf = f.data_ptr<float>();
    float* pd = d.data_ptr<float>();
    const int64_t n = th.numel();
    at::parallel_for(0, n, 65536, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const float ang = scaled ? x[i] / a : x[i];
            const float u = ang + 1.0f;
            const float y = u - 4.0f * std::floor(u * 0.25f);
            pf[i] = 1.0f - std::fabs(y - 2.0f);
            pd[i] = 1.0f - 2.0f * std::floor(y * 0.5f);
        }
    });
    return {f, d};
}
"""

_ext = None
_tried = False


def get_extension():
    """The compiled module (``tri_forward``) or ``None`` when disabled or the build failed."""
    global _ext, _tried
    if _tried:
        return _ext
    _tried = True
    if not ENABLED:
        return None
    try:
        from torch.utils.cpp_extension import load_inline

        _ext = load_inline(
            name="plasticity_tri_fused_v1",
            cpp_sources=[_SRC],
            functions=["tri_forward"],
            extra_cflags=["-O3", "-ffp-contract=off", "-fno-fast-math"],
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - depends on the machine's toolchain
        warnings.warn(f"triangle-wave fused kernel unavailable, using the eager map ({type(exc).__name__}: {exc})")
        _ext = None
    return _ext


def tri_forward(theta: torch.Tensor, amplitude: torch.Tensor, scaled: bool) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(f, f')`` of the triangle wave at ``theta / amplitude`` (scaled) or ``theta``; ``None`` if the fused
    path does not apply (small tensor, non-float32, non-CPU, or no compiler)."""
    if not ENABLED or theta.numel() < MIN_NUMEL or theta.dtype != torch.float32 or not theta.device.type == "cpu":
        return None
    if amplitude.numel() != 1:
        return None
    ext = get_extension()
    if ext is None:
        return None
    f, d = ext.tri_forward(theta.detach(), float(amplitude), bool(scaled))
    return f.view_as(theta), d.view_as(theta)
