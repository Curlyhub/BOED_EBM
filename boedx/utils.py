"""
Shared numerical and I/O utilities used throughout BOEDX.

Nothing in this module imports from other BOEDX sub-modules; it is safe to
import everywhere without circular-dependency concerns.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random-number generators for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# File system helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """Create *path* (and any missing parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Tensor utilities
# ---------------------------------------------------------------------------

def sanitize(x: torch.Tensor, clip: float = 1e4) -> torch.Tensor:
    """Replace NaN / ±Inf values and hard-clamp to [-clip, clip].

    This is used defensively throughout the training loop to prevent silent
    divergence from propagating through the computation graph.
    """
    return torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)


def gaussian_logpdf(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor | float,
) -> torch.Tensor:
    """Element-wise log-probability under N(mean, std²)."""
    if not torch.is_tensor(std):
        std = torch.tensor(std, dtype=mean.dtype, device=mean.device)
    var = std * std
    log_two_pi_var = torch.log(
        2.0 * torch.tensor(math.pi, dtype=mean.dtype, device=mean.device) * var
    )
    return -0.5 * (((x - mean) ** 2) / var + log_two_pi_var)


def logmeanexp_t(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable log-mean-exp along *dim*."""
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak / soft update: target ← (1-τ)·target + τ·source.

    Uses ``lerp_`` for a single fused kernel per parameter tensor, which is
    faster than the equivalent ``mul_`` + ``add_`` pair.
    """
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.lerp_(sp.data, tau)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def mean_std_ci95(x: np.ndarray) -> Dict[str, float]:
    """Return mean, std, and 95 % CI bounds for a 1-D array.

    Uses ddof=1 (Bessel-corrected) standard deviation and the normal
    approximation (z=1.96) for the CI — appropriate for n ≥ 3 seeds.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    se = std / math.sqrt(max(len(x), 1))
    ci95 = 1.96 * se
    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
    }


def paired_summary(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Paired-difference statistics b − a (mean, std, 95 % CI, n)."""
    diff = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    out = mean_std_ci95(diff)
    out["n"] = int(len(diff))
    return out
