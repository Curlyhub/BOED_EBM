from __future__ import annotations

import copy
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
try:
    from common_boed_source_expressive_policy import save_standard_plots
except Exception:
    def save_standard_plots(*args, **kwargs):
        return None

# ============================================================
# Generic utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def sanitize(x: torch.Tensor, clip: float = 1e4) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)



_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_logpdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(std):
        std = torch.tensor(std, dtype=mean.dtype, device=mean.device)
    var = std * std
    return -0.5 * (((x - mean) ** 2) / var + torch.log(var) + _LOG_2PI)



def mean_std_ci95(x: np.ndarray) -> Dict[str, float]:
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
    diff = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    out = mean_std_ci95(diff)
    out["n"] = int(len(diff))
    return out



def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    # lerp_ performs tp += tau*(sp-tp) in a single fused CUDA kernel, vs the
    # original mul_+add_ which launches two kernels per parameter tensor.
    # Mathematically identical to: tp = (1-tau)*tp + tau*sp.
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.lerp_(sp.data, tau)



def logmeanexp_t(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])


def phase_gate(time_frac: Optional[torch.Tensor], start_frac: float) -> Optional[torch.Tensor]:
    """Smooth late-horizon gate in [0,1].

    0 before start_frac, then smoothly rises to 1 at the horizon.
    Accepts shape (B,) or (B,1). Returns shape (B,1).
    """
    if time_frac is None:
        return None
    tf = time_frac if time_frac.dim() > 1 else time_frac.unsqueeze(-1)
    tf = tf.clamp(0.0, 1.0)
    if start_frac >= 1.0:
        return torch.zeros_like(tf)
    x = ((tf - start_frac) / max(1.0 - start_frac, 1e-6)).clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ============================================================
# Common model components
# ============================================================


class CachedFilterBackbone(nn.Module):
    """Backbone over cached filter logits.

    We intentionally work from cached logits updated online by the environment.
    This keeps the trainer experiment-agnostic and lets each environment decide
    what its exact or approximate filter should be.
    """

    def forward_from_logits(self, cached_logits: torch.Tensor) -> torch.Tensor:
        return torch.exp(F.log_softmax(cached_logits, dim=-1))


class EnergyNet(nn.Module):
    def __init__(self, hist_dim: int, theta_dim: int, hidden: int = 128):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.theta_net = nn.Sequential(nn.Linear(theta_dim, hidden), nn.ReLU())
        self.out = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, hist_feat: torch.Tensor, theta_bank: torch.Tensor) -> torch.Tensor:
        B = hist_feat.shape[0]
        H = theta_bank.shape[0]
        s = sanitize(self.state_net(hist_feat), 100.0)[:, None, :].expand(B, H, -1)
        th = sanitize(self.theta_net(theta_bank), 100.0).unsqueeze(0).expand(B, H, -1)
        e = self.out(torch.cat([s, th], dim=-1)).squeeze(-1)
        return sanitize(e, 50.0)


class CrossInteractionEnergyNet(nn.Module):
    def __init__(self, hist_dim: int, theta_dim: int, hidden: int = 128):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.theta_net = nn.Sequential(
            nn.Linear(theta_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hist_feat: torch.Tensor, theta_bank: torch.Tensor) -> torch.Tensor:
        B = hist_feat.shape[0]
        H = theta_bank.shape[0]
        s = sanitize(self.state_net(hist_feat), 100.0)[:, None, :].expand(B, H, -1)
        th = sanitize(self.theta_net(theta_bank), 100.0).unsqueeze(0).expand(B, H, -1)
        fused = torch.cat([s, th, s * th, torch.abs(s - th)], dim=-1)
        e = self.out(fused).squeeze(-1)
        return sanitize(e, 50.0)




class _ThetaEncodingCacheMixin:
    """Caches theta-side encodings in eval mode for a fixed hypothesis bank."""

    def _init_theta_cache(self) -> None:
        self._theta_cache_key: Optional[Tuple[int, torch.device, torch.dtype, Tuple[int, ...], int]] = None
        self._theta_cache_value: Optional[torch.Tensor] = None

    def _maybe_get_theta_cache(self, theta_bank: torch.Tensor) -> Optional[torch.Tensor]:
        if self.training:
            return None
        key = (theta_bank.data_ptr(), theta_bank.device, theta_bank.dtype, tuple(theta_bank.shape), self.n_sources)
        if self._theta_cache_key == key and self._theta_cache_value is not None:
            return self._theta_cache_value
        return None

    def _store_theta_cache(self, theta_bank: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            return value
        key = (theta_bank.data_ptr(), theta_bank.device, theta_bank.dtype, tuple(theta_bank.shape), self.n_sources)
        self._theta_cache_key = key
        self._theta_cache_value = value.detach()
        return value


# ============================================================
# Geometry-aware EBM variants for MULTI-SOURCE localisation
# ============================================================
#
# Design history and why these exist:
#
# The original EnergyNet / CrossInteractionEnergyNet above were designed
# and validated on the 1-D source localisation problem, where theta lives
# in R^1 and the notion of "position" and "distance" are the same scalar.
# There a generic MLP over raw theta works: the network trivially learns
# that the likelihood is a smooth scalar function of theta, no further
# geometric structure is needed.
#
# In 2-D source localisation the latent is theta = (theta_1, theta_2)
# with each theta_i in R^2. The likelihood has THREE structural properties
# that a raw-theta MLP is geometry-oblivious to:
#
#   (A) Source-swap symmetry.  mu(theta_1, theta_2, d) = mu(theta_2, theta_1, d).
#       Hence the true posterior is permutation-invariant under (theta_1 <-> theta_2).
#       A generic MLP over raw R^4 theta can only approximate this symmetry
#       and wastes capacity doing so.
#
#   (B) Distance-based information.  The per-source intensity is 1/(m + ||theta_i - d||^2).
#       What actually drives the likelihood is Euclidean distance in R^2,
#       not raw coordinates. The geometric variants below do NOT receive the
#       full design history explicitly; they only inject source-side geometric
#       structure (coordinates, norms, pairwise distances) while the design
#       history remains compressed in hist_feat.
#
#   (C) Two-source factorisation.  mu is a sum of per-source terms.
#       A network that processes each source INDEPENDENTLY and then SUMS
#       their contributions matches this factorisation structurally.
#
# The classes below implement (A) and (C) via a deep-sets aggregator
# (Zaheer et al., 2017) and add source-side geometric features related to (B).
# specifically parametrised for K-source problems where the latent theta
# factors as (theta_1, ..., theta_K), each theta_i in R^d_source (typically 2).
#
# For 1-D single-source or non-multi-source problems, use the original
# EnergyNet / CrossInteractionEnergyNet above — they remain correct and
# are the historical baseline (good results on prey and on 1-D source).

class SymmetricSourceEnergyNet(_ThetaEncodingCacheMixin, nn.Module):
    """
    Permutation-invariant energy net for K-source localisation problems.

    Architecture (deep-sets style):
      1. Each source theta_i in R^{d_source} is augmented with geometric
         features (norm, optional pairwise distances), then encoded by
         a SHARED MLP phi_theta. Shared weights + sum aggregation gives
         permutation invariance by construction.
      2. hist_feat is encoded by phi_hist.
      3. The energy is computed from [phi_hist(h), sum_i phi_theta(theta_i)]
         through an output MLP.

    Invariance:
      E(h, (theta_1, ..., theta_K)) = E(h, (theta_sigma(1), ..., theta_sigma(K)))
      for any permutation sigma.

    Parameters:
      hist_dim:   size of hist_feat input vector (bank size for quotient filter)
      theta_dim:  total theta dimension (e.g. 4 for 2 sources x 2D)
      n_sources:  K (2 in the source-location benchmark)
      hidden:     hidden width of MLPs
      add_pairwise_dist: if True and K >= 2, include ||theta_i - theta_j||
                        as a scalar feature per source. This breaks strict
                        per-source symmetry but is still permutation-invariant
                        overall because the same set of pairwise distances is
                        available to every source encoder.
    """

    def __init__(
        self,
        hist_dim: int,
        theta_dim: int,
        hidden: int = 128,
        n_sources: int = 2,
        add_pairwise_dist: bool = True,
    ):
        super().__init__()
        if theta_dim % n_sources != 0:
            raise ValueError(
                f"SymmetricSourceEnergyNet requires theta_dim ({theta_dim}) "
                f"divisible by n_sources ({n_sources})."
            )
        self.n_sources = int(n_sources)
        self.d_source = theta_dim // n_sources
        self.add_pairwise_dist = bool(add_pairwise_dist) and n_sources >= 2

        # Per-source feature: [theta_i (d_source), ||theta_i|| (1),
        #                      mean pairwise dist to other sources (1, optional)]
        self._per_source_feat_dim = self.d_source + 1 + (1 if self.add_pairwise_dist else 0)
        self._init_theta_cache()

        self.state_net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # SHARED per-source encoder phi_theta — this is what enforces symmetry
        self.theta_net = nn.Sequential(
            nn.Linear(self._per_source_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _augment_theta(self, theta_bank: torch.Tensor) -> torch.Tensor:
        """
        Augment each source with geometric features.
        Input: theta_bank of shape (H, K * d_source)
        Output: (H, K, per_source_feat_dim)
        """
        H = theta_bank.shape[0]
        K = self.n_sources
        d = self.d_source
        # split into (H, K, d_source)
        th = theta_bank.view(H, K, d)
        # per-source norm: (H, K, 1)
        norms = th.norm(dim=-1, keepdim=True)
        pieces = [th, norms]
        if self.add_pairwise_dist:
            # mean pairwise dist from source i to all other sources
            # th: (H, K, d); compute pairwise (H, K, K) distances
            diff = th.unsqueeze(2) - th.unsqueeze(1)          # (H, K, K, d)
            pairwise = diff.norm(dim=-1)                      # (H, K, K)
            # exclude self-distance (which is 0) from the mean:
            # sum over j != i, divide by (K-1)
            mean_other = (pairwise.sum(dim=-1, keepdim=True)) / max(K - 1, 1)  # (H, K, 1)
            pieces.append(mean_other)
        return torch.cat(pieces, dim=-1)                      # (H, K, feat_dim)

    def forward(self, hist_feat: torch.Tensor, theta_bank: torch.Tensor) -> torch.Tensor:
        B = hist_feat.shape[0]
        H = theta_bank.shape[0]
        th_agg = self._maybe_get_theta_cache(theta_bank)
        if th_agg is None:
            th_feat = self._augment_theta(theta_bank)
            th_enc = sanitize(self.theta_net(th_feat), 100.0)
            th_agg = self._store_theta_cache(theta_bank, th_enc.sum(dim=1))
        s = sanitize(self.state_net(hist_feat), 100.0)
        # broadcast and concat
        s_exp = s[:, None, :].expand(B, H, -1)
        th_exp = th_agg[None, :, :].expand(B, H, -1)
        e = self.out(torch.cat([s_exp, th_exp], dim=-1)).squeeze(-1)
        return sanitize(e, 50.0)


class SymmetricSourceCrossNet(_ThetaEncodingCacheMixin, nn.Module):
    """
    Permutation-invariant cross-interaction energy net for K-source problems.

    Adds the [s, th, s*th, |s - th|] interaction fusion from
    CrossInteractionEnergyNet on top of the per-source deep-sets encoder.
    The per-source representation remains permutation-invariant because
    the interaction is applied to the AGGREGATED th vector, not to
    individual sources.

    Empirically the cross interaction helps when the history features
    encode position-like information that can usefully be combined with
    theta position features via element-wise products.
    """

    def __init__(
        self,
        hist_dim: int,
        theta_dim: int,
        hidden: int = 128,
        n_sources: int = 2,
        add_pairwise_dist: bool = True,
    ):
        super().__init__()
        if theta_dim % n_sources != 0:
            raise ValueError(
                f"SymmetricSourceCrossNet requires theta_dim ({theta_dim}) "
                f"divisible by n_sources ({n_sources})."
            )
        self.n_sources = int(n_sources)
        self.d_source = theta_dim // n_sources
        self.add_pairwise_dist = bool(add_pairwise_dist) and n_sources >= 2
        self._per_source_feat_dim = self.d_source + 1 + (1 if self.add_pairwise_dist else 0)
        self._init_theta_cache()

        self.state_net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.theta_net = nn.Sequential(
            nn.Linear(self._per_source_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _augment_theta(self, theta_bank: torch.Tensor) -> torch.Tensor:
        H = theta_bank.shape[0]
        K = self.n_sources
        d = self.d_source
        th = theta_bank.view(H, K, d)
        norms = th.norm(dim=-1, keepdim=True)
        pieces = [th, norms]
        if self.add_pairwise_dist:
            diff = th.unsqueeze(2) - th.unsqueeze(1)
            pairwise = diff.norm(dim=-1)
            mean_other = (pairwise.sum(dim=-1, keepdim=True)) / max(K - 1, 1)
            pieces.append(mean_other)
        return torch.cat(pieces, dim=-1)

    def forward(self, hist_feat: torch.Tensor, theta_bank: torch.Tensor) -> torch.Tensor:
        B = hist_feat.shape[0]
        H = theta_bank.shape[0]
        th_agg = self._maybe_get_theta_cache(theta_bank)
        if th_agg is None:
            th_feat = self._augment_theta(theta_bank)
            th_enc = sanitize(self.theta_net(th_feat), 100.0)
            th_agg = self._store_theta_cache(theta_bank, th_enc.sum(dim=1))
        s = sanitize(self.state_net(hist_feat), 100.0)
        s_exp = s[:, None, :].expand(B, H, -1)
        th_exp = th_agg[None, :, :].expand(B, H, -1)
        fused = torch.cat([s_exp, th_exp, s_exp * th_exp, torch.abs(s_exp - th_exp)], dim=-1)
        e = self.out(fused).squeeze(-1)
        return sanitize(e, 50.0)


class ApsiHead(nn.Module):
    def __init__(self, hist_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, hist_feat: torch.Tensor) -> torch.Tensor:
        return sanitize(self.net(hist_feat), 50.0)


class TanhGaussianActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        dropout: float = 0.0,
        phase_adaptive: bool = False,
        phase_start_frac: float = 0.6,
        phase_strength: float = 1.0,
        late_std_scale: float = 1.0,
    ):
        super().__init__()
        self.input_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.phase_adaptive = bool(phase_adaptive)
        self.phase_start_frac = float(phase_start_frac)
        self.phase_strength = float(phase_strength)
        self.late_std_scale = float(late_std_scale)
        if self.phase_adaptive:
            self.late_mean_delta = nn.Linear(hidden, action_dim)
            self.late_log_std_delta = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor, time_frac: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = sanitize(self.net(self.input_dropout(state)), 100.0)
        mean = sanitize(self.mean(h), 20.0)
        log_std = sanitize(self.log_std(h), 5.0)
        gate = phase_gate(time_frac, self.phase_start_frac) if self.phase_adaptive else None
        if gate is not None:
            mean = mean + self.phase_strength * gate * sanitize(self.late_mean_delta(h), 20.0)
            log_std = log_std + self.phase_strength * gate * sanitize(self.late_log_std_delta(h), 5.0)
            if self.late_std_scale != 1.0:
                scale = (1.0 - gate) + gate * self.late_std_scale
                log_std = log_std + torch.log(scale.clamp_min(1e-4))
        log_std = sanitize(log_std, 5.0).clamp(-5.0, 1.0)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        time_frac: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(state, time_frac=time_frac)
        std = log_std.exp().clamp(min=1e-4, max=10.0)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        u = torch.tanh(z)
        action = u * action_scale + action_bias
        log_prob = normal.log_prob(z) - torch.log(1.0 - u.pow(2) + 1e-6)
        log_prob = sanitize(log_prob, 100.0).sum(dim=-1, keepdim=True)
        deterministic = torch.tanh(mean) * action_scale + action_bias
        return action, log_prob, deterministic


class MixtureTanhGaussianActor(nn.Module):
    """Shared expressive continuous actor with a K-component tanh-Gaussian mixture.

    Adds an optional late-horizon residual head that is activated only in the
    final part of the horizon, to let the policy become more disciplined late
    without perturbing the early exploratory regime.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        dropout: float = 0.0,
        n_components: int = 4,
        phase_adaptive: bool = False,
        phase_start_frac: float = 0.6,
        phase_strength: float = 1.0,
        late_std_scale: float = 1.0,
        late_mix_temp: float = 1.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.n_components = int(n_components)
        if self.n_components < 2:
            raise ValueError('MixtureTanhGaussianActor requires n_components >= 2')
        self.input_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mix_logits = nn.Linear(hidden, self.n_components)
        self.mean = nn.Linear(hidden, self.n_components * self.action_dim)
        self.log_std = nn.Linear(hidden, self.n_components * self.action_dim)
        self.phase_adaptive = bool(phase_adaptive)
        self.phase_start_frac = float(phase_start_frac)
        self.phase_strength = float(phase_strength)
        self.late_std_scale = float(late_std_scale)
        self.late_mix_temp = float(late_mix_temp)
        if self.phase_adaptive:
            self.late_mix_logits_delta = nn.Linear(hidden, self.n_components)
            self.late_mean_delta = nn.Linear(hidden, self.n_components * self.action_dim)
            self.late_log_std_delta = nn.Linear(hidden, self.n_components * self.action_dim)

    def forward(self, state: torch.Tensor, time_frac: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = sanitize(self.net(self.input_dropout(state)), 100.0)
        logits = sanitize(self.mix_logits(h), 20.0)
        mean = sanitize(self.mean(h), 20.0).view(-1, self.n_components, self.action_dim)
        log_std = sanitize(self.log_std(h), 5.0).view(-1, self.n_components, self.action_dim)
        gate = phase_gate(time_frac, self.phase_start_frac) if self.phase_adaptive else None
        if gate is not None:
            logits = logits + self.phase_strength * gate * sanitize(self.late_mix_logits_delta(h), 20.0)
            mean = mean + self.phase_strength * gate.unsqueeze(-1) * sanitize(self.late_mean_delta(h), 20.0).view(-1, self.n_components, self.action_dim)
            log_std = log_std + self.phase_strength * gate.unsqueeze(-1) * sanitize(self.late_log_std_delta(h), 5.0).view(-1, self.n_components, self.action_dim)
            if self.late_std_scale != 1.0:
                scale = ((1.0 - gate) + gate * self.late_std_scale).unsqueeze(-1)
                log_std = log_std + torch.log(scale.clamp_min(1e-4))
            if self.late_mix_temp != 1.0:
                temp = ((1.0 - gate) + gate * self.late_mix_temp).clamp_min(1e-4)
                logits = logits / temp
        log_std = sanitize(log_std, 5.0).clamp(-5.0, 1.0)
        return logits, mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        time_frac: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, mean, log_std = self(state, time_frac=time_frac)
        std = log_std.exp().clamp(min=1e-4, max=10.0)
        normal = torch.distributions.Normal(mean, std)
        z_all = normal.rsample()
        g = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        z = (g.unsqueeze(-1) * z_all).sum(dim=1)
        u = torch.tanh(z)
        action = u * action_scale + action_bias
        z_exp = z.unsqueeze(1)
        comp_log_prob = normal.log_prob(z_exp).sum(dim=-1)
        log_mix = F.log_softmax(logits, dim=-1)
        log_prob_z = torch.logsumexp(log_mix + comp_log_prob, dim=-1, keepdim=True)
        log_det = torch.log(1.0 - u.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = sanitize(log_prob_z - log_det, 100.0)
        det_idx = torch.argmax(logits, dim=-1)
        det_mean = mean[torch.arange(mean.shape[0], device=mean.device), det_idx]
        deterministic = torch.tanh(det_mean) * action_scale + action_bias
        return action, log_prob, deterministic



class SequenceTransformerTanhGaussianActor(nn.Module):
    """Path-aware continuous actor for NES BOED policies.

    The input state is expected to start with the flattened raw history block:
        actions[0:H, action_dim], obs[0:H]
    followed by context features such as time, last observation, quotient/filter
    state and EBM belief features. The actor turns the path into sequence tokens,
    appends a context token, and predicts a tanh-Gaussian mixture policy from a
    learned CLS token. It keeps the same sample() interface as the MLP/MoG actors.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        horizon: int,
        hidden: int = 256,
        dropout: float = 0.0,
        n_components: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        phase_adaptive: bool = False,
        phase_start_frac: float = 0.6,
        phase_strength: float = 1.0,
        late_std_scale: float = 1.0,
        late_mix_temp: float = 1.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.raw_path_dim = self.horizon * self.action_dim + self.horizon
        self.context_dim = max(1, self.state_dim - self.raw_path_dim)
        self.n_components = int(n_components)
        if self.n_components < 2:
            raise ValueError("SequenceTransformerTanhGaussianActor requires n_components >= 2")
        self.d_model = int(d_model)
        self.input_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.step_proj = nn.Linear(self.action_dim + 1, self.d_model)
        self.context_proj = nn.Linear(self.context_dim, self.d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.horizon + 2, self.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        head_hidden = max(int(hidden), self.d_model)
        self.head = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, head_hidden), nn.GELU())
        self.mix_logits = nn.Linear(head_hidden, self.n_components)
        self.mean = nn.Linear(head_hidden, self.n_components * self.action_dim)
        self.log_std = nn.Linear(head_hidden, self.n_components * self.action_dim)
        self.phase_adaptive = bool(phase_adaptive)
        self.phase_start_frac = float(phase_start_frac)
        self.phase_strength = float(phase_strength)
        self.late_std_scale = float(late_std_scale)
        self.late_mix_temp = float(late_mix_temp)
        if self.phase_adaptive:
            self.late_mix_logits_delta = nn.Linear(head_hidden, self.n_components)
            self.late_mean_delta = nn.Linear(head_hidden, self.n_components * self.action_dim)
            self.late_log_std_delta = nn.Linear(head_hidden, self.n_components * self.action_dim)

    def _split_state(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = state.shape[0]
        state = self.input_dropout(state)
        if state.shape[-1] < self.raw_path_dim:
            pad = torch.zeros(B, self.raw_path_dim - state.shape[-1], dtype=state.dtype, device=state.device)
            state = torch.cat([state, pad], dim=-1)
        actions_flat = state[:, : self.horizon * self.action_dim]
        obs_flat = state[:, self.horizon * self.action_dim : self.raw_path_dim]
        actions = actions_flat.reshape(B, self.horizon, self.action_dim)
        obs = obs_flat.reshape(B, self.horizon, 1)
        path = torch.cat([actions, obs], dim=-1)
        ctx = state[:, self.raw_path_dim:]
        if ctx.shape[-1] < self.context_dim:
            pad = torch.zeros(B, self.context_dim - ctx.shape[-1], dtype=state.dtype, device=state.device)
            ctx = torch.cat([ctx, pad], dim=-1)
        elif ctx.shape[-1] > self.context_dim:
            ctx = ctx[:, : self.context_dim]
        return path, ctx

    def forward(self, state: torch.Tensor, time_frac: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = state.shape[0]
        path, ctx = self._split_state(sanitize(state, 1e3))
        step_tok = self.step_proj(path)
        ctx_tok = self.context_proj(ctx).unsqueeze(1)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, step_tok, ctx_tok], dim=1)
        tokens = tokens + self.pos[:, : tokens.shape[1], :]
        z = sanitize(self.encoder(tokens), 100.0)[:, 0]
        h = sanitize(self.head(z), 100.0)
        logits = sanitize(self.mix_logits(h), 20.0)
        mean = sanitize(self.mean(h), 20.0).view(-1, self.n_components, self.action_dim)
        log_std = sanitize(self.log_std(h), 5.0).view(-1, self.n_components, self.action_dim)
        gate = phase_gate(time_frac, self.phase_start_frac) if self.phase_adaptive else None
        if gate is not None:
            logits = logits + self.phase_strength * gate * sanitize(self.late_mix_logits_delta(h), 20.0)
            mean = mean + self.phase_strength * gate.unsqueeze(-1) * sanitize(self.late_mean_delta(h), 20.0).view(-1, self.n_components, self.action_dim)
            log_std = log_std + self.phase_strength * gate.unsqueeze(-1) * sanitize(self.late_log_std_delta(h), 5.0).view(-1, self.n_components, self.action_dim)
            if self.late_std_scale != 1.0:
                scale = ((1.0 - gate) + gate * self.late_std_scale).unsqueeze(-1)
                log_std = log_std + torch.log(scale.clamp_min(1e-4))
            if self.late_mix_temp != 1.0:
                temp = ((1.0 - gate) + gate * self.late_mix_temp).clamp_min(1e-4)
                logits = logits / temp
        log_std = sanitize(log_std, 5.0).clamp(-5.0, 1.0)
        return logits, mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        time_frac: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, mean, log_std = self(state, time_frac=time_frac)
        std = log_std.exp().clamp(min=1e-4, max=10.0)
        normal = torch.distributions.Normal(mean, std)
        z_all = normal.rsample()
        g = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        z = (g.unsqueeze(-1) * z_all).sum(dim=1)
        u = torch.tanh(z)
        action = u * action_scale + action_bias
        z_exp = z.unsqueeze(1)
        comp_log_prob = normal.log_prob(z_exp).sum(dim=-1)
        log_mix = F.log_softmax(logits, dim=-1)
        log_prob_z = torch.logsumexp(log_mix + comp_log_prob, dim=-1, keepdim=True)
        log_det = torch.log(1.0 - u.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = sanitize(log_prob_z - log_det, 100.0)
        det_idx = torch.argmax(logits, dim=-1)
        det_mean = mean[torch.arange(mean.shape[0], device=mean.device), det_idx]
        deterministic = torch.tanh(det_mean) * action_scale + action_bias
        return action, log_prob, deterministic




class DualBranchMixtureTanhGaussianActor(nn.Module):
    """EBM-specific actor with separate quotient/base and belief branches.

    This is only used when --ebm-dual-branch-actor is enabled for EBM variants.
    It prevents NES from having to discover from a flat concatenation that the
    quotient/filter block and belief/modal block have different geometries.
    """

    def __init__(
        self,
        base_dim: int,
        belief_dim: int,
        action_dim: int,
        hidden: int = 256,
        dropout: float = 0.0,
        n_components: int = 4,
        phase_adaptive: bool = False,
        phase_start_frac: float = 0.6,
        phase_strength: float = 1.0,
        late_std_scale: float = 1.0,
        late_mix_temp: float = 1.0,
    ):
        super().__init__()
        self.base_dim = int(base_dim)
        self.belief_dim = int(belief_dim)
        self.action_dim = int(action_dim)
        self.n_components = int(n_components)
        if self.belief_dim <= 0:
            raise ValueError("DualBranchMixtureTanhGaussianActor requires belief_dim > 0")
        if self.n_components < 2:
            raise ValueError("DualBranchMixtureTanhGaussianActor requires n_components >= 2")
        self.input_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        branch_hidden = max(32, hidden // 2)
        self.base_net = nn.Sequential(
            nn.Linear(self.base_dim, branch_hidden), nn.ReLU(),
            nn.Linear(branch_hidden, branch_hidden), nn.ReLU(),
        )
        self.belief_net = nn.Sequential(
            nn.Linear(self.belief_dim, branch_hidden), nn.ReLU(),
            nn.Linear(branch_hidden, branch_hidden), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * branch_hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mix_logits = nn.Linear(hidden, self.n_components)
        self.mean = nn.Linear(hidden, self.n_components * self.action_dim)
        self.log_std = nn.Linear(hidden, self.n_components * self.action_dim)
        self.phase_adaptive = bool(phase_adaptive)
        self.phase_start_frac = float(phase_start_frac)
        self.phase_strength = float(phase_strength)
        self.late_std_scale = float(late_std_scale)
        self.late_mix_temp = float(late_mix_temp)
        if self.phase_adaptive:
            self.late_mix_logits_delta = nn.Linear(hidden, self.n_components)
            self.late_mean_delta = nn.Linear(hidden, self.n_components * self.action_dim)
            self.late_log_std_delta = nn.Linear(hidden, self.n_components * self.action_dim)

    def forward(self, state: torch.Tensor, time_frac: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.input_dropout(state)
        base = state[..., :self.base_dim]
        belief = state[..., self.base_dim:self.base_dim + self.belief_dim]
        hb = sanitize(self.base_net(base), 100.0)
        hz = sanitize(self.belief_net(belief), 100.0)
        h = sanitize(self.fusion(torch.cat([hb, hz], dim=-1)), 100.0)
        logits = sanitize(self.mix_logits(h), 20.0)
        mean = sanitize(self.mean(h), 20.0).view(-1, self.n_components, self.action_dim)
        log_std = sanitize(self.log_std(h), 5.0).view(-1, self.n_components, self.action_dim)
        gate = phase_gate(time_frac, self.phase_start_frac) if self.phase_adaptive else None
        if gate is not None:
            logits = logits + self.phase_strength * gate * sanitize(self.late_mix_logits_delta(h), 20.0)
            mean = mean + self.phase_strength * gate.unsqueeze(-1) * sanitize(self.late_mean_delta(h), 20.0).view(-1, self.n_components, self.action_dim)
            log_std = log_std + self.phase_strength * gate.unsqueeze(-1) * sanitize(self.late_log_std_delta(h), 5.0).view(-1, self.n_components, self.action_dim)
            if self.late_std_scale != 1.0:
                scale = ((1.0 - gate) + gate * self.late_std_scale).unsqueeze(-1)
                log_std = log_std + torch.log(scale.clamp_min(1e-4))
            if self.late_mix_temp != 1.0:
                temp = ((1.0 - gate) + gate * self.late_mix_temp).clamp_min(1e-4)
                logits = logits / temp
        log_std = sanitize(log_std, 5.0).clamp(-5.0, 1.0)
        return logits, mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        time_frac: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, mean, log_std = self(state, time_frac=time_frac)
        std = log_std.exp().clamp(min=1e-4, max=10.0)
        normal = torch.distributions.Normal(mean, std)
        z_all = normal.rsample()
        g = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        z = (g.unsqueeze(-1) * z_all).sum(dim=1)
        u = torch.tanh(z)
        action = u * action_scale + action_bias
        z_exp = z.unsqueeze(1)
        comp_log_prob = normal.log_prob(z_exp).sum(dim=-1)
        log_mix = F.log_softmax(logits, dim=-1)
        log_prob_z = torch.logsumexp(log_mix + comp_log_prob, dim=-1, keepdim=True)
        log_det = torch.log(1.0 - u.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = sanitize(log_prob_z - log_det, 100.0)
        det_idx = torch.argmax(logits, dim=-1)
        det_mean = mean[torch.arange(mean.shape[0], device=mean.device), det_idx]
        deterministic = torch.tanh(det_mean) * action_scale + action_bias
        return action, log_prob, deterministic

class DiscreteCategoricalActor(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.num_actions = int(num_actions)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.logits_head = nn.Linear(hidden, self.num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = sanitize(self.net(state), 100.0)
        return sanitize(self.logits_head(h), 20.0)

    def probs_and_log_probs(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self(state)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        return probs, log_probs

    def sample(
        self,
        state: torch.Tensor,
        action_values: torch.Tensor,
        time_frac: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self(state)
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        log_prob = dist.log_prob(idx).unsqueeze(-1)
        action = action_values[idx]
        det_idx = torch.argmax(logits, dim=-1)
        deterministic = action_values[det_idx]
        return action, log_prob, deterministic


class QCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sanitize(state, 1e3), sanitize(action, 1e3)], dim=-1)
        return sanitize(self.net(x), 1e4)


def uses_discrete_actor(env: "GenericBankBOEDEnv") -> bool:
    return getattr(env, "name", "") == "prey_population" and int(getattr(env, "action_dim", 0)) == 1


def get_discrete_action_values(env: "GenericBankBOEDEnv", device: torch.device) -> torch.Tensor:
    low = int(round(float(env.action_low[0].detach().cpu())))
    high = int(round(float(env.action_high[0].detach().cpu())))
    return torch.arange(low, high + 1, dtype=torch.float32, device=device).unsqueeze(-1)


def evaluate_q_over_discrete_actions(
    qnet: nn.Module,
    state: torch.Tensor,
    action_values: torch.Tensor,
) -> torch.Tensor:
    B = state.shape[0]
    N = action_values.shape[0]
    state_rep = state[:, None, :].expand(B, N, -1).reshape(B * N, -1)
    action_rep = action_values[None, :, :].expand(B, N, -1).reshape(B * N, -1)
    q = qnet(state_rep, action_rep).reshape(B, N)
    return q


class ReplayBuffer:
    """
    Pre-allocated numpy-backed replay buffer.

    Functionally identical to the previous list-of-dicts version:
    - Same sampling distribution (uniform np.random.randint over valid range)
    - Same output dict keys, shapes, and dtypes
    - Same FIFO eviction when capacity is reached

    The only change is the internal storage: each field is a contiguous
    numpy array of shape (capacity, *field_shape). This eliminates the
    per-sample Python object loop that was calling np.stack() 19 times
    per batch (once per field). On the source-location task with H=2401
    this is where the CPU was spending most of its time.

    Intentionally NOT using pin_memory(): calling pin_memory() on a fresh
    tensor allocation per batch is ~milliseconds of CUDA API overhead per
    call and becomes a net regression at high update frequencies.
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = int(capacity)
        self._size = 0
        self._ptr = 0
        self._arrays: Dict[str, np.ndarray] = {}
        # preserve long dtype for t_idx fields
        self._long_keys = {"t_idx", "next_t_idx"}

    def _init_arrays(self, item: Dict) -> None:
        for k, v in item.items():
            v_arr = np.asarray(v)
            # store scalar fields as (capacity,) -> will be reshaped (B,1) on sample
            shape = (self.capacity,) + v_arr.shape
            # keep floats as float32, ints as int64 (for t_idx)
            dtype = np.int64 if k in self._long_keys else np.float32
            self._arrays[k] = np.zeros(shape, dtype=dtype)

    def add(self, item: Dict) -> None:
        if not self._arrays:
            self._init_arrays(item)
        for k, v in item.items():
            self._arrays[k][self._ptr] = v
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self._size, size=batch_size)
        out: Dict[str, torch.Tensor] = {}
        for k, arr in self._arrays.items():
            sampled = arr[idxs]  # one fancy-index, contiguous copy
            if k in self._long_keys:
                t = torch.tensor(sampled, dtype=torch.long, device=device)
            else:
                t = torch.tensor(sampled, dtype=torch.float32, device=device)
            # scalar fields (original stored 0-dim per sample) arrive as (B,)
            # need to restore (B,1) shape to match old behaviour
            if t.dim() == 1 and k not in self._long_keys:
                t = t.unsqueeze(-1)
            out[k] = t
        return out


# ============================================================
# Generic bank-based sequential BOED environment
# ============================================================


@dataclass
class GenericTrainConfig:
    episodes: int = 300
    eval_episodes: int = 60
    batch_size: int = 128
    replay_size: int = 50000
    warmup_episodes: int = 20
    updates_per_step: int = 1
    gamma: float = 1.0
    tau: float = 0.01
    alpha: float = 0.1
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_ebm: float = 3e-4
    hidden_rl: int = 256
    hidden_ebm: int = 512
    grad_clip: float = 10.0
    apsi_coef: float = 0.1
    ebm_update_every: int = 4
    print_every: int = 25
    device: str = "cpu"
    seeds: str = "0,1,2"
    # --- Model selection on top of the final evaluation ---
    selection_eval_episodes: int = 40
    selection_every: int = 100
    selection_start_episode: int = 100
    selection_return_weight: float = 1.0
    selection_belief_kl_weight: float = 0.10
    selection_belief_map_weight: float = 0.25
    selection_belief_mean_weight: float = 0.50
    # --- Regularization (added to combat over-fitting in EBM variants on 2D source) ---
    # actor_weight_decay: L2 on actor params. 0.0 = off. 1e-4 is a safe starting value.
    # actor_dropout: dropout prob on actor's state input. 0.0 = off. 0.1-0.2 typical.
    # Both apply only to the actor (which is where over-fitting manifests); EBM and
    # critic are left alone to preserve belief tracking quality and Q-learning stability.
    actor_weight_decay: float = 0.0
    actor_dropout: float = 0.0
    # Continuous actor family shared by Blau and EBM variants.
    # "gaussian" keeps the historical single-Gaussian SAC actor.
    # "mog" uses a tractable mixture-of-Gaussians actor with exact mixture log-prob.
    actor_family: str = "gaussian"
    actor_mixture_components: int = 4
    # EBM-only actor overrides. Empty/zero means reuse the shared actor settings.
    ebm_actor_family: str = ""
    ebm_hidden_rl: int = 0
    ebm_actor_mixture_components: int = 0
    ebm_dual_branch_actor: bool = False
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_layers: int = 2
    transformer_ff: int = 128
    phase_adaptive_actor: bool = False
    phase_start_frac: float = 0.6
    phase_strength: float = 1.0
    late_std_scale: float = 0.5
    late_mix_temp: float = 0.75


@dataclass
class BeliefConfig:
    # How the EBM belief relates to the policy state.
    mode: str = "distilled_detached"
    # Which belief features the actor sees on top of hist_feat / minimal_base.
    # "legacy"  — mean + entropy (+ optional A) — the 1-D-era default.
    # "moments" — mean + diag(cov) + upper_triu(cov) + entropy (+ A). Better in R^D >= 2.
    # "modal"   — "moments" features PLUS top-K bank atoms and their probabilities.
    #             The selected atoms are sorted by bank index, so the representation is
    #             deterministic as long as the hypothesis bank itself is canonically
    #             ordered. This is important for multi-source / multi-modal posteriors.
    feature_mode: str = "legacy"

    # Which EBM architecture to instantiate for the "control" and "cross" variants.
    # "standard"  — historical EnergyNet / CrossInteractionEnergyNet. Generic MLP over raw theta.
    #               Correct choice for 1-D source and for non-multi-source problems (e.g. prey).
    # "geometric" — SymmetricSourceEnergyNet / SymmetricSourceCrossNet. Permutation-invariant
    #               per-source deep-sets encoder with distance features. Designed for
    #               K-source localisation in R^d_source (e.g. 2-D source with K=2).
    ebm_architecture: str = "standard"

    # Parameters used only when ebm_architecture == "geometric".
    # For 2-D source localisation: n_sources=2, source_dim=2.
    # For 3-D source with 2 sources: n_sources=2, source_dim=3.
    # The network will check that theta_dim == n_sources * source_dim.
    n_sources: int = 1
    source_dim: int = 0  # 0 = infer from theta_dim / n_sources
    add_pairwise_dist: bool = True

    # Parameters used only when feature_mode == "modal".
    # Number of top-probability bank atoms to expose to the actor.
    modal_top_k: int = 4

    # NES-specific actor-side adaptation. The EBM can still be trained against the
    # full posterior while the actor consumes a more compact, smoother belief
    # representation that is easier for NES to exploit.
    # "" means: reuse feature_mode / modal_top_k.
    nes_actor_feature_mode: str = ""
    nes_actor_modal_top_k: int = 0
    nes_cross_compact_belief: bool = False
    include_raw_history_for_ebm_actor: bool = False


class GenericBankBOEDEnv:
    """Generic bank-based BOED environment with dense SPCE-style reward.

    Each concrete environment must supply:
    - action bounds and clipping
    - latent prior sampling
    - observation sampling
    - scalar log-likelihood
    - batched bank/trajectory log-likelihoods
    - optional auxiliary state update
    """

    name: str = "generic"
    action_dim: int = 1
    theta_dim: int = 1

    def __init__(self, device: torch.device):
        self.device = device
        self.hypothesis_bank = self.build_hypothesis_bank().to(device)
        self.H = int(self.hypothesis_bank.shape[0])
        self.theta_dim = int(self.hypothesis_bank.shape[1])
        self.prior_bank_logits = self.build_prior_bank_logits().to(device)
        self.action_low = torch.tensor(self.get_action_low(), dtype=torch.float32, device=device)
        self.action_high = torch.tensor(self.get_action_high(), dtype=torch.float32, device=device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0
        self.reset()

    # ----- expected overrides -----
    def get_horizon(self) -> int:
        raise NotImplementedError

    def get_action_low(self) -> np.ndarray:
        raise NotImplementedError

    def get_action_high(self) -> np.ndarray:
        raise NotImplementedError

    def build_hypothesis_bank(self) -> torch.Tensor:
        raise NotImplementedError

    def build_prior_bank_logits(self) -> torch.Tensor:
        raise NotImplementedError

    def sample_theta(self) -> np.ndarray:
        raise NotImplementedError

    def sample_prior_thetas(self, n: int) -> torch.Tensor:
        raise NotImplementedError

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample_observation(self, theta: np.ndarray, action: np.ndarray) -> float:
        raise NotImplementedError

    def loglik_scalar(self, obs: float, theta: np.ndarray, action: np.ndarray) -> float:
        raise NotImplementedError

    def trajectory_loglik_thetas(
        self, actions: np.ndarray, obs: np.ndarray, thetas: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def bank_loglik_single(self, obs_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ----- optional overrides -----
    def current_aux_state(self) -> np.ndarray:
        return np.zeros((0,), dtype=np.float32)

    def before_episode(self) -> None:
        pass

    def after_step_update_aux(self, prev_action: Optional[np.ndarray], action: np.ndarray, obs: float) -> Tuple[bool, float]:
        """Return (terminate_now, reward_adjustment)."""
        return False, 0.0

    def observation_to_feature_scalar(self, obs: float) -> float:
        return float(obs)

    # ----- generic logic -----
    def posterior_bank(self) -> torch.Tensor:
        log_probs = F.log_softmax(self._posterior_logu, dim=0)
        return torch.exp(log_probs)

    def equivalence_bank(self) -> torch.Tensor:
        log_probs = F.log_softmax(self._equiv_logu, dim=0)
        return torch.exp(log_probs)

    def posterior_init_logits(self) -> torch.Tensor:
        """Exact posterior filter always starts from the discrete prior over the bank."""
        return self.prior_bank_logits.clone()

    def equivalence_filter_mode(self) -> str:
        mode = getattr(self, "exact_filter", "likelihood")
        if mode not in {"likelihood", "posterior"}:
            raise ValueError(f"Unsupported exact_filter={mode!r}. Expected 'likelihood' or 'posterior'.")
        return mode

    def equivalence_init_logits(self) -> torch.Tensor:
        """Initial logits for the control-side filter.

        - likelihood: prior-free accumulation of log-likelihoods
        - posterior:  prior-weighted filter, included only as an explicit ablation

        Keeping these paths separate avoids the previous bug where the control-side
        filter silently collapsed to the posterior in all cases.
        """
        mode = self.equivalence_filter_mode()
        if mode == "posterior":
            return self.prior_bank_logits.clone()
        return torch.zeros(self.H, dtype=torch.float32, device=self.device)

    def accumulate_filter_logits(self, cur_logits: torch.Tensor, obs_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        ll_bank = self.bank_loglik_single(obs_t, action_t)
        new_logits = torch.nan_to_num(cur_logits + ll_bank, nan=0.0, posinf=1e4, neginf=-1e4)
        return new_logits - torch.max(new_logits)

    def project_equivalence_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Optional environment hook for quotient/equivalence projections.

        The default is identity. Environments with known state-label symmetries can
        override this to aggregate states inside an exact equivalence class.
        """
        return logits

    def posterior_update_logits(self, cur_logits: torch.Tensor, obs_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        return self.accumulate_filter_logits(cur_logits, obs_t, action_t)

    def equivalence_update_logits(self, cur_logits: torch.Tensor, obs_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        logits = self.accumulate_filter_logits(cur_logits, obs_t, action_t)
        return self.project_equivalence_logits(logits)

    def reset(self) -> List[Tuple[np.ndarray, float]]:
        self.before_episode()
        self.theta0 = self.sample_theta()
        n = self.get_n_contrastive() + 1
        self.contrastive_thetas = [self.theta0] + [self.sample_theta() for _ in range(self.get_n_contrastive())]
        self._contrastive_t = torch.tensor(np.stack(self.contrastive_thetas, axis=0), dtype=torch.float32, device=self.device)
        self.logC = np.zeros(n, dtype=np.float64)
        self.history: List[Tuple[np.ndarray, float]] = []
        self.t = 0
        self.last_obs = 0.0
        self.last_action: Optional[np.ndarray] = None
        self._posterior_logu = self.posterior_init_logits()
        self._equiv_logu = self.equivalence_init_logits()
        return self.history

    def get_n_contrastive(self) -> int:
        return getattr(self, "n_contrastive", 32)

    def belief_distance(self, theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
        """Default geometry for belief errors: Euclidean distance in latent space."""
        return torch.linalg.norm(theta_a - theta_b, dim=-1)

    def step(self, action: np.ndarray) -> Tuple[float, float, bool, Dict]:
        action = self.clip_action(np.asarray(action, dtype=np.float32))
        obs = self.sample_observation(self.theta0, action)
        prev_logsum = float(np.log(np.exp(self.logC - self.logC.max()).sum()) + self.logC.max())
        ll_real = self.loglik_scalar(obs, self.theta0, action)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device)

        ll_all = self.trajectory_loglik_thetas(
            np.asarray([action], dtype=np.float32),
            np.asarray([obs], dtype=np.float32),
            self._contrastive_t,
        ).detach().cpu().numpy().astype(np.float64)
        self.logC = self.logC + ll_all
        new_logsum = float(np.log(np.exp(self.logC - self.logC.max()).sum()) + self.logC.max())
        reward = ll_real - new_logsum + prev_logsum

        with torch.no_grad():
            self._posterior_logu = self.posterior_update_logits(self._posterior_logu, obs_t, action_t)
            self._equiv_logu = self.equivalence_update_logits(self._equiv_logu, obs_t, action_t)

        terminate_now, reward_adjustment = self.after_step_update_aux(self.last_action, action, obs)
        reward = float(reward + reward_adjustment)

        self.history.append((action.copy(), float(obs)))
        self.last_obs = float(self.observation_to_feature_scalar(obs))
        self.last_action = action.copy()
        self.t += 1
        done = self.t >= self.get_horizon() or terminate_now
        info = {"theta0": self.theta0.copy(), "history": list(self.history), "reward_dense": reward}
        return obs, reward, done, info


# ============================================================
# Trainer helpers
# ============================================================


def build_base_state(
    hist_feat: torch.Tensor,
    t_idx: torch.Tensor,
    horizon: int,
    last_obs: torch.Tensor,
    aux_state: torch.Tensor,
) -> torch.Tensor:
    time_feat = t_idx.float().unsqueeze(-1) / float(max(horizon, 1))
    pieces = [hist_feat, time_feat, last_obs]
    if aux_state.shape[-1] > 0:
        pieces.append(aux_state)
    return sanitize(torch.cat(pieces, dim=-1), 1e3)



def build_minimal_state(
    t_idx: torch.Tensor,
    horizon: int,
    last_obs: torch.Tensor,
    aux_state: torch.Tensor,
) -> torch.Tensor:
    time_feat = t_idx.float().unsqueeze(-1) / float(max(horizon, 1))
    pieces = [time_feat, last_obs]
    if aux_state.shape[-1] > 0:
        pieces.append(aux_state)
    return sanitize(torch.cat(pieces, dim=-1), 1e3)



def build_raw_history_state(
    actions: torch.Tensor,
    obs: torch.Tensor,
    t_idx: torch.Tensor,
    horizon: int,
    last_obs: torch.Tensor,
    aux_state: torch.Tensor,
) -> torch.Tensor:
    B = actions.shape[0]
    flat_actions = actions.reshape(B, -1)
    flat_obs = obs.reshape(B, -1)
    time_feat = t_idx.float().unsqueeze(-1) / float(max(horizon, 1))
    pieces = [flat_actions, flat_obs, time_feat, last_obs]
    if aux_state.shape[-1] > 0:
        pieces.append(aux_state)
    return sanitize(torch.cat(pieces, dim=-1), 1e3)



def posterior_probs_from_energy(energy: torch.Tensor) -> torch.Tensor:
    return torch.exp(F.log_softmax(-energy, dim=-1))



def belief_kl_divergence(target_probs: torch.Tensor, pred_probs: torch.Tensor) -> torch.Tensor:
    target = target_probs.clamp_min(1e-12)
    pred = pred_probs.clamp_min(1e-12)
    return (target * (torch.log(target) - torch.log(pred))).sum(dim=-1)



def belief_l1_error(target_probs: torch.Tensor, pred_probs: torch.Tensor) -> torch.Tensor:
    return torch.abs(target_probs - pred_probs).sum(dim=-1)



def belief_feature_dim(theta_dim: int, feature_mode: str, modal_top_k: int = 4) -> int:
    if feature_mode == "legacy":
        return theta_dim + 2
    if feature_mode == "moments":
        upper = theta_dim * (theta_dim - 1) // 2
        return theta_dim + theta_dim + upper + 1 + 1
    if feature_mode == "modal":
        # "moments" features PLUS top-K (position + prob) from the bank.
        # Rationale: on multi-modal posteriors (e.g. 2-source), the posterior MEAN
        # is a misleading summary — it points at the midpoint of the modes rather
        # than at any mode. Top-K exposes the actual modes to the actor.
        upper = theta_dim * (theta_dim - 1) // 2
        moments_dim = theta_dim + theta_dim + upper + 1 + 1
        modal_dim = modal_top_k * (theta_dim + 1)  # K atoms of (theta, prob)
        return moments_dim + modal_dim
    raise ValueError(f"Unknown belief feature mode: {feature_mode}")



def belief_features_from_probs(
    probs: torch.Tensor,
    theta_bank: torch.Tensor,
    A_scalar: Optional[torch.Tensor] = None,
    feature_mode: str = "legacy",
    modal_top_k: int = 4,
) -> torch.Tensor:
    log_probs = torch.log(probs.clamp_min(1e-12))
    mean = probs @ theta_bank
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
    if feature_mode == "legacy":
        pieces = [mean, entropy]
        if A_scalar is not None:
            pieces.append(sanitize(A_scalar, 50.0))
        return sanitize(torch.cat(pieces, dim=-1), 1e3)
    if feature_mode not in {"moments", "modal"}:
        raise ValueError(f"Unknown belief feature mode: {feature_mode}")

    xc = theta_bank.unsqueeze(0) - mean.unsqueeze(1)
    cov = torch.einsum("bh,bhd,bhe->bde", probs, xc, xc)
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    upper_terms = []
    D = theta_bank.shape[-1]
    for i in range(D):
        for j in range(i + 1, D):
            upper_terms.append(cov[:, i, j:j+1])
    pieces = [mean, diag]
    if upper_terms:
        pieces.append(torch.cat(upper_terms, dim=-1))
    pieces.append(entropy)
    if A_scalar is not None:
        pieces.append(sanitize(A_scalar, 50.0))

    if feature_mode == "modal":
        # Extract top-K atoms by probability. Each atom contributes (theta, prob).
        # We sort the selected atoms by BANK INDEX, not by a single coordinate.
        # This is deterministic and robust as long as the environment builds the
        # hypothesis bank in a canonical order.
        B, H = probs.shape
        K = int(modal_top_k)
        K_eff = min(K, H)
        top_probs, top_idx = torch.topk(probs, k=K_eff, dim=-1)   # (B, K)
        top_theta = theta_bank[top_idx]                           # (B, K, D)
        sort_idx = torch.argsort(top_idx, dim=-1)
        idx_expand = sort_idx.unsqueeze(-1).expand(-1, -1, D)
        top_theta_sorted = torch.gather(top_theta, dim=1, index=idx_expand)
        top_probs_sorted = torch.gather(top_probs, dim=1, index=sort_idx)
        # flatten: (B, K*(D+1))
        modal_flat = torch.cat(
            [top_theta_sorted.reshape(B, -1), top_probs_sorted.reshape(B, -1)], dim=-1
        )
        # if K_eff < K, pad to (B, K*(D+1)) with zeros to keep fixed dim
        target_dim = K * (D + 1)
        if modal_flat.shape[-1] < target_dim:
            pad = torch.zeros(B, target_dim - modal_flat.shape[-1],
                              dtype=modal_flat.dtype, device=modal_flat.device)
            modal_flat = torch.cat([modal_flat, pad], dim=-1)
        pieces.append(modal_flat)

    return sanitize(torch.cat(pieces, dim=-1), 1e3)



def make_raw_state(env: GenericBankBOEDEnv, need_probs: bool = False) -> Dict:
    horizon = env.get_horizon()
    actions = np.zeros((horizon, env.action_dim), dtype=np.float32)
    obs = np.zeros(horizon, dtype=np.float32)
    for i, (a, y) in enumerate(env.history):
        actions[i] = a
        obs[i] = y
    out = {
        "actions": actions,
        "obs": obs,
        "length": len(env.history),
        "last_obs": float(env.last_obs),
        "t_idx": env.t,
        "aux_state": env.current_aux_state().astype(np.float32),
        "posterior_logits": env._posterior_logu.detach().cpu().numpy().astype(np.float32),
        "filter_logits": env._equiv_logu.detach().cpu().numpy().astype(np.float32),
    }
    if need_probs:
        out["posterior"] = env.posterior_bank().detach().cpu().numpy().astype(np.float32)
        out["filter_probs"] = env.equivalence_bank().detach().cpu().numpy().astype(np.float32)
    return out


@torch.no_grad()
def raw_state_to_policy_state(
    variant: str,
    raw_state: Dict,
    filter_backbone: CachedFilterBackbone,
    env: GenericBankBOEDEnv,
    device: torch.device,
    energy_net: Optional[nn.Module],
    apsi_head: Optional[nn.Module],
    belief_cfg: Optional[BeliefConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    batch = {
        "last_obs": torch.tensor([[raw_state["last_obs"]]], dtype=torch.float32, device=device),
        "t_idx": torch.tensor([raw_state["t_idx"]], dtype=torch.long, device=device),
        "aux_state": torch.tensor(raw_state["aux_state"][None], dtype=torch.float32, device=device),
        "actions": torch.tensor(raw_state["actions"][None], dtype=torch.float32, device=device),
        "obs": torch.tensor(raw_state["obs"][None], dtype=torch.float32, device=device),
        "posterior_logits": torch.tensor(raw_state["posterior_logits"][None], dtype=torch.float32, device=device),
        "filter_probs": torch.tensor(raw_state["filter_probs"][None], dtype=torch.float32, device=device),
        "filter_logits": torch.tensor(raw_state["filter_logits"][None], dtype=torch.float32, device=device),
    }
    return compute_state_from_batch(
        variant=variant,
        filter_backbone=filter_backbone,
        batch=batch,
        env=env,
        energy_net=energy_net,
        apsi_head=apsi_head,
        belief_cfg=belief_cfg,
        use_next=False,
    )




def variant_uses_ebm(variant: str) -> bool:
    return variant in {
        "ours_ebm_control",
        "ours_ebm_cross",
        "ours_ebm_control_filter",
        "ours_ebm_control_posterior",
        "ours_ebm_cross_filter",
        "ours_ebm_cross_posterior",
    }


def variant_uses_cross_ebm(variant: str) -> bool:
    return variant in {"ours_ebm_cross", "ours_ebm_cross_filter", "ours_ebm_cross_posterior"}


def history_logits_from_batch(variant: str, batch: Dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    """Select which exact history summary the actor/belief should consume.

    Variants ending in `_posterior` are backed by the exact posterior bank logits.
    Variants ending in `_filter` are backed by the control/filter logits. Historical
    variants (`ours_ebm_control`, `ours_ebm_cross`) keep consuming `filter_logits`
    for backwards compatibility. `control_posterior_exact` and `control_filter_exact`
    provide a clean ablation without any EBM at all.
    """
    if variant == "control_posterior_exact" or variant.endswith("_posterior"):
        return batch[f"{prefix}posterior_logits"]
    if variant == "control_filter_exact" or variant.endswith("_filter"):
        return batch[f"{prefix}filter_logits"]
    if variant in {"ours_ebm_control", "ours_ebm_cross"}:
        return batch[f"{prefix}filter_logits"]
    raise ValueError(f"Unknown variant for history selection: {variant}")


def nes_actor_belief_spec(variant: str, belief_cfg: BeliefConfig) -> Tuple[str, int]:
    feature_mode = belief_cfg.feature_mode
    modal_top_k = belief_cfg.modal_top_k
    if belief_cfg.nes_cross_compact_belief and variant_uses_cross_ebm(variant):
        feature_mode = "moments"
        modal_top_k = min(max(belief_cfg.modal_top_k, 1), 2)
    if belief_cfg.nes_actor_feature_mode:
        feature_mode = belief_cfg.nes_actor_feature_mode
    if belief_cfg.nes_actor_modal_top_k > 0:
        modal_top_k = belief_cfg.nes_actor_modal_top_k
    return feature_mode, modal_top_k

def actor_aligned_belief_diagnostics(
    variant: str,
    belief_cfg: BeliefConfig,
    exact_probs: torch.Tensor,
    pred_probs: torch.Tensor,
    theta_bank: torch.Tensor,
    env: GenericBankBOEDEnv,
    true_theta: torch.Tensor,
) -> Dict[str, float]:
    actor_feature_mode, actor_modal_top_k = nes_actor_belief_spec(variant, belief_cfg)
    exact_feat = belief_features_from_probs(
        probs=exact_probs,
        theta_bank=theta_bank,
        A_scalar=None,
        feature_mode=actor_feature_mode,
        modal_top_k=actor_modal_top_k,
    )
    pred_feat = belief_features_from_probs(
        probs=pred_probs,
        theta_bank=theta_bank,
        A_scalar=None,
        feature_mode=actor_feature_mode,
        modal_top_k=actor_modal_top_k,
    )
    pred_map = theta_bank[torch.argmax(pred_probs, dim=-1)]
    exact_map = theta_bank[torch.argmax(exact_probs, dim=-1)]
    return {
        "actor_belief_feature_mae": float(torch.mean(torch.abs(pred_feat - exact_feat)).detach().cpu()),
        "actor_belief_prob_l1": float(torch.abs(exact_probs - pred_probs).sum(dim=-1).mean().detach().cpu()),
        "actor_belief_map_to_exact_distance": float(env.belief_distance(pred_map, exact_map).mean().detach().cpu()),
        "actor_belief_map_to_true_distance": float(env.belief_distance(pred_map, true_theta).mean().detach().cpu()),
    }

def compute_state_from_batch(
    variant: str,
    filter_backbone: CachedFilterBackbone,
    batch: Dict[str, torch.Tensor],
    env: GenericBankBOEDEnv,
    energy_net: Optional[nn.Module],
    apsi_head: Optional[nn.Module],
    belief_cfg: Optional[BeliefConfig] = None,
    use_next: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    belief_cfg = belief_cfg or BeliefConfig()
    prefix = "next_" if use_next else ""
    last_obs = batch[f"{prefix}last_obs"]
    t_idx = batch[f"{prefix}t_idx"]
    aux_state = batch[f"{prefix}aux_state"]
    actions = batch[f"{prefix}actions"]
    obs = batch[f"{prefix}obs"]

    if variant == "blau_approx":
        raw_state = build_raw_history_state(actions, obs, t_idx, env.get_horizon(), last_obs, aux_state)
        return raw_state, raw_state, None, None

    selected_logits = history_logits_from_batch(variant, batch, prefix)
    hist_feat = filter_backbone.forward_from_logits(selected_logits)
    quotient_base = build_base_state(hist_feat, t_idx, env.get_horizon(), last_obs, aux_state)

    # Exact control ablations: same quotient-style policy state, but no EBM.
    if variant in {"control_filter_exact", "control_posterior_exact"}:
        return quotient_base, hist_feat, None, None

    if energy_net is None or apsi_head is None or belief_cfg.mode == "exact":
        return quotient_base, hist_feat, None, None

    energy = energy_net(hist_feat, env.hypothesis_bank)
    A = apsi_head(hist_feat)
    probs = posterior_probs_from_energy(energy)
    actor_feature_mode, actor_modal_top_k = nes_actor_belief_spec(variant, belief_cfg)
    belief = belief_features_from_probs(
        probs=probs,
        theta_bank=env.hypothesis_bank,
        A_scalar=A,
        feature_mode=actor_feature_mode,
        modal_top_k=actor_modal_top_k,
    )
    raw_prefix = build_raw_history_state(actions, obs, t_idx, env.get_horizon(), last_obs, aux_state) if getattr(belief_cfg, "include_raw_history_for_ebm_actor", False) else None
    if belief_cfg.mode == "distilled_detached":
        belief = belief.detach()
        base_state = torch.cat([quotient_base, belief], dim=-1)
        state = sanitize(torch.cat([raw_prefix, base_state], dim=-1), 1e3) if raw_prefix is not None else sanitize(base_state, 1e3)
    elif belief_cfg.mode == "distilled_e2e":
        base_state = torch.cat([quotient_base, belief], dim=-1)
        state = sanitize(torch.cat([raw_prefix, base_state], dim=-1), 1e3) if raw_prefix is not None else sanitize(base_state, 1e3)
    elif belief_cfg.mode == "learned_only":
        minimal_base = build_minimal_state(t_idx, env.get_horizon(), last_obs, aux_state)
        base_state = torch.cat([minimal_base, belief], dim=-1)
        state = sanitize(torch.cat([raw_prefix, base_state], dim=-1), 1e3) if raw_prefix is not None else sanitize(base_state, 1e3)
    else:
        raise ValueError(f"Unknown belief mode: {belief_cfg.mode}")
    return state, hist_feat, energy, A



def discrete_bank_ig_from_logits(filter_logits: np.ndarray, prior_bank_logits: torch.Tensor) -> float:
    logits_t = torch.tensor(filter_logits, dtype=torch.float32, device=prior_bank_logits.device)
    log_post = F.log_softmax(logits_t, dim=-1)
    post = torch.exp(log_post)
    return float((post * (log_post - prior_bank_logits)).sum().detach().cpu())



def estimate_spce_prefix(
    env: GenericBankBOEDEnv,
    actions: np.ndarray,
    obs: np.ndarray,
    true_theta: np.ndarray,
    L: int,
) -> float:
    theta0 = torch.tensor(true_theta[None], dtype=torch.float32, device=env.device)
    ll_true = env.trajectory_loglik_thetas(actions, obs, theta0)[0]
    ctr = env.sample_prior_thetas(L)
    ll_ctr = env.trajectory_loglik_thetas(actions, obs, ctr)
    all_ll = torch.cat([ll_true.view(1), ll_ctr], dim=0)
    est = math.log(L + 1.0) + ll_true - torch.logsumexp(all_ll, dim=0)
    return float(est.detach().cpu())



def estimate_snmc_style_upper_prefix(
    env: GenericBankBOEDEnv,
    actions: np.ndarray,
    obs: np.ndarray,
    true_theta: np.ndarray,
    L: int,
) -> float:
    theta0 = torch.tensor(true_theta[None], dtype=torch.float32, device=env.device)
    ll_true = env.trajectory_loglik_thetas(actions, obs, theta0)[0]
    nested = env.sample_prior_thetas(L)
    ll_nested = env.trajectory_loglik_thetas(actions, obs, nested)
    log_marg_est = logmeanexp_t(ll_nested, dim=0)
    est = ll_true - log_marg_est
    return float(est.detach().cpu())



def _actor_hparams_for_variant(variant: str, train_cfg: GenericTrainConfig) -> Tuple[str, int, int, bool]:
    if variant_uses_ebm(variant):
        family = train_cfg.ebm_actor_family or train_cfg.actor_family
        hidden = int(train_cfg.ebm_hidden_rl) if int(train_cfg.ebm_hidden_rl) > 0 else int(train_cfg.hidden_rl)
        comps = int(train_cfg.ebm_actor_mixture_components) if int(train_cfg.ebm_actor_mixture_components) > 0 else int(train_cfg.actor_mixture_components)
        return family, hidden, comps, bool(train_cfg.ebm_dual_branch_actor)
    return train_cfg.actor_family, int(train_cfg.hidden_rl), int(train_cfg.actor_mixture_components), False


def _make_continuous_actor_for_variant(
    variant: str,
    state_dim: int,
    base_dim: int,
    belief_dim: int,
    action_dim: int,
    train_cfg: GenericTrainConfig,
    device: torch.device,
) -> nn.Module:
    family, hidden, comps, dual = _actor_hparams_for_variant(variant, train_cfg)
    if dual:
        if family != "mog":
            raise ValueError("--ebm-dual-branch-actor currently requires --ebm-actor-family mog")
        return DualBranchMixtureTanhGaussianActor(
            base_dim=base_dim, belief_dim=belief_dim, action_dim=action_dim, hidden=hidden,
            dropout=train_cfg.actor_dropout, n_components=comps,
            phase_adaptive=train_cfg.phase_adaptive_actor, phase_start_frac=train_cfg.phase_start_frac,
            phase_strength=train_cfg.phase_strength, late_std_scale=train_cfg.late_std_scale,
            late_mix_temp=train_cfg.late_mix_temp,
        ).to(device)
    if family == "transformer":
        return SequenceTransformerTanhGaussianActor(
            state_dim=state_dim, action_dim=action_dim, horizon=getattr(train_cfg, "sequence_horizon", 30),
            hidden=hidden, dropout=train_cfg.actor_dropout, n_components=comps,
            d_model=train_cfg.transformer_d_model, nhead=train_cfg.transformer_nhead,
            num_layers=train_cfg.transformer_layers, dim_feedforward=train_cfg.transformer_ff,
            phase_adaptive=train_cfg.phase_adaptive_actor, phase_start_frac=train_cfg.phase_start_frac,
            phase_strength=train_cfg.phase_strength, late_std_scale=train_cfg.late_std_scale,
            late_mix_temp=train_cfg.late_mix_temp,
        ).to(device)
    if family == "mog":
        return MixtureTanhGaussianActor(
            state_dim=state_dim, action_dim=action_dim, hidden=hidden, dropout=train_cfg.actor_dropout,
            n_components=comps, phase_adaptive=train_cfg.phase_adaptive_actor,
            phase_start_frac=train_cfg.phase_start_frac, phase_strength=train_cfg.phase_strength,
            late_std_scale=train_cfg.late_std_scale, late_mix_temp=train_cfg.late_mix_temp,
        ).to(device)
    if family == "gaussian":
        return TanhGaussianActor(
            state_dim=state_dim, action_dim=action_dim, hidden=hidden, dropout=train_cfg.actor_dropout,
            phase_adaptive=train_cfg.phase_adaptive_actor, phase_start_frac=train_cfg.phase_start_frac,
            phase_strength=train_cfg.phase_strength, late_std_scale=train_cfg.late_std_scale,
        ).to(device)
    raise ValueError(f"Unknown actor_family={family!r}")


def build_modules(
    variant: str,
    env: GenericBankBOEDEnv,
    train_cfg: GenericTrainConfig,
    device: torch.device,
    belief_cfg: Optional[BeliefConfig] = None,
):
    belief_cfg = belief_cfg or BeliefConfig()
    filter_backbone = CachedFilterBackbone().to(device)
    hist_dim = env.H
    aux_dim = int(env.current_aux_state().shape[0])
    quotient_base_state_dim = hist_dim + 1 + 1 + aux_dim
    raw_history_state_dim = env.get_horizon() * env.action_dim + env.get_horizon() + 1 + 1 + aux_dim
    minimal_base_state_dim = 1 + 1 + aux_dim
    energy_net = None
    apsi_head = None
    belief_dim = 0

    if variant_uses_ebm(variant) and belief_cfg.mode != "exact":
        use_cross = variant_uses_cross_ebm(variant)
        if belief_cfg.ebm_architecture == "geometric":
            source_dim = belief_cfg.source_dim or (env.theta_dim // max(belief_cfg.n_sources, 1))
            if belief_cfg.n_sources * source_dim != env.theta_dim:
                raise ValueError(
                    f"Geometric EBM requires n_sources*source_dim == theta_dim, "
                    f"got {belief_cfg.n_sources}*{source_dim} != {env.theta_dim}"
                )
            ebm_cls = SymmetricSourceCrossNet if use_cross else SymmetricSourceEnergyNet
            energy_net = ebm_cls(
                hist_dim=hist_dim,
                theta_dim=env.theta_dim,
                hidden=train_cfg.hidden_ebm,
                n_sources=belief_cfg.n_sources,
                add_pairwise_dist=belief_cfg.add_pairwise_dist,
            ).to(device)
        else:
            ebm_cls = CrossInteractionEnergyNet if use_cross else EnergyNet
            energy_net = ebm_cls(hist_dim=hist_dim, theta_dim=env.theta_dim, hidden=train_cfg.hidden_ebm).to(device)
        apsi_head = ApsiHead(hist_dim=hist_dim, hidden=train_cfg.hidden_ebm).to(device)
        actor_feature_mode, actor_modal_top_k = nes_actor_belief_spec(variant, belief_cfg)
        belief_dim = belief_feature_dim(env.theta_dim, actor_feature_mode, actor_modal_top_k)
    elif variant not in {
        "blau_approx",
        "control_filter_exact",
        "control_posterior_exact",
        "ours_ebm_control",
        "ours_ebm_cross",
        "ours_ebm_control_filter",
        "ours_ebm_control_posterior",
        "ours_ebm_cross_filter",
        "ours_ebm_cross_posterior",
    }:
        raise ValueError(f"Unknown variant: {variant}")

    actor_family, actor_hidden, actor_comps, _ = _actor_hparams_for_variant(variant, train_cfg)
    if variant == "blau_approx":
        state_dim = raw_history_state_dim
    elif variant_uses_ebm(variant) and actor_family == "transformer":
        base_no_path = minimal_base_state_dim if belief_cfg.mode == "learned_only" and belief_dim > 0 else quotient_base_state_dim
        state_dim = raw_history_state_dim + base_no_path + belief_dim
    elif belief_cfg.mode == "learned_only" and belief_dim > 0:
        state_dim = minimal_base_state_dim + belief_dim
    else:
        state_dim = quotient_base_state_dim + belief_dim
    base_dim_for_actor = minimal_base_state_dim if belief_cfg.mode == "learned_only" and belief_dim > 0 else quotient_base_state_dim
    if uses_discrete_actor(env):
        action_values = get_discrete_action_values(env, device)
        actor = DiscreteCategoricalActor(state_dim=state_dim, num_actions=action_values.shape[0], hidden=actor_hidden).to(device)
    else:
        actor = _make_continuous_actor_for_variant(
            variant=variant, state_dim=state_dim, base_dim=base_dim_for_actor, belief_dim=belief_dim,
            action_dim=env.action_dim, train_cfg=train_cfg, device=device,
        )
    q1 = QCritic(state_dim=state_dim, action_dim=env.action_dim, hidden=train_cfg.hidden_rl).to(device)
    q2 = QCritic(state_dim=state_dim, action_dim=env.action_dim, hidden=train_cfg.hidden_rl).to(device)
    q1_tgt = QCritic(state_dim=state_dim, action_dim=env.action_dim, hidden=train_cfg.hidden_rl).to(device)
    q2_tgt = QCritic(state_dim=state_dim, action_dim=env.action_dim, hidden=train_cfg.hidden_rl).to(device)
    q1_tgt.load_state_dict(q1.state_dict())
    q2_tgt.load_state_dict(q2.state_dict())
    actor_params = list(actor.parameters())
    if (
        energy_net is not None
        and apsi_head is not None
        and belief_cfg.mode in {"distilled_e2e", "learned_only"}
    ):
        actor_params += list(energy_net.parameters()) + list(apsi_head.parameters())
    actor_optim = torch.optim.Adam(actor_params, lr=train_cfg.lr_actor, weight_decay=train_cfg.actor_weight_decay)
    critic_optim = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=train_cfg.lr_critic)
    ebm_optim = None
    if energy_net is not None and apsi_head is not None:
        ebm_optim = torch.optim.Adam(list(energy_net.parameters()) + list(apsi_head.parameters()), lr=train_cfg.lr_ebm)
    return filter_backbone, actor, q1, q2, q1_tgt, q2_tgt, actor_optim, critic_optim, energy_net, apsi_head, ebm_optim




@dataclass
class NESConfig:
    generations: int = 200
    population_size: int = 48
    rollout_episodes_per_candidate: int = 2
    eval_episodes: int = 150
    lr_mu: float = 0.04
    lr_sigma: float = 0.10
    sigma_init: float = 0.03
    sigma_final: float = 0.005
    sigma_schedule: str = "exp"  # constant | linear | exp
    mirrored_sampling: bool = True
    utility_mode: str = "nes"    # nes | centered_ranks
    optimizer: str = "adam"      # adam | rmsprop | sgd
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    sigma_adapt_on_success: bool = False
    sigma_success_target: float = 0.20
    sigma_adapt_rate: float = 0.05
    print_every: int = 10
    device: str = "cpu"
    seeds: str = "0,1,2"
    selection_eval_episodes: int = 40
    selection_every: int = 10
    selection_start_generation: int = 10
    selection_top_k: int = 3
    selection_final_eval_episodes: int = 0
    selection_return_weight: float = 1.0
    selection_belief_kl_weight: float = 0.05
    selection_belief_map_weight: float = 0.15
    selection_belief_mean_weight: float = 0.25
    # Cross-only model selection refinement: reward exact-metric checkpoints instead
    # of over-valuing beliefs that look clean only in the filtered geometry. Blau and
    # other non-cross variants keep the historical selection score unchanged.
    cross_selection_bank_ig_weight: float = 0.30
    cross_selection_spce_weight: float = 0.20
    cross_selection_filter_bank_ig_weight: float = 0.00
    cross_selection_gap_penalty_weight: float = 0.35
    cross_selection_belief_kl_weight: float = 0.02
    cross_selection_belief_map_weight: float = 0.04
    cross_selection_belief_mean_weight: float = 0.04
    ebm_updates_per_generation: int = 10
    ebm_batch_size: int = 128
    ebm_data_episodes: int = 8
    ebm_pretrain_episodes: int = 0
    ebm_pretrain_updates: int = 0
    ebm_update_every_generations: int = 1
    freeze_ebm: bool = False
    ebm_freeze_after_generation: int = -1
    parallel_candidates: bool = False
    n_jobs: int = 1
    parallel_backend: str = "threading"  # threading | loky
    use_common_random_numbers: bool = False
    common_random_numbers_seed_stride: int = 1000003
    reevaluate_top_candidates: int = 0
    reevaluate_top_episodes: int = 0


class ParamOptimizer:
    def __init__(self, shape: torch.Size, cfg: NESConfig, lr: float, device: torch.device):
        self.kind = cfg.optimizer
        self.lr = float(lr)
        self.beta1 = float(cfg.beta1)
        self.beta2 = float(cfg.beta2)
        self.eps = float(cfg.eps)
        self.m = torch.zeros(shape, device=device)
        self.v = torch.zeros(shape, device=device)
        self.t = 0

    def step(self, params: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        if self.kind == "sgd":
            return params + self.lr * grad
        if self.kind == "rmsprop":
            self.v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            return params + self.lr * grad / (self.v.sqrt() + self.eps)
        if self.kind == "adam":
            self.t += 1
            self.m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            self.v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
            m_hat = self.m / (1.0 - self.beta1 ** self.t)
            v_hat = self.v / (1.0 - self.beta2 ** self.t)
            return params + self.lr * m_hat / (v_hat.sqrt() + self.eps)
        raise ValueError(f"Unknown optimizer={self.kind!r}")




class SingleStateTensorBuffer:
    """Preallocated single-sample tensors reused in rollout and eval hot paths."""

    def __init__(self, env: GenericBankBOEDEnv, device: torch.device):
        self.device = device
        self.last_obs = torch.empty((1, 1), dtype=torch.float32, device=device)
        self.t_idx = torch.empty((1,), dtype=torch.long, device=device)
        self.aux_state = torch.empty((1, int(env.current_aux_state().shape[0])), dtype=torch.float32, device=device)
        self.actions = torch.empty((1, env.get_horizon(), env.action_dim), dtype=torch.float32, device=device)
        self.obs = torch.empty((1, env.get_horizon()), dtype=torch.float32, device=device)
        self.posterior_logits = torch.empty((1, env.H), dtype=torch.float32, device=device)
        self.filter_logits = torch.empty((1, env.H), dtype=torch.float32, device=device)
        self.posterior = torch.empty((1, env.H), dtype=torch.float32, device=device)
        self.filter_probs = torch.empty((1, env.H), dtype=torch.float32, device=device)

    def update(self, raw: Dict, need_probs: bool = False) -> Dict[str, torch.Tensor]:
        self.last_obs[0, 0] = float(raw["last_obs"])
        self.t_idx[0] = int(raw["t_idx"])
        if self.aux_state.numel() > 0:
            self.aux_state[0].copy_(torch.from_numpy(raw["aux_state"]).to(self.device))
        self.actions[0].copy_(torch.from_numpy(raw["actions"]).to(self.device))
        self.obs[0].copy_(torch.from_numpy(raw["obs"]).to(self.device))
        self.posterior_logits[0].copy_(torch.from_numpy(raw["posterior_logits"]).to(self.device))
        self.filter_logits[0].copy_(torch.from_numpy(raw["filter_logits"]).to(self.device))
        batch = {
            "last_obs": self.last_obs,
            "t_idx": self.t_idx,
            "aux_state": self.aux_state,
            "actions": self.actions,
            "obs": self.obs,
            "posterior_logits": self.posterior_logits,
            "filter_logits": self.filter_logits,
        }
        if need_probs:
            self.posterior[0].copy_(torch.from_numpy(raw["posterior"]).to(self.device))
            if "filter_probs" in raw:
                self.filter_probs[0].copy_(torch.from_numpy(raw["filter_probs"]).to(self.device))
            batch["posterior"] = self.posterior
            batch["filter_probs"] = self.filter_probs
        return batch


def _centered_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    ranks = np.empty_like(x)
    order = np.argsort(x)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    ranks /= max(len(x) - 1, 1)
    ranks -= 0.5
    return ranks


def _nes_utilities(n: int) -> np.ndarray:
    ranks = np.arange(1, n + 1, dtype=np.float64)
    util = np.maximum(0.0, np.log(n / 2.0 + 1.0) - np.log(ranks))
    util /= util.sum()
    util -= 1.0 / n
    return util


def _schedule_sigma(gen: int, total: int, init_sigma: float, final_sigma: float, mode: str) -> float:
    if mode == "constant" or total <= 1:
        return float(init_sigma)
    frac = min(max((gen - 1) / max(total - 1, 1), 0.0), 1.0)
    if mode == "linear":
        return float(init_sigma + frac * (final_sigma - init_sigma))
    if mode == "exp":
        if init_sigma <= 0 or final_sigma <= 0:
            raise ValueError("exp sigma schedule requires positive sigma_init and sigma_final")
        return float(init_sigma * ((final_sigma / init_sigma) ** frac))
    raise ValueError(f"Unknown sigma_schedule={mode!r}")


def _ebm_is_frozen(nes_cfg: NESConfig, gen: Optional[int] = None) -> bool:
    if nes_cfg.freeze_ebm:
        return True
    if gen is None:
        return False
    return nes_cfg.ebm_freeze_after_generation >= 0 and gen >= nes_cfg.ebm_freeze_after_generation


def _selection_score(
    variant: str,
    eval_metrics: Dict[str, float],
    cfg: NESConfig,
    cross_diag_metrics: Optional[Dict[str, float]] = None,
) -> float:
    # Historical criterion for Blau / non-cross variants remains unchanged.
    if not variant_uses_cross_ebm(variant):
        return float(cfg.selection_return_weight * float(eval_metrics.get("avg_return", 0.0)))

    # Cross-specific selection: only head-to-head exact-task metrics drive model
    # selection. Cross-only diagnostics are logged separately and can be used as
    # very light tie-breakers, but they should not dominate the exact objective.
    score = cfg.selection_return_weight * float(eval_metrics.get("avg_return", 0.0))
    score += cfg.cross_selection_bank_ig_weight * float(eval_metrics.get("avg_bank_ig", 0.0))
    score += cfg.cross_selection_spce_weight * float(eval_metrics.get("avg_spce_lower", 0.0))
    score += cfg.cross_selection_filter_bank_ig_weight * float(eval_metrics.get("avg_filter_bank_ig", 0.0))

    filter_bank = float(eval_metrics.get("avg_filter_bank_ig", 0.0))
    exact_bank = float(eval_metrics.get("avg_bank_ig", 0.0))
    score -= cfg.cross_selection_gap_penalty_weight * max(filter_bank - exact_bank, 0.0)

    if cross_diag_metrics is not None:
        if "avg_actor_belief_map_to_exact_distance" in cross_diag_metrics:
            score -= cfg.cross_selection_belief_map_weight * float(cross_diag_metrics["avg_actor_belief_map_to_exact_distance"])
        if "avg_actor_belief_feature_mae" in cross_diag_metrics:
            score -= cfg.cross_selection_belief_mean_weight * float(cross_diag_metrics["avg_actor_belief_feature_mae"])
        if "avg_actor_belief_prob_l1" in cross_diag_metrics:
            score -= cfg.cross_selection_belief_kl_weight * float(cross_diag_metrics["avg_actor_belief_prob_l1"])
    return float(score)


def _flatten_params(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def _set_flat_params(module: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in module.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n


def _clone_module_state(module: Optional[nn.Module]) -> Optional[Dict[str, torch.Tensor]]:
    if module is None:
        return None
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _load_module_state(module: Optional[nn.Module], state: Optional[Dict[str, torch.Tensor]]) -> None:
    if module is None or state is None:
        return
    module.load_state_dict(state)


def _build_policy_modules(
    variant: str,
    env: GenericBankBOEDEnv,
    train_cfg: GenericTrainConfig,
    device: torch.device,
    belief_cfg: Optional[BeliefConfig] = None,
):
    belief_cfg = belief_cfg or BeliefConfig()
    filter_backbone = CachedFilterBackbone().to(device)
    hist_dim = env.H
    aux_dim = int(env.current_aux_state().shape[0])
    quotient_base_state_dim = hist_dim + 1 + 1 + aux_dim
    raw_history_state_dim = env.get_horizon() * env.action_dim + env.get_horizon() + 1 + 1 + aux_dim
    minimal_base_state_dim = 1 + 1 + aux_dim
    energy_net = None
    apsi_head = None
    belief_dim = 0

    if variant_uses_ebm(variant) and belief_cfg.mode != "exact":
        use_cross = variant_uses_cross_ebm(variant)
        if belief_cfg.ebm_architecture == "geometric":
            source_dim = belief_cfg.source_dim or (env.theta_dim // max(belief_cfg.n_sources, 1))
            if belief_cfg.n_sources * source_dim != env.theta_dim:
                raise ValueError(
                    f"Geometric EBM requires n_sources*source_dim == theta_dim, got {belief_cfg.n_sources}*{source_dim} != {env.theta_dim}"
                )
            ebm_cls = SymmetricSourceCrossNet if use_cross else SymmetricSourceEnergyNet
            energy_net = ebm_cls(
                hist_dim=hist_dim,
                theta_dim=env.theta_dim,
                hidden=train_cfg.hidden_ebm,
                n_sources=belief_cfg.n_sources,
                add_pairwise_dist=belief_cfg.add_pairwise_dist,
            ).to(device)
        else:
            ebm_cls = CrossInteractionEnergyNet if use_cross else EnergyNet
            energy_net = ebm_cls(hist_dim=hist_dim, theta_dim=env.theta_dim, hidden=train_cfg.hidden_ebm).to(device)
        apsi_head = ApsiHead(hist_dim=hist_dim, hidden=train_cfg.hidden_ebm).to(device)
        actor_feature_mode, actor_modal_top_k = nes_actor_belief_spec(variant, belief_cfg)
        belief_dim = belief_feature_dim(env.theta_dim, actor_feature_mode, actor_modal_top_k)

    actor_family, actor_hidden, actor_comps, _ = _actor_hparams_for_variant(variant, train_cfg)
    if variant == "blau_approx":
        state_dim = raw_history_state_dim
    elif variant_uses_ebm(variant) and actor_family == "transformer":
        base_no_path = minimal_base_state_dim if belief_cfg.mode == "learned_only" and belief_dim > 0 else quotient_base_state_dim
        state_dim = raw_history_state_dim + base_no_path + belief_dim
    elif belief_cfg.mode == "learned_only" and belief_dim > 0:
        state_dim = minimal_base_state_dim + belief_dim
    else:
        state_dim = quotient_base_state_dim + belief_dim
    base_dim_for_actor = minimal_base_state_dim if belief_cfg.mode == "learned_only" and belief_dim > 0 else quotient_base_state_dim
    if uses_discrete_actor(env):
        action_values = get_discrete_action_values(env, device)
        actor = DiscreteCategoricalActor(state_dim=state_dim, num_actions=action_values.shape[0], hidden=actor_hidden).to(device)
    else:
        actor = _make_continuous_actor_for_variant(
            variant=variant, state_dim=state_dim, base_dim=base_dim_for_actor, belief_dim=belief_dim,
            action_dim=env.action_dim, train_cfg=train_cfg, device=device,
        )
    return filter_backbone, actor, energy_net, apsi_head


def compute_state_from_batch_nes(
    variant: str,
    filter_backbone: CachedFilterBackbone,
    batch: Dict[str, torch.Tensor],
    env: GenericBankBOEDEnv,
    energy_net: Optional[nn.Module],
    apsi_head: Optional[nn.Module],
    belief_cfg: Optional[BeliefConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    belief_cfg = belief_cfg or BeliefConfig()
    last_obs = batch["last_obs"]
    t_idx = batch["t_idx"]
    aux_state = batch["aux_state"]
    actions = batch["actions"]
    obs = batch["obs"]

    if variant == "blau_approx":
        raw_state = build_raw_history_state(actions, obs, t_idx, env.get_horizon(), last_obs, aux_state)
        return raw_state, raw_state, None, None

    selected_logits = history_logits_from_batch(variant, batch, "")
    hist_feat = filter_backbone.forward_from_logits(selected_logits)
    quotient_base = build_base_state(hist_feat, t_idx, env.get_horizon(), last_obs, aux_state)

    if variant in {"control_filter_exact", "control_posterior_exact"}:
        return quotient_base, hist_feat, None, None

    if energy_net is None or apsi_head is None or belief_cfg.mode == "exact":
        return quotient_base, hist_feat, None, None

    energy = energy_net(hist_feat, env.hypothesis_bank)
    A = apsi_head(hist_feat)
    probs = posterior_probs_from_energy(energy)
    actor_feature_mode, actor_modal_top_k = nes_actor_belief_spec(variant, belief_cfg)
    belief = belief_features_from_probs(
        probs=probs,
        theta_bank=env.hypothesis_bank,
        A_scalar=A,
        feature_mode=actor_feature_mode,
        modal_top_k=actor_modal_top_k,
    )
    raw_prefix = build_raw_history_state(actions, obs, t_idx, env.get_horizon(), last_obs, aux_state) if getattr(belief_cfg, "include_raw_history_for_ebm_actor", False) else None
    if belief_cfg.mode == "distilled_detached":
        belief = belief.detach()
        base_state = torch.cat([quotient_base, belief], dim=-1)
        state = sanitize(torch.cat([raw_prefix, base_state], dim=-1), 1e3) if raw_prefix is not None else sanitize(base_state, 1e3)
    elif belief_cfg.mode == "distilled_e2e":
        base_state = torch.cat([quotient_base, belief], dim=-1)
        state = sanitize(torch.cat([raw_prefix, base_state], dim=-1), 1e3) if raw_prefix is not None else sanitize(base_state, 1e3)
    elif belief_cfg.mode == "learned_only":
        minimal_base = build_minimal_state(t_idx, env.get_horizon(), last_obs, aux_state)
        base_state = torch.cat([minimal_base, belief], dim=-1)
        state = sanitize(torch.cat([raw_prefix, base_state], dim=-1), 1e3) if raw_prefix is not None else sanitize(base_state, 1e3)
    else:
        raise ValueError(f"Unknown belief mode: {belief_cfg.mode}")
    return state, hist_feat, energy, A


def _episode_rollout(
    variant: str,
    env_factory: Callable[[torch.device], GenericBankBOEDEnv],
    filter_backbone: CachedFilterBackbone,
    actor: nn.Module,
    energy_net: Optional[nn.Module],
    apsi_head: Optional[nn.Module],
    device: torch.device,
    belief_cfg: Optional[BeliefConfig],
    deterministic: bool = False,
    collect_raw_states: bool = False,
    episode_seed: Optional[int] = None,
) -> Tuple[float, List[Dict]]:
    if episode_seed is not None:
        set_seed(int(episode_seed))
    env = env_factory(device)
    action_scale = env.action_scale
    action_bias = env.action_bias
    discrete_action_values = get_discrete_action_values(env, device) if uses_discrete_actor(env) else None

    modules = [filter_backbone, actor]
    if energy_net is not None:
        modules.append(energy_net)
    if apsi_head is not None:
        modules.append(apsi_head)
    prev_modes = [m.training for m in modules]
    for m in modules:
        m.eval()

    raw_states: List[Dict] = []
    try:
        env.reset()
        need_probs = bool(collect_raw_states)
        raw = make_raw_state(env, need_probs=need_probs)
        single = SingleStateTensorBuffer(env, device)
        done = False
        total_return = 0.0
        while not done:
            if collect_raw_states:
                raw_states.append(dict(raw))
            batch = single.update(raw, need_probs=False)
            state_t, _, _, _ = compute_state_from_batch_nes(
                variant, filter_backbone, batch, env, energy_net, apsi_head, belief_cfg=belief_cfg
            )
            time_frac = batch["t_idx"].float() / float(max(env.get_horizon(), 1))
            with torch.no_grad():
                if discrete_action_values is not None:
                    act, _, det = actor.sample(state_t, action_values=discrete_action_values, time_frac=time_frac)
                else:
                    act, _, det = actor.sample(state_t, action_scale=action_scale, action_bias=action_bias, time_frac=time_frac)
            chosen = det if deterministic else act
            action = chosen.squeeze(0).detach().cpu().numpy().astype(np.float32)
            _, reward, done, _ = env.step(action)
            raw = make_raw_state(env, need_probs=need_probs)
            total_return += float(reward)
    finally:
        for m, mode in zip(modules, prev_modes):
            m.train(mode)

    return total_return, raw_states


def _batch_from_raw_states(raw_states: Sequence[Dict], device: torch.device) -> Dict[str, torch.Tensor]:
    def arr(key: str) -> np.ndarray:
        return np.stack([rs[key] for rs in raw_states], axis=0)

    return {
        "last_obs": torch.tensor([[rs["last_obs"]] for rs in raw_states], dtype=torch.float32, device=device),
        "t_idx": torch.tensor([rs["t_idx"] for rs in raw_states], dtype=torch.long, device=device),
        "aux_state": torch.tensor(arr("aux_state"), dtype=torch.float32, device=device),
        "actions": torch.tensor(arr("actions"), dtype=torch.float32, device=device),
        "obs": torch.tensor(arr("obs"), dtype=torch.float32, device=device),
        "posterior": torch.tensor(arr("posterior"), dtype=torch.float32, device=device) if all("posterior" in rs for rs in raw_states) else torch.softmax(torch.tensor(arr("posterior_logits"), dtype=torch.float32, device=device), dim=-1),
        "posterior_logits": torch.tensor(arr("posterior_logits"), dtype=torch.float32, device=device),
        "filter_logits": torch.tensor(arr("filter_logits"), dtype=torch.float32, device=device),
    }


def _collect_random_raw_states(
    env_factory: Callable[[torch.device], GenericBankBOEDEnv],
    device: torch.device,
    n_episodes: int,
) -> List[Dict]:
    collected: List[Dict] = []
    if n_episodes <= 0:
        return collected
    env = env_factory(device)
    for _ in range(n_episodes):
        env.reset()
        done = False
        while not done:
            raw = make_raw_state(env, need_probs=True)
            collected.append(raw)
            if uses_discrete_actor(env):
                vals = get_discrete_action_values(env, device)
                idx = np.random.randint(0, vals.shape[0])
                action = vals[idx].detach().cpu().numpy().astype(np.float32)
            else:
                action = np.random.uniform(
                    env.action_low.detach().cpu().numpy(),
                    env.action_high.detach().cpu().numpy(),
                ).astype(np.float32)
            _, _, done, _ = env.step(action)
    return collected


def _supervised_update_ebm(
    variant: str,
    env: GenericBankBOEDEnv,
    filter_backbone: CachedFilterBackbone,
    energy_net: Optional[nn.Module],
    apsi_head: Optional[nn.Module],
    raw_states: Sequence[Dict],
    device: torch.device,
    belief_cfg: BeliefConfig,
    cfg: NESConfig,
) -> None:
    if energy_net is None or apsi_head is None or len(raw_states) == 0:
        return
    energy_net.train()
    apsi_head.train()
    optim = torch.optim.Adam(list(energy_net.parameters()) + list(apsi_head.parameters()), lr=3e-4)
    raw_states = list(raw_states)
    H = len(raw_states)
    bs = min(cfg.ebm_batch_size, H)
    for _ in range(max(cfg.ebm_updates_per_generation, 0)):
        idx = np.random.randint(0, H, size=bs)
        batch = _batch_from_raw_states([raw_states[i] for i in idx], device=device)
        selected_logits = history_logits_from_batch(variant, batch, "")
        hist_feat = filter_backbone.forward_from_logits(selected_logits)
        energy = energy_net(hist_feat, env.hypothesis_bank)
        A = apsi_head(hist_feat)
        log_probs = F.log_softmax(-energy, dim=-1)
        target_probs = batch["posterior"]
        loss = -(target_probs * log_probs).sum(dim=-1).mean() + 0.1 * A.pow(2).mean()
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(energy_net.parameters()) + list(apsi_head.parameters()), 10.0)
        optim.step()


def evaluate_modules_nes(
    variant: str,
    env_factory: Callable[[torch.device], GenericBankBOEDEnv],
    filter_backbone: CachedFilterBackbone,
    actor: nn.Module,
    energy_net: Optional[nn.Module],
    apsi_head: Optional[nn.Module],
    device: torch.device,
    train_cfg: GenericTrainConfig,
    spce_L: int,
    snmc_L: int,
    belief_cfg: Optional[BeliefConfig] = None,
    n_eval_episodes: Optional[int] = None,
    collect_paths: bool = True,
) -> Dict[str, object]:
    belief_cfg = belief_cfg or BeliefConfig()
    n_eval = int(n_eval_episodes if n_eval_episodes is not None else train_cfg.eval_episodes)
    env = env_factory(device)

    modules = [filter_backbone, actor]
    if energy_net is not None:
        modules.append(energy_net)
    if apsi_head is not None:
        modules.append(apsi_head)
    prev_modes = [m.training for m in modules]
    for m in modules:
        m.eval()

    eval_returns: List[float] = []
    actor_belief_feature_maes: List[float] = []
    actor_belief_prob_l1s: List[float] = []
    actor_belief_map_to_exact_errs: List[float] = []
    actor_belief_map_to_true_errs: List[float] = []
    full_belief_mean_errs: List[float] = []
    full_belief_kl_errs: List[float] = []
    full_belief_l1_errs: List[float] = []
    full_belief_map_to_exact_errs: List[float] = []
    full_belief_map_to_true_errs: List[float] = []
    bank_ig_finals: List[float] = []
    filter_bank_ig_finals: List[float] = []
    spce_lower_finals: List[float] = []
    snmc_upper_finals: List[float] = []
    bank_ig_paths: List[List[float]] = []
    filter_bank_ig_paths: List[List[float]] = []
    spce_paths: List[List[float]] = []
    snmc_paths: List[List[float]] = []

    try:
        for _ in range(n_eval):
            env.reset()
            need_probs = energy_net is not None
            raw = make_raw_state(env, need_probs=need_probs)
            single = SingleStateTensorBuffer(env, device)
            done = False
            ep_return = 0.0
            ep_bank_path: List[float] = []
            ep_filter_bank_path: List[float] = []
            ep_spce_path: List[float] = []
            ep_snmc_path: List[float] = []

            while not done:
                batch = single.update(raw, need_probs=need_probs)
                state_t, _, energy_t, _ = compute_state_from_batch_nes(
                    variant, filter_backbone, batch, env, energy_net, apsi_head, belief_cfg=belief_cfg
                )
                if energy_net is not None and energy_t is not None:
                    exact_probs_t = batch["posterior"]
                    pred_probs = posterior_probs_from_energy(energy_t)
                    exact_mean_t = exact_probs_t @ env.hypothesis_bank
                    pred_mean = pred_probs @ env.hypothesis_bank
                    full_belief_mean_errs.append(float(torch.mean(torch.abs(pred_mean - exact_mean_t)).detach().cpu()))
                    full_belief_kl_errs.append(float(((exact_probs_t.clamp_min(1e-12)) * (torch.log(exact_probs_t.clamp_min(1e-12)) - torch.log(pred_probs.clamp_min(1e-12)))).sum(dim=-1).mean().detach().cpu()))
                    full_belief_l1_errs.append(float(torch.abs(exact_probs_t - pred_probs).sum(dim=-1).mean().detach().cpu()))
                    pred_map = env.hypothesis_bank[torch.argmax(pred_probs, dim=-1)]
                    exact_map = env.hypothesis_bank[torch.argmax(exact_probs_t, dim=-1)]
                    true_theta_t = torch.tensor(env.theta0[None], dtype=torch.float32, device=device)
                    full_belief_map_to_exact_errs.append(float(env.belief_distance(pred_map, exact_map).mean().detach().cpu()))
                    full_belief_map_to_true_errs.append(float(env.belief_distance(pred_map, true_theta_t).mean().detach().cpu()))
                    diag = actor_aligned_belief_diagnostics(
                        variant=variant,
                        belief_cfg=belief_cfg,
                        exact_probs=exact_probs_t,
                        pred_probs=pred_probs,
                        theta_bank=env.hypothesis_bank,
                        env=env,
                        true_theta=true_theta_t,
                    )
                    actor_belief_feature_maes.append(diag["actor_belief_feature_mae"])
                    actor_belief_prob_l1s.append(diag["actor_belief_prob_l1"])
                    actor_belief_map_to_exact_errs.append(diag["actor_belief_map_to_exact_distance"])
                    actor_belief_map_to_true_errs.append(diag["actor_belief_map_to_true_distance"])
                time_frac = batch["t_idx"].float() / float(max(env.get_horizon(), 1))
                if uses_discrete_actor(env):
                    action_values = get_discrete_action_values(env, device)
                    _, _, det = actor.sample(state_t, action_values=action_values, time_frac=time_frac)
                else:
                    _, _, det = actor.sample(state_t, action_scale=env.action_scale, action_bias=env.action_bias, time_frac=time_frac)
                action = det.squeeze(0).detach().cpu().numpy().astype(np.float32)
                _, reward, done, _ = env.step(action)
                raw = make_raw_state(env, need_probs=need_probs)
                ep_return += reward
                prefix_actions = raw["actions"][: raw["length"]]
                prefix_obs = raw["obs"][: raw["length"]]
                ep_bank_path.append(discrete_bank_ig_from_logits(raw["posterior_logits"], env.prior_bank_logits))
                ep_filter_bank_path.append(discrete_bank_ig_from_logits(raw["filter_logits"], env.prior_bank_logits))
                ep_spce_path.append(estimate_spce_prefix(env, prefix_actions, prefix_obs, env.theta0, spce_L))
                if snmc_L > 0:
                    ep_snmc_path.append(estimate_snmc_style_upper_prefix(env, prefix_actions, prefix_obs, env.theta0, snmc_L))

            eval_returns.append(ep_return)
            bank_ig_finals.append(ep_bank_path[-1])
            filter_bank_ig_finals.append(ep_filter_bank_path[-1])
            spce_lower_finals.append(ep_spce_path[-1])
            if snmc_L > 0 and ep_snmc_path:
                snmc_upper_finals.append(ep_snmc_path[-1])
            if collect_paths:
                bank_ig_paths.append(ep_bank_path)
                filter_bank_ig_paths.append(ep_filter_bank_path)
                spce_paths.append(ep_spce_path)
                if snmc_L > 0 and ep_snmc_path:
                    snmc_paths.append(ep_snmc_path)
    finally:
        for m, mode in zip(modules, prev_modes):
            m.train(mode)

    eval_dict: Dict[str, float] = {
        "avg_return": float(np.mean(eval_returns)),
        "std_return": float(np.std(eval_returns, ddof=1)) if len(eval_returns) > 1 else 0.0,
        "avg_bank_ig": float(np.mean(bank_ig_finals)),
        "avg_filter_bank_ig": float(np.mean(filter_bank_ig_finals)),
        "avg_spce_lower": float(np.mean(spce_lower_finals)),
    }
    if snmc_L > 0 and snmc_upper_finals:
        eval_dict["avg_snmc_style_upper"] = float(np.mean(snmc_upper_finals))
    out: Dict[str, object] = {"eval": eval_dict}
    if actor_belief_feature_maes:
        out["cross_diagnostics"] = {
            "avg_actor_belief_feature_mae": float(np.mean(actor_belief_feature_maes)),
            "avg_actor_belief_prob_l1": float(np.mean(actor_belief_prob_l1s)),
            "avg_actor_belief_map_to_exact_distance": float(np.mean(actor_belief_map_to_exact_errs)),
            "avg_actor_belief_map_to_true_distance": float(np.mean(actor_belief_map_to_true_errs)),
            "avg_full_belief_mean_error": float(np.mean(full_belief_mean_errs)),
            "avg_full_belief_kl": float(np.mean(full_belief_kl_errs)),
            "avg_full_belief_l1": float(np.mean(full_belief_l1_errs)),
            "avg_full_belief_map_to_exact_distance": float(np.mean(full_belief_map_to_exact_errs)),
            "avg_full_belief_map_to_true_distance": float(np.mean(full_belief_map_to_true_errs)),
        }

    if collect_paths:
        paths = {
            "bank_ig_mean_path": np.mean(np.array(bank_ig_paths, dtype=np.float64), axis=0).tolist(),
            "filter_bank_ig_mean_path": np.mean(np.array(filter_bank_ig_paths, dtype=np.float64), axis=0).tolist(),
            "spce_lower_mean_path": np.mean(np.array(spce_paths, dtype=np.float64), axis=0).tolist(),
        }
        if snmc_L > 0 and snmc_paths:
            paths["snmc_style_upper_mean_path"] = np.mean(np.array(snmc_paths, dtype=np.float64), axis=0).tolist()
        out["paths"] = paths
    return out


def _candidate_return_from_flat(
    flat_params_cpu: np.ndarray,
    variant: str,
    env_factory: Callable[[torch.device], GenericBankBOEDEnv],
    train_cfg: GenericTrainConfig,
    device_str: str,
    belief_cfg: Optional[BeliefConfig],
    filter_state: Optional[Dict[str, torch.Tensor]],
    actor_state: Optional[Dict[str, torch.Tensor]],
    energy_state: Optional[Dict[str, torch.Tensor]],
    apsi_state: Optional[Dict[str, torch.Tensor]],
    stochastic: bool,
    rollout_episodes: int,
    episode_seeds: Optional[Sequence[int]] = None,
) -> float:
    device = torch.device(device_str)
    env = env_factory(device)
    filter_backbone, actor, energy_net, apsi_head = _build_policy_modules(variant, env, train_cfg, device, belief_cfg=belief_cfg)
    _load_module_state(filter_backbone, filter_state)
    _load_module_state(actor, actor_state)
    _load_module_state(energy_net, energy_state)
    _load_module_state(apsi_head, apsi_state)
    flat = torch.tensor(flat_params_cpu, dtype=torch.float32, device=device)
    _set_flat_params(actor, flat)
    vals = []
    for ep in range(rollout_episodes):
        ep_seed = None if episode_seeds is None else int(episode_seeds[ep])
        ret, _ = _episode_rollout(
            variant, env_factory, filter_backbone, actor, energy_net, apsi_head, device,
            belief_cfg=belief_cfg, deterministic=not stochastic, collect_raw_states=False, episode_seed=ep_seed
        )
        vals.append(ret)
    return float(np.mean(vals))




def _generation_episode_seeds(base_seed: int, gen: int, count: int, stride: int) -> List[int]:
    start = int(base_seed) * int(stride) + int(gen) * int(stride)
    return [start + i for i in range(int(count))]


def train_one_seed_nes(
    variant: str,
    env_factory: Callable[[torch.device], GenericBankBOEDEnv],
    train_cfg: GenericTrainConfig,
    nes_cfg: NESConfig,
    seed: int,
    output_dir: str,
    spce_L: int,
    snmc_L: int,
    belief_cfg: Optional[BeliefConfig] = None,
) -> Dict:
    set_seed(seed)
    device = torch.device(nes_cfg.device if nes_cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    env = env_factory(device)
    filter_backbone, actor, energy_net, apsi_head = _build_policy_modules(variant, env, train_cfg, device, belief_cfg=belief_cfg)

    if energy_net is not None and apsi_head is not None and (not _ebm_is_frozen(nes_cfg, None)):
        pre_states = _collect_random_raw_states(env_factory, device, nes_cfg.ebm_pretrain_episodes)
        if pre_states:
            pre_cfg = NESConfig(**{**nes_cfg.__dict__})
            if nes_cfg.ebm_pretrain_updates > 0:
                pre_cfg.ebm_updates_per_generation = nes_cfg.ebm_pretrain_updates
            _supervised_update_ebm(
                variant=variant,
                env=env,
                filter_backbone=filter_backbone,
                energy_net=energy_net,
                apsi_head=apsi_head,
                raw_states=pre_states,
                device=device,
                belief_cfg=belief_cfg or BeliefConfig(),
                cfg=pre_cfg,
            )

    mu = _flatten_params(actor).to(device)
    mu_opt = ParamOptimizer(mu.shape, nes_cfg, nes_cfg.lr_mu, device)
    log_sigma = torch.tensor(math.log(max(nes_cfg.sigma_init, 1e-8)), dtype=torch.float32, device=device)
    sigma_offset = 0.0

    pop = int(nes_cfg.population_size)
    if nes_cfg.mirrored_sampling and pop % 2 != 0:
        raise ValueError("population_size must be even when mirrored_sampling=True")
    half = pop // 2 if nes_cfg.mirrored_sampling else pop
    dim = int(mu.numel())
    if nes_cfg.use_common_random_numbers and nes_cfg.parallel_candidates and nes_cfg.parallel_backend == "threading" and nes_cfg.n_jobs > 1:
        print(f"[{env.name}:{variant}:nes] seed={seed} common random numbers disable threaded parallel candidate eval for correctness")
    effective_parallel = bool(nes_cfg.parallel_candidates) and not (
        nes_cfg.use_common_random_numbers and nes_cfg.parallel_backend == "threading" and nes_cfg.n_jobs > 1
    )

    generation_returns: List[float] = []
    sigma_history: List[float] = []
    success_history: List[float] = []

    best_score = -float("inf")
    best_generation = 0
    best_actor_state = _clone_module_state(actor)
    best_energy_state = _clone_module_state(energy_net)
    best_apsi_state = _clone_module_state(apsi_head)
    best_eval_preview: Optional[Dict[str, float]] = None
    top_candidates: List[Dict[str, object]] = []

    def maybe_run_selection(gen_no: int) -> None:
        nonlocal best_score, best_generation, best_actor_state, best_energy_state, best_apsi_state, best_eval_preview, top_candidates
        if nes_cfg.selection_eval_episodes <= 0:
            return
        if gen_no < nes_cfg.selection_start_generation:
            return
        if gen_no % nes_cfg.selection_every != 0 and gen_no != nes_cfg.generations:
            return
        preview_bundle = evaluate_modules_nes(
            variant=variant,
            env_factory=env_factory,
            filter_backbone=filter_backbone,
            actor=actor,
            energy_net=energy_net,
            apsi_head=apsi_head,
            device=device,
            train_cfg=train_cfg,
            spce_L=spce_L,
            snmc_L=snmc_L,
            belief_cfg=belief_cfg,
            n_eval_episodes=nes_cfg.selection_eval_episodes,
            collect_paths=False,
        )
        preview = preview_bundle["eval"]
        preview_diag = preview_bundle.get("cross_diagnostics")
        score = _selection_score(variant, preview, nes_cfg, cross_diag_metrics=preview_diag)
        cand = {
            "score": float(score),
            "generation": int(gen_no),
            "actor_state": _clone_module_state(actor),
            "energy_state": _clone_module_state(energy_net),
            "apsi_state": _clone_module_state(apsi_head),
            "preview_eval": dict(preview),
            "preview_cross_diagnostics": dict(preview_diag) if preview_diag is not None else None,
        }
        top_candidates.append(cand)
        top_candidates.sort(key=lambda d: float(d["score"]), reverse=True)
        keep_k = max(int(nes_cfg.selection_top_k), 1)
        top_candidates = top_candidates[:keep_k]
        if score > best_score:
            best_score = score
            best_generation = gen_no
            best_actor_state = cand["actor_state"]
            best_energy_state = cand["energy_state"]
            best_apsi_state = cand["apsi_state"]
            best_eval_preview = dict(preview)

    for gen in range(1, nes_cfg.generations + 1):
        base_sigma = _schedule_sigma(gen, nes_cfg.generations, nes_cfg.sigma_init, nes_cfg.sigma_final, nes_cfg.sigma_schedule)
        sigma = float(base_sigma * math.exp(sigma_offset))
        sigma = max(sigma, 1e-6)
        sigma_history.append(sigma)
        generation_episode_seeds = None
        if nes_cfg.use_common_random_numbers:
            seed_count = max(
                int(nes_cfg.rollout_episodes_per_candidate),
                int(nes_cfg.reevaluate_top_episodes),
                1,
            )
            generation_episode_seeds = _generation_episode_seeds(seed, gen, seed_count, nes_cfg.common_random_numbers_seed_stride)
        candidate_episode_seeds = None if generation_episode_seeds is None else generation_episode_seeds[: int(nes_cfg.rollout_episodes_per_candidate)]

        z_pos = torch.randn(half, dim, device=device)
        if nes_cfg.mirrored_sampling:
            z_all = torch.cat([z_pos, -z_pos], dim=0)
        else:
            z_all = z_pos
        candidates = mu.unsqueeze(0) + sigma * z_all

        filter_state = _clone_module_state(filter_backbone)
        actor_state = _clone_module_state(actor)
        energy_state = _clone_module_state(energy_net)
        apsi_state = _clone_module_state(apsi_head)

        if effective_parallel:
            backend = nes_cfg.parallel_backend
            n_jobs = nes_cfg.n_jobs
            fits = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_candidate_return_from_flat)(
                    flat.detach().cpu().numpy(),
                    variant,
                    env_factory,
                    train_cfg,
                    str(device),
                    belief_cfg,
                    filter_state,
                    actor_state,
                    energy_state,
                    apsi_state,
                    True,
                    nes_cfg.rollout_episodes_per_candidate,
                    candidate_episode_seeds,
                )
                for flat in candidates
            )
            fitness = np.asarray(fits, dtype=np.float64)
        else:
            fitness = np.zeros(pop, dtype=np.float64)
            for i in range(pop):
                _set_flat_params(actor, candidates[i])
                vals = []
                for ep in range(nes_cfg.rollout_episodes_per_candidate):
                    ep_seed = None if candidate_episode_seeds is None else int(candidate_episode_seeds[ep])
                    ret, _ = _episode_rollout(
                        variant, env_factory, filter_backbone, actor, energy_net, apsi_head, device,
                        belief_cfg=belief_cfg, deterministic=False, collect_raw_states=False, episode_seed=ep_seed
                    )
                    vals.append(ret)
                fitness[i] = float(np.mean(vals))
            _set_flat_params(actor, mu)

        if nes_cfg.reevaluate_top_candidates > 0 and nes_cfg.reevaluate_top_episodes > 0 and (
            nes_cfg.reevaluate_top_episodes != nes_cfg.rollout_episodes_per_candidate or not nes_cfg.use_common_random_numbers
        ):
            top_n = min(int(nes_cfg.reevaluate_top_candidates), pop)
            if top_n > 0:
                top_idx = np.argsort(-fitness)[:top_n]
                reeval_episode_seeds = None if generation_episode_seeds is None else generation_episode_seeds[: int(nes_cfg.reevaluate_top_episodes)]
                if effective_parallel:
                    reeval = Parallel(n_jobs=nes_cfg.n_jobs, backend=nes_cfg.parallel_backend)(
                        delayed(_candidate_return_from_flat)(
                            candidates[i].detach().cpu().numpy(),
                            variant,
                            env_factory,
                            train_cfg,
                            str(device),
                            belief_cfg,
                            filter_state,
                            actor_state,
                            energy_state,
                            apsi_state,
                            True,
                            nes_cfg.reevaluate_top_episodes,
                            reeval_episode_seeds,
                        )
                        for i in top_idx
                    )
                    fitness[top_idx] = np.asarray(reeval, dtype=np.float64)
                else:
                    for i in top_idx:
                        _set_flat_params(actor, candidates[i])
                        vals = []
                        for ep in range(nes_cfg.reevaluate_top_episodes):
                            ep_seed = None if reeval_episode_seeds is None else int(reeval_episode_seeds[ep])
                            ret, _ = _episode_rollout(
                                variant, env_factory, filter_backbone, actor, energy_net, apsi_head, device,
                                belief_cfg=belief_cfg, deterministic=False, collect_raw_states=False, episode_seed=ep_seed
                            )
                            vals.append(ret)
                        fitness[i] = float(np.mean(vals))
                    _set_flat_params(actor, mu)

        if nes_cfg.utility_mode == "centered_ranks":
            utilities = _centered_ranks(fitness)
        else:
            order = np.argsort(-fitness)
            u_sorted = _nes_utilities(pop)
            utilities = np.empty(pop, dtype=np.float64)
            utilities[order] = u_sorted

        util_t = torch.tensor(utilities, dtype=torch.float32, device=device)
        z_mean = torch.sum(util_t[:, None] * z_all, dim=0)
        grad_mu = sigma * z_mean

        sq_term = (z_all.pow(2).mean(dim=1) - 1.0)
        grad_log_sigma = torch.sum(util_t * sq_term)

        mu = mu_opt.step(mu, grad_mu)
        log_sigma = log_sigma + nes_cfg.lr_sigma * grad_log_sigma

        if nes_cfg.sigma_adapt_on_success:
            center_fit = _candidate_return_from_flat(
                mu.detach().cpu().numpy(),
                variant,
                env_factory,
                train_cfg,
                str(device),
                belief_cfg,
                filter_state,
                actor_state,
                energy_state,
                apsi_state,
                False,
                max(1, nes_cfg.rollout_episodes_per_candidate),
                candidate_episode_seeds,
            )
            success = float(np.mean(fitness > center_fit))
            sigma_offset += nes_cfg.sigma_adapt_rate * (success - nes_cfg.sigma_success_target)
        else:
            success = float(np.mean(fitness > fitness.mean()))
        success_history.append(success)

        sigma_from_grad = float(log_sigma.exp().detach().cpu())
        # blend explicit sigma dynamics with schedule instead of letting one dominate entirely
        sigma = max(1e-6, math.sqrt(base_sigma * sigma_from_grad) * math.exp(sigma_offset))
        log_sigma = torch.tensor(math.log(sigma), dtype=torch.float32, device=device)

        _set_flat_params(actor, mu)
        generation_returns.append(float(fitness.mean()))

        if energy_net is not None and apsi_head is not None and (not _ebm_is_frozen(nes_cfg, gen)) and nes_cfg.ebm_data_episodes > 0 and (gen % max(nes_cfg.ebm_update_every_generations, 1) == 0):
            collected: List[Dict] = []
            for _ in range(nes_cfg.ebm_data_episodes):
                _, states = _episode_rollout(
                    variant, env_factory, filter_backbone, actor, energy_net, apsi_head, device,
                    belief_cfg=belief_cfg, deterministic=False, collect_raw_states=True
                )
                collected.extend(states)
            _supervised_update_ebm(
                variant=variant,
                env=env,
                filter_backbone=filter_backbone,
                energy_net=energy_net,
                apsi_head=apsi_head,
                raw_states=collected,
                device=device,
                belief_cfg=belief_cfg or BeliefConfig(),
                cfg=nes_cfg,
            )

        if gen % nes_cfg.print_every == 0:
            print(f"[{env.name}:{variant}:nes] seed={seed} gen={gen}/{nes_cfg.generations} mean_fit={fitness.mean():.4f} sigma={sigma:.5f} success={success:.3f}")
        maybe_run_selection(gen)

    last_eval_bundle = evaluate_modules_nes(
        variant=variant,
        env_factory=env_factory,
        filter_backbone=filter_backbone,
        actor=actor,
        energy_net=energy_net,
        apsi_head=apsi_head,
        device=device,
        train_cfg=train_cfg,
        spce_L=spce_L,
        snmc_L=snmc_L,
        belief_cfg=belief_cfg,
        n_eval_episodes=nes_cfg.eval_episodes,
        collect_paths=False,
    )

    final_eval_eps = int(nes_cfg.selection_final_eval_episodes) if int(nes_cfg.selection_final_eval_episodes) > 0 else int(nes_cfg.eval_episodes)
    final_candidates: List[Dict[str, object]] = []
    if top_candidates:
        for cand in top_candidates:
            _load_module_state(actor, cand["actor_state"])
            _load_module_state(energy_net, cand["energy_state"])
            _load_module_state(apsi_head, cand["apsi_state"])
            final_bundle = evaluate_modules_nes(
                variant=variant,
                env_factory=env_factory,
                filter_backbone=filter_backbone,
                actor=actor,
                energy_net=energy_net,
                apsi_head=apsi_head,
                device=device,
                train_cfg=train_cfg,
                spce_L=spce_L,
                snmc_L=snmc_L,
                belief_cfg=belief_cfg,
                n_eval_episodes=final_eval_eps,
                collect_paths=False,
            )
            final_eval = final_bundle["eval"]
            final_diag = final_bundle.get("cross_diagnostics")
            final_candidates.append({
                "generation": int(cand["generation"]),
                "preview_score": float(cand["score"]),
                "preview_eval": dict(cand["preview_eval"]),
                "preview_cross_diagnostics": cand.get("preview_cross_diagnostics"),
                "final_eval": dict(final_eval),
                "final_cross_diagnostics": dict(final_diag) if final_diag is not None else None,
                "final_score": float(_selection_score(variant, final_eval, nes_cfg, cross_diag_metrics=final_diag)),
                "actor_state": cand["actor_state"],
                "energy_state": cand["energy_state"],
                "apsi_state": cand["apsi_state"],
            })
        final_candidates.sort(key=lambda d: float(d["final_score"]), reverse=True)
        chosen = final_candidates[0]
        best_generation = int(chosen["generation"])
        best_score = float(chosen["final_score"])
        best_actor_state = chosen["actor_state"]
        best_energy_state = chosen["energy_state"]
        best_apsi_state = chosen["apsi_state"]
        best_eval_preview = dict(chosen["preview_eval"])
    _load_module_state(actor, best_actor_state)
    _load_module_state(energy_net, best_energy_state)
    _load_module_state(apsi_head, best_apsi_state)

    selected_bundle = evaluate_modules_nes(
        variant=variant,
        env_factory=env_factory,
        filter_backbone=filter_backbone,
        actor=actor,
        energy_net=energy_net,
        apsi_head=apsi_head,
        device=device,
        train_cfg=train_cfg,
        spce_L=spce_L,
        snmc_L=snmc_L,
        belief_cfg=belief_cfg,
        n_eval_episodes=nes_cfg.eval_episodes,
        collect_paths=True,
    )

    out: Dict[str, object] = {
        "train": {
            "episode_returns": generation_returns,
            "sigma_history": sigma_history,
            "success_rate_history": success_history,
        },
        "eval": selected_bundle["eval"],
        "eval_last": last_eval_bundle["eval"],
        "cross_diagnostics": selected_bundle.get("cross_diagnostics"),
        "cross_diagnostics_last": last_eval_bundle.get("cross_diagnostics"),
        "selection": {
            "best_generation": int(best_generation if best_generation > 0 else nes_cfg.generations),
            "best_score": float(best_score if best_score > -float("inf") else _selection_score(variant, last_eval_bundle["eval"], nes_cfg, cross_diag_metrics=last_eval_bundle.get("cross_diagnostics"))),
            "best_preview_eval": best_eval_preview if best_eval_preview is not None else dict(last_eval_bundle["eval"]),
            "selection_eval_episodes": int(nes_cfg.selection_eval_episodes),
            "selection_final_eval_episodes": int(final_eval_eps),
            "selection_every": int(nes_cfg.selection_every),
            "selection_start_generation": int(nes_cfg.selection_start_generation),
            "selection_top_k": int(max(nes_cfg.selection_top_k, 1)),
            "top_candidates": [
                {
                    "generation": int(c["generation"]),
                    "preview_score": float(c["preview_score"]),
                    "preview_eval": c["preview_eval"],
                    "preview_cross_diagnostics": c.get("preview_cross_diagnostics"),
                    "final_score": float(c["final_score"]),
                    "final_eval": c["final_eval"],
                    "final_cross_diagnostics": c.get("final_cross_diagnostics"),
                }
                for c in final_candidates
            ],
        },
        "variant": variant,
        "seed": seed,
        "paths": selected_bundle["paths"],
    }
    seed_dir = os.path.join(output_dir, variant, f"seed_{seed}")
    ensure_dir(seed_dir)
    with open(os.path.join(seed_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def run_experiment_suite_nes(
    experiment_name: str,
    env_factory: Callable[[torch.device], GenericBankBOEDEnv],
    output_dir: str,
    train_cfg: GenericTrainConfig,
    nes_cfg: NESConfig,
    seeds: Sequence[int],
    variants: Sequence[str],
    spce_L: int,
    snmc_L: int,
    belief_cfg: Optional[BeliefConfig] = None,
) -> Dict:
    ensure_dir(output_dir)
    all_results: Dict[str, List[Dict]] = {}
    sample_env = env_factory(torch.device(nes_cfg.device if nes_cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")))

    for variant in variants:
        print("\n============================")
        print(f"NES {experiment_name} variant: {variant}")
        print("============================")
        variant_results = []
        for seed in seeds:
            variant_results.append(train_one_seed_nes(variant, env_factory, train_cfg, nes_cfg, seed, output_dir, spce_L, snmc_L, belief_cfg=belief_cfg))
        all_results[variant] = variant_results

    summary: Dict[str, object] = {}
    tracked_fields = [
        "avg_return",
        "avg_bank_ig",
        "avg_filter_bank_ig",
        "avg_spce_lower",
        "avg_snmc_style_upper",
    ]
    cross_diag_fields = [
        "avg_actor_belief_feature_mae",
        "avg_actor_belief_prob_l1",
        "avg_actor_belief_map_to_exact_distance",
        "avg_actor_belief_map_to_true_distance",
        "avg_full_belief_mean_error",
        "avg_full_belief_kl",
        "avg_full_belief_l1",
        "avg_full_belief_map_to_exact_distance",
        "avg_full_belief_map_to_true_distance",
    ]
    for variant, results in all_results.items():
        for field in tracked_fields:
            vals = [r["eval"][field] for r in results if field in r["eval"]]
            if vals:
                summary[f"{variant}_{field}"] = mean_std_ci95(np.array(vals, dtype=np.float64))
            vals_last = [r["eval_last"][field] for r in results if field in r.get("eval_last", {})]
            if vals_last:
                summary[f"{variant}_last_{field}"] = mean_std_ci95(np.array(vals_last, dtype=np.float64))
        for field in cross_diag_fields:
            vals = [r["cross_diagnostics"][field] for r in results if r.get("cross_diagnostics") and field in r["cross_diagnostics"]]
            if vals:
                summary[f"{variant}_{field}"] = mean_std_ci95(np.array(vals, dtype=np.float64))
            vals_last = [r["cross_diagnostics_last"][field] for r in results if r.get("cross_diagnostics_last") and field in r["cross_diagnostics_last"]]
            if vals_last:
                summary[f"{variant}_last_{field}"] = mean_std_ci95(np.array(vals_last, dtype=np.float64))
    if "blau_approx" in all_results:
        blau = np.array([r["eval"]["avg_return"] for r in all_results["blau_approx"]], dtype=np.float64)
        for variant in variants:
            if variant == "blau_approx":
                continue
            cur = np.array([r["eval"]["avg_return"] for r in all_results[variant]], dtype=np.float64)
            summary[f"paired_return_diff_{variant}_minus_blau_approx"] = paired_summary(blau, cur)

    summary["experiment_name"] = experiment_name
    summary["variants"] = list(variants)
    summary["seeds"] = list(seeds)
    summary["horizon"] = sample_env.get_horizon()
    summary["spce_L"] = spce_L
    summary["snmc_L"] = snmc_L
    if belief_cfg is not None:
        summary["belief_config"] = {
            "mode": belief_cfg.mode,
            "feature_mode": belief_cfg.feature_mode,
            "ebm_architecture": belief_cfg.ebm_architecture,
            "n_sources": belief_cfg.n_sources,
            "source_dim": belief_cfg.source_dim,
            "add_pairwise_dist": belief_cfg.add_pairwise_dist,
            "modal_top_k": belief_cfg.modal_top_k,
            "include_raw_history_for_ebm_actor": belief_cfg.include_raw_history_for_ebm_actor,
        }
    summary["nes_config"] = {
        "generations": nes_cfg.generations,
        "population_size": nes_cfg.population_size,
        "rollout_episodes_per_candidate": nes_cfg.rollout_episodes_per_candidate,
        "lr_mu": nes_cfg.lr_mu,
        "lr_sigma": nes_cfg.lr_sigma,
        "sigma_init": nes_cfg.sigma_init,
        "sigma_final": nes_cfg.sigma_final,
        "sigma_schedule": nes_cfg.sigma_schedule,
        "utility_mode": nes_cfg.utility_mode,
        "optimizer": nes_cfg.optimizer,
        "mirrored_sampling": nes_cfg.mirrored_sampling,
        "ebm_updates_per_generation": nes_cfg.ebm_updates_per_generation,
        "ebm_data_episodes": nes_cfg.ebm_data_episodes,
        "ebm_pretrain_episodes": nes_cfg.ebm_pretrain_episodes,
        "ebm_pretrain_updates": nes_cfg.ebm_pretrain_updates,
        "ebm_update_every_generations": nes_cfg.ebm_update_every_generations,
        "cross_selection_bank_ig_weight": nes_cfg.cross_selection_bank_ig_weight,
        "cross_selection_spce_weight": nes_cfg.cross_selection_spce_weight,
        "cross_selection_filter_bank_ig_weight": nes_cfg.cross_selection_filter_bank_ig_weight,
        "cross_selection_gap_penalty_weight": nes_cfg.cross_selection_gap_penalty_weight,
        "cross_selection_belief_kl_weight": nes_cfg.cross_selection_belief_kl_weight,
        "cross_selection_belief_map_weight": nes_cfg.cross_selection_belief_map_weight,
        "cross_selection_belief_mean_weight": nes_cfg.cross_selection_belief_mean_weight,
        "ebm_freeze_after_generation": nes_cfg.ebm_freeze_after_generation,
        "actor_family": train_cfg.actor_family,
        "actor_mixture_components": train_cfg.actor_mixture_components,
        "hidden_rl": train_cfg.hidden_rl,
        "ebm_actor_family": train_cfg.ebm_actor_family,
        "ebm_hidden_rl": train_cfg.ebm_hidden_rl,
        "ebm_actor_mixture_components": train_cfg.ebm_actor_mixture_components,
        "ebm_dual_branch_actor": train_cfg.ebm_dual_branch_actor,
        "transformer_d_model": train_cfg.transformer_d_model,
        "transformer_nhead": train_cfg.transformer_nhead,
        "transformer_layers": train_cfg.transformer_layers,
        "transformer_ff": train_cfg.transformer_ff,
    }

    with open(os.path.join(output_dir, "summary_multi_seed.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_standard_plots(output_dir, all_results, summary, sample_env.get_horizon())
    print("\n=== Multi-seed summary ===")
    print(json.dumps(summary, indent=2))
    return summary


# Backward-compatible aliases for experiment script.
EvolutionConfig = NESConfig
run_experiment_suite_evolution = run_experiment_suite_nes

