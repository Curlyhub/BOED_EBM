"""
Neural network modules used by all BOEDX variants.

Module hierarchy
----------------
Layer 0 – base utilities (``sanitize`` from utils)
Layer 1 – EBM energy heads (EnergyNet, CrossInteractionEnergyNet)
Layer 2 – geometry-aware EBM heads for K-source problems
           (SymmetricSourceEnergyNet, SymmetricSourceCrossNet)
           Both gain an eval-mode theta-encoding cache via _ThetaEncodingCacheMixin.
Layer 3 – composite EBM wrapper (BetaContrastiveEnergyNet)
Layer 4 – policy actors
           Continuous (SAC + NES): TanhGaussianActor, MixtureTanhGaussianActor
           NES-only:  SequenceTransformerTanhGaussianActor,
                      DualBranchMixtureTanhGaussianActor
           Discrete:  DiscreteCategoricalActor, DiscreteMoECategoricalActor
Layer 5 – Q-critics (QCritic)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from boedx.utils import sanitize


# ---------------------------------------------------------------------------
# Filter backbone (shared across all variants)
# ---------------------------------------------------------------------------

class CachedFilterBackbone(nn.Module):
    """Convert cached filter log-logits to a probability vector.

    The filter logits are updated online by the environment (one step at a
    time) and stored in the replay buffer.  This module converts them to
    normalised probabilities, which serve as the *history feature* for the
    quotient-state policy.
    """

    def forward_from_logits(self, cached_logits: torch.Tensor) -> torch.Tensor:
        return torch.exp(F.log_softmax(cached_logits, dim=-1))


# ---------------------------------------------------------------------------
# Generic EBM energy heads (1-D / non-multi-source problems)
# ---------------------------------------------------------------------------

class EnergyNet(nn.Module):
    """Standard MLP energy E(h, θ).

    Suitable for 1-D single-source problems and non-geometric latent spaces.
    For multi-source 2-D problems use ``SymmetricSourceEnergyNet`` instead.

    Args:
        hist_dim:  Dimension of the history feature vector h.
        theta_dim: Dimension of a single latent hypothesis θ.
        hidden:    Width of hidden layers.
    """

    def __init__(self, hist_dim: int, theta_dim: int, hidden: int = 128):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.theta_net = nn.Sequential(nn.Linear(theta_dim, hidden), nn.ReLU())
        self.out = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hist_feat: torch.Tensor, theta_bank: torch.Tensor) -> torch.Tensor:
        """Return energy matrix of shape (B, H)."""
        B = hist_feat.shape[0]
        H = theta_bank.shape[0]
        s = sanitize(self.state_net(hist_feat), 100.0)[:, None, :].expand(B, H, -1)
        th = sanitize(self.theta_net(theta_bank), 100.0).unsqueeze(0).expand(B, H, -1)
        e = self.out(torch.cat([s, th], dim=-1)).squeeze(-1)
        return sanitize(e, 50.0)


class CrossInteractionEnergyNet(nn.Module):
    """EBM energy with explicit cross-interaction features [s, θ, s·θ, |s-θ|].

    The element-wise interaction terms help the network discover alignment
    between history features and hypothesis coordinates without requiring the
    outer MLP to learn it implicitly.
    """

    def __init__(self, hist_dim: int, theta_dim: int, hidden: int = 128):
        super().__init__()
        self.state_net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.theta_net = nn.Sequential(
            nn.Linear(theta_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
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


# ---------------------------------------------------------------------------
# Geometry-aware EBM heads for K-source localisation
# ---------------------------------------------------------------------------
#
# Motivation (2-D source localisation):
#
#   (A) Source-swap symmetry.  μ(θ₁, θ₂, d) = μ(θ₂, θ₁, d).
#       The posterior is permutation-invariant under (θ₁ ↔ θ₂).
#       A generic MLP over raw ℝ⁴ must waste capacity learning this.
#
#   (B) Distance-based likelihood.  Per-source intensity ∝ 1/(m + ‖θᵢ-d‖²).
#       What drives the likelihood is Euclidean distance, not raw coords.
#
#   (C) Per-source factorisation.  μ is a sum of per-source terms.
#
# Both classes implement (A) and (C) via a Deep Sets aggregator
# (Zaheer et al., 2017) and inject geometric features related to (B).


def phase_gate(
    time_frac: Optional[torch.Tensor], start_frac: float
) -> Optional[torch.Tensor]:
    """Smooth late-horizon gate in [0, 1].

    Returns 0 before *start_frac* of the horizon, then smoothly rises to 1
    using a cubic Hermite interpolant (3x² − 2x³).  Returns None when
    *time_frac* is None so callers can skip the late-head computation entirely.

    Args:
        time_frac:  (B,) or (B, 1) tensor in [0, 1]; None disables gating.
        start_frac: Fraction of the horizon at which the gate begins to open.
    """
    if time_frac is None:
        return None
    tf = time_frac if time_frac.dim() > 1 else time_frac.unsqueeze(-1)
    tf = tf.clamp(0.0, 1.0)
    if start_frac >= 1.0:
        return torch.zeros_like(tf)
    x = ((tf - start_frac) / max(1.0 - start_frac, 1e-6)).clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class _ThetaEncodingCacheMixin:
    """Cache the theta-side encoding in eval mode for a fixed hypothesis bank.

    The hypothesis bank never changes during a single rollout or evaluation
    pass, so the costly per-source encoding can be computed once and reused
    across all time steps.  The cache is keyed by the tensor's storage pointer,
    device, dtype, shape, and source count, so it automatically invalidates
    whenever the bank is replaced.

    Only active in ``eval()`` mode — training forward passes always recompute.
    """

    def _init_theta_cache(self) -> None:
        self._theta_cache_key: Optional[Tuple] = None
        self._theta_cache_value: Optional[torch.Tensor] = None

    def _maybe_get_theta_cache(
        self, theta_bank: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.training:
            return None
        key = (
            theta_bank.data_ptr(),
            theta_bank.device,
            theta_bank.dtype,
            tuple(theta_bank.shape),
            self.n_sources,
        )
        if self._theta_cache_key == key and self._theta_cache_value is not None:
            return self._theta_cache_value
        return None

    def _store_theta_cache(
        self, theta_bank: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            return value
        key = (
            theta_bank.data_ptr(),
            theta_bank.device,
            theta_bank.dtype,
            tuple(theta_bank.shape),
            self.n_sources,
        )
        self._theta_cache_key = key
        self._theta_cache_value = value.detach()
        return value


class SymmetricSourceEnergyNet(_ThetaEncodingCacheMixin, nn.Module):
    """Permutation-invariant energy net for K-source localisation.

    Architecture (Deep Sets style):
      1. Each source θᵢ ∈ ℝ^{d_source} is augmented with norm and optional
         mean pairwise distances, then encoded by a **shared** MLP φ_θ.
         Shared weights + sum aggregation gives permutation invariance.
      2. History h is encoded by φ_h.
      3. Energy = out([φ_h(h),  Σᵢ φ_θ(θᵢ)]).

    The theta-side encoding is cached in eval mode for speed (see
    ``_ThetaEncodingCacheMixin``).

    Args:
        hist_dim:         Dimension of history feature vector.
        theta_dim:        Total θ dimension (K × d_source, e.g. 4 for K=2, d=2).
        hidden:           Hidden layer width.
        n_sources:        K (number of sources).
        add_pairwise_dist: Include mean pairwise distance as a per-source feature.
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
                f"theta_dim ({theta_dim}) must be divisible by n_sources ({n_sources})."
            )
        self.n_sources = int(n_sources)
        self.d_source = theta_dim // n_sources
        self.add_pairwise_dist = bool(add_pairwise_dist) and n_sources >= 2
        # Per-source feature: [θᵢ, ‖θᵢ‖, (optional) mean pairwise dist]
        feat_dim = self.d_source + 1 + (1 if self.add_pairwise_dist else 0)
        self._per_source_feat_dim = feat_dim
        self._init_theta_cache()

        self.state_net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # SHARED per-source encoder — this is what enforces permutation invariance
        self.theta_net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _augment_theta(self, theta_bank: torch.Tensor) -> torch.Tensor:
        """Augment raw coordinates with norm and pairwise distance features.

        Args:
            theta_bank: (H, K * d_source)
        Returns:
            (H, K, per_source_feat_dim)
        """
        H = theta_bank.shape[0]
        th = theta_bank.view(H, self.n_sources, self.d_source)
        norms = th.norm(dim=-1, keepdim=True)
        pieces = [th, norms]
        if self.add_pairwise_dist:
            # Mean Euclidean distance from source i to all other sources
            diff = th.unsqueeze(2) - th.unsqueeze(1)          # (H, K, K, d)
            pairwise = diff.norm(dim=-1)                       # (H, K, K)
            mean_other = pairwise.sum(dim=-1, keepdim=True) / max(self.n_sources - 1, 1)
            pieces.append(mean_other)
        return torch.cat(pieces, dim=-1)

    def forward(self, hist_feat: torch.Tensor, theta_bank: torch.Tensor) -> torch.Tensor:
        B = hist_feat.shape[0]
        H = theta_bank.shape[0]
        th_agg = self._maybe_get_theta_cache(theta_bank)
        if th_agg is None:
            th_feat = self._augment_theta(theta_bank)
            # Shared per-source encoding then Deep Sets sum aggregation
            th_enc = sanitize(self.theta_net(th_feat), 100.0)
            th_agg = self._store_theta_cache(theta_bank, th_enc.sum(dim=1))
        s = sanitize(self.state_net(hist_feat), 100.0)
        s_exp = s[:, None, :].expand(B, H, -1)
        th_exp = th_agg[None, :, :].expand(B, H, -1)
        e = self.out(torch.cat([s_exp, th_exp], dim=-1)).squeeze(-1)
        return sanitize(e, 50.0)


class SymmetricSourceCrossNet(_ThetaEncodingCacheMixin, nn.Module):
    """Permutation-invariant energy net with cross-interaction fusion.

    Extends ``SymmetricSourceEnergyNet`` by applying the cross-interaction
    features [s, θ_agg, s·θ_agg, |s-θ_agg|] from ``CrossInteractionEnergyNet``
    to the *aggregated* source representation.  Permutation invariance is
    preserved because the interaction acts on the post-aggregation vector.
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
                f"theta_dim ({theta_dim}) must be divisible by n_sources ({n_sources})."
            )
        self.n_sources = int(n_sources)
        self.d_source = theta_dim // n_sources
        self.add_pairwise_dist = bool(add_pairwise_dist) and n_sources >= 2
        feat_dim = self.d_source + 1 + (1 if self.add_pairwise_dist else 0)
        self._per_source_feat_dim = feat_dim
        self._init_theta_cache()

        self.state_net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.theta_net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def _augment_theta(self, theta_bank: torch.Tensor) -> torch.Tensor:
        H = theta_bank.shape[0]
        th = theta_bank.view(H, self.n_sources, self.d_source)
        norms = th.norm(dim=-1, keepdim=True)
        pieces = [th, norms]
        if self.add_pairwise_dist:
            diff = th.unsqueeze(2) - th.unsqueeze(1)
            pairwise = diff.norm(dim=-1)
            mean_other = pairwise.sum(dim=-1, keepdim=True) / max(self.n_sources - 1, 1)
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


# ---------------------------------------------------------------------------
# Auxiliary scalar head
# ---------------------------------------------------------------------------

class ApsiHead(nn.Module):
    """Scalar ψ(h) auxiliary head used for the EBM partition-function baseline."""

    def __init__(self, hist_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, hist_feat: torch.Tensor) -> torch.Tensor:
        return sanitize(self.net(hist_feat), 50.0)


# ---------------------------------------------------------------------------
# Beta-contrastive hybrid EBM
# ---------------------------------------------------------------------------

class BetaContrastiveEnergyNet(nn.Module):
    """Hybrid EBM blending a posterior branch and a contrastive branch.

    The energy at time-step t is:

        E_t(θ) = β_t · E_post(θ | h_t)  +  (1−β_t) · E_ctr(θ | h_t, c_t)

    where:
      - β_t is supplied externally and typically decays from ≈1 to ≈0.
      - h_t is the quotient history feature (posterior filter probabilities).
      - c_t is a compact summary derived from the running contrastive particle set.

    This allows early-trajectory behaviour to be driven by the posterior (low
    variance but biased to the discrete bank) and late-trajectory behaviour to
    incorporate the richer contrastive signal.
    """

    def __init__(
        self,
        hist_dim: int,
        contrastive_dim: int,
        theta_dim: int,
        hidden: int = 128,
        use_cross: bool = False,
        ebm_architecture: str = "standard",
        n_sources: int = 1,
        add_pairwise_dist: bool = True,
    ):
        super().__init__()
        if ebm_architecture == "geometric":
            ebm_cls = SymmetricSourceCrossNet if use_cross else SymmetricSourceEnergyNet
            kwargs: dict = dict(
                hist_dim=hist_dim,
                theta_dim=theta_dim,
                hidden=hidden,
                n_sources=n_sources,
                add_pairwise_dist=add_pairwise_dist,
            )
        else:
            ebm_cls = CrossInteractionEnergyNet if use_cross else EnergyNet
            kwargs = dict(hist_dim=hist_dim, theta_dim=theta_dim, hidden=hidden)

        self.posterior_net = ebm_cls(**kwargs)
        self.posterior_apsi = ApsiHead(hist_dim=hist_dim, hidden=hidden)
        # Adapter projects (h, c, β) → h-shaped vector for the contrastive EBM
        self.contrastive_adapter = nn.Sequential(
            nn.Linear(hist_dim + contrastive_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hist_dim), nn.ReLU(),
        )
        ctr_kwargs = dict(kwargs)
        ctr_kwargs["hist_dim"] = hist_dim
        self.contrastive_net = ebm_cls(**ctr_kwargs)
        self.contrastive_apsi = ApsiHead(hist_dim=hist_dim, hidden=hidden)

    def forward_beta(
        self,
        hist_feat: torch.Tensor,
        contrastive_feat: torch.Tensor,
        theta_bank: torch.Tensor,
        beta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the blended energy and scalar baseline.

        Returns:
            energy: (B, H) blended energy matrix.
            A:      (B, 1) blended scalar baseline.
            extras: dict with per-branch tensors for debugging / logging.
        """
        beta = beta.clamp(0.0, 1.0)
        if beta.dim() == 1:
            beta = beta.unsqueeze(-1)
        ctr_hist = sanitize(
            self.contrastive_adapter(torch.cat([hist_feat, contrastive_feat, beta], dim=-1)),
            100.0,
        )
        e_post = self.posterior_net(hist_feat, theta_bank)
        A_post = self.posterior_apsi(hist_feat)
        e_ctr = self.contrastive_net(ctr_hist, theta_bank)
        A_ctr = self.contrastive_apsi(ctr_hist)
        beta_h = beta.expand(-1, e_post.shape[-1])
        energy = sanitize(beta_h * e_post + (1.0 - beta_h) * e_ctr, 50.0)
        A = sanitize(beta * A_post + (1.0 - beta) * A_ctr, 50.0)
        extras = {
            "energy_post": e_post,
            "energy_ctr": e_ctr,
            "A_post": A_post,
            "A_ctr": A_ctr,
            "contrastive_hist": ctr_hist,
        }
        return energy, A, extras


# ---------------------------------------------------------------------------
# Policy actors — continuous
# ---------------------------------------------------------------------------

class TanhGaussianActor(nn.Module):
    """Single-Gaussian SAC actor with tanh squashing.

    Supports an optional late-horizon *phase-adaptive* residual head that is
    activated by a smooth cubic gate based on the normalised time ``time_frac``.
    When ``phase_adaptive=False`` the actor behaves exactly as the original
    single-Gaussian actor used by the RL trainer.

    Args:
        state_dim:       Dimension of the policy state input.
        action_dim:      Dimension of the action.
        hidden:          Hidden layer width.
        dropout:         Input dropout probability (0 = disabled).
        phase_adaptive:  Enable the late-horizon residual head.
        phase_start_frac: Fraction of the horizon at which the gate opens.
        phase_strength:  Multiplier on the residual correction.
        late_std_scale:  Multiplicative scale applied to std late in the horizon.
    """

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
        # Dropout on the input only; keeps belief tracking and Q-value stable
        self.input_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
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

    def forward(
        self,
        state: torch.Tensor,
        time_frac: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        """Reparameterised sample, log-prob, and deterministic action."""
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
    """Expressive K-component mixture-of-Gaussians SAC actor with tanh squashing.

    Motivation:
      A single-Gaussian policy has limited expressiveness on multi-modal action
      preferences arising from belief-rich EBM states.  A MoG actor is strictly
      more expressive while remaining fully tractable — the log-probability is
      the exact log-sum-exp of K Gaussian log-probs.

      The same policy family is used for **all** variants (Blau, EBM-control,
      EBM-cross, beta-contrastive) to ensure a fair comparison.

    Phase-adaptive extension:
      When ``phase_adaptive=True``, a separate late-horizon residual head
      (mean, log-std, and mixture-logit corrections) is activated by a smooth
      cubic gate keyed on the normalised time ``time_frac``.  This lets NES
      policies become more decisive late in the horizon without perturbing the
      early exploratory regime.

    Training:
      Component selection uses a straight-through Gumbel-Softmax for a
      pathwise actor gradient while maintaining the discrete mixture semantics.
      Log-probabilities use the *exact* mixture density at the sampled pre-tanh
      value, avoiding the approximation of the single-component log-prob trick.
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
        if n_components < 2:
            raise ValueError("MixtureTanhGaussianActor requires n_components >= 2.")
        self.action_dim = int(action_dim)
        self.n_components = int(n_components)
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

    def forward(
        self,
        state: torch.Tensor,
        time_frac: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mix_logits, per-component means, per-component log_stds)."""
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
        """Straight-through Gumbel-Softmax sample with exact mixture log-prob."""
        logits, mean, log_std = self(state, time_frac=time_frac)
        std = log_std.exp().clamp(min=1e-4, max=10.0)
        normal = torch.distributions.Normal(mean, std)
        z_all = normal.rsample()                              # (B, K, A)
        # Straight-through hard Gumbel sample for pathwise gradient
        g = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)  # (B, K)
        z = (g.unsqueeze(-1) * z_all).sum(dim=1)              # (B, A)
        u = torch.tanh(z)
        action = u * action_scale + action_bias
        # Exact mixture log-density at the sampled pre-tanh z
        z_exp = z.unsqueeze(1)                                # (B, 1, A)
        comp_log_prob = normal.log_prob(z_exp).sum(dim=-1)    # (B, K)
        log_mix = F.log_softmax(logits, dim=-1)
        log_prob_z = torch.logsumexp(log_mix + comp_log_prob, dim=-1, keepdim=True)
        log_det = torch.log(1.0 - u.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = sanitize(log_prob_z - log_det, 100.0)
        # Deterministic action uses the highest-weight component mean
        det_idx = torch.argmax(logits, dim=-1)
        det_mean = mean[torch.arange(mean.shape[0], device=mean.device), det_idx]
        deterministic = torch.tanh(det_mean) * action_scale + action_bias
        return action, log_prob, deterministic


class SequenceTransformerTanhGaussianActor(nn.Module):
    """Path-aware continuous actor for NES BOED policies.

    The policy state is structured as:
        [raw_actions (H × action_dim), raw_obs (H,), context_features]

    The actor tokenises the raw path (one token per time step), appends a
    learned context token and a CLS token, then reads the CLS representation
    through a Transformer encoder to predict a tanh-Gaussian mixture policy.

    This architecture lets the actor attend directly to individual past
    measurements rather than consuming only a compressed filter summary,
    which is especially beneficial with NES where the actor can be expressive
    without worrying about SAC's off-policy correction.

    The ``sample()`` interface is identical to ``MixtureTanhGaussianActor`` so
    it is a drop-in replacement in the NES training loop.

    Args:
        state_dim:       Full policy state dimension (raw path + context).
        action_dim:      Dimension of a single action.
        horizon:         Trajectory length T (used to parse the state vector).
        hidden:          Width of the head MLP after the Transformer.
        dropout:         Attention and input dropout probability.
        n_components:    Number of mixture components (≥ 2).
        d_model:         Transformer token embedding dimension.
        nhead:           Number of attention heads.
        num_layers:      Number of Transformer encoder layers.
        dim_feedforward: Feedforward expansion in each Transformer layer.
        phase_adaptive:  Enable phase-adaptive late-horizon correction.
        phase_start_frac: Fraction of horizon at which gate opens.
        phase_strength:  Multiplier on the late-horizon residual.
        late_std_scale:  Scale applied to std late in the horizon.
        late_mix_temp:   Temperature applied to mixture logits late in horizon.
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
        if n_components < 2:
            raise ValueError("SequenceTransformerTanhGaussianActor requires n_components >= 2")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.raw_path_dim = self.horizon * self.action_dim + self.horizon
        self.context_dim = max(1, self.state_dim - self.raw_path_dim)
        self.n_components = int(n_components)
        self.d_model = int(d_model)
        self.input_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        # Per-step token: (action, obs) projected to d_model
        self.step_proj = nn.Linear(self.action_dim + 1, self.d_model)
        # Context token: remaining state features projected to d_model
        self.context_proj = nn.Linear(self.context_dim, self.d_model)
        # Learned CLS token and positional embeddings
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
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
        )
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

    def _split_state(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse the flat state vector into (path_tokens, context)."""
        B = state.shape[0]
        state = self.input_dropout(state)
        # Pad if the state is shorter than expected (early in trajectory)
        if state.shape[-1] < self.raw_path_dim:
            pad = torch.zeros(
                B, self.raw_path_dim - state.shape[-1],
                dtype=state.dtype, device=state.device,
            )
            state = torch.cat([state, pad], dim=-1)
        actions_flat = state[:, : self.horizon * self.action_dim]
        obs_flat = state[:, self.horizon * self.action_dim : self.raw_path_dim]
        actions = actions_flat.reshape(B, self.horizon, self.action_dim)
        obs = obs_flat.reshape(B, self.horizon, 1)
        path = torch.cat([actions, obs], dim=-1)                # (B, T, action_dim+1)
        ctx = state[:, self.raw_path_dim:]
        # Clamp / pad context to the expected dimension
        if ctx.shape[-1] < self.context_dim:
            pad = torch.zeros(
                B, self.context_dim - ctx.shape[-1],
                dtype=state.dtype, device=state.device,
            )
            ctx = torch.cat([ctx, pad], dim=-1)
        elif ctx.shape[-1] > self.context_dim:
            ctx = ctx[:, : self.context_dim]
        return path, ctx

    def forward(
        self,
        state: torch.Tensor,
        time_frac: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = state.shape[0]
        path, ctx = self._split_state(sanitize(state, 1e3))
        step_tok = self.step_proj(path)                        # (B, T, d_model)
        ctx_tok = self.context_proj(ctx).unsqueeze(1)          # (B, 1, d_model)
        cls = self.cls.expand(B, -1, -1)                       # (B, 1, d_model)
        tokens = torch.cat([cls, step_tok, ctx_tok], dim=1)    # (B, T+2, d_model)
        tokens = tokens + self.pos[:, : tokens.shape[1], :]
        z = sanitize(self.encoder(tokens), 100.0)[:, 0]        # CLS output (B, d_model)
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
        """Gumbel-Softmax sample with exact mixture log-prob (same as MoG actor)."""
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

    Rather than processing the concatenated [base_state, belief_features] vector
    through a single MLP, this actor routes them through separate encoders before
    fusing.  This prevents NES from having to discover from a flat concatenation
    that the quotient/filter block and belief/modal block carry different geometry.

    Only used when ``--ebm-dual-branch-actor`` is enabled for EBM variants.

    Args:
        base_dim:        Dimension of the quotient / minimal base state.
        belief_dim:      Dimension of the EBM belief features.
        action_dim:      Dimension of the action.
        hidden:          Width of the fusion MLP (each branch is hidden//2).
        dropout:         Input dropout probability.
        n_components:    Number of mixture components (≥ 2).
        phase_adaptive:  Enable phase-adaptive late-horizon correction.
        phase_start_frac: Fraction of horizon at which gate opens.
        phase_strength:  Multiplier on the late-horizon residual.
        late_std_scale:  Scale applied to std late in the horizon.
        late_mix_temp:   Temperature applied to mixture logits late in horizon.
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
        if belief_dim <= 0:
            raise ValueError("DualBranchMixtureTanhGaussianActor requires belief_dim > 0")
        if n_components < 2:
            raise ValueError("DualBranchMixtureTanhGaussianActor requires n_components >= 2")
        self.base_dim = int(base_dim)
        self.belief_dim = int(belief_dim)
        self.action_dim = int(action_dim)
        self.n_components = int(n_components)
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

    def forward(
        self,
        state: torch.Tensor,
        time_frac: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.input_dropout(state)
        base = state[..., : self.base_dim]
        belief = state[..., self.base_dim : self.base_dim + self.belief_dim]
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


# ---------------------------------------------------------------------------
# Policy actors — discrete
# ---------------------------------------------------------------------------

class DiscreteCategoricalActor(nn.Module):
    """Categorical actor for discrete action spaces (e.g., prey-population env)."""

    def __init__(self, state_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.num_actions = int(num_actions)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.logits_head = nn.Linear(hidden, self.num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return sanitize(self.logits_head(sanitize(self.net(state), 100.0)), 20.0)

    def probs_and_log_probs(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self(state)
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.exp(log_probs), log_probs

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
        deterministic = action_values[torch.argmax(logits, dim=-1)]
        return action, log_prob, deterministic


class DiscreteMoECategoricalActor(nn.Module):
    """Mixture-of-experts categorical actor for discrete action spaces."""

    def __init__(
        self, state_dim: int, num_actions: int, hidden: int = 256, n_experts: int = 4
    ):
        super().__init__()
        self.num_actions = int(num_actions)
        self.n_experts = int(n_experts)
        if self.n_experts < 1:
            raise ValueError("DiscreteMoECategoricalActor requires n_experts >= 1.")
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.expert_logits_head = nn.Linear(hidden, self.n_experts * self.num_actions)
        self.gate_logits_head = nn.Linear(hidden, self.n_experts)

    def _mixture_log_probs(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = sanitize(self.net(state), 100.0)
        expert_logits = sanitize(
            self.expert_logits_head(hidden).view(-1, self.n_experts, self.num_actions), 20.0
        )
        gate_logits = sanitize(self.gate_logits_head(hidden), 20.0)
        expert_log_probs = F.log_softmax(expert_logits, dim=-1)
        gate_log_probs = F.log_softmax(gate_logits, dim=-1)
        log_probs = torch.logsumexp(
            gate_log_probs.unsqueeze(-1) + expert_log_probs, dim=1
        )
        return torch.exp(log_probs), log_probs

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        _, log_probs = self._mixture_log_probs(state)
        return log_probs

    def probs_and_log_probs(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._mixture_log_probs(state)

    def sample(
        self,
        state: torch.Tensor,
        action_values: torch.Tensor,
        time_frac: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, log_probs = self.probs_and_log_probs(state)
        idx = torch.distributions.Categorical(probs=probs).sample()
        log_prob = log_probs.gather(-1, idx.unsqueeze(-1))
        action = action_values[idx]
        deterministic = action_values[torch.argmax(log_probs, dim=-1)]
        return action, log_prob, deterministic


# ---------------------------------------------------------------------------
# Soft Q-critic
# ---------------------------------------------------------------------------

class QCritic(nn.Module):
    """Double-Q soft critic: Q(s, a) → scalar value."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sanitize(state, 1e3), sanitize(action, 1e3)], dim=-1)
        return sanitize(self.net(x), 1e4)
