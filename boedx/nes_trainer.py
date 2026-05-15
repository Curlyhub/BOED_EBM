"""
Natural Evolution Strategy (NES) training loop for BOEDX.

This module implements the OpenAI-ES style population-based optimisation loop
used as an alternative to SAC for BOED policy search.  NES is fundamentally
different from the RL trainer:

  - **No replay buffer** — candidates are evaluated via fresh episode rollouts.
  - **No Q-critics** — the EBM is trained offline from rollout data via a
    supervised cross-entropy objective against the exact posterior.
  - **Population perturbation** — a population of actor parameters is sampled
    around a mean ``mu`` with isotropic Gaussian noise; fitness scores are
    computed per candidate and used to update ``mu`` via a natural gradient
    estimate.
  - **Model selection** — periodic evaluation snapshots keep the top-K
    checkpoints; at the end the best is re-evaluated and saved.

Entry points
------------
``train_one_seed_nes``     — train one (variant, seed) pair.
``run_experiment_suite_nes``  — sweep all (variant × seed) pairs and aggregate.

Aliases
-------
``NESConfig = EvolutionConfig`` (backward-compatible name used in legacy scripts)
``run_experiment_suite_evolution = run_experiment_suite_nes``
"""

from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from boedx.env import BeliefConfig, GenericBankBOEDEnv, GenericTrainConfig
from boedx.models import (
    ApsiHead,
    CachedFilterBackbone,
    CrossInteractionEnergyNet,
    DiscreteCategoricalActor,
    DualBranchMixtureTanhGaussianActor,
    EnergyNet,
    MixtureTanhGaussianActor,
    QCritic,
    SequenceTransformerTanhGaussianActor,
    SymmetricSourceCrossNet,
    SymmetricSourceEnergyNet,
    TanhGaussianActor,
)
from boedx.state import (
    belief_feature_dim,
    belief_features_from_probs,
    build_base_state,
    build_minimal_state,
    build_raw_history_state,
    history_logits_from_batch,
    make_raw_state,
    posterior_probs_from_energy,
    variant_uses_cross_ebm,
    variant_uses_ebm,
)
from boedx.trainer import (
    discrete_bank_ig_from_logits,
    estimate_snmc_style_upper_prefix,
    estimate_spce_prefix,
    get_discrete_action_values,
    uses_discrete_actor,
)
from boedx.utils import (
    ensure_dir,
    mean_std_ci95,
    paired_summary,
    set_seed,
)


# ---------------------------------------------------------------------------
# NES configuration
# ---------------------------------------------------------------------------

@dataclass
class NESConfig:
    """Hyper-parameters for the NES optimisation loop.

    Attributes mirror the CLI flags of ``boedx-source-location-nes`` and
    can be passed directly to ``run_experiment_suite_nes``.
    """

    # Population and budget
    generations: int = 200
    population_size: int = 48             # must be even when mirrored_sampling=True
    rollout_episodes_per_candidate: int = 2
    eval_episodes: int = 150

    # Step-size (sigma) parameters
    lr_mu: float = 0.04                   # learning rate for mean update
    lr_sigma: float = 0.10               # learning rate for log-sigma update
    sigma_init: float = 0.03
    sigma_final: float = 0.005
    sigma_schedule: str = "exp"          # "constant" | "linear" | "exp"

    # Sampling and utility
    mirrored_sampling: bool = True        # antithetic noise for variance reduction
    utility_mode: str = "nes"            # "nes" (log-rank) | "centered_ranks"

    # Meta-optimizer for mu updates
    optimizer: str = "adam"              # "adam" | "rmsprop" | "sgd"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Optional adaptive sigma scaling based on population success rate
    sigma_adapt_on_success: bool = False
    sigma_success_target: float = 0.20
    sigma_adapt_rate: float = 0.05

    # Logging
    print_every: int = 10
    device: str = "cpu"
    seeds: str = "0,1,2"

    # Model selection: keep top-K checkpoints and pick the best at the end
    selection_eval_episodes: int = 40
    selection_every: int = 10
    selection_start_generation: int = 10
    selection_top_k: int = 3
    selection_final_eval_episodes: int = 0   # 0 = reuse selection_eval_episodes
    selection_return_weight: float = 1.0
    selection_belief_kl_weight: float = 0.05
    selection_belief_map_weight: float = 0.15
    selection_belief_mean_weight: float = 0.25

    # Cross-variant selection weights (replace the defaults for cross variants)
    cross_selection_bank_ig_weight: float = 0.30
    cross_selection_spce_weight: float = 0.20
    cross_selection_filter_bank_ig_weight: float = 0.00
    cross_selection_gap_penalty_weight: float = 0.35
    cross_selection_belief_kl_weight: float = 0.02
    cross_selection_belief_map_weight: float = 0.04
    cross_selection_belief_mean_weight: float = 0.04

    # EBM offline training (supervised, from rollout data)
    ebm_updates_per_generation: int = 10
    ebm_batch_size: int = 128
    ebm_data_episodes: int = 8           # rollouts to collect per EBM update cycle
    ebm_pretrain_episodes: int = 0       # random pre-training rollouts before NES
    ebm_pretrain_updates: int = 0        # SGD steps during pre-training
    ebm_update_every_generations: int = 1
    freeze_ebm: bool = False             # never update EBM during NES
    ebm_freeze_after_generation: int = -1  # freeze after this many generations (-1=off)

    # Parallelism (requires joblib)
    parallel_candidates: bool = False
    n_jobs: int = 1
    parallel_backend: str = "threading"  # "threading" | "loky"

    # Common random numbers for variance reduction across candidates
    use_common_random_numbers: bool = False
    common_random_numbers_seed_stride: int = 1000003

    # Top-candidate re-evaluation for low-variance fitness ranking
    reevaluate_top_candidates: int = 0
    reevaluate_top_episodes: int = 0


# ---------------------------------------------------------------------------
# Meta-optimizer for NES mean update
# ---------------------------------------------------------------------------

class ParamOptimizer:
    """Adam / RMSProp / SGD update rule for the NES population mean.

    Operates on flat parameter tensors rather than ``nn.Module`` parameters to
    keep the NES loop free of autograd overhead.
    """

    def __init__(
        self,
        shape: torch.Size,
        cfg: NESConfig,
        lr: float,
        device: torch.device,
    ):
        self.kind = cfg.optimizer
        self.lr = float(lr)
        self.beta1 = float(cfg.beta1)
        self.beta2 = float(cfg.beta2)
        self.eps = float(cfg.eps)
        self.m = torch.zeros(shape, device=device)
        self.v = torch.zeros(shape, device=device)
        self.t = 0

    def step(self, params: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Return updated params given the gradient estimate."""
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


# ---------------------------------------------------------------------------
# Pre-allocated state buffer (eval / rollout hot-path)
# ---------------------------------------------------------------------------

class SingleStateTensorBuffer:
    """Pre-allocated single-sample tensors reused in the rollout hot path.

    Avoids per-step tensor allocations during episode rollouts and evaluations.
    Call ``update(raw_state)`` to fill the buffer from a plain-dict state
    snapshot; the method returns a batch dict suitable for
    ``compute_state_from_batch_nes``.
    """

    def __init__(self, env: GenericBankBOEDEnv, device: torch.device):
        self.device = device
        self.last_obs = torch.empty((1, 1), dtype=torch.float32, device=device)
        self.t_idx = torch.empty((1,), dtype=torch.long, device=device)
        self.aux_state = torch.empty(
            (1, int(env.current_aux_state().shape[0])),
            dtype=torch.float32, device=device,
        )
        self.actions = torch.empty(
            (1, env.get_horizon(), env.action_dim),
            dtype=torch.float32, device=device,
        )
        self.obs = torch.empty(
            (1, env.get_horizon()), dtype=torch.float32, device=device
        )
        self.posterior_logits = torch.empty(
            (1, env.H), dtype=torch.float32, device=device
        )
        self.filter_logits = torch.empty(
            (1, env.H), dtype=torch.float32, device=device
        )
        self.posterior = torch.empty(
            (1, env.H), dtype=torch.float32, device=device
        )
        self.filter_probs = torch.empty(
            (1, env.H), dtype=torch.float32, device=device
        )

    def update(
        self, raw: Dict, need_probs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Fill the buffer from a raw state dict and return a batch dict."""
        self.last_obs[0, 0] = float(raw["last_obs"])
        self.t_idx[0] = int(raw["t_idx"])
        if self.aux_state.numel() > 0:
            self.aux_state[0].copy_(
                torch.from_numpy(raw["aux_state"]).to(self.device)
            )
        self.actions[0].copy_(
            torch.from_numpy(raw["actions"]).to(self.device)
        )
        self.obs[0].copy_(
            torch.from_numpy(raw["obs"]).to(self.device)
        )
        self.posterior_logits[0].copy_(
            torch.from_numpy(raw["posterior_logits"]).to(self.device)
        )
        self.filter_logits[0].copy_(
            torch.from_numpy(raw["filter_logits"]).to(self.device)
        )
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
            self.posterior[0].copy_(
                torch.from_numpy(raw["posterior"]).to(self.device)
            )
            if "filter_probs" in raw:
                self.filter_probs[0].copy_(
                    torch.from_numpy(raw["filter_probs"]).to(self.device)
                )
            batch["posterior"] = self.posterior
            batch["filter_probs"] = self.filter_probs
        return batch


# ---------------------------------------------------------------------------
# NES-specific state computation
# ---------------------------------------------------------------------------

def nes_actor_belief_spec(
    variant: str, belief_cfg: BeliefConfig
) -> Tuple[str, int]:
    """Return the (feature_mode, modal_top_k) that the NES actor should consume.

    The EBM can be trained against the full posterior while the actor receives
    a more compact, smoother belief representation that is easier for NES to
    exploit.  This function resolves the NES-specific overrides in
    ``BeliefConfig`` for the given variant.
    """
    feature_mode = belief_cfg.feature_mode
    modal_top_k = belief_cfg.modal_top_k
    # For cross-variants, optionally compress the belief to moments + top-2
    if belief_cfg.nes_cross_compact_belief and variant_uses_cross_ebm(variant):
        feature_mode = "moments"
        modal_top_k = min(max(belief_cfg.modal_top_k, 1), 2)
    # NES-specific overrides take precedence if set
    if belief_cfg.nes_actor_feature_mode:
        feature_mode = belief_cfg.nes_actor_feature_mode
    if belief_cfg.nes_actor_modal_top_k > 0:
        modal_top_k = belief_cfg.nes_actor_modal_top_k
    return feature_mode, modal_top_k


def compute_state_from_batch_nes(
    variant: str,
    filter_backbone: CachedFilterBackbone,
    batch: Dict[str, torch.Tensor],
    env: GenericBankBOEDEnv,
    energy_net: Optional[nn.Module],
    apsi_head: Optional[nn.Module],
    belief_cfg: Optional[BeliefConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """NES-specific state computation (no ``use_next`` indirection).

    Returns (state, hist_feat, energy, A) following the same contract as the
    RL version, but adds NES-specific actor belief overrides and an optional
    raw history prefix for transformer actors.
    """
    belief_cfg = belief_cfg or BeliefConfig()
    last_obs = batch["last_obs"]
    t_idx = batch["t_idx"]
    aux_state = batch["aux_state"]
    actions = batch["actions"]
    obs = batch["obs"]

    if variant == "blau_approx":
        raw_state = build_raw_history_state(
            actions, obs, t_idx, env.get_horizon(), last_obs, aux_state
        )
        return raw_state, raw_state, None, None

    selected_logits = history_logits_from_batch(variant, batch, "")
    hist_feat = filter_backbone.forward_from_logits(selected_logits)
    quotient_base = build_base_state(
        hist_feat, t_idx, env.get_horizon(), last_obs, aux_state
    )

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
    # Transformer actors need the raw path prepended so they can parse it as tokens
    raw_prefix = (
        build_raw_history_state(actions, obs, t_idx, env.get_horizon(), last_obs, aux_state)
        if belief_cfg.include_raw_history_for_ebm_actor
        else None
    )
    if belief_cfg.mode == "distilled_detached":
        belief = belief.detach()
        base_state = torch.cat([quotient_base, belief], dim=-1)
    elif belief_cfg.mode in {"distilled_e2e", "learned_only"}:
        if belief_cfg.mode == "learned_only":
            minimal_base = build_minimal_state(
                t_idx, env.get_horizon(), last_obs, aux_state
            )
            base_state = torch.cat([minimal_base, belief], dim=-1)
        else:
            base_state = torch.cat([quotient_base, belief], dim=-1)
    else:
        raise ValueError(f"Unknown belief mode: {belief_cfg.mode!r}")

    if raw_prefix is not None:
        from boedx.utils import sanitize as _sanitize
        state = _sanitize(torch.cat([raw_prefix, base_state], dim=-1), 1e3)
    else:
        from boedx.utils import sanitize as _sanitize
        state = _sanitize(base_state, 1e3)
    return state, hist_feat, energy, A


# ---------------------------------------------------------------------------
# Module construction (NES — no Q-critics)
# ---------------------------------------------------------------------------

def _actor_hparams_for_variant(
    variant: str, train_cfg: GenericTrainConfig
) -> Tuple[str, int, int, bool]:
    """Return (family, hidden, n_components, dual_branch) for the actor."""
    if variant_uses_ebm(variant):
        family = train_cfg.ebm_actor_family or train_cfg.actor_family
        hidden = int(train_cfg.ebm_hidden_rl) if int(train_cfg.ebm_hidden_rl) > 0 else int(train_cfg.hidden_rl)
        comps = (
            int(train_cfg.ebm_actor_mixture_components)
            if int(train_cfg.ebm_actor_mixture_components) > 0
            else int(train_cfg.actor_mixture_components)
        )
        return family, hidden, comps, bool(train_cfg.ebm_dual_branch_actor)
    return train_cfg.actor_family, int(train_cfg.hidden_rl), int(train_cfg.actor_mixture_components), False


def _make_continuous_actor(
    variant: str,
    state_dim: int,
    base_dim: int,
    belief_dim: int,
    action_dim: int,
    train_cfg: GenericTrainConfig,
    device: torch.device,
) -> nn.Module:
    """Instantiate the correct continuous actor architecture."""
    family, hidden, comps, dual = _actor_hparams_for_variant(variant, train_cfg)
    phase_kw = dict(
        phase_adaptive=train_cfg.phase_adaptive_actor,
        phase_start_frac=train_cfg.phase_start_frac,
        phase_strength=train_cfg.phase_strength,
        late_std_scale=train_cfg.late_std_scale,
    )
    if dual:
        if family != "mog":
            raise ValueError("--ebm-dual-branch-actor currently requires --ebm-actor-family mog")
        return DualBranchMixtureTanhGaussianActor(
            base_dim=base_dim, belief_dim=belief_dim, action_dim=action_dim,
            hidden=hidden, dropout=train_cfg.actor_dropout, n_components=comps,
            late_mix_temp=train_cfg.late_mix_temp, **phase_kw,
        ).to(device)
    if family == "transformer":
        seq_horizon = getattr(train_cfg, "sequence_horizon", train_cfg.episodes)
        return SequenceTransformerTanhGaussianActor(
            state_dim=state_dim, action_dim=action_dim, horizon=seq_horizon,
            hidden=hidden, dropout=train_cfg.actor_dropout, n_components=comps,
            d_model=train_cfg.transformer_d_model, nhead=train_cfg.transformer_nhead,
            num_layers=train_cfg.transformer_layers,
            dim_feedforward=train_cfg.transformer_ff,
            late_mix_temp=train_cfg.late_mix_temp, **phase_kw,
        ).to(device)
    if family == "mog":
        return MixtureTanhGaussianActor(
            state_dim=state_dim, action_dim=action_dim, hidden=hidden,
            dropout=train_cfg.actor_dropout, n_components=comps,
            late_mix_temp=train_cfg.late_mix_temp, **phase_kw,
        ).to(device)
    if family == "gaussian":
        return TanhGaussianActor(
            state_dim=state_dim, action_dim=action_dim, hidden=hidden,
            dropout=train_cfg.actor_dropout, **phase_kw,
        ).to(device)
    raise ValueError(f"Unknown actor_family={family!r}")


def _build_policy_modules(
    variant: str,
    env: GenericBankBOEDEnv,
    train_cfg: GenericTrainConfig,
    device: torch.device,
    belief_cfg: Optional[BeliefConfig] = None,
) -> Tuple[CachedFilterBackbone, nn.Module, Optional[nn.Module], Optional[nn.Module]]:
    """Build (filter_backbone, actor, energy_net, apsi_head) — no Q-critics."""
    belief_cfg = belief_cfg or BeliefConfig()
    filter_backbone = CachedFilterBackbone().to(device)
    hist_dim = env.H
    aux_dim = int(env.current_aux_state().shape[0])
    quotient_base_dim = hist_dim + 1 + 1 + aux_dim
    raw_history_dim = env.get_horizon() * env.action_dim + env.get_horizon() + 1 + 1 + aux_dim
    minimal_base_dim = 1 + 1 + aux_dim
    energy_net: Optional[nn.Module] = None
    apsi_head: Optional[nn.Module] = None
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
                hist_dim=hist_dim, theta_dim=env.theta_dim,
                hidden=train_cfg.hidden_ebm,
                n_sources=belief_cfg.n_sources,
                add_pairwise_dist=belief_cfg.add_pairwise_dist,
            ).to(device)
        else:
            ebm_cls_std = CrossInteractionEnergyNet if use_cross else EnergyNet
            energy_net = ebm_cls_std(
                hist_dim=hist_dim, theta_dim=env.theta_dim, hidden=train_cfg.hidden_ebm
            ).to(device)
        apsi_head = ApsiHead(hist_dim=hist_dim, hidden=train_cfg.hidden_ebm).to(device)
        actor_feature_mode, actor_modal_top_k = nes_actor_belief_spec(variant, belief_cfg)
        belief_dim = belief_feature_dim(env.theta_dim, actor_feature_mode, actor_modal_top_k)

    actor_family, _, _, _ = _actor_hparams_for_variant(variant, train_cfg)
    if variant == "blau_approx":
        state_dim = raw_history_dim
    elif variant_uses_ebm(variant) and actor_family == "transformer":
        # Transformer actor prepends the full raw history then adds base + belief
        base_no_path = (
            minimal_base_dim if belief_cfg.mode == "learned_only" and belief_dim > 0
            else quotient_base_dim
        )
        state_dim = raw_history_dim + base_no_path + belief_dim
    elif belief_cfg.mode == "learned_only" and belief_dim > 0:
        state_dim = minimal_base_dim + belief_dim
    else:
        state_dim = quotient_base_dim + belief_dim
    base_dim_for_actor = (
        minimal_base_dim if belief_cfg.mode == "learned_only" and belief_dim > 0
        else quotient_base_dim
    )

    if uses_discrete_actor(env):
        action_values = get_discrete_action_values(env, device)
        actor: nn.Module = DiscreteCategoricalActor(
            state_dim=state_dim, num_actions=action_values.shape[0],
            hidden=int(train_cfg.hidden_rl),
        ).to(device)
    else:
        actor = _make_continuous_actor(
            variant=variant, state_dim=state_dim, base_dim=base_dim_for_actor,
            belief_dim=belief_dim, action_dim=env.action_dim,
            train_cfg=train_cfg, device=device,
        )
    return filter_backbone, actor, energy_net, apsi_head


# ---------------------------------------------------------------------------
# NES utility functions
# ---------------------------------------------------------------------------

def _centered_ranks(x: np.ndarray) -> np.ndarray:
    """Normalise fitness values to centered ranks in [-0.5, 0.5]."""
    x = np.asarray(x, dtype=np.float64)
    ranks = np.empty_like(x)
    order = np.argsort(x)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    ranks /= max(len(x) - 1, 1)
    ranks -= 0.5
    return ranks


def _nes_utilities(n: int) -> np.ndarray:
    """Compute the log-rank utility weights used by the NES gradient estimator."""
    ranks = np.arange(1, n + 1, dtype=np.float64)
    util = np.maximum(0.0, np.log(n / 2.0 + 1.0) - np.log(ranks))
    util /= util.sum()
    util -= 1.0 / n
    return util


def _schedule_sigma(
    gen: int, total: int, init_sigma: float, final_sigma: float, mode: str
) -> float:
    """Return the scheduled sigma for this generation."""
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
    """Return True if the EBM should not be updated at this generation."""
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
    """Compute the scalar model-selection score for a given checkpoint."""
    # Blau and non-cross variants: use return only
    if not variant_uses_cross_ebm(variant):
        return float(cfg.selection_return_weight * float(eval_metrics.get("avg_return", 0.0)))
    # Cross-specific selection: exact-task metrics dominate, diagnostics are tie-breakers
    score = cfg.selection_return_weight * float(eval_metrics.get("avg_return", 0.0))
    score += cfg.cross_selection_bank_ig_weight * float(eval_metrics.get("avg_bank_ig", 0.0))
    score += cfg.cross_selection_spce_weight * float(eval_metrics.get("avg_spce_lower", 0.0))
    score += cfg.cross_selection_filter_bank_ig_weight * float(eval_metrics.get("avg_filter_bank_ig", 0.0))
    filter_bank = float(eval_metrics.get("avg_filter_bank_ig", 0.0))
    exact_bank = float(eval_metrics.get("avg_bank_ig", 0.0))
    score -= cfg.cross_selection_gap_penalty_weight * max(filter_bank - exact_bank, 0.0)
    if cross_diag_metrics is not None:
        score -= cfg.cross_selection_belief_map_weight * float(
            cross_diag_metrics.get("avg_actor_belief_map_to_exact_distance", 0.0)
        )
        score -= cfg.cross_selection_belief_mean_weight * float(
            cross_diag_metrics.get("avg_actor_belief_feature_mae", 0.0)
        )
        score -= cfg.cross_selection_belief_kl_weight * float(
            cross_diag_metrics.get("avg_actor_belief_prob_l1", 0.0)
        )
    return float(score)


# ---------------------------------------------------------------------------
# Parameter manipulation helpers
# ---------------------------------------------------------------------------

def _flatten_params(module: nn.Module) -> torch.Tensor:
    """Return all parameters as a single flat tensor."""
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()])


def _set_flat_params(module: nn.Module, flat: torch.Tensor) -> None:
    """Write a flat parameter vector back into the module in-place."""
    offset = 0
    with torch.no_grad():
        for p in module.parameters():
            n = p.numel()
            p.copy_(flat[offset : offset + n].view_as(p))
            offset += n


def _clone_module_state(
    module: Optional[nn.Module],
) -> Optional[Dict[str, torch.Tensor]]:
    """Return a CPU copy of the module's state dict, or None."""
    if module is None:
        return None
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _load_module_state(
    module: Optional[nn.Module],
    state: Optional[Dict[str, torch.Tensor]],
) -> None:
    """Load a previously cloned state dict back into a module."""
    if module is None or state is None:
        return
    module.load_state_dict(state)


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

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
    """Run one episode and return (total_return, list_of_raw_states).

    All modules are temporarily set to eval mode for the duration of the
    rollout, then restored to their original training state.

    Args:
        deterministic:     Use the actor's deterministic (argmax) action.
        collect_raw_states: If True, record a raw state snapshot at each step.
        episode_seed:      If set, fix the RNG at the start of this episode.
    """
    if episode_seed is not None:
        set_seed(int(episode_seed))
    env = env_factory(device)
    action_scale = env.action_scale
    action_bias = env.action_bias
    discrete_action_values = (
        get_discrete_action_values(env, device) if uses_discrete_actor(env) else None
    )
    modules = [m for m in [filter_backbone, actor, energy_net, apsi_head] if m is not None]
    prev_modes = [m.training for m in modules]
    for m in modules:
        m.eval()

    raw_states: List[Dict] = []
    try:
        env.reset()
        need_probs = collect_raw_states
        raw = _make_raw_state_nes(env, need_probs=need_probs)
        single = SingleStateTensorBuffer(env, device)
        done = False
        total_return = 0.0
        while not done:
            if collect_raw_states:
                raw_states.append(dict(raw))
            batch = single.update(raw, need_probs=False)
            state_t, _, _, _ = compute_state_from_batch_nes(
                variant, filter_backbone, batch, env, energy_net, apsi_head,
                belief_cfg=belief_cfg,
            )
            time_frac = batch["t_idx"].float() / float(max(env.get_horizon(), 1))
            with torch.no_grad():
                if discrete_action_values is not None:
                    act, _, det = actor.sample(
                        state_t, action_values=discrete_action_values, time_frac=time_frac
                    )
                else:
                    act, _, det = actor.sample(
                        state_t, action_scale=action_scale, action_bias=action_bias,
                        time_frac=time_frac,
                    )
            chosen = det if deterministic else act
            action = chosen.squeeze(0).detach().cpu().numpy().astype(np.float32)
            _, reward, done, _ = env.step(action)
            raw = _make_raw_state_nes(env, need_probs=need_probs)
            total_return += float(reward)
    finally:
        for m, mode in zip(modules, prev_modes):
            m.train(mode)
    return total_return, raw_states


def _make_raw_state_nes(env: GenericBankBOEDEnv, need_probs: bool = False) -> Dict:
    """State snapshot for the NES rollout hot path.

    Lighter than the RL version: skips the contrastive particle fields that
    the NES loop does not need.  Optionally includes posterior / filter probs
    required for offline EBM training.
    """
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


# ---------------------------------------------------------------------------
# EBM offline training
# ---------------------------------------------------------------------------

def _batch_from_raw_states(
    raw_states: Sequence[Dict], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Stack a list of raw state dicts into a batched tensor dict."""

    def arr(key: str) -> np.ndarray:
        return np.stack([rs[key] for rs in raw_states], axis=0)

    has_post = all("posterior" in rs for rs in raw_states)
    posterior = (
        torch.tensor(arr("posterior"), dtype=torch.float32, device=device)
        if has_post
        else torch.softmax(
            torch.tensor(arr("posterior_logits"), dtype=torch.float32, device=device),
            dim=-1,
        )
    )
    return {
        "last_obs": torch.tensor(
            [[rs["last_obs"]] for rs in raw_states], dtype=torch.float32, device=device
        ),
        "t_idx": torch.tensor(
            [rs["t_idx"] for rs in raw_states], dtype=torch.long, device=device
        ),
        "aux_state": torch.tensor(arr("aux_state"), dtype=torch.float32, device=device),
        "actions": torch.tensor(arr("actions"), dtype=torch.float32, device=device),
        "obs": torch.tensor(arr("obs"), dtype=torch.float32, device=device),
        "posterior": posterior,
        "posterior_logits": torch.tensor(
            arr("posterior_logits"), dtype=torch.float32, device=device
        ),
        "filter_logits": torch.tensor(
            arr("filter_logits"), dtype=torch.float32, device=device
        ),
    }


def _collect_random_raw_states(
    env_factory: Callable[[torch.device], GenericBankBOEDEnv],
    device: torch.device,
    n_episodes: int,
) -> List[Dict]:
    """Collect state snapshots from random-policy rollouts for EBM pre-training."""
    collected: List[Dict] = []
    if n_episodes <= 0:
        return collected
    env = env_factory(device)
    for _ in range(n_episodes):
        env.reset()
        done = False
        while not done:
            raw = _make_raw_state_nes(env, need_probs=True)
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
    """Train the EBM offline via cross-entropy against the exact posterior.

    The exact posterior comes from the per-step filter logits accumulated
    during episode rollouts.  The EBM is trained to reproduce this posterior
    distribution from the compressed history feature.
    """
    if energy_net is None or apsi_head is None or len(raw_states) == 0:
        return
    energy_net.train()
    apsi_head.train()
    optim = torch.optim.Adam(
        list(energy_net.parameters()) + list(apsi_head.parameters()), lr=3e-4
    )
    raw_list = list(raw_states)
    H = len(raw_list)
    bs = min(cfg.ebm_batch_size, H)
    for _ in range(max(cfg.ebm_updates_per_generation, 0)):
        idx = np.random.randint(0, H, size=bs)
        batch = _batch_from_raw_states([raw_list[i] for i in idx], device=device)
        selected_logits = history_logits_from_batch(variant, batch, "")
        hist_feat = filter_backbone.forward_from_logits(selected_logits)
        energy = energy_net(hist_feat, env.hypothesis_bank)
        A = apsi_head(hist_feat)
        log_probs = F.log_softmax(-energy, dim=-1)
        target_probs = batch["posterior"]
        loss = -(target_probs * log_probs).sum(dim=-1).mean() + 0.1 * A.pow(2).mean()
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(energy_net.parameters()) + list(apsi_head.parameters()), 10.0
        )
        optim.step()


# ---------------------------------------------------------------------------
# Cross-variant belief diagnostics (NES-specific)
# ---------------------------------------------------------------------------

def _actor_aligned_belief_diagnostics(
    variant: str,
    belief_cfg: BeliefConfig,
    exact_probs: torch.Tensor,
    pred_probs: torch.Tensor,
    theta_bank: torch.Tensor,
    env: GenericBankBOEDEnv,
    true_theta: torch.Tensor,
) -> Dict[str, float]:
    """Per-step belief accuracy as the actor sees it (using actor-specific features)."""
    actor_feature_mode, actor_modal_top_k = nes_actor_belief_spec(variant, belief_cfg)
    exact_feat = belief_features_from_probs(
        probs=exact_probs, theta_bank=theta_bank, A_scalar=None,
        feature_mode=actor_feature_mode, modal_top_k=actor_modal_top_k,
    )
    pred_feat = belief_features_from_probs(
        probs=pred_probs, theta_bank=theta_bank, A_scalar=None,
        feature_mode=actor_feature_mode, modal_top_k=actor_modal_top_k,
    )
    pred_map = theta_bank[torch.argmax(pred_probs, dim=-1)]
    exact_map = theta_bank[torch.argmax(exact_probs, dim=-1)]
    return {
        "actor_belief_feature_mae": float(
            torch.mean(torch.abs(pred_feat - exact_feat)).detach().cpu()
        ),
        "actor_belief_prob_l1": float(
            torch.abs(exact_probs - pred_probs).sum(dim=-1).mean().detach().cpu()
        ),
        "actor_belief_map_to_exact_distance": float(
            env.belief_distance(pred_map, exact_map).mean().detach().cpu()
        ),
        "actor_belief_map_to_true_distance": float(
            env.belief_distance(pred_map, true_theta).mean().detach().cpu()
        ),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

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
    """Evaluate the NES policy and EBM over multiple episodes.

    Returns a dict with:
      ``eval``             — scalar evaluation metrics.
      ``cross_diagnostics`` — belief accuracy metrics (only for cross variants).
      ``paths``            — per-step path arrays (only when collect_paths=True).
    """
    belief_cfg = belief_cfg or BeliefConfig()
    n_eval = int(n_eval_episodes if n_eval_episodes is not None else train_cfg.eval_episodes)
    env = env_factory(device)
    modules = [m for m in [filter_backbone, actor, energy_net, apsi_head] if m is not None]
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
            raw = _make_raw_state_nes(env, need_probs=need_probs)
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
                    variant, filter_backbone, batch, env, energy_net, apsi_head,
                    belief_cfg=belief_cfg,
                )
                if energy_net is not None and energy_t is not None:
                    exact_probs_t = batch["posterior"]
                    pred_probs = posterior_probs_from_energy(energy_t)
                    exact_mean_t = exact_probs_t @ env.hypothesis_bank
                    pred_mean = pred_probs @ env.hypothesis_bank
                    full_belief_mean_errs.append(
                        float(torch.mean(torch.abs(pred_mean - exact_mean_t)).detach().cpu())
                    )
                    full_belief_kl_errs.append(float(
                        (exact_probs_t.clamp_min(1e-12) * (
                            torch.log(exact_probs_t.clamp_min(1e-12))
                            - torch.log(pred_probs.clamp_min(1e-12))
                        )).sum(dim=-1).mean().detach().cpu()
                    ))
                    full_belief_l1_errs.append(float(
                        torch.abs(exact_probs_t - pred_probs).sum(dim=-1).mean().detach().cpu()
                    ))
                    pred_map = env.hypothesis_bank[torch.argmax(pred_probs, dim=-1)]
                    exact_map = env.hypothesis_bank[torch.argmax(exact_probs_t, dim=-1)]
                    true_theta_t = torch.tensor(
                        env.theta0[None], dtype=torch.float32, device=device
                    )
                    full_belief_map_to_exact_errs.append(float(
                        env.belief_distance(pred_map, exact_map).mean().detach().cpu()
                    ))
                    full_belief_map_to_true_errs.append(float(
                        env.belief_distance(pred_map, true_theta_t).mean().detach().cpu()
                    ))
                    diag = _actor_aligned_belief_diagnostics(
                        variant=variant, belief_cfg=belief_cfg,
                        exact_probs=exact_probs_t, pred_probs=pred_probs,
                        theta_bank=env.hypothesis_bank, env=env,
                        true_theta=true_theta_t,
                    )
                    actor_belief_feature_maes.append(diag["actor_belief_feature_mae"])
                    actor_belief_prob_l1s.append(diag["actor_belief_prob_l1"])
                    actor_belief_map_to_exact_errs.append(diag["actor_belief_map_to_exact_distance"])
                    actor_belief_map_to_true_errs.append(diag["actor_belief_map_to_true_distance"])

                time_frac = batch["t_idx"].float() / float(max(env.get_horizon(), 1))
                if uses_discrete_actor(env):
                    action_values = get_discrete_action_values(env, device)
                    _, _, det = actor.sample(
                        state_t, action_values=action_values, time_frac=time_frac
                    )
                else:
                    _, _, det = actor.sample(
                        state_t, action_scale=env.action_scale,
                        action_bias=env.action_bias, time_frac=time_frac,
                    )
                action = det.squeeze(0).detach().cpu().numpy().astype(np.float32)
                _, reward, done, _ = env.step(action)
                raw = _make_raw_state_nes(env, need_probs=need_probs)
                ep_return += reward
                prefix_actions = raw["actions"][: raw["length"]]
                prefix_obs = raw["obs"][: raw["length"]]
                ep_bank_path.append(
                    discrete_bank_ig_from_logits(raw["posterior_logits"], env.prior_bank_logits)
                )
                ep_filter_bank_path.append(
                    discrete_bank_ig_from_logits(raw["filter_logits"], env.prior_bank_logits)
                )
                ep_spce_path.append(
                    estimate_spce_prefix(env, prefix_actions, prefix_obs, env.theta0, spce_L)
                )
                if snmc_L > 0:
                    ep_snmc_path.append(
                        estimate_snmc_style_upper_prefix(
                            env, prefix_actions, prefix_obs, env.theta0, snmc_L
                        )
                    )
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
        paths: Dict[str, object] = {
            "bank_ig_mean_path": np.mean(np.array(bank_ig_paths, dtype=np.float64), axis=0).tolist(),
            "filter_bank_ig_mean_path": np.mean(np.array(filter_bank_ig_paths, dtype=np.float64), axis=0).tolist(),
            "spce_lower_mean_path": np.mean(np.array(spce_paths, dtype=np.float64), axis=0).tolist(),
        }
        if snmc_L > 0 and snmc_paths:
            paths["snmc_style_upper_mean_path"] = np.mean(
                np.array(snmc_paths, dtype=np.float64), axis=0
            ).tolist()
        out["paths"] = paths
    return out


# ---------------------------------------------------------------------------
# Parallel candidate evaluation helpers
# ---------------------------------------------------------------------------

def _generation_episode_seeds(
    base_seed: int, gen: int, count: int, stride: int
) -> List[int]:
    start = int(base_seed) * int(stride) + int(gen) * int(stride)
    return [start + i for i in range(int(count))]


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
    """Evaluate a candidate parameter vector by rolling out fresh episodes.

    This function is designed to be called in a subprocess (via joblib) so it
    reconstructs all modules from scratch rather than relying on shared state.
    """
    device = torch.device(device_str)
    env = env_factory(device)
    filter_backbone, actor, energy_net, apsi_head = _build_policy_modules(
        variant, env, train_cfg, device, belief_cfg=belief_cfg
    )
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
            belief_cfg=belief_cfg, deterministic=not stochastic,
            collect_raw_states=False, episode_seed=ep_seed,
        )
        vals.append(ret)
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Main NES training loop (one variant, one seed)
# ---------------------------------------------------------------------------

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
    """Run the NES optimisation loop for one (variant, seed) pair.

    Saves ``result.json`` to ``output_dir/<variant>/seed_<seed>/`` and
    returns the result dict.
    """
    set_seed(seed)
    device = torch.device(
        nes_cfg.device
        if nes_cfg.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    env = env_factory(device)
    filter_backbone, actor, energy_net, apsi_head = _build_policy_modules(
        variant, env, train_cfg, device, belief_cfg=belief_cfg
    )

    # EBM pre-training from random rollouts
    if energy_net is not None and apsi_head is not None and not _ebm_is_frozen(nes_cfg, None):
        pre_states = _collect_random_raw_states(env_factory, device, nes_cfg.ebm_pretrain_episodes)
        if pre_states:
            pre_cfg = NESConfig(**{k: v for k, v in nes_cfg.__dict__.items()})
            if nes_cfg.ebm_pretrain_updates > 0:
                pre_cfg.ebm_updates_per_generation = nes_cfg.ebm_pretrain_updates
            _supervised_update_ebm(
                variant=variant, env=env, filter_backbone=filter_backbone,
                energy_net=energy_net, apsi_head=apsi_head, raw_states=pre_states,
                device=device, belief_cfg=belief_cfg or BeliefConfig(), cfg=pre_cfg,
            )

    # Initialise NES state
    mu = _flatten_params(actor).to(device)
    mu_opt = ParamOptimizer(mu.shape, nes_cfg, nes_cfg.lr_mu, device)
    log_sigma = torch.tensor(
        math.log(max(nes_cfg.sigma_init, 1e-8)), dtype=torch.float32, device=device
    )
    sigma_offset = 0.0

    pop = int(nes_cfg.population_size)
    if nes_cfg.mirrored_sampling and pop % 2 != 0:
        raise ValueError("population_size must be even when mirrored_sampling=True")
    half = pop // 2 if nes_cfg.mirrored_sampling else pop
    dim = int(mu.numel())

    # Parallel candidate eval is only safe without threading + CRN (race on RNG)
    effective_parallel = bool(nes_cfg.parallel_candidates) and not (
        nes_cfg.use_common_random_numbers
        and nes_cfg.parallel_backend == "threading"
        and nes_cfg.n_jobs > 1
    )
    if effective_parallel and not _JOBLIB_AVAILABLE:
        effective_parallel = False

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
        nonlocal best_score, best_generation, best_actor_state, best_energy_state
        nonlocal best_apsi_state, best_eval_preview, top_candidates
        if nes_cfg.selection_eval_episodes <= 0:
            return
        if gen_no < nes_cfg.selection_start_generation:
            return
        if gen_no % nes_cfg.selection_every != 0 and gen_no != nes_cfg.generations:
            return
        preview_bundle = evaluate_modules_nes(
            variant=variant, env_factory=env_factory, filter_backbone=filter_backbone,
            actor=actor, energy_net=energy_net, apsi_head=apsi_head, device=device,
            train_cfg=train_cfg, spce_L=spce_L, snmc_L=snmc_L, belief_cfg=belief_cfg,
            n_eval_episodes=nes_cfg.selection_eval_episodes, collect_paths=False,
        )
        preview = preview_bundle["eval"]
        preview_diag = preview_bundle.get("cross_diagnostics")
        score = _selection_score(variant, preview, nes_cfg, cross_diag_metrics=preview_diag)
        cand: Dict[str, object] = {
            "score": float(score),
            "generation": int(gen_no),
            "actor_state": _clone_module_state(actor),
            "energy_state": _clone_module_state(energy_net),
            "apsi_state": _clone_module_state(apsi_head),
            "preview_eval": dict(preview),
            "preview_cross_diagnostics": dict(preview_diag) if preview_diag else None,
        }
        top_candidates.append(cand)
        top_candidates.sort(key=lambda d: float(d["score"]), reverse=True)
        keep_k = max(int(nes_cfg.selection_top_k), 1)
        top_candidates[:] = top_candidates[:keep_k]
        if score > best_score:
            best_score = score
            best_generation = gen_no
            best_actor_state = cand["actor_state"]
            best_energy_state = cand["energy_state"]
            best_apsi_state = cand["apsi_state"]
            best_eval_preview = dict(preview)

    # ---- Main NES loop ----
    for gen in range(1, nes_cfg.generations + 1):
        base_sigma = _schedule_sigma(
            gen, nes_cfg.generations, nes_cfg.sigma_init, nes_cfg.sigma_final,
            nes_cfg.sigma_schedule,
        )
        sigma = float(max(base_sigma * math.exp(sigma_offset), 1e-6))
        sigma_history.append(sigma)
        log_sigma = torch.tensor(math.log(sigma), dtype=torch.float32, device=device)

        generation_episode_seeds = None
        if nes_cfg.use_common_random_numbers:
            seed_count = max(
                int(nes_cfg.rollout_episodes_per_candidate),
                int(nes_cfg.reevaluate_top_episodes),
                1,
            )
            generation_episode_seeds = _generation_episode_seeds(
                seed, gen, seed_count, nes_cfg.common_random_numbers_seed_stride
            )
        candidate_episode_seeds = (
            None if generation_episode_seeds is None
            else generation_episode_seeds[: int(nes_cfg.rollout_episodes_per_candidate)]
        )

        z_pos = torch.randn(half, dim, device=device)
        z_all = torch.cat([z_pos, -z_pos], dim=0) if nes_cfg.mirrored_sampling else z_pos
        candidates = mu.unsqueeze(0) + sigma * z_all

        filter_state = _clone_module_state(filter_backbone)
        actor_state = _clone_module_state(actor)
        energy_state = _clone_module_state(energy_net)
        apsi_state = _clone_module_state(apsi_head)

        # Evaluate candidates
        if effective_parallel:
            fits = Parallel(n_jobs=nes_cfg.n_jobs, backend=nes_cfg.parallel_backend)(
                delayed(_candidate_return_from_flat)(
                    flat.detach().cpu().numpy(), variant, env_factory, train_cfg,
                    str(device), belief_cfg, filter_state, actor_state,
                    energy_state, apsi_state, True,
                    nes_cfg.rollout_episodes_per_candidate, candidate_episode_seeds,
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
                    ep_seed = (
                        None if candidate_episode_seeds is None
                        else int(candidate_episode_seeds[ep])
                    )
                    ret, _ = _episode_rollout(
                        variant, env_factory, filter_backbone, actor,
                        energy_net, apsi_head, device, belief_cfg=belief_cfg,
                        deterministic=False, collect_raw_states=False, episode_seed=ep_seed,
                    )
                    vals.append(ret)
                fitness[i] = float(np.mean(vals))
            _set_flat_params(actor, mu)

        # Optional top-candidate re-evaluation for variance reduction
        if (
            nes_cfg.reevaluate_top_candidates > 0
            and nes_cfg.reevaluate_top_episodes > 0
            and (
                nes_cfg.reevaluate_top_episodes != nes_cfg.rollout_episodes_per_candidate
                or not nes_cfg.use_common_random_numbers
            )
        ):
            top_n = min(int(nes_cfg.reevaluate_top_candidates), pop)
            if top_n > 0:
                top_idx = np.argsort(-fitness)[:top_n]
                reeval_seeds = (
                    None if generation_episode_seeds is None
                    else generation_episode_seeds[: int(nes_cfg.reevaluate_top_episodes)]
                )
                if effective_parallel:
                    reeval = Parallel(
                        n_jobs=nes_cfg.n_jobs, backend=nes_cfg.parallel_backend
                    )(
                        delayed(_candidate_return_from_flat)(
                            candidates[i].detach().cpu().numpy(), variant, env_factory,
                            train_cfg, str(device), belief_cfg, filter_state,
                            actor_state, energy_state, apsi_state, True,
                            nes_cfg.reevaluate_top_episodes, reeval_seeds,
                        )
                        for i in top_idx
                    )
                    fitness[top_idx] = np.asarray(reeval, dtype=np.float64)
                else:
                    for i in top_idx:
                        _set_flat_params(actor, candidates[i])
                        vals = []
                        for ep in range(nes_cfg.reevaluate_top_episodes):
                            ep_seed = (
                                None if reeval_seeds is None else int(reeval_seeds[ep])
                            )
                            ret, _ = _episode_rollout(
                                variant, env_factory, filter_backbone, actor,
                                energy_net, apsi_head, device, belief_cfg=belief_cfg,
                                deterministic=False, collect_raw_states=False,
                                episode_seed=ep_seed,
                            )
                            vals.append(ret)
                        fitness[i] = float(np.mean(vals))
                    _set_flat_params(actor, mu)

        # NES gradient update
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

        sq_term = z_all.pow(2).mean(dim=1) - 1.0
        grad_log_sigma = torch.sum(util_t * sq_term)

        mu = mu_opt.step(mu, grad_mu)
        log_sigma = log_sigma + nes_cfg.lr_sigma * grad_log_sigma

        # Optional success-rate-based sigma adaptation
        if nes_cfg.sigma_adapt_on_success:
            center_fit = _candidate_return_from_flat(
                mu.detach().cpu().numpy(), variant, env_factory, train_cfg, str(device),
                belief_cfg, filter_state, actor_state, energy_state, apsi_state,
                False, max(1, nes_cfg.rollout_episodes_per_candidate),
                candidate_episode_seeds,
            )
            success = float(np.mean(fitness > center_fit))
            sigma_offset += nes_cfg.sigma_adapt_rate * (success - nes_cfg.sigma_success_target)
        else:
            success = float(np.mean(fitness > fitness.mean()))
        success_history.append(success)

        # Blend schedule with gradient update (avoid either dominating)
        sigma = max(1e-6, math.sqrt(base_sigma * float(log_sigma.exp().detach().cpu())) * math.exp(sigma_offset))
        log_sigma = torch.tensor(math.log(sigma), dtype=torch.float32, device=device)

        _set_flat_params(actor, mu)
        generation_returns.append(float(fitness.mean()))

        # Offline EBM training from policy rollouts
        if (
            energy_net is not None
            and apsi_head is not None
            and not _ebm_is_frozen(nes_cfg, gen)
            and nes_cfg.ebm_data_episodes > 0
            and gen % max(nes_cfg.ebm_update_every_generations, 1) == 0
        ):
            collected: List[Dict] = []
            for _ in range(nes_cfg.ebm_data_episodes):
                _, states = _episode_rollout(
                    variant, env_factory, filter_backbone, actor, energy_net, apsi_head,
                    device, belief_cfg=belief_cfg, deterministic=False,
                    collect_raw_states=True,
                )
                collected.extend(states)
            _supervised_update_ebm(
                variant=variant, env=env, filter_backbone=filter_backbone,
                energy_net=energy_net, apsi_head=apsi_head, raw_states=collected,
                device=device, belief_cfg=belief_cfg or BeliefConfig(), cfg=nes_cfg,
            )

        if gen % nes_cfg.print_every == 0:
            print(
                f"[{env.name}:{variant}:nes] seed={seed} gen={gen}/{nes_cfg.generations}"
                f" mean_fit={fitness.mean():.4f} sigma={sigma:.5f} success={success:.3f}"
            )
        maybe_run_selection(gen)

    # Final evaluation at the last generation
    last_eval_bundle = evaluate_modules_nes(
        variant=variant, env_factory=env_factory, filter_backbone=filter_backbone,
        actor=actor, energy_net=energy_net, apsi_head=apsi_head, device=device,
        train_cfg=train_cfg, spce_L=spce_L, snmc_L=snmc_L, belief_cfg=belief_cfg,
        n_eval_episodes=nes_cfg.eval_episodes, collect_paths=False,
    )

    # Re-evaluate all top-K candidates with more episodes for final selection
    final_eval_eps = int(nes_cfg.selection_final_eval_episodes) if int(nes_cfg.selection_final_eval_episodes) > 0 else int(nes_cfg.eval_episodes)
    final_candidates: List[Dict[str, object]] = []
    if top_candidates:
        for cand in top_candidates:
            _load_module_state(actor, cand["actor_state"])
            _load_module_state(energy_net, cand["energy_state"])
            _load_module_state(apsi_head, cand["apsi_state"])
            final_bundle = evaluate_modules_nes(
                variant=variant, env_factory=env_factory, filter_backbone=filter_backbone,
                actor=actor, energy_net=energy_net, apsi_head=apsi_head, device=device,
                train_cfg=train_cfg, spce_L=spce_L, snmc_L=snmc_L, belief_cfg=belief_cfg,
                n_eval_episodes=final_eval_eps, collect_paths=False,
            )
            final_eval = final_bundle["eval"]
            final_diag = final_bundle.get("cross_diagnostics")
            final_candidates.append({
                "generation": int(cand["generation"]),
                "preview_score": float(cand["score"]),
                "preview_eval": dict(cand["preview_eval"]),
                "preview_cross_diagnostics": cand.get("preview_cross_diagnostics"),
                "final_eval": dict(final_eval),
                "final_cross_diagnostics": dict(final_diag) if final_diag else None,
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
        variant=variant, env_factory=env_factory, filter_backbone=filter_backbone,
        actor=actor, energy_net=energy_net, apsi_head=apsi_head, device=device,
        train_cfg=train_cfg, spce_L=spce_L, snmc_L=snmc_L, belief_cfg=belief_cfg,
        n_eval_episodes=nes_cfg.eval_episodes, collect_paths=True,
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
            "best_score": float(
                best_score if best_score > -float("inf")
                else _selection_score(variant, last_eval_bundle["eval"], nes_cfg, cross_diag_metrics=last_eval_bundle.get("cross_diagnostics"))
            ),
            "best_preview_eval": best_eval_preview if best_eval_preview else dict(last_eval_bundle["eval"]),
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


# ---------------------------------------------------------------------------
# Multi-seed experiment suite
# ---------------------------------------------------------------------------

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
    """Run all (variant × seed) combinations and write summary JSON.

    Args:
        experiment_name: Identifies the experiment in printed logs.
        env_factory:     Callable ``device → GenericBankBOEDEnv``.
        output_dir:      Root directory for results.
        train_cfg:       Actor / EBM architecture hyper-parameters.
        nes_cfg:         NES optimisation hyper-parameters.
        seeds:           List of random seeds to run per variant.
        variants:        List of variant names to compare.
        spce_L:          Number of contrastive samples for SPCE lower bound.
        snmc_L:          Number of samples for SNMC-style upper bound (0 = skip).
        belief_cfg:      EBM belief configuration.

    Returns:
        Summary dict written to ``output_dir/summary_multi_seed.json``.
    """
    ensure_dir(output_dir)
    all_results: Dict[str, List[Dict]] = {}
    _dev = nes_cfg.device if nes_cfg.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    sample_env = env_factory(torch.device(_dev))

    for variant in variants:
        print(f"\n============================\nNES {experiment_name} variant: {variant}\n============================")
        variant_results = []
        for seed in seeds:
            variant_results.append(
                train_one_seed_nes(
                    variant, env_factory, train_cfg, nes_cfg, seed, output_dir,
                    spce_L, snmc_L, belief_cfg=belief_cfg,
                )
            )
        all_results[variant] = variant_results

    # Aggregate statistics
    summary: Dict[str, object] = {}
    tracked_fields = [
        "avg_return", "avg_bank_ig", "avg_filter_bank_ig",
        "avg_spce_lower", "avg_snmc_style_upper",
    ]
    cross_diag_fields = [
        "avg_actor_belief_feature_mae", "avg_actor_belief_prob_l1",
        "avg_actor_belief_map_to_exact_distance", "avg_actor_belief_map_to_true_distance",
        "avg_full_belief_mean_error", "avg_full_belief_kl", "avg_full_belief_l1",
        "avg_full_belief_map_to_exact_distance", "avg_full_belief_map_to_true_distance",
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
            vals_cd = [
                r["cross_diagnostics"][field]
                for r in results
                if r.get("cross_diagnostics") and field in r["cross_diagnostics"]
            ]
            if vals_cd:
                summary[f"{variant}_{field}"] = mean_std_ci95(np.array(vals_cd, dtype=np.float64))
            vals_cd_last = [
                r["cross_diagnostics_last"][field]
                for r in results
                if r.get("cross_diagnostics_last") and field in r["cross_diagnostics_last"]
            ]
            if vals_cd_last:
                summary[f"{variant}_last_{field}"] = mean_std_ci95(np.array(vals_cd_last, dtype=np.float64))

    if "blau_approx" in all_results:
        blau = np.array(
            [r["eval"]["avg_return"] for r in all_results["blau_approx"]], dtype=np.float64
        )
        for variant in variants:
            if variant == "blau_approx":
                continue
            cur = np.array(
                [r["eval"]["avg_return"] for r in all_results[variant]], dtype=np.float64
            )
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
    print("\n=== Multi-seed NES summary ===")
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Backward-compatible aliases for legacy experiment scripts
# ---------------------------------------------------------------------------

#: Alias kept for backward compatibility with older experiment scripts.
EvolutionConfig = NESConfig

#: Alias kept for backward compatibility with older experiment scripts.
run_experiment_suite_evolution = run_experiment_suite_nes
