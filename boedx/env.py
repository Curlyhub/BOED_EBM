"""
Abstract base class for bank-based sequential BOED environments plus
the two shared configuration dataclasses.

Concrete environments subclass ``GenericBankBOEDEnv`` and implement:
  - ``get_horizon`` / ``get_action_low`` / ``get_action_high``
  - ``build_hypothesis_bank`` / ``build_prior_bank_logits``
  - ``sample_theta`` / ``sample_prior_thetas``
  - ``clip_action``
  - ``sample_observation`` / ``loglik_scalar``
  - ``trajectory_loglik_thetas`` / ``bank_loglik_single``

Optional hooks:
  - ``current_aux_state`` — extra state appended to the policy input.
  - ``before_episode`` — called at the start of each episode reset.
  - ``after_step_update_aux`` — budget / cost tracking per step.
  - ``observation_to_feature_scalar`` — custom obs normalisation.
  - ``project_equivalence_logits`` — symmetry-aware filter projection.
  - ``belief_distance`` — custom geometry for MAP-error metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from boedx.utils import gaussian_logpdf  # noqa: F401  (re-exported for subclasses)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GenericTrainConfig:
    """Hyper-parameters controlling training, evaluation, and model selection."""

    # Training budget
    episodes: int = 300
    eval_episodes: int = 60
    # Replay and optimisation
    batch_size: int = 128
    replay_size: int = 50_000
    warmup_episodes: int = 20
    updates_per_step: int = 1
    gamma: float = 1.0
    tau: float = 0.01
    alpha: float = 0.1
    # Learning rates
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_ebm: float = 3e-4
    # Network sizes
    hidden_rl: int = 256
    hidden_ebm: int = 512
    # Gradient and loss coefficients
    grad_clip: float = 10.0
    apsi_coef: float = 0.1
    ebm_update_every: int = 4
    # Logging
    print_every: int = 25
    device: str = "cpu"
    seeds: str = "0,1,2"
    # Model selection: periodically evaluate and keep the best checkpoint
    selection_eval_episodes: int = 40
    selection_every: int = 100
    selection_start_episode: int = 100
    selection_return_weight: float = 1.0
    selection_bank_ig_weight: float = 0.0
    selection_spce_weight: float = 0.0
    selection_survival_risk_weight: float = 0.0
    selection_fallback_weight: float = 0.0
    selection_belief_kl_weight: float = 0.10
    selection_belief_mean_weight: float = 0.50
    selection_belief_map_weight: float = 0.25
    # Actor regularisation — applies to the actor only (not EBM / critic)
    actor_weight_decay: float = 0.0
    actor_dropout: float = 0.0
    # Policy family shared by all variants for a fair comparison
    # "gaussian"        — single-Gaussian SAC actor (historical default)
    # "mog"             — mixture-of-Gaussians actor (more expressive)
    # "categorical"     — discrete categorical actor (prey-population default)
    # "categorical_moe" — discrete mixture-of-experts categorical actor
    # "transformer"     — NES-only path-aware Transformer actor
    actor_family: str = "gaussian"
    actor_mixture_components: int = 4

    # EBM-only actor overrides (NES). Empty/0 means reuse the shared actor settings.
    ebm_actor_family: str = ""
    ebm_hidden_rl: int = 0
    ebm_actor_mixture_components: int = 0
    # Separate encoder branches for quotient state and belief features (NES).
    ebm_dual_branch_actor: bool = False

    # Transformer actor hyper-parameters (NES only; ignored by SAC trainer).
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_layers: int = 2
    transformer_ff: int = 128

    # Phase-adaptive late-horizon residual head for continuous actors.
    # When True, a smooth cubic gate activates a correction head for the
    # final portion of the horizon, improving late-phase precision in NES.
    phase_adaptive_actor: bool = False
    phase_start_frac: float = 0.6   # fraction of horizon at which gate opens
    phase_strength: float = 1.0     # multiplier on the late correction
    late_std_scale: float = 0.5     # std scale factor applied late in horizon
    late_mix_temp: float = 0.75     # mixture temperature applied late in horizon


@dataclass
class BeliefConfig:
    """Configuration for the EBM belief module and feature extraction."""

    # How the EBM belief is coupled to the policy
    # "exact"               — skip EBM; use the exact filter probabilities directly
    # "distilled_detached"  — EBM trained offline, gradient blocked to actor
    # "distilled_e2e"       — EBM co-trained end-to-end with actor
    # "learned_only"        — actor sees only belief features (no quotient base)
    mode: str = "distilled_detached"

    # Which belief summary features the actor receives
    # "legacy"  — posterior mean + entropy  (1-D era default)
    # "moments" — mean + diag(Σ) + upper-triangular Σ + entropy
    # "modal"   — "moments" plus the top-K highest-probability bank atoms
    feature_mode: str = "legacy"

    # EBM architecture
    # "standard"  — generic MLP (EnergyNet / CrossInteractionEnergyNet)
    # "geometric" — permutation-invariant Deep Sets (SymmetricSource*)
    ebm_architecture: str = "standard"

    # Parameters for "geometric" architecture (K-source problems)
    n_sources: int = 1
    source_dim: int = 0           # 0 → inferred as theta_dim // n_sources
    add_pairwise_dist: bool = True

    # Number of top-probability atoms to expose for "modal" feature mode
    modal_top_k: int = 4

    # NES-specific actor-side overrides.  The EBM can still be trained against
    # the full posterior while the actor consumes a more compact representation.
    # Empty string / 0 means: reuse feature_mode / modal_top_k unchanged.
    nes_actor_feature_mode: str = ""      # override feature_mode for the actor
    nes_actor_modal_top_k: int = 0        # override modal_top_k for the actor
    # When True, cross-variant actors receive a compact moments-only belief
    # (top_k capped at 2) regardless of nes_actor_feature_mode.
    nes_cross_compact_belief: bool = False
    # Prepend the full raw (action, obs) history to the EBM actor state (NES
    # transformer actor needs this to parse the path as sequence tokens).
    include_raw_history_for_ebm_actor: bool = False

    # Beta schedule for beta-contrastive variants:
    #   β_t = β_end + (β_start - β_end) · (1 - t/(T-1))^β_power
    beta_start: float = 0.95
    beta_end: float = 0.05
    beta_power: float = 1.5


# ---------------------------------------------------------------------------
# Abstract environment
# ---------------------------------------------------------------------------

class GenericBankBOEDEnv:
    """Abstract bank-based sequential BOED environment with SPCE-style reward.

    The reward at each step is the one-step SPCE increment, approximated via a
    running contrastive particle set:

        r_t = log p(y_t | θ₀, d_t)  −  Δ log Z_t

    where Z_t is estimated by the contrastive log-weights ``logC``.
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

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by subclasses
    # ------------------------------------------------------------------

    def get_horizon(self) -> int:
        raise NotImplementedError

    def get_action_low(self) -> np.ndarray:
        raise NotImplementedError

    def get_action_high(self) -> np.ndarray:
        raise NotImplementedError

    def build_hypothesis_bank(self) -> torch.Tensor:
        """Return an (H, theta_dim) tensor of all candidate hypotheses."""
        raise NotImplementedError

    def build_prior_bank_logits(self) -> torch.Tensor:
        """Return (H,) log-prior weights over the hypothesis bank."""
        raise NotImplementedError

    def sample_theta(self) -> np.ndarray:
        """Draw a single latent θ₀ from the prior."""
        raise NotImplementedError

    def sample_prior_thetas(self, n: int) -> torch.Tensor:
        """Draw n i.i.d. samples from the prior (for SPCE / SNMC estimation)."""
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
        """Return (N,) log-likelihoods for N hypotheses over the full trajectory."""
        raise NotImplementedError

    def bank_loglik_single(
        self, obs_t: torch.Tensor, action_t: torch.Tensor
    ) -> torch.Tensor:
        """Return (H,) log-likelihood of a single (obs, action) pair for each bank atom."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional hooks with sensible defaults
    # ------------------------------------------------------------------

    def current_aux_state(self) -> np.ndarray:
        """Return environment-specific auxiliary features appended to policy input."""
        return np.zeros((0,), dtype=np.float32)

    def before_episode(self) -> None:
        """Called at the start of each episode (before θ₀ is sampled)."""

    def after_step_update_aux(
        self, prev_action: Optional[np.ndarray], action: np.ndarray, obs: float
    ) -> Tuple[bool, float]:
        """Called after every step to update budget / cost counters.

        Returns:
            (terminate_now, reward_adjustment)
        """
        return False, 0.0

    def observation_to_feature_scalar(self, obs: float) -> float:
        """Transform raw obs to the scalar stored in the policy state (default: identity)."""
        return float(obs)

    def project_equivalence_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Quotient-space projection for the control-filter logits (default: identity)."""
        return logits

    def belief_distance(self, theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
        """Distance between two hypothesis tensors for MAP-error metrics."""
        return torch.linalg.norm(theta_a - theta_b, dim=-1)

    # ------------------------------------------------------------------
    # Filter / posterior management
    # ------------------------------------------------------------------

    def posterior_bank(self) -> torch.Tensor:
        return torch.exp(F.log_softmax(self._posterior_logu, dim=0))

    def equivalence_bank(self) -> torch.Tensor:
        return torch.exp(F.log_softmax(self._equiv_logu, dim=0))

    def posterior_init_logits(self) -> torch.Tensor:
        return self.prior_bank_logits.clone()

    def equivalence_filter_mode(self) -> str:
        mode = getattr(self, "exact_filter", "likelihood")
        if mode not in {"likelihood", "posterior"}:
            raise ValueError(f"Unsupported exact_filter={mode!r}.")
        return mode

    def equivalence_init_logits(self) -> torch.Tensor:
        """Initial logits for the control-side filter.

        ``"likelihood"`` accumulates raw log-likelihoods (no prior bias).
        ``"posterior"``  initialises with the prior (explicit ablation only).
        """
        if self.equivalence_filter_mode() == "posterior":
            return self.prior_bank_logits.clone()
        return torch.zeros(self.H, dtype=torch.float32, device=self.device)

    def accumulate_filter_logits(
        self, cur_logits: torch.Tensor, obs_t: torch.Tensor, action_t: torch.Tensor
    ) -> torch.Tensor:
        ll_bank = self.bank_loglik_single(obs_t, action_t)
        new_logits = torch.nan_to_num(cur_logits + ll_bank, nan=0.0, posinf=1e4, neginf=-1e4)
        return new_logits - torch.max(new_logits)

    def posterior_update_logits(
        self, cur_logits: torch.Tensor, obs_t: torch.Tensor, action_t: torch.Tensor
    ) -> torch.Tensor:
        return self.accumulate_filter_logits(cur_logits, obs_t, action_t)

    def equivalence_update_logits(
        self, cur_logits: torch.Tensor, obs_t: torch.Tensor, action_t: torch.Tensor
    ) -> torch.Tensor:
        logits = self.accumulate_filter_logits(cur_logits, obs_t, action_t)
        return self.project_equivalence_logits(logits)

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def get_n_contrastive(self) -> int:
        return getattr(self, "n_contrastive", 32)

    def reset(self) -> List[Tuple[np.ndarray, float]]:
        self.before_episode()
        self.theta0 = self.sample_theta()
        n_ctr = self.get_n_contrastive()
        self.contrastive_thetas = [self.theta0] + [
            self.sample_theta() for _ in range(n_ctr)
        ]
        self._contrastive_t = torch.tensor(
            np.stack(self.contrastive_thetas, axis=0), dtype=torch.float32, device=self.device
        )
        self.logC = np.zeros(n_ctr + 1, dtype=np.float64)
        self.history: List[Tuple[np.ndarray, float]] = []
        self.t = 0
        self.last_obs = 0.0
        self.last_action: Optional[np.ndarray] = None
        self._posterior_logu = self.posterior_init_logits()
        self._equiv_logu = self.equivalence_init_logits()
        return self.history

    def step(self, action: np.ndarray) -> Tuple[float, float, bool, Dict]:
        """Execute one design choice and return (obs, reward, done, info)."""
        action = self.clip_action(np.asarray(action, dtype=np.float32))
        obs = self.sample_observation(self.theta0, action)

        # SPCE reward: log p(y|θ₀,d) − Δ log Z  (contrastive estimate)
        prev_logsum = float(
            np.log(np.exp(self.logC - self.logC.max()).sum()) + self.logC.max()
        )
        ll_real = self.loglik_scalar(obs, self.theta0, action)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device)
        ll_all = (
            self.trajectory_loglik_thetas(
                np.asarray([action], dtype=np.float32),
                np.asarray([obs], dtype=np.float32),
                self._contrastive_t,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        self.logC = self.logC + ll_all
        new_logsum = float(
            np.log(np.exp(self.logC - self.logC.max()).sum()) + self.logC.max()
        )
        reward = ll_real - new_logsum + prev_logsum

        with torch.no_grad():
            self._posterior_logu = self.posterior_update_logits(
                self._posterior_logu, obs_t, action_t
            )
            self._equiv_logu = self.equivalence_update_logits(
                self._equiv_logu, obs_t, action_t
            )

        terminate_now, reward_adjustment = self.after_step_update_aux(
            self.last_action, action, obs
        )
        reward = float(reward + reward_adjustment)

        self.history.append((action.copy(), float(obs)))
        self.last_obs = float(self.observation_to_feature_scalar(obs))
        self.last_action = action.copy()
        self.t += 1
        done = self.t >= self.get_horizon() or terminate_now
        info = {
            "theta0": self.theta0.copy(),
            "history": list(self.history),
            "reward_dense": reward,
        }
        return obs, reward, done, info
