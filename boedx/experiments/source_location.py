"""
2-D two-source localisation benchmark.

Task description
----------------
A sensor takes sequential scalar measurements at 2-D design locations d_t.
Each measurement y_t is drawn from a log-normal model driven by two hidden
point sources at θ₁, θ₂ ∈ ℝ²:

    μ(θ, d) = bkg + 1/(m + ‖θ₁ - d‖²)  +  1/(m + ‖θ₂ - d‖²)
    y_t ~ N(log μ(θ, d_t), σ²)

The goal is to maximise the total information gain (EIG) about the source
positions over a horizon of T experiments.

Compared variants
-----------------
+---------------------------------------+--------------------------------------------+
| Variant name                          | Policy-state representation                |
+=======================================+============================================+
| ``blau_approx``                       | Flattened raw (action, obs) history        |
+---------------------------------------+--------------------------------------------+
| ``ours_ebm_control_posterior``        | Quotient + EBM-control (posterior)         |
| ``ours_ebm_control_beta_contrastive`` | Quotient + EBM-control (β-contrastive)     |
| ``ours_ebm_cross_posterior``          | Quotient + EBM-cross (posterior)           |
| ``ours_ebm_cross_beta_contrastive``   | Quotient + EBM-cross (β-contrastive)       |
+---------------------------------------+--------------------------------------------+

Canonical pair symmetry
-----------------------
The two sources are unordered — swapping θ₁ ↔ θ₂ gives the same physical
configuration.  All sampling and distance computations canonicalise pairs
so that the first source is lexicographically ≤ the second.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from boedx.env import BeliefConfig, GenericBankBOEDEnv, GenericTrainConfig
from boedx.trainer import run_experiment_suite
from boedx.utils import gaussian_logpdf


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SourceLocationConfig:
    """Hyper-parameters for the 2-D two-source localisation environment."""

    horizon: int = 30
    prior_std: float = 1.0
    design_min: float = -4.0
    design_max: float = 4.0
    # Observation model: bkg + 1/(m + dist²)
    bkg: float = 0.1
    m: float = 1e-4
    sigma: float = 0.35
    # Contrastive particle count (affects SPCE reward variance)
    n_contrastive: int = 128
    # Discrete hypothesis bank (grid over ℝ²)
    bank_grid_min: float = -3.0
    bank_grid_max: float = 3.0
    bank_grid_size: int = 7   # grid_size² single-source positions → ~K*(K+1)/2 pairs
    # Filter mode for the control-side exact filter
    exact_filter: str = "likelihood"   # "likelihood" or "posterior"
    # Auxiliary budget state (disabled by default)
    enable_aux_state: bool = False
    budget_total: float = 24.0
    move_cost_coef: float = 1.0
    probe_cost: float = 0.35
    budget_violation_penalty: float = -2.0


# ---------------------------------------------------------------------------
# Canonical pair helpers
# ---------------------------------------------------------------------------

def _canonical_pair_numpy(theta: np.ndarray) -> np.ndarray:
    """Sort θ = (θ₁, θ₂) so that θ₁ ≤ θ₂ lexicographically."""
    theta = np.asarray(theta, dtype=np.float32).reshape(2, 2)
    order = np.lexsort((theta[:, 1], theta[:, 0]))
    return theta[order].reshape(-1).astype(np.float32)


def _canonical_pair_torch(theta: torch.Tensor) -> torch.Tensor:
    """Batched canonical sort for (B, 4) latent pairs."""
    orig_shape = theta.shape
    th = theta.reshape(-1, 2, 2)
    x0, y0 = th[:, 0, 0], th[:, 0, 1]
    x1, y1 = th[:, 1, 0], th[:, 1, 1]
    swap = (x0 > x1) | ((x0 == x1) & (y0 > y1))
    out = th.clone()
    out[swap] = th[swap][:, [1, 0], :]
    return out.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SourceLocalization2DEnv(GenericBankBOEDEnv):
    """2-D two-source localisation BOED environment.

    The hypothesis bank enumerates all unordered pairs of grid positions.
    For a K×K grid with K=7 this gives K²=49 single-source positions and
    49×50/2 = 1225 unordered pairs (including diagonal i=j pairs where the
    two sources coincide).
    """

    name = "source_location"
    action_dim = 2           # 2-D design location d_t ∈ ℝ²

    def __init__(self, cfg: SourceLocationConfig, device: torch.device):
        self.cfg = cfg
        self.n_contrastive = cfg.n_contrastive
        self.exact_filter = cfg.exact_filter
        self.remaining_budget = float(cfg.budget_total)
        self.cumulative_cost = 0.0
        super().__init__(device=device)

    # -- Environment interface --

    def get_horizon(self) -> int:
        return self.cfg.horizon

    def get_action_low(self) -> np.ndarray:
        return np.full((2,), self.cfg.design_min, dtype=np.float32)

    def get_action_high(self) -> np.ndarray:
        return np.full((2,), self.cfg.design_max, dtype=np.float32)

    def _single_positions(self) -> torch.Tensor:
        g = torch.linspace(self.cfg.bank_grid_min, self.cfg.bank_grid_max, self.cfg.bank_grid_size)
        xs, ys = torch.meshgrid(g, g, indexing="ij")
        return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).float()

    def build_hypothesis_bank(self) -> torch.Tensor:
        """Enumerate all unordered (θ₁, θ₂) grid pairs."""
        singles = self._single_positions()
        p = singles.shape[0]
        pairs = [
            torch.cat([singles[i], singles[j]], dim=0)
            for i in range(p) for j in range(i, p)
        ]
        return torch.stack(pairs, dim=0).float()

    def build_prior_bank_logits(self) -> torch.Tensor:
        """Log-prior over the bank, corrected for pair multiplicity."""
        bank = self.hypothesis_bank
        s1, s2 = bank[:, 0:2], bank[:, 2:4]
        std = self.cfg.prior_std
        logp = gaussian_logpdf(s1, torch.zeros_like(s1), std).sum(dim=-1)
        logp += gaussian_logpdf(s2, torch.zeros_like(s2), std).sum(dim=-1)
        # Unordered pairs with s1 ≠ s2 represent two ordered configurations
        same = (s1[:, 0] == s2[:, 0]) & (s1[:, 1] == s2[:, 1])
        log_mult = torch.where(same, torch.zeros_like(logp), torch.full_like(logp, math.log(2.0)))
        logp = logp + log_mult
        return (logp - torch.logsumexp(logp, dim=0)).float()

    def sample_theta(self) -> np.ndarray:
        th = np.random.randn(4).astype(np.float32) * self.cfg.prior_std
        return _canonical_pair_numpy(th)

    def sample_prior_thetas(self, n: int) -> torch.Tensor:
        th = torch.randn(n, 4, device=self.device) * self.cfg.prior_std
        return _canonical_pair_torch(th)

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.cfg.design_min, self.cfg.design_max).astype(np.float32)

    # -- Observation model --

    def _mu(self, theta: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """Expected intensity μ(θ, d) = bkg + Σᵢ 1/(m + ‖θᵢ - d‖²)."""
        s1, s2 = theta[..., 0:2], theta[..., 2:4]
        dist1 = ((s1 - d) ** 2).sum(dim=-1)
        dist2 = ((s2 - d) ** 2).sum(dim=-1)
        return self.cfg.bkg + 1.0 / (self.cfg.m + dist1) + 1.0 / (self.cfg.m + dist2)

    def sample_observation(self, theta: np.ndarray, action: np.ndarray) -> float:
        th = torch.tensor(theta, dtype=torch.float32, device=self.device)
        d = torch.tensor(action, dtype=torch.float32, device=self.device)
        mean = torch.log(self._mu(th, d))
        return float((mean + torch.randn((), device=self.device) * self.cfg.sigma).detach().cpu())

    def loglik_scalar(self, obs: float, theta: np.ndarray, action: np.ndarray) -> float:
        th = torch.tensor(theta, dtype=torch.float32, device=self.device)
        d = torch.tensor(action, dtype=torch.float32, device=self.device)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device)
        return float(gaussian_logpdf(x, torch.log(self._mu(th, d)), self.cfg.sigma).detach().cpu())

    def trajectory_loglik_thetas(
        self, actions: np.ndarray, obs: np.ndarray, thetas: torch.Tensor
    ) -> torch.Tensor:
        if len(obs) == 0:
            return torch.zeros(thetas.shape[0], device=thetas.device)
        a = torch.tensor(actions[: len(obs)], dtype=torch.float32, device=thetas.device)
        y = torch.tensor(obs[: len(obs)], dtype=torch.float32, device=thetas.device)
        mu = self._mu(thetas[:, None, :], a[None, :, :])
        return gaussian_logpdf(y[None, :], torch.log(mu), self.cfg.sigma).sum(dim=-1)

    def bank_loglik_single(
        self, obs_t: torch.Tensor, action_t: torch.Tensor
    ) -> torch.Tensor:
        bank = self.hypothesis_bank
        dist1 = ((bank[:, 0:2] - action_t) ** 2).sum(dim=-1)
        dist2 = ((bank[:, 2:4] - action_t) ** 2).sum(dim=-1)
        mu_bank = self.cfg.bkg + 1.0 / (self.cfg.m + dist1) + 1.0 / (self.cfg.m + dist2)
        return gaussian_logpdf(obs_t, torch.log(mu_bank), self.cfg.sigma)

    # -- Geometry for belief evaluation --

    def belief_distance(self, theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
        """Permutation-invariant distance: min(direct, swapped) / √2."""
        a = theta_a.reshape(-1, 2, 2)
        b = theta_b.reshape(-1, 2, 2)
        direct = torch.sqrt(
            ((a[:, 0] - b[:, 0]) ** 2).sum(dim=-1) + ((a[:, 1] - b[:, 1]) ** 2).sum(dim=-1)
        )
        swapped = torch.sqrt(
            ((a[:, 0] - b[:, 1]) ** 2).sum(dim=-1) + ((a[:, 1] - b[:, 0]) ** 2).sum(dim=-1)
        )
        return torch.minimum(direct, swapped) / np.sqrt(2.0)

    # -- Budget / auxiliary state --

    def before_episode(self) -> None:
        self.remaining_budget = float(self.cfg.budget_total)
        self.cumulative_cost = 0.0

    def current_aux_state(self) -> np.ndarray:
        if not self.cfg.enable_aux_state:
            return np.zeros((0,), dtype=np.float32)
        total = max(self.cfg.budget_total, 1e-6)
        return np.array(
            [self.remaining_budget / total, self.cumulative_cost / total], dtype=np.float32
        )

    def after_step_update_aux(
        self, prev_action: np.ndarray | None, action: np.ndarray, obs: float
    ) -> tuple:
        del obs
        if not self.cfg.enable_aux_state:
            return False, 0.0
        move_cost = (
            0.0 if prev_action is None
            else self.cfg.move_cost_coef * float(np.linalg.norm(action - prev_action))
        )
        step_cost = self.cfg.probe_cost + move_cost
        self.remaining_budget -= step_cost
        self.cumulative_cost += step_cost
        if self.remaining_budget < 0.0:
            return True, float(self.cfg.budget_violation_penalty)
        return False, 0.0


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

VARIANTS: list[str] = [
    "blau_approx",
    "ours_ebm_control_posterior",
    "ours_ebm_control_beta_contrastive",
    "ours_ebm_cross_posterior",
    "ours_ebm_cross_beta_contrastive",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2-D two-source localisation BOED benchmark."
    )
    p.add_argument("--episodes",        type=int,   default=300)
    p.add_argument("--eval-episodes",   type=int,   default=40)
    p.add_argument("--seeds",           type=str,   default="0,1,2")
    p.add_argument("--device",          type=str,   default="auto")
    p.add_argument("--output-dir",      type=str,   default="./outputs/source_location_compare")
    p.add_argument("--variants",        type=str,   default=",".join(VARIANTS))
    p.add_argument("--horizon",         type=int,   default=30)
    p.add_argument("--prior-std",       type=float, default=1.0)
    p.add_argument("--design-min",      type=float, default=-4.0)
    p.add_argument("--design-max",      type=float, default=4.0)
    p.add_argument("--bank-grid-size",  type=int,   default=7)
    p.add_argument("--bank-grid-min",   type=float, default=-3.0)
    p.add_argument("--bank-grid-max",   type=float, default=3.0)
    p.add_argument("--sigma",           type=float, default=0.35)
    p.add_argument("--hidden-rl",       type=int,   default=1024,
                   help="Actor/critic width — wider is safer for 2-D source.")
    p.add_argument("--hidden-ebm",      type=int,   default=256)
    p.add_argument("--batch-size",      type=int,   default=128)
    p.add_argument("--gamma",           type=float, default=1.0)
    p.add_argument("--spce-L",          type=int,   default=1024)
    p.add_argument("--snmc-L",          type=int,   default=512)
    p.add_argument("--exact-filter",    type=str,   default="likelihood",
                   choices=["likelihood", "posterior"])
    p.add_argument("--belief-mode",     type=str,   default="distilled_detached",
                   choices=["exact", "distilled_detached", "distilled_e2e", "learned_only"])
    p.add_argument("--belief-feature-mode", type=str, default="modal",
                   choices=["legacy", "moments", "modal"])
    p.add_argument("--ebm-architecture",   type=str,   default="geometric",
                   choices=["standard", "geometric"])
    p.add_argument("--n-sources",       type=int,   default=2)
    p.add_argument("--source-dim",      type=int,   default=2)
    p.add_argument("--modal-top-k",     type=int,   default=4)
    p.add_argument("--add-pairwise-dist",  action="store_true", default=True)
    p.add_argument("--no-pairwise-dist",   dest="add_pairwise_dist", action="store_false")
    p.add_argument("--n-contrastive",   type=int,   default=128)
    p.add_argument("--beta-start",      type=float, default=0.95)
    p.add_argument("--beta-end",        type=float, default=0.05)
    p.add_argument("--beta-power",      type=float, default=1.5)
    p.add_argument("--selection-eval-episodes",    type=int,   default=40)
    p.add_argument("--selection-every",            type=int,   default=100)
    p.add_argument("--selection-start-episode",    type=int,   default=100)
    p.add_argument("--selection-return-weight",    type=float, default=1.0)
    p.add_argument("--selection-belief-kl-weight", type=float, default=0.10)
    p.add_argument("--selection-belief-map-weight",type=float, default=0.25)
    p.add_argument("--selection-belief-mean-weight",type=float,default=0.50)
    p.add_argument("--actor-weight-decay", type=float, default=0.0)
    p.add_argument("--actor-dropout",      type=float, default=0.0)
    p.add_argument("--actor-family",       type=str,   default="mog",
                   choices=["gaussian", "mog"])
    p.add_argument("--actor-mixture-components", type=int, default=4)
    p.add_argument("--enable-aux-state",   action="store_true")
    p.add_argument("--budget-total",       type=float, default=24.0)
    p.add_argument("--move-cost-coef",     type=float, default=1.0)
    p.add_argument("--probe-cost",         type=float, default=0.35)
    p.add_argument("--budget-violation-penalty", type=float, default=-2.0)
    return p.parse_args()


def main() -> None:
    """CLI entry point: ``boedx-source-location`` or ``python -m boedx.experiments.source_location``."""
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    cfg = SourceLocationConfig(
        horizon=args.horizon,
        prior_std=args.prior_std,
        design_min=args.design_min,
        design_max=args.design_max,
        sigma=args.sigma,
        bank_grid_size=args.bank_grid_size,
        bank_grid_min=args.bank_grid_min,
        bank_grid_max=args.bank_grid_max,
        exact_filter=args.exact_filter,
        n_contrastive=args.n_contrastive,
        enable_aux_state=args.enable_aux_state,
        budget_total=args.budget_total,
        move_cost_coef=args.move_cost_coef,
        probe_cost=args.probe_cost,
        budget_violation_penalty=args.budget_violation_penalty,
    )
    train_cfg = GenericTrainConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        batch_size=args.batch_size,
        device=device,
        seeds=args.seeds,
        hidden_rl=args.hidden_rl,
        hidden_ebm=args.hidden_ebm,
        gamma=args.gamma,
        print_every=25,
        selection_eval_episodes=args.selection_eval_episodes,
        selection_every=args.selection_every,
        selection_start_episode=args.selection_start_episode,
        selection_return_weight=args.selection_return_weight,
        selection_belief_kl_weight=args.selection_belief_kl_weight,
        selection_belief_map_weight=args.selection_belief_map_weight,
        selection_belief_mean_weight=args.selection_belief_mean_weight,
        actor_weight_decay=args.actor_weight_decay,
        actor_dropout=args.actor_dropout,
        actor_family=args.actor_family,
        actor_mixture_components=args.actor_mixture_components,
    )
    seeds: Sequence[int] = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants: list[str] = [v.strip() for v in args.variants.split(",") if v.strip()]
    belief_cfg = BeliefConfig(
        mode=args.belief_mode,
        feature_mode=args.belief_feature_mode,
        ebm_architecture=args.ebm_architecture,
        n_sources=args.n_sources,
        source_dim=args.source_dim,
        modal_top_k=args.modal_top_k,
        add_pairwise_dist=args.add_pairwise_dist,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_power=args.beta_power,
    )

    def factory(dev: torch.device) -> SourceLocalization2DEnv:
        return SourceLocalization2DEnv(cfg=cfg, device=dev)

    summary = run_experiment_suite(
        experiment_name="source_location_compare",
        env_factory=factory,
        output_dir=args.output_dir,
        train_cfg=train_cfg,
        seeds=seeds,
        variants=variants,
        spce_L=args.spce_L,
        snmc_L=args.snmc_L,
        belief_cfg=belief_cfg,
    )
    summary["config"] = vars(args)
    with open(os.path.join(args.output_dir, "run_config_and_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
