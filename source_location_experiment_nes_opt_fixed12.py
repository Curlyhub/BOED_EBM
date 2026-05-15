from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from boedx.env import BeliefConfig, GenericBankBOEDEnv, GenericTrainConfig
from boedx.nes_trainer import NESConfig, run_experiment_suite_nes
from boedx.utils import gaussian_logpdf


@dataclass
class SourceLocationConfig:
    horizon: int = 30
    prior_std: float = 1.0
    design_min: float = -4.0
    design_max: float = 4.0
    bkg: float = 0.1
    m: float = 1e-4
    sigma: float = 0.35
    n_contrastive: int = 128
    bank_grid_min: float = -3.0
    bank_grid_max: float = 3.0
    bank_grid_size: int = 7
    exact_filter: str = "likelihood"
    enable_aux_state: bool = False
    budget_total: float = 24.0
    move_cost_coef: float = 1.0
    probe_cost: float = 0.35
    budget_violation_penalty: float = -2.0


def _canonical_pair_numpy(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float32).reshape(2, 2)
    order = np.lexsort((theta[:, 1], theta[:, 0]))
    return theta[order].reshape(-1).astype(np.float32)


def _canonical_pair_torch(theta: torch.Tensor) -> torch.Tensor:
    orig_shape = theta.shape
    th = theta.reshape(-1, 2, 2)
    x0, y0 = th[:, 0, 0], th[:, 0, 1]
    x1, y1 = th[:, 1, 0], th[:, 1, 1]
    swap = (x0 > x1) | ((x0 == x1) & (y0 > y1))
    out = th.clone()
    out[swap] = th[swap][:, [1, 0], :]
    return out.reshape(orig_shape)


class SourceLocalization2DEnv(GenericBankBOEDEnv):
    name = "source_location"
    action_dim = 2

    def __init__(self, cfg: SourceLocationConfig, device: torch.device):
        self.cfg = cfg
        self.n_contrastive = cfg.n_contrastive
        self.exact_filter = cfg.exact_filter
        self.remaining_budget = float(cfg.budget_total)
        self.cumulative_cost = 0.0
        super().__init__(device=device)

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
        singles = self._single_positions()
        p = singles.shape[0]
        pairs = []
        for i in range(p):
            for j in range(i, p):
                pairs.append(torch.cat([singles[i], singles[j]], dim=0))
        return torch.stack(pairs, dim=0).float()

    def build_prior_bank_logits(self) -> torch.Tensor:
        bank = self.hypothesis_bank
        s1, s2 = bank[:, 0:2], bank[:, 2:4]
        std = self.cfg.prior_std
        logp = gaussian_logpdf(s1, torch.zeros_like(s1), std).sum(dim=-1)
        logp += gaussian_logpdf(s2, torch.zeros_like(s2), std).sum(dim=-1)
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

    def _mu(self, theta: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        s1 = theta[..., 0:2]
        s2 = theta[..., 2:4]
        dist1 = ((s1 - d) ** 2).sum(dim=-1)
        dist2 = ((s2 - d) ** 2).sum(dim=-1)
        return self.cfg.bkg + 1.0 / (self.cfg.m + dist1) + 1.0 / (self.cfg.m + dist2)

    def sample_observation(self, theta: np.ndarray, action: np.ndarray) -> float:
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
        d_t = torch.tensor(action, dtype=torch.float32, device=self.device)
        mean = torch.log(self._mu(theta_t, d_t))
        noise = torch.randn((), device=self.device) * self.cfg.sigma
        return float((mean + noise).detach().cpu())

    def loglik_scalar(self, obs: float, theta: np.ndarray, action: np.ndarray) -> float:
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
        d_t = torch.tensor(action, dtype=torch.float32, device=self.device)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device)
        mean = torch.log(self._mu(theta_t, d_t))
        return float(gaussian_logpdf(x, mean, self.cfg.sigma).detach().cpu())

    def trajectory_loglik_thetas(self, actions: np.ndarray, obs: np.ndarray, thetas: torch.Tensor) -> torch.Tensor:
        if len(obs) == 0:
            return torch.zeros(thetas.shape[0], device=thetas.device)
        a = torch.tensor(actions[: len(obs)], dtype=torch.float32, device=thetas.device)
        y = torch.tensor(obs[: len(obs)], dtype=torch.float32, device=thetas.device)
        mu = self._mu(thetas[:, None, :], a[None, :, :])
        mean = torch.log(mu)
        ll = gaussian_logpdf(y[None, :], mean, self.cfg.sigma)
        return ll.sum(dim=-1)

    def bank_loglik_single(self, obs_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        bank = self.hypothesis_bank
        dist1 = ((bank[:, 0:2] - action_t) ** 2).sum(dim=-1)
        dist2 = ((bank[:, 2:4] - action_t) ** 2).sum(dim=-1)
        mu_bank = self.cfg.bkg + 1.0 / (self.cfg.m + dist1) + 1.0 / (self.cfg.m + dist2)
        mean_bank = torch.log(mu_bank)
        return gaussian_logpdf(obs_t, mean_bank, self.cfg.sigma)

    def belief_distance(self, theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
        a = theta_a.reshape(-1, 2, 2)
        b = theta_b.reshape(-1, 2, 2)
        direct = torch.sqrt(((a[:, 0] - b[:, 0]) ** 2).sum(dim=-1) + ((a[:, 1] - b[:, 1]) ** 2).sum(dim=-1))
        swapped = torch.sqrt(((a[:, 0] - b[:, 1]) ** 2).sum(dim=-1) + ((a[:, 1] - b[:, 0]) ** 2).sum(dim=-1))
        return torch.minimum(direct, swapped) / np.sqrt(2.0)

    def before_episode(self) -> None:
        self.remaining_budget = float(self.cfg.budget_total)
        self.cumulative_cost = 0.0

    def current_aux_state(self) -> np.ndarray:
        if not self.cfg.enable_aux_state:
            return np.zeros((0,), dtype=np.float32)
        total = max(self.cfg.budget_total, 1e-6)
        return np.array([self.remaining_budget / total, self.cumulative_cost / total], dtype=np.float32)

    def after_step_update_aux(self, prev_action: np.ndarray | None, action: np.ndarray, obs: float):
        del obs
        if not self.cfg.enable_aux_state:
            return False, 0.0
        move_cost = 0.0 if prev_action is None else self.cfg.move_cost_coef * float(np.linalg.norm(action - prev_action))
        step_cost = self.cfg.probe_cost + move_cost
        self.remaining_budget -= step_cost
        self.cumulative_cost += step_cost
        if self.remaining_budget < 0.0:
            return True, float(self.cfg.budget_violation_penalty)
        return False, 0.0


VARIANTS = [
    "blau_approx",
    "control_posterior_exact",
    "control_filter_exact",
    "ours_ebm_cross_posterior",
    "ours_ebm_cross_filter",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2D source-localisation benchmark with NES-specific optimisation.")
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--population-size", type=int, default=48)
    p.add_argument("--rollout-episodes-per-candidate", type=int, default=2)
    p.add_argument("--eval-episodes", type=int, default=150)
    p.add_argument("--nes-lr-mu", type=float, default=0.04)
    p.add_argument("--nes-lr-sigma", type=float, default=0.10)
    p.add_argument("--nes-sigma-init", type=float, default=0.03)
    p.add_argument("--nes-sigma-final", type=float, default=0.005)
    p.add_argument("--nes-sigma-schedule", type=str, default="exp", choices=["constant", "linear", "exp"])
    p.add_argument("--nes-mirrored-sampling", action="store_true", default=True)
    p.add_argument("--no-nes-mirrored-sampling", dest="nes_mirrored_sampling", action="store_false")
    p.add_argument("--nes-utility-mode", type=str, default="nes", choices=["nes", "centered_ranks"])
    p.add_argument("--nes-optimizer", type=str, default="adam", choices=["adam", "rmsprop", "sgd"])
    p.add_argument("--nes-beta1", type=float, default=0.9)
    p.add_argument("--nes-beta2", type=float, default=0.999)
    p.add_argument("--nes-eps", type=float, default=1e-8)
    p.add_argument("--nes-sigma-adapt-on-success", action="store_true")
    p.add_argument("--nes-sigma-success-target", type=float, default=0.20)
    p.add_argument("--nes-sigma-adapt-rate", type=float, default=0.05)
    p.add_argument("--ebm-updates-per-generation", type=int, default=10)
    p.add_argument("--ebm-batch-size", type=int, default=128)
    p.add_argument("--ebm-data-episodes", type=int, default=8)
    p.add_argument("--ebm-pretrain-episodes", type=int, default=0)
    p.add_argument("--ebm-pretrain-updates", type=int, default=0)
    p.add_argument("--ebm-update-every-generations", type=int, default=1)
    p.add_argument("--freeze-ebm", action="store_true")
    p.add_argument("--ebm-freeze-after-generation", type=int, default=-1)
    p.add_argument("--use-common-random-numbers", action="store_true")
    p.add_argument("--common-random-numbers-seed-stride", type=int, default=1000003)
    p.add_argument("--reevaluate-top-candidates", type=int, default=0)
    p.add_argument("--reevaluate-top-episodes", type=int, default=0)
    p.add_argument("--parallel-candidates", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--parallel-backend", type=str, default="threading", choices=["threading", "loky"])
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="./outputs/source_location_nes")
    p.add_argument("--variants", type=str, default=",".join(["blau_approx", "ours_ebm_cross_posterior"]))
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--prior-std", type=float, default=1.0)
    p.add_argument("--design-min", type=float, default=-4.0)
    p.add_argument("--design-max", type=float, default=4.0)
    p.add_argument("--bank-grid-size", type=int, default=7)
    p.add_argument("--bank-grid-min", type=float, default=-3.0)
    p.add_argument("--bank-grid-max", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--hidden-rl", type=int, default=256)
    p.add_argument("--hidden-ebm", type=int, default=256)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--spce-L", type=int, default=1024)
    p.add_argument("--snmc-L", type=int, default=512)
    p.add_argument("--exact-filter", type=str, default="likelihood", choices=["likelihood", "posterior"])
    p.add_argument("--belief-mode", type=str, default="distilled_detached", choices=["exact", "distilled_detached", "distilled_e2e", "learned_only"])
    p.add_argument("--belief-feature-mode", type=str, default="modal", choices=["legacy", "moments", "modal"])
    p.add_argument("--ebm-architecture", type=str, default="geometric", choices=["standard", "geometric"])
    p.add_argument("--n-sources", type=int, default=2)
    p.add_argument("--source-dim", type=int, default=2)
    p.add_argument("--modal-top-k", type=int, default=4)
    p.add_argument("--nes-actor-feature-mode", type=str, default="", choices=["", "legacy", "moments", "modal"])
    p.add_argument("--nes-actor-modal-top-k", type=int, default=0)
    p.add_argument("--nes-cross-compact-belief", action="store_true")
    p.add_argument("--add-pairwise-dist", action="store_true", default=True)
    p.add_argument("--no-pairwise-dist", dest="add_pairwise_dist", action="store_false")
    p.add_argument("--n-contrastive", type=int, default=128)
    p.add_argument("--selection-eval-episodes", type=int, default=40)
    p.add_argument("--selection-every", type=int, default=10)
    p.add_argument("--selection-start-generation", type=int, default=10)
    p.add_argument("--selection-top-k", type=int, default=3)
    p.add_argument("--selection-final-eval-episodes", type=int, default=0)
    p.add_argument("--selection-return-weight", type=float, default=1.0)
    p.add_argument("--selection-belief-kl-weight", type=float, default=0.05)
    p.add_argument("--selection-belief-map-weight", type=float, default=0.15)
    p.add_argument("--selection-belief-mean-weight", type=float, default=0.25)
    p.add_argument("--cross-selection-bank-ig-weight", type=float, default=0.30)
    p.add_argument("--cross-selection-spce-weight", type=float, default=0.20)
    p.add_argument("--cross-selection-filter-bank-ig-weight", type=float, default=0.00)
    p.add_argument("--cross-selection-gap-penalty-weight", type=float, default=0.35)
    p.add_argument("--cross-selection-belief-kl-weight", type=float, default=0.02)
    p.add_argument("--cross-selection-belief-map-weight", type=float, default=0.04)
    p.add_argument("--cross-selection-belief-mean-weight", type=float, default=0.04)
    p.add_argument("--actor-family", type=str, default="mog", choices=["gaussian", "mog", "transformer"])
    p.add_argument("--actor-mixture-components", type=int, default=4)
    p.add_argument("--ebm-actor-family", type=str, default="", choices=["", "gaussian", "mog", "transformer"])
    p.add_argument("--ebm-hidden-rl", type=int, default=0)
    p.add_argument("--ebm-actor-mixture-components", type=int, default=0)
    p.add_argument("--ebm-dual-branch-actor", action="store_true")
    p.add_argument("--transformer-d-model", type=int, default=64)
    p.add_argument("--transformer-nhead", type=int, default=4)
    p.add_argument("--transformer-layers", type=int, default=2)
    p.add_argument("--transformer-ff", type=int, default=128)
    p.add_argument("--phase-adaptive-actor", action="store_true")
    p.add_argument("--phase-start-frac", type=float, default=0.6)
    p.add_argument("--phase-strength", type=float, default=1.0)
    p.add_argument("--late-std-scale", type=float, default=0.5)
    p.add_argument("--late-mix-temp", type=float, default=0.75)
    p.add_argument("--enable-aux-state", action="store_true")
    p.add_argument("--budget-total", type=float, default=24.0)
    p.add_argument("--move-cost-coef", type=float, default=1.0)
    p.add_argument("--probe-cost", type=float, default=0.35)
    p.add_argument("--budget-violation-penalty", type=float, default=-2.0)
    return p.parse_args()


def main() -> None:
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
        episodes=args.generations,
        eval_episodes=args.eval_episodes,
        device=device,
        seeds=args.seeds,
        hidden_rl=args.hidden_rl,
        hidden_ebm=args.hidden_ebm,
        gamma=args.gamma,
        print_every=10,
        actor_family=args.actor_family,
        actor_mixture_components=args.actor_mixture_components,
        ebm_actor_family=args.ebm_actor_family,
        ebm_hidden_rl=args.ebm_hidden_rl,
        ebm_actor_mixture_components=args.ebm_actor_mixture_components,
        ebm_dual_branch_actor=args.ebm_dual_branch_actor,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_layers=args.transformer_layers,
        transformer_ff=args.transformer_ff,
        phase_adaptive_actor=args.phase_adaptive_actor,
        phase_start_frac=args.phase_start_frac,
        phase_strength=args.phase_strength,
        late_std_scale=args.late_std_scale,
        late_mix_temp=args.late_mix_temp,
    )
    # Used internally by the transformer actor to parse path tokens.
    train_cfg.sequence_horizon = args.horizon

    nes_cfg = NESConfig(
        generations=args.generations,
        population_size=args.population_size,
        rollout_episodes_per_candidate=args.rollout_episodes_per_candidate,
        eval_episodes=args.eval_episodes,
        lr_mu=args.nes_lr_mu,
        lr_sigma=args.nes_lr_sigma,
        sigma_init=args.nes_sigma_init,
        sigma_final=args.nes_sigma_final,
        sigma_schedule=args.nes_sigma_schedule,
        mirrored_sampling=args.nes_mirrored_sampling,
        utility_mode=args.nes_utility_mode,
        optimizer=args.nes_optimizer,
        beta1=args.nes_beta1,
        beta2=args.nes_beta2,
        eps=args.nes_eps,
        sigma_adapt_on_success=args.nes_sigma_adapt_on_success,
        sigma_success_target=args.nes_sigma_success_target,
        sigma_adapt_rate=args.nes_sigma_adapt_rate,
        device=device,
        seeds=args.seeds,
        selection_eval_episodes=args.selection_eval_episodes,
        selection_every=args.selection_every,
        selection_start_generation=args.selection_start_generation,
        selection_top_k=args.selection_top_k,
        selection_final_eval_episodes=args.selection_final_eval_episodes,
        selection_return_weight=args.selection_return_weight,
        selection_belief_kl_weight=args.selection_belief_kl_weight,
        selection_belief_map_weight=args.selection_belief_map_weight,
        selection_belief_mean_weight=args.selection_belief_mean_weight,
        cross_selection_bank_ig_weight=args.cross_selection_bank_ig_weight,
        cross_selection_spce_weight=args.cross_selection_spce_weight,
        cross_selection_filter_bank_ig_weight=args.cross_selection_filter_bank_ig_weight,
        cross_selection_gap_penalty_weight=args.cross_selection_gap_penalty_weight,
        cross_selection_belief_kl_weight=args.cross_selection_belief_kl_weight,
        cross_selection_belief_map_weight=args.cross_selection_belief_map_weight,
        cross_selection_belief_mean_weight=args.cross_selection_belief_mean_weight,
        ebm_updates_per_generation=args.ebm_updates_per_generation,
        ebm_batch_size=args.ebm_batch_size,
        ebm_data_episodes=args.ebm_data_episodes,
        ebm_pretrain_episodes=args.ebm_pretrain_episodes,
        ebm_pretrain_updates=args.ebm_pretrain_updates,
        ebm_update_every_generations=args.ebm_update_every_generations,
        freeze_ebm=args.freeze_ebm,
        ebm_freeze_after_generation=args.ebm_freeze_after_generation,
        use_common_random_numbers=args.use_common_random_numbers,
        common_random_numbers_seed_stride=args.common_random_numbers_seed_stride,
        reevaluate_top_candidates=args.reevaluate_top_candidates,
        reevaluate_top_episodes=args.reevaluate_top_episodes,
        parallel_candidates=args.parallel_candidates,
        n_jobs=args.n_jobs,
        parallel_backend=args.parallel_backend,
    )
    seeds: Sequence[int] = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    belief_cfg = BeliefConfig(
        mode=args.belief_mode,
        feature_mode=args.belief_feature_mode,
        ebm_architecture=args.ebm_architecture,
        n_sources=args.n_sources,
        source_dim=args.source_dim,
        modal_top_k=args.modal_top_k,
        add_pairwise_dist=args.add_pairwise_dist,
        nes_actor_feature_mode=args.nes_actor_feature_mode,
        nes_actor_modal_top_k=args.nes_actor_modal_top_k,
        nes_cross_compact_belief=args.nes_cross_compact_belief,
        include_raw_history_for_ebm_actor=(args.ebm_actor_family == "transformer" or args.actor_family == "transformer"),
    )

    def factory(dev: torch.device) -> SourceLocalization2DEnv:
        return SourceLocalization2DEnv(cfg=cfg, device=dev)

    summary = run_experiment_suite_nes(
        experiment_name="source_location_nes",
        env_factory=factory,
        output_dir=args.output_dir,
        train_cfg=train_cfg,
        nes_cfg=nes_cfg,
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
