"""
Dose-response benchmark — sequential BOED with a Hill/log-logistic model.

Task description
----------------
The experimenter sequentially chooses a log-dose z_t = log x_t and observes a
continuous biological response

    y_t ~ Normal(mu(z_t; theta), sigma^2)

where the mean response is a Hill/log-logistic curve

    mu(z; theta) = E0 + s * Delta * sigmoid(h * (z - eta)),

with s = +1 for increasing response and s = -1 for decreasing/toxic response.
The unknown parameters are

    theta = (E0, log Delta, eta, log h)                 if sigma is fixed
    theta = (E0, log Delta, eta, log h, log sigma)      if --infer-noise

where eta = log EC50. The action is the log-dose itself, constrained to
[log(dose_min), log(dose_max)]. This keeps the optimisation continuous while
respecting positive physical doses.

Compared variants
-----------------
+--------------------------------+--------------------------------------------+
| Variant name                   | Policy-state representation                |
+================================+============================================+
| ``blau_approx``                | Flattened raw (action, obs) history        |
| ``control_posterior_exact``    | Exact posterior/filter quotient state      |
| ``ours_ebm_control_posterior`` | Quotient + EBM-control posterior belief    |
| ``ours_ebm_cross_posterior``   | Quotient + EBM-cross posterior belief      |
+--------------------------------+--------------------------------------------+

This benchmark is deliberately simple and biologically interpretable: the
policy learns where to dose in order to reduce uncertainty about baseline,
effect size, EC50, and Hill slope.
"""

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
from boedx.homeostatic import HomeostaticConfig
from boedx.trainer import run_experiment_suite


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DoseResponseConfig:
    """Hyper-parameters for the Hill/log-logistic dose-response BOED task."""

    # Sequential-design horizon
    horizon: int = 20

    # Physical dose range. The action used by the policy is log-dose.
    dose_min: float = 0.01
    dose_max: float = 100.0

    # Observation noise. Used directly unless infer_noise=True.
    sigma: float = 0.08
    infer_noise: bool = False

    # Response direction: +1 for efficacy/increasing response, -1 for toxicity/decreasing response.
    response_direction: str = "increasing"  # "increasing" or "decreasing"

    # Prior over E0, log Delta, eta=log EC50, log h, and optionally log sigma.
    prior_e0_mean: float = 0.0
    prior_e0_sd: float = 0.05
    prior_delta_mean: float = 0.8
    prior_log_delta_sd: float = 0.25
    prior_hill_mean: float = 1.5
    prior_log_hill_sd: float = 0.4
    prior_log_sigma_sd: float = 0.35

    # Hypothesis bank and contrastive particles.
    bank_size: int = 512
    n_contrastive: int = 128

    # Filter initialisation: "likelihood" accumulates raw log-likelihoods.
    exact_filter: str = "likelihood"

    @property
    def log_dose_min(self) -> float:
        return float(math.log(self.dose_min))

    @property
    def log_dose_max(self) -> float:
        return float(math.log(self.dose_max))

    @property
    def eta_mean(self) -> float:
        return 0.5 * (self.log_dose_min + self.log_dose_max)

    @property
    def eta_sd(self) -> float:
        # Roughly covers the tested log-dose range by +/- 1.5 sd.
        return (self.log_dose_max - self.log_dose_min) / 3.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class DoseResponseEnv(GenericBankBOEDEnv):
    """Continuous log-dose BOED environment for Hill/log-logistic response."""

    name = "dose_response"
    action_dim = 1

    def __init__(self, cfg: DoseResponseConfig, device: torch.device):
        self.cfg = cfg
        self.n_contrastive = cfg.n_contrastive
        self.exact_filter = cfg.exact_filter
        super().__init__(device=device)

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def get_horizon(self) -> int:
        return self.cfg.horizon

    def get_action_low(self) -> np.ndarray:
        return np.array([self.cfg.log_dose_min], dtype=np.float32)

    def get_action_high(self) -> np.ndarray:
        return np.array([self.cfg.log_dose_max], dtype=np.float32)

    def _prior_components_torch(self, n: int, device: torch.device) -> torch.Tensor:
        e0 = torch.randn(n, 1, device=device) * self.cfg.prior_e0_sd + self.cfg.prior_e0_mean

        log_delta = (
            torch.randn(n, 1, device=device) * self.cfg.prior_log_delta_sd
            + math.log(self.cfg.prior_delta_mean)
        )

        eta = torch.randn(n, 1, device=device) * self.cfg.eta_sd + self.cfg.eta_mean

        log_h = (
            torch.randn(n, 1, device=device) * self.cfg.prior_log_hill_sd
            + math.log(self.cfg.prior_hill_mean)
        )

        if not self.cfg.infer_noise:
            return torch.cat([e0, log_delta, eta, log_h], dim=-1)

        log_sigma = (
            torch.randn(n, 1, device=device) * self.cfg.prior_log_sigma_sd
            + math.log(self.cfg.sigma)
        )
        return torch.cat([e0, log_delta, eta, log_h, log_sigma], dim=-1)

    def sample_theta(self) -> np.ndarray:
        return self._prior_components_torch(1, self.device).squeeze(0).detach().cpu().numpy().astype(np.float32)

    def sample_prior_thetas(self, n: int) -> torch.Tensor:
        return self._prior_components_torch(n, self.device)

    def build_hypothesis_bank(self) -> torch.Tensor:
        """Sample a finite bank from the prior and store it on CPU."""
        return self.sample_prior_thetas(self.cfg.bank_size).detach().cpu()

    def build_prior_bank_logits(self) -> torch.Tensor:
        """Uniform prior over a prior-sampled bank."""
        return torch.full(
            (self.cfg.bank_size,),
            -math.log(self.cfg.bank_size),
            dtype=torch.float32,
        )

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip log-dose to the admissible design interval."""
        return np.clip(action, self.cfg.log_dose_min, self.cfg.log_dose_max).astype(np.float32)

    # ------------------------------------------------------------------
    # Hill/log-logistic observation model
    # ------------------------------------------------------------------

    def _decode_theta(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode unconstrained theta coordinates into interpretable parameters."""
        e0 = theta[..., 0]
        delta = torch.exp(theta[..., 1]).clamp_min(1e-8)
        eta = theta[..., 2]
        h = torch.exp(theta[..., 3]).clamp_min(1e-4)
        if self.cfg.infer_noise:
            sigma = torch.exp(theta[..., 4]).clamp_min(1e-5)
        else:
            sigma = torch.full_like(e0, float(self.cfg.sigma))
        return e0, delta, eta, h, sigma

    def _mean_response(self, log_dose: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Hill/log-logistic mean response at log-dose z = log x."""
        e0, delta, eta, h, _sigma = self._decode_theta(theta)
        sign = 1.0 if self.cfg.response_direction == "increasing" else -1.0
        q = torch.sigmoid(h * (log_dose - eta))
        return e0 + sign * delta * q

    @staticmethod
    def _normal_logpdf(x: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.clamp_min(1e-8)
        z = (x - mean) / sigma
        return -0.5 * (z * z + 2.0 * torch.log(sigma) + math.log(2.0 * math.pi))

    def sample_observation(self, theta: np.ndarray, action: np.ndarray) -> float:
        th = torch.tensor(theta, dtype=torch.float32, device=self.device)
        z = torch.tensor(float(action[0]), dtype=torch.float32, device=self.device)
        mu = self._mean_response(z, th)
        _e0, _delta, _eta, _h, sigma = self._decode_theta(th)
        y = mu + torch.randn((), device=self.device) * sigma
        return float(y.detach().cpu())

    def loglik_scalar(self, obs: float, theta: np.ndarray, action: np.ndarray) -> float:
        th = torch.tensor(theta, dtype=torch.float32, device=self.device)
        z = torch.tensor(float(action[0]), dtype=torch.float32, device=self.device)
        y = torch.tensor(float(obs), dtype=torch.float32, device=self.device)
        mu = self._mean_response(z, th)
        _e0, _delta, _eta, _h, sigma = self._decode_theta(th)
        return float(self._normal_logpdf(y, mu, sigma).detach().cpu())

    def trajectory_loglik_thetas(
        self, actions: np.ndarray, obs: np.ndarray, thetas: torch.Tensor
    ) -> torch.Tensor:
        """Sum Normal log-likelihoods over a prefix trajectory for each theta."""
        T = len(obs)
        if T == 0:
            return torch.zeros(thetas.shape[0], device=thetas.device)

        z = torch.tensor(actions[:T, 0], dtype=torch.float32, device=thetas.device)  # (T,)
        y = torch.tensor(obs[:T], dtype=torch.float32, device=thetas.device)         # (T,)

        # Broadcast: theta particles H x 1 x D, log-doses 1 x T.
        th = thetas[:, None, :]
        z_bt = z[None, :]
        mu = self._mean_response(z_bt, th)  # (H, T)
        _e0, _delta, _eta, _h, sigma = self._decode_theta(thetas)
        sigma_bt = sigma[:, None]
        return self._normal_logpdf(y[None, :], mu, sigma_bt).sum(dim=-1)

    def bank_loglik_single(
        self, obs_t: torch.Tensor, action_t: torch.Tensor
    ) -> torch.Tensor:
        """Normal log-likelihood for a single (obs, action) pair over the bank."""
        bank = self.hypothesis_bank.to(self.device)
        z = action_t.reshape(()).to(self.device)
        mu = self._mean_response(z, bank)
        _e0, _delta, _eta, _h, sigma = self._decode_theta(bank)
        return self._normal_logpdf(obs_t.to(self.device), mu, sigma)

    # ------------------------------------------------------------------
    # Diagnostics / geometry
    # ------------------------------------------------------------------

    def observation_to_feature_scalar(self, obs: float) -> float:
        """Keep response features numerically mild for raw-history baselines."""
        return float(np.clip(obs, -2.0, 2.0))

    def belief_distance(self, theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
        """Scale-aware parameter distance for belief-quality diagnostics."""
        scale_vals = [
            max(self.cfg.prior_e0_sd, 1e-6),
            max(self.cfg.prior_log_delta_sd, 1e-6),
            max(self.cfg.eta_sd, 1e-6),
            max(self.cfg.prior_log_hill_sd, 1e-6),
        ]
        if self.cfg.infer_noise:
            scale_vals.append(max(self.cfg.prior_log_sigma_sd, 1e-6))
        scale = torch.tensor(scale_vals, dtype=torch.float32, device=theta_a.device)
        diff = (theta_a - theta_b) / scale
        return torch.sqrt(torch.mean(diff * diff, dim=-1).clamp_min(0.0))

    def dose_from_action(self, action: np.ndarray | torch.Tensor) -> np.ndarray:
        """Convenience helper for plotting: convert log-dose action to physical dose."""
        z = torch.as_tensor(action, dtype=torch.float32)
        return torch.exp(z).detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

VARIANTS: list[str] = [
    "blau_approx",
    "control_posterior_exact",
    "ours_ebm_control_posterior",
    "ours_ebm_cross_posterior",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hill/log-logistic dose-response BOED benchmark."
    )
    p.add_argument("--episodes",             type=int,   default=1200)
    p.add_argument("--eval-episodes",        type=int,   default=80)
    p.add_argument("--seeds",                type=str,   default="0,1,2")
    p.add_argument("--device",               type=str,   default="auto")
    p.add_argument("--output-dir",           type=str,   default="./outputs/dose_response")
    p.add_argument("--variants",             type=str,   default=",".join(VARIANTS))

    p.add_argument("--horizon",              type=int,   default=20)
    p.add_argument("--dose-min",             type=float, default=0.01)
    p.add_argument("--dose-max",             type=float, default=100.0)
    p.add_argument("--sigma",                type=float, default=0.08)
    p.add_argument("--infer-noise",          action="store_true")
    p.add_argument("--response-direction",   type=str,   default="increasing",
                   choices=["increasing", "decreasing"])

    p.add_argument("--bank-size",            type=int,   default=512)
    p.add_argument("--hidden-ebm",           type=int,   default=256)
    p.add_argument("--hidden-rl",            type=int,   default=256)
    p.add_argument("--batch-size",           type=int,   default=128)
    p.add_argument("--actor-family",         type=str,   default="mog",
                   choices=["gaussian", "mog"])
    p.add_argument("--actor-mixture-components", type=int, default=4)
    p.add_argument("--gamma",                type=float, default=0.95)
    p.add_argument("--ebm-update-every",     type=int,   default=4)

    p.add_argument("--spce-L",               type=int,   default=256)
    p.add_argument("--snmc-L",               type=int,   default=0)
    p.add_argument("--n-contrastive",        type=int,   default=128)
    p.add_argument("--exact-filter",         type=str,   default="likelihood",
                   choices=["likelihood", "posterior"])

    p.add_argument("--belief-mode",          type=str,   default="distilled_detached",
                   choices=["exact", "distilled_detached", "distilled_e2e", "learned_only"])
    p.add_argument("--belief-feature-mode",  type=str,   default="moments",
                   choices=["legacy", "moments", "modal"])
    p.add_argument("--modal-top-k",          type=int,   default=4)
    p.add_argument("--ebm-moe-enabled",      action="store_true")
    p.add_argument("--ebm-moe-experts",      type=str,   default="identity,standard,cross")
    p.add_argument("--ebm-moe-router-hidden", type=int,  default=128)
    p.add_argument("--ebm-moe-router-temp",  type=float, default=1.0)
    p.add_argument("--ebm-moe-entropy-reg",  type=float, default=0.0)
    p.add_argument("--ebm-moe-mode",         type=str,   default="measure_mixture",
                   choices=["measure_mixture", "energy_blend"])

    # Prior controls. Defaults are for normalised responses in roughly [0, 1].
    p.add_argument("--prior-e0-mean",        type=float, default=0.0)
    p.add_argument("--prior-e0-sd",          type=float, default=0.05)
    p.add_argument("--prior-delta-mean",     type=float, default=0.8)
    p.add_argument("--prior-log-delta-sd",   type=float, default=0.25)
    p.add_argument("--prior-hill-mean",      type=float, default=1.5)
    p.add_argument("--prior-log-hill-sd",    type=float, default=0.4)
    p.add_argument("--prior-log-sigma-sd",   type=float, default=0.35)

    # Checkpoint selection. Enabled by default because this benchmark is intended
    # to be presentation-friendly under reduced training budgets.
    p.add_argument("--selection-start-episode", type=int, default=300)
    p.add_argument("--selection-every", type=int, default=100)
    p.add_argument("--selection-eval-episodes", type=int, default=20)
    p.add_argument("--selection-return-weight", type=float, default=1.0)
    p.add_argument("--selection-bank-ig-weight", type=float, default=0.0)
    p.add_argument("--selection-spce-weight", type=float, default=0.0)
    p.add_argument("--selection-belief-kl-weight", type=float, default=0.05)
    p.add_argument("--selection-belief-mean-weight", type=float, default=0.10)
    p.add_argument("--selection-belief-map-weight", type=float, default=0.05)

    return p.parse_args()


def main() -> None:
    """CLI entry point: ``boedx-dose-response``."""
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    cfg = DoseResponseConfig(
        horizon=args.horizon,
        dose_min=args.dose_min,
        dose_max=args.dose_max,
        sigma=args.sigma,
        infer_noise=args.infer_noise,
        response_direction=args.response_direction,
        prior_e0_mean=args.prior_e0_mean,
        prior_e0_sd=args.prior_e0_sd,
        prior_delta_mean=args.prior_delta_mean,
        prior_log_delta_sd=args.prior_log_delta_sd,
        prior_hill_mean=args.prior_hill_mean,
        prior_log_hill_sd=args.prior_log_hill_sd,
        prior_log_sigma_sd=args.prior_log_sigma_sd,
        bank_size=args.bank_size,
        n_contrastive=args.n_contrastive,
        exact_filter=args.exact_filter,
    )

    train_cfg = GenericTrainConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        batch_size=args.batch_size,
        device=device,
        seeds=args.seeds,
        hidden_rl=args.hidden_rl,
        hidden_ebm=args.hidden_ebm,
        actor_family=args.actor_family,
        actor_mixture_components=args.actor_mixture_components,
        gamma=args.gamma,
        ebm_update_every=args.ebm_update_every,
        print_every=25,
        selection_start_episode=args.selection_start_episode,
        selection_every=args.selection_every,
        selection_eval_episodes=args.selection_eval_episodes,
        selection_return_weight=args.selection_return_weight,
        selection_bank_ig_weight=args.selection_bank_ig_weight,
        selection_spce_weight=args.selection_spce_weight,
        selection_belief_kl_weight=args.selection_belief_kl_weight,
        selection_belief_mean_weight=args.selection_belief_mean_weight,
        selection_belief_map_weight=args.selection_belief_map_weight,
    )

    seeds: Sequence[int] = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    belief_cfg = BeliefConfig(
        mode=args.belief_mode,
        feature_mode=args.belief_feature_mode,
        modal_top_k=args.modal_top_k,
        ebm_moe_enabled=args.ebm_moe_enabled or any(v.startswith("ours_ebm_moe") for v in variants),
        ebm_moe_experts=args.ebm_moe_experts,
        ebm_moe_router_hidden=args.ebm_moe_router_hidden,
        ebm_moe_router_temp=args.ebm_moe_router_temp,
        ebm_moe_entropy_reg=args.ebm_moe_entropy_reg,
        ebm_moe_mode=args.ebm_moe_mode,
    )

    def factory(dev: torch.device) -> DoseResponseEnv:
        return DoseResponseEnv(cfg=cfg, device=dev)

    summary = run_experiment_suite(
        experiment_name="dose_response",
        env_factory=factory,
        output_dir=args.output_dir,
        train_cfg=train_cfg,
        seeds=seeds,
        variants=variants,
        spce_L=args.spce_L,
        snmc_L=args.snmc_L,
        belief_cfg=belief_cfg,
        homeostatic_cfg=HomeostaticConfig(enabled=False, mode="none"),
    )
    summary["model"] = {
        "mean": "E0 + sign * Delta * sigmoid(h * (log_dose - log_EC50))",
        "theta_fixed_noise": ["E0", "log_delta", "log_EC50", "log_h"],
        "theta_infer_noise": ["E0", "log_delta", "log_EC50", "log_h", "log_sigma"],
        "action": "log_dose",
        "physical_dose_range": [args.dose_min, args.dose_max],
    }
    summary["config"] = vars(args)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_config_and_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
