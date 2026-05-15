"""
Prey population benchmark — discrete sequential BOED experiment.

Task description
----------------
A predator–prey system follows the Holling type-II functional-response ODE:

    dN/dt = −(a N²) / (1 + a Th N²)

where N(t) is the prey population and T_f is the fixed observation window.
The experimenter chooses d_t ∈ {1, …, 300} initial prey individuals and
observes:

    y_t ~ Binomial(d_t, p_T),    p_T = (N_0 − N_T) / N_0

The unknown parameters are θ = (log a, log Th) drawn from independent
Gaussian priors:  log a, log Th ~ N(−1.4, 1.35²).

The goal is to maximise total information gain (EIG) about the predator
functional-response parameters over a horizon of T experiments.

Closed-form ODE solution
------------------------
The ODE admits a closed-form solution used in place of the Runge–Kutta
integrator for speed and numerical stability:

    a Th N_T² + C N_T − 1 = 0,    C = aT + 1/N_0 − a Th N_0

solved as N_T = (−C + √(C² + 4 a Th)) / (2 a Th), with a stable limit
N_T = 1 / (aT + 1/N_0) when Th → 0.

Compared variants
-----------------
+---------------------+-----------------------------------------------+
| Variant name        | Policy-state representation                   |
+=====================+===============================================+
| ``blau_approx``     | Flattened raw (action, obs) history           |
+---------------------+-----------------------------------------------+
| ``ours_ebm_control``| Quotient state + EBM-control belief           |
| ``ours_ebm_cross``  | Quotient state + EBM-cross belief             |
+---------------------+-----------------------------------------------+

Discrete action space
---------------------
The design is an integer initial prey count d_t ∈ {1, …, 300}.
This environment flags ``name == "prey_population"`` so the trainer
automatically selects a ``DiscreteCategoricalActor`` instead of the
continuous SAC actor used for source-localisation.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from boedx.env import BeliefConfig, GenericBankBOEDEnv, GenericTrainConfig
from boedx.trainer import run_experiment_suite


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreyConfig:
    """Hyper-parameters for the prey-population BOED environment."""

    # Sequential-design horizon
    horizon: int = 10
    # Discrete action range: initial prey count d_t ∈ {design_min, …, design_max}
    design_min: float = 1.0
    design_max: float = 300.0
    # ODE parameters
    final_time: float = 24.0    # observation window length (hours)
    ode_steps: int = 120        # kept for reference; closed-form is used instead
    # Hypothesis bank
    bank_size: int = 512
    # Contrastive particle count (affects SPCE reward variance)
    n_contrastive: int = 128
    # Filter initialisation: "likelihood" accumulates raw log-likelihoods
    exact_filter: str = "likelihood"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PreyPopulationEnv(GenericBankBOEDEnv):
    """Prey-population discrete BOED environment.

    The hypothesis bank is a random sample of size ``bank_size`` from the
    log-normal prior.  Observations are Binomial counts of consumed prey.
    The action space is a finite set of integer initial prey counts.

    Because the action is a 1-D integer, the trainer dispatch logic in
    ``boedx.trainer.uses_discrete_actor`` automatically selects a
    ``DiscreteCategoricalActor`` over {design_min, …, design_max}.
    """

    name = "prey_population"
    action_dim = 1

    def __init__(self, cfg: PreyConfig, device: torch.device):
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
        return np.array([self.cfg.design_min], dtype=np.float32)

    def get_action_high(self) -> np.ndarray:
        return np.array([self.cfg.design_max], dtype=np.float32)

    def sample_theta(self) -> np.ndarray:
        """Draw (log a, log Th) from N(−1.4, 1.35²) independently."""
        log_a = np.random.randn() * 1.35 - 1.4
        log_Th = np.random.randn() * 1.35 - 1.4
        return np.array([log_a, log_Th], dtype=np.float32)

    def sample_prior_thetas(self, n: int) -> torch.Tensor:
        log_a = torch.randn(n, 1, device=self.device) * 1.35 - 1.4
        log_Th = torch.randn(n, 1, device=self.device) * 1.35 - 1.4
        return torch.cat([log_a, log_Th], dim=-1)

    def build_hypothesis_bank(self) -> torch.Tensor:
        """Sample ``bank_size`` hypotheses from the prior (stored on CPU)."""
        return self.sample_prior_thetas(self.cfg.bank_size).detach().cpu()

    def build_prior_bank_logits(self) -> torch.Tensor:
        """Uniform prior logits over the bank (log(1/H) per atom)."""
        return torch.full(
            (self.cfg.bank_size,),
            -math.log(self.cfg.bank_size),
            dtype=torch.float32,
        )

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip to [design_min, design_max] and round to the nearest integer."""
        return np.round(
            np.clip(action, self.cfg.design_min, self.cfg.design_max)
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # ODE and likelihood
    # ------------------------------------------------------------------

    def _integrate_terminal_population(
        self, N0: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """Closed-form terminal prey population N_T.

        Solves  a Th N_T² + C N_T − 1 = 0  where
            C = a T_f + 1/N_0 − a Th N_0.

        Falls back to the Th→0 limit N_T = 1/(aT_f + 1/N_0) when a Th
        is negligible to avoid division by near-zero.
        """
        a = torch.exp(theta[..., 0])
        Th = torch.exp(theta[..., 1])
        N0 = N0.float().clamp_min(1e-8)
        aTh = a * Th
        C = a * self.cfg.final_time + 1.0 / N0 - aTh * N0

        disc = (C * C + 4.0 * aTh).clamp_min(0.0)
        NT_main = (-C + torch.sqrt(disc)) / (2.0 * aTh.clamp_min(1e-12))
        NT_limit = 1.0 / (a * self.cfg.final_time + 1.0 / N0)
        return torch.where(aTh > 1e-10, NT_main, NT_limit).clamp_min(0.0)

    def _binom_consumed_loglik(
        self, y: torch.Tensor, N0: torch.Tensor, pT: torch.Tensor
    ) -> torch.Tensor:
        """Binomial log-likelihood of y consumed prey out of N0 total."""
        pT = pT.clamp(1e-6, 1.0 - 1e-6)
        return torch.distributions.Binomial(total_count=N0, probs=pT).log_prob(y.float())

    def sample_observation(self, theta: np.ndarray, action: np.ndarray) -> float:
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
        N0 = torch.tensor(float(action[0]), dtype=torch.float32, device=self.device)
        NT = self._integrate_terminal_population(N0, theta_t)
        pT = ((N0 - NT) / N0).clamp(1e-6, 1.0 - 1e-6)
        return float(
            torch.distributions.Binomial(total_count=N0, probs=pT).sample().detach().cpu()
        )

    def loglik_scalar(self, obs: float, theta: np.ndarray, action: np.ndarray) -> float:
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
        N0 = torch.tensor(float(action[0]), dtype=torch.float32, device=self.device)
        y = torch.tensor(float(obs), dtype=torch.float32, device=self.device)
        NT = self._integrate_terminal_population(N0, theta_t)
        pT = ((N0 - NT) / N0).clamp(1e-6, 1.0 - 1e-6)
        return float(self._binom_consumed_loglik(y, N0, pT).detach().cpu())

    def trajectory_loglik_thetas(
        self, actions: np.ndarray, obs: np.ndarray, thetas: torch.Tensor
    ) -> torch.Tensor:
        """Sum Binomial log-likelihoods over all time steps for each hypothesis."""
        T = len(obs)
        if T == 0:
            return torch.zeros(thetas.shape[0], device=thetas.device)
        a_t = torch.tensor(actions[:T, 0], dtype=torch.float32, device=thetas.device)
        y_t = torch.tensor(obs[:T], dtype=torch.float32, device=thetas.device)
        out = torch.zeros(thetas.shape[0], dtype=torch.float32, device=thetas.device)
        for t in range(T):
            N0 = a_t[t].expand(thetas.shape[0])
            NT = self._integrate_terminal_population(N0, thetas)
            pT = ((N0 - NT) / N0).clamp(1e-6, 1.0 - 1e-6)
            out = out + self._binom_consumed_loglik(y_t[t].expand_as(pT), N0, pT)
        return out

    def bank_loglik_single(
        self, obs_t: torch.Tensor, action_t: torch.Tensor
    ) -> torch.Tensor:
        """Binomial log-likelihood for a single (obs, action) pair over the full bank."""
        N0 = action_t[0].expand(self.cfg.bank_size)
        theta_bank = self.hypothesis_bank.to(self.device)
        NT = self._integrate_terminal_population(N0, theta_bank)
        pT = ((N0 - NT) / N0).clamp(1e-6, 1.0 - 1e-6)
        return self._binom_consumed_loglik(obs_t.expand_as(pT), N0, pT)


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

VARIANTS: list[str] = [
    "blau_approx",
    "ours_ebm_control",
    "ours_ebm_cross",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prey-population discrete BOED benchmark."
    )
    p.add_argument("--episodes",             type=int,   default=250)
    p.add_argument("--eval-episodes",        type=int,   default=40)
    p.add_argument("--seeds",                type=str,   default="0,1,2")
    p.add_argument("--device",               type=str,   default="auto")
    p.add_argument("--output-dir",           type=str,   default="./outputs/prey_population")
    p.add_argument("--variants",             type=str,   default=",".join(VARIANTS))
    p.add_argument("--horizon",              type=int,   default=10)
    p.add_argument("--bank-size",            type=int,   default=512)
    p.add_argument("--hidden-ebm",           type=int,   default=128)
    p.add_argument("--gamma",                type=float, default=0.95)
    p.add_argument("--spce-L",               type=int,   default=128)
    p.add_argument("--snmc-L",               type=int,   default=128)
    p.add_argument("--belief-mode",          type=str,   default="distilled_detached",
                   choices=["exact", "distilled_detached", "distilled_e2e", "learned_only"])
    p.add_argument("--belief-feature-mode",  type=str,   default="legacy",
                   choices=["legacy", "moments"])
    return p.parse_args()


def main() -> None:
    """CLI entry point: ``boedx-prey-population``."""
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = PreyConfig(horizon=args.horizon, bank_size=args.bank_size)
    train_cfg = GenericTrainConfig(
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        device=device,
        seeds=args.seeds,
        hidden_ebm=args.hidden_ebm,
        gamma=args.gamma,
        print_every=25,
    )
    seeds: Sequence[int] = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    belief_cfg = BeliefConfig(
        mode=args.belief_mode,
        feature_mode=args.belief_feature_mode,
    )

    def factory(dev: torch.device) -> PreyPopulationEnv:
        return PreyPopulationEnv(cfg=cfg, device=dev)

    import json
    summary = run_experiment_suite(
        experiment_name="prey_population",
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
