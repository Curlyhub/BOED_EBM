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
from boedx.trainer import run_experiment_suite


@dataclass
class PreyConfig:
    horizon: int = 10
    design_min: float = 1.0
    design_max: float = 300.0
    final_time: float = 24.0
    ode_steps: int = 120
    bank_size: int = 512
    n_contrastive: int = 128
    exact_filter: str = "likelihood"


class PreyPopulationEnv(GenericBankBOEDEnv):
    name = "prey_population"
    action_dim = 1
    # NOTE: this small framework currently uses a rounded continuous actor for d in {1,...,300}.
    # Blau's original RL implementation used a discrete-action head for prey.

    def __init__(self, cfg: PreyConfig, device: torch.device):
        self.cfg = cfg
        self.n_contrastive = cfg.n_contrastive
        self.exact_filter = cfg.exact_filter
        super().__init__(device=device)

    def get_horizon(self) -> int:
        return self.cfg.horizon

    def get_action_low(self) -> np.ndarray:
        return np.array([self.cfg.design_min], dtype=np.float32)

    def get_action_high(self) -> np.ndarray:
        return np.array([self.cfg.design_max], dtype=np.float32)

    def sample_theta(self) -> np.ndarray:
        log_a = np.random.randn() * 1.35 - 1.4
        log_Th = np.random.randn() * 1.35 - 1.4
        return np.array([log_a, log_Th], dtype=np.float32)

    def sample_prior_thetas(self, n: int) -> torch.Tensor:
        log_a = torch.randn(n, 1, device=self.device) * 1.35 - 1.4
        log_Th = torch.randn(n, 1, device=self.device) * 1.35 - 1.4
        return torch.cat([log_a, log_Th], dim=-1)

    def build_hypothesis_bank(self) -> torch.Tensor:
        return self.sample_prior_thetas(self.cfg.bank_size).detach().cpu()

    def build_prior_bank_logits(self) -> torch.Tensor:
        return torch.full((self.cfg.bank_size,), -math.log(self.cfg.bank_size), dtype=torch.float32)

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        clipped = np.clip(action, self.cfg.design_min, self.cfg.design_max)
        clipped = np.round(clipped).astype(np.float32)
        return clipped

    def old_integrate_terminal_population(self, N0: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        a = torch.exp(theta[..., 0])
        Th = torch.exp(theta[..., 1])
        N = N0.float()
        dt = self.cfg.final_time / float(self.cfg.ode_steps)
        for _ in range(self.cfg.ode_steps):
            def f(x: torch.Tensor) -> torch.Tensor:
                return -(a * x * x) / (1.0 + a * Th * x * x)

            k1 = f(N)
            k2 = f(N + 0.5 * dt * k1)
            k3 = f(N + 0.5 * dt * k2)
            k4 = f(N + dt * k3)
            N = (N + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).clamp_min(0.0)
        return N

    def _integrate_terminal_population(self, N0: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        a = torch.exp(theta[..., 0])
        Th = torch.exp(theta[..., 1])

        N0 = N0.float().clamp_min(1e-8)
        aTh = a * Th
        C = a * self.cfg.final_time + 1.0 / N0 - aTh * N0

        # Closed-form solution for:
        # dN/dt = -(a N^2) / (1 + a Th N^2)
        #
        # which implies:
        # 1/N_T - a Th N_T = a T + 1/N_0 - a Th N_0 = C
        #
        # so:
        # aTh * N_T^2 + C * N_T - 1 = 0

        disc = (C * C + 4.0 * aTh).clamp_min(0.0)
        sqrt_disc = torch.sqrt(disc)

        # Main branch when aTh is not tiny
        NT_main = (-C + sqrt_disc) / (2.0 * aTh.clamp_min(1e-12))

        # Stable limit when Th -> 0:
        # dN/dt = -a N^2  =>  N_T = N0 / (1 + a T N0) = 1 / (aT + 1/N0)
        NT_limit = 1.0 / (a * self.cfg.final_time + 1.0 / N0)

        NT = torch.where(aTh > 1e-10, NT_main, NT_limit)
        return NT.clamp_min(0.0)

    def _binom_consumed_loglik(self, y: torch.Tensor, N0: torch.Tensor, pT: torch.Tensor) -> torch.Tensor:
        total = N0
        y = y.float()
        pT = pT.clamp(1e-6, 1.0 - 1e-6)
        dist = torch.distributions.Binomial(total_count=total, probs=pT)
        return dist.log_prob(y)

    def sample_observation(self, theta: np.ndarray, action: np.ndarray) -> float:
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
        N0 = torch.tensor(float(action[0]), dtype=torch.float32, device=self.device)
        NT = self._integrate_terminal_population(N0, theta_t)
        pT = ((N0 - NT) / N0).clamp(1e-6, 1.0 - 1e-6)
        y = torch.distributions.Binomial(total_count=N0, probs=pT).sample()
        return float(y.detach().cpu())

    def loglik_scalar(self, obs: float, theta: np.ndarray, action: np.ndarray) -> float:
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
        N0 = torch.tensor(float(action[0]), dtype=torch.float32, device=self.device)
        y = torch.tensor(float(obs), dtype=torch.float32, device=self.device)
        NT = self._integrate_terminal_population(N0, theta_t)
        pT = ((N0 - NT) / N0).clamp(1e-6, 1.0 - 1e-6)
        ll = self._binom_consumed_loglik(y, N0, pT)
        return float(ll.detach().cpu())

    def trajectory_loglik_thetas(self, actions: np.ndarray, obs: np.ndarray, thetas: torch.Tensor) -> torch.Tensor:
        if len(obs) == 0:
            return torch.zeros(thetas.shape[0], device=thetas.device)
        a = torch.tensor(actions[: len(obs), 0], dtype=torch.float32, device=thetas.device)
        y = torch.tensor(obs[: len(obs)], dtype=torch.float32, device=thetas.device)
        out = torch.zeros(thetas.shape[0], dtype=torch.float32, device=thetas.device)
        for t in range(len(obs)):
            N0 = a[t].expand(thetas.shape[0])
            NT = self._integrate_terminal_population(N0, thetas)
            pT = ((N0 - NT) / N0).clamp(1e-6, 1.0 - 1e-6)
            out = out + self._binom_consumed_loglik(y[t].expand_as(pT), N0, pT)
        return out

    def bank_loglik_single(self, obs_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        N0 = action_t[0].expand(self.cfg.bank_size)
        theta_bank = self.hypothesis_bank.to(self.device)
        NT = self._integrate_terminal_population(N0, theta_bank)
        pT = ((N0 - NT) / N0).clamp(1e-6, 1.0 - 1e-6)
        return self._binom_consumed_loglik(obs_t.expand_as(pT), N0, pT)

#that lines was update
VARIANTS = ["blau_approx", "ours_ebm_control", "ours_ebm_cross"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prey population benchmark with Blau-style and EBM variants.")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--eval-episodes", type=int, default=120)
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="./outputs/prey_population")
    p.add_argument("--variants", type=str, default=",".join(VARIANTS))
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--bank-size", type=int, default=512)
    p.add_argument("--hidden-ebm", type=int, default=512)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--spce-L", type=int, default=1024)
    p.add_argument("--snmc-L", type=int, default=512)
    p.add_argument("--belief-mode", type=str, default="learned_only", choices=["exact", "distilled_detached", "distilled_e2e", "learned_only"])
    p.add_argument("--belief-feature-mode", type=str, default="moments", choices=["legacy", "moments"])
    return p.parse_args()


def main() -> None:
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
    belief_cfg = BeliefConfig(mode=args.belief_mode, feature_mode=args.belief_feature_mode)

    def factory(dev: torch.device) -> PreyPopulationEnv:
        return PreyPopulationEnv(cfg=cfg, device=dev)

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
