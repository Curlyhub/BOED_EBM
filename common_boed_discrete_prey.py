from __future__ import annotations

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



def gaussian_logpdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(std):
        std = torch.tensor(std, dtype=mean.dtype, device=mean.device)
    var = std * std
    return -0.5 * (((x - mean) ** 2) / var + torch.log(2.0 * torch.tensor(math.pi, dtype=mean.dtype, device=mean.device) * var))



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
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)



def logmeanexp_t(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim) - math.log(x.shape[dim])


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


class ApsiHead(nn.Module):
    def __init__(self, hist_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hist_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, hist_feat: torch.Tensor) -> torch.Tensor:
        return sanitize(self.net(hist_feat), 50.0)


class TanhGaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = sanitize(self.net(state), 100.0)
        mean = sanitize(self.mean(h), 20.0)
        log_std = sanitize(self.log_std(h), 5.0).clamp(-5.0, 1.0)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp().clamp(min=1e-4, max=10.0)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        u = torch.tanh(z)
        action = u * action_scale + action_bias
        log_prob = normal.log_prob(z) - torch.log(1.0 - u.pow(2) + 1e-6)
        log_prob = sanitize(log_prob, 100.0).sum(dim=-1, keepdim=True)
        deterministic = torch.tanh(mean) * action_scale + action_bias
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
    def __init__(self, capacity: int = 100000):
        self.capacity = int(capacity)
        self.storage: List[Dict] = []
        self.ptr = 0

    def add(self, item: Dict) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.capacity

    def __len__(self) -> int:
        return len(self.storage)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, len(self.storage), size=batch_size)
        batch = [self.storage[i] for i in idxs]

        def arr(key: str) -> np.ndarray:
            return np.stack([b[key] for b in batch], axis=0)

        return {
            "last_obs": torch.tensor([[b["last_obs"]] for b in batch], dtype=torch.float32, device=device),
            "t_idx": torch.tensor([b["t_idx"] for b in batch], dtype=torch.long, device=device),
            "aux_state": torch.tensor(arr("aux_state"), dtype=torch.float32, device=device),
            "actions": torch.tensor(arr("actions"), dtype=torch.float32, device=device),
            "obs": torch.tensor(arr("obs"), dtype=torch.float32, device=device),
            "posterior": torch.tensor(arr("posterior"), dtype=torch.float32, device=device),
            "posterior_logits": torch.tensor(arr("posterior_logits"), dtype=torch.float32, device=device),
            "filter_logits": torch.tensor(arr("filter_logits"), dtype=torch.float32, device=device),
            "next_last_obs": torch.tensor([[b["next_last_obs"]] for b in batch], dtype=torch.float32, device=device),
            "next_t_idx": torch.tensor([b["next_t_idx"] for b in batch], dtype=torch.long, device=device),
            "next_aux_state": torch.tensor(arr("next_aux_state"), dtype=torch.float32, device=device),
            "next_actions": torch.tensor(arr("next_actions"), dtype=torch.float32, device=device),
            "next_obs": torch.tensor(arr("next_obs"), dtype=torch.float32, device=device),
            "next_posterior": torch.tensor(arr("next_posterior"), dtype=torch.float32, device=device),
            "next_posterior_logits": torch.tensor(arr("next_posterior_logits"), dtype=torch.float32, device=device),
            "next_filter_logits": torch.tensor(arr("next_filter_logits"), dtype=torch.float32, device=device),
            "reward": torch.tensor([[b["reward"]] for b in batch], dtype=torch.float32, device=device),
            "done": torch.tensor([[b["done"]] for b in batch], dtype=torch.float32, device=device),
            "action_taken": torch.tensor(arr("action_taken"), dtype=torch.float32, device=device),
        }


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
    print_every: int = 25
    device: str = "cpu"
    seeds: str = "0,1,2"


@dataclass
class BeliefConfig:
    mode: str = "distilled_detached"
    feature_mode: str = "legacy"


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

    def _initial_filter_logits(self) -> torch.Tensor:
        if self.use_posterior_filter():
            return self.prior_bank_logits.clone()
        return torch.zeros(self.H, dtype=torch.float32, device=self.device)

    def _update_filter_logits(self, cur_logits: torch.Tensor, obs_t: torch.Tensor, action_t: torch.Tensor) -> torch.Tensor:
        ll_bank = self.bank_loglik_single(obs_t, action_t)
        new_logits = torch.nan_to_num(cur_logits + ll_bank, nan=0.0, posinf=1e4, neginf=-1e4)
        return new_logits - torch.max(new_logits)

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
        init_logits = self._initial_filter_logits()
        self._posterior_logu = init_logits.clone()
        self._equiv_logu = init_logits.clone()
        return self.history

    def get_n_contrastive(self) -> int:
        return getattr(self, "n_contrastive", 32)

    def use_posterior_filter(self) -> bool:
        return getattr(self, "exact_filter", "likelihood") == "posterior"

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
            self._posterior_logu = self._update_filter_logits(self._posterior_logu, obs_t, action_t)
            self._equiv_logu = self._update_filter_logits(self._equiv_logu, obs_t, action_t)

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



def belief_feature_dim(theta_dim: int, feature_mode: str) -> int:
    if feature_mode == "legacy":
        return theta_dim + 2
    if feature_mode == "moments":
        upper = theta_dim * (theta_dim - 1) // 2
        return theta_dim + theta_dim + upper + 1 + 1
    raise ValueError(f"Unknown belief feature mode: {feature_mode}")



def belief_features_from_probs(
    probs: torch.Tensor,
    theta_bank: torch.Tensor,
    A_scalar: Optional[torch.Tensor] = None,
    feature_mode: str = "legacy",
) -> torch.Tensor:
    log_probs = torch.log(probs.clamp_min(1e-12))
    mean = probs @ theta_bank
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
    if feature_mode == "legacy":
        pieces = [mean, entropy]
        if A_scalar is not None:
            pieces.append(sanitize(A_scalar, 50.0))
        return sanitize(torch.cat(pieces, dim=-1), 1e3)
    if feature_mode != "moments":
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
    return sanitize(torch.cat(pieces, dim=-1), 1e3)



def make_raw_state(env: GenericBankBOEDEnv) -> Dict:
    horizon = env.get_horizon()
    actions = np.zeros((horizon, env.action_dim), dtype=np.float32)
    obs = np.zeros(horizon, dtype=np.float32)
    for i, (a, y) in enumerate(env.history):
        actions[i] = a
        obs[i] = y
    posterior = env.posterior_bank().detach().cpu().numpy().astype(np.float32)
    posterior_logits = env._posterior_logu.detach().cpu().numpy().astype(np.float32)
    filter_logits = env._equiv_logu.detach().cpu().numpy().astype(np.float32)
    aux_state = env.current_aux_state().astype(np.float32)
    return {
        "actions": actions,
        "obs": obs,
        "length": len(env.history),
        "last_obs": float(env.last_obs),
        "t_idx": env.t,
        "aux_state": aux_state,
        "posterior": posterior,
        "posterior_logits": posterior_logits,
        "filter_logits": filter_logits,
    }


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

    cached = batch[f"{prefix}filter_logits"]
    hist_feat = filter_backbone.forward_from_logits(cached)
    quotient_base = build_base_state(hist_feat, t_idx, env.get_horizon(), last_obs, aux_state)
    if energy_net is None or apsi_head is None or belief_cfg.mode == "exact":
        return quotient_base, hist_feat, None, None
    energy = energy_net(hist_feat, env.hypothesis_bank)
    A = apsi_head(hist_feat)
    probs = posterior_probs_from_energy(energy)
    belief = belief_features_from_probs(
        probs=probs,
        theta_bank=env.hypothesis_bank,
        A_scalar=A,
        feature_mode=belief_cfg.feature_mode,
    )
    if belief_cfg.mode == "distilled_detached":
        belief = belief.detach()
        state = sanitize(torch.cat([quotient_base, belief], dim=-1), 1e3)
    elif belief_cfg.mode == "distilled_e2e":
        state = sanitize(torch.cat([quotient_base, belief], dim=-1), 1e3)
    elif belief_cfg.mode == "learned_only":
        minimal_base = build_minimal_state(t_idx, env.get_horizon(), last_obs, aux_state)
        state = sanitize(torch.cat([minimal_base, belief], dim=-1), 1e3)
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
    if variant == "ours_ebm_control" and belief_cfg.mode != "exact":
        energy_net = EnergyNet(hist_dim=hist_dim, theta_dim=env.theta_dim, hidden=train_cfg.hidden_ebm).to(device)
        apsi_head = ApsiHead(hist_dim=hist_dim, hidden=train_cfg.hidden_ebm).to(device)
        belief_dim = belief_feature_dim(env.theta_dim, belief_cfg.feature_mode)
    elif variant == "ours_ebm_cross" and belief_cfg.mode != "exact":
        energy_net = CrossInteractionEnergyNet(hist_dim=hist_dim, theta_dim=env.theta_dim, hidden=train_cfg.hidden_ebm).to(device)
        apsi_head = ApsiHead(hist_dim=hist_dim, hidden=train_cfg.hidden_ebm).to(device)
        belief_dim = belief_feature_dim(env.theta_dim, belief_cfg.feature_mode)
    elif variant not in {"blau_approx", "ours_ebm_control", "ours_ebm_cross"}:
        raise ValueError(f"Unknown variant: {variant}")

    if variant == "blau_approx":
        state_dim = raw_history_state_dim
    elif belief_cfg.mode == "learned_only" and belief_dim > 0:
        state_dim = minimal_base_state_dim + belief_dim
    else:
        state_dim = quotient_base_state_dim + belief_dim
    if uses_discrete_actor(env):
        action_values = get_discrete_action_values(env, device)
        actor = DiscreteCategoricalActor(state_dim=state_dim, num_actions=action_values.shape[0], hidden=train_cfg.hidden_rl).to(device)
    else:
        actor = TanhGaussianActor(state_dim=state_dim, action_dim=env.action_dim, hidden=train_cfg.hidden_rl).to(device)
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
    actor_optim = torch.optim.Adam(actor_params, lr=train_cfg.lr_actor)
    critic_optim = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=train_cfg.lr_critic)
    ebm_optim = None
    if energy_net is not None and apsi_head is not None:
        ebm_optim = torch.optim.Adam(list(energy_net.parameters()) + list(apsi_head.parameters()), lr=train_cfg.lr_ebm)
    return filter_backbone, actor, q1, q2, q1_tgt, q2_tgt, actor_optim, critic_optim, energy_net, apsi_head, ebm_optim



def train_one_seed(
    variant: str,
    env_factory: Callable[[torch.device], GenericBankBOEDEnv],
    train_cfg: GenericTrainConfig,
    seed: int,
    output_dir: str,
    spce_L: int,
    snmc_L: int,
    belief_cfg: Optional[BeliefConfig] = None,
) -> Dict:
    set_seed(seed)
    device = torch.device(train_cfg.device if train_cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    env = env_factory(device)
    (
        filter_backbone,
        actor,
        q1,
        q2,
        q1_tgt,
        q2_tgt,
        actor_optim,
        critic_optim,
        energy_net,
        apsi_head,
        ebm_optim,
    ) = build_modules(variant, env, train_cfg, device, belief_cfg=belief_cfg)
    replay = ReplayBuffer(capacity=train_cfg.replay_size)
    action_scale = env.action_scale
    action_bias = env.action_bias
    discrete_action_values = get_discrete_action_values(env, device) if uses_discrete_actor(env) else None
    episode_returns: List[float] = []

    for ep in range(train_cfg.episodes):
        env.reset()
        raw = make_raw_state(env)
        done = False
        ep_return = 0.0
        while not done:
            state_t, _, _, _ = raw_state_to_policy_state(variant, raw, filter_backbone, env, device, energy_net, apsi_head, belief_cfg=belief_cfg)
            if ep < train_cfg.warmup_episodes:
                if discrete_action_values is not None:
                    idx = np.random.randint(0, discrete_action_values.shape[0])
                    action = discrete_action_values[idx].detach().cpu().numpy().astype(np.float32)
                else:
                    action = np.random.uniform(env.action_low.detach().cpu().numpy(), env.action_high.detach().cpu().numpy()).astype(np.float32)
            else:
                with torch.no_grad():
                    if discrete_action_values is not None:
                        act, _, _ = actor.sample(state_t, action_values=discrete_action_values)
                    else:
                        act, _, _ = actor.sample(state_t, action_scale=action_scale, action_bias=action_bias)
                action = act.squeeze(0).detach().cpu().numpy().astype(np.float32)
            _, reward, done, _ = env.step(action)
            next_raw = make_raw_state(env)
            replay.add(
                {
                    "last_obs": raw["last_obs"],
                    "t_idx": raw["t_idx"],
                    "aux_state": raw["aux_state"],
                    "actions": raw["actions"],
                    "obs": raw["obs"],
                    "posterior": raw["posterior"],
                    "posterior_logits": raw["posterior_logits"],
                    "filter_logits": raw["filter_logits"],
                    "next_last_obs": next_raw["last_obs"],
                    "next_t_idx": next_raw["t_idx"],
                    "next_aux_state": next_raw["aux_state"],
                    "next_actions": next_raw["actions"],
                    "next_obs": next_raw["obs"],
                    "next_posterior": next_raw["posterior"],
                    "next_posterior_logits": next_raw["posterior_logits"],
                    "next_filter_logits": next_raw["filter_logits"],
                    "reward": np.float32(reward),
                    "done": np.float32(done),
                    "action_taken": action.astype(np.float32),
                }
            )
            raw = next_raw
            ep_return += reward
            if len(replay) >= train_cfg.batch_size:
                for _ in range(train_cfg.updates_per_step):
                    batch = replay.sample(train_cfg.batch_size, device=device)

                    # Critic update: detach the policy state so the critic does not
                    # reuse the learned-belief graph.
                    critic_state, _, _, _ = compute_state_from_batch(
                        variant, filter_backbone, batch, env, energy_net, apsi_head, belief_cfg=belief_cfg, use_next=False
                    )
                    critic_state = critic_state.detach()
                    with torch.no_grad():
                        next_state, _, _, _ = compute_state_from_batch(
                            variant, filter_backbone, batch, env, energy_net, apsi_head, belief_cfg=belief_cfg, use_next=True
                        )
                        if discrete_action_values is not None:
                            next_probs, next_log_probs = actor.probs_and_log_probs(next_state)
                            next_q1_all = evaluate_q_over_discrete_actions(q1_tgt, next_state, discrete_action_values)
                            next_q2_all = evaluate_q_over_discrete_actions(q2_tgt, next_state, discrete_action_values)
                            next_v = (next_probs * (torch.min(next_q1_all, next_q2_all) - train_cfg.alpha * next_log_probs)).sum(dim=-1, keepdim=True)
                            target = batch["reward"] + (1.0 - batch["done"]) * train_cfg.gamma * next_v
                        else:
                            next_action, next_logp, _ = actor.sample(next_state, action_scale=action_scale, action_bias=action_bias)
                            next_q = torch.min(q1_tgt(next_state, next_action), q2_tgt(next_state, next_action))
                            target = batch["reward"] + (1.0 - batch["done"]) * train_cfg.gamma * (
                                next_q - train_cfg.alpha * next_logp
                            )
                    q1_pred = q1(critic_state, batch["action_taken"])
                    q2_pred = q2(critic_state, batch["action_taken"])
                    critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
                    critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), train_cfg.grad_clip)
                    critic_optim.step()

                    # Actor update: recompute a fresh graph so end-to-end belief modes
                    # can propagate policy gradients into the belief model.
                    actor_state, _, _, _ = compute_state_from_batch(
                        variant, filter_backbone, batch, env, energy_net, apsi_head, belief_cfg=belief_cfg, use_next=False
                    )
                    if discrete_action_values is not None:
                        probs, log_probs = actor.probs_and_log_probs(actor_state)
                        q1_all = evaluate_q_over_discrete_actions(q1, actor_state, discrete_action_values)
                        q2_all = evaluate_q_over_discrete_actions(q2, actor_state, discrete_action_values)
                        actor_loss = (probs * (train_cfg.alpha * log_probs - torch.min(q1_all, q2_all))).sum(dim=-1).mean()
                    else:
                        new_action, logp, _ = actor.sample(actor_state, action_scale=action_scale, action_bias=action_bias)
                        actor_loss = (train_cfg.alpha * logp - torch.min(q1(actor_state, new_action), q2(actor_state, new_action))).mean()
                    actor_optim.zero_grad()
                    actor_loss.backward()
                    actor_grad_params = list(actor.parameters())
                    if energy_net is not None and apsi_head is not None and belief_cfg is not None and belief_cfg.mode in {"distilled_e2e", "learned_only"}:
                        actor_grad_params += list(energy_net.parameters()) + list(apsi_head.parameters())
                    nn.utils.clip_grad_norm_(actor_grad_params, train_cfg.grad_clip)
                    actor_optim.step()

                    if energy_net is not None and apsi_head is not None and ebm_optim is not None:
                        _, _, energy, A = compute_state_from_batch(
                            variant, filter_backbone, batch, env, energy_net, apsi_head, belief_cfg=belief_cfg, use_next=False
                        )
                        if energy is not None and A is not None:
                            log_probs = F.log_softmax(-energy, dim=-1)
                            target_probs = batch["posterior"]
                            ebm_loss = (-(target_probs * log_probs).sum(dim=-1).mean() + train_cfg.apsi_coef * A.pow(2).mean())
                            ebm_optim.zero_grad()
                            ebm_loss.backward()
                            nn.utils.clip_grad_norm_(list(energy_net.parameters()) + list(apsi_head.parameters()), train_cfg.grad_clip)
                            ebm_optim.step()

                    soft_update(q1_tgt, q1, train_cfg.tau)
                    soft_update(q2_tgt, q2, train_cfg.tau)
        episode_returns.append(float(ep_return))
        if (ep + 1) % train_cfg.print_every == 0:
            recent = np.mean(episode_returns[-train_cfg.print_every:])
            print(f"[{env.name}:{variant}] seed={seed} episode={ep+1}/{train_cfg.episodes} avg_return_recent={recent:.4f}")

    eval_returns: List[float] = []
    belief_errs: List[float] = []
    bank_ig_finals: List[float] = []
    spce_lower_finals: List[float] = []
    snmc_upper_finals: List[float] = []
    bank_ig_paths: List[List[float]] = []
    spce_paths: List[List[float]] = []
    snmc_paths: List[List[float]] = []

    for _ in range(train_cfg.eval_episodes):
        env.reset()
        raw = make_raw_state(env)
        done = False
        ep_return = 0.0
        ep_bank_path: List[float] = []
        ep_spce_path: List[float] = []
        ep_snmc_path: List[float] = []

        while not done:
            state_t, _, energy_t, A_t = raw_state_to_policy_state(variant, raw, filter_backbone, env, device, energy_net, apsi_head, belief_cfg=belief_cfg)
            if energy_net is not None and energy_t is not None and A_t is not None:
                exact_probs_t = torch.tensor(raw["posterior"][None], dtype=torch.float32, device=device)
                exact_mean_t = exact_probs_t @ env.hypothesis_bank
                pred_probs = posterior_probs_from_energy(energy_t)
                pred_mean = pred_probs @ env.hypothesis_bank
                belief_errs.append(float(torch.mean(torch.abs(pred_mean - exact_mean_t)).detach().cpu()))
            with torch.no_grad():
                if discrete_action_values is not None:
                    _, _, det = actor.sample(state_t, action_values=discrete_action_values)
                else:
                    _, _, det = actor.sample(state_t, action_scale=action_scale, action_bias=action_bias)
            action = det.squeeze(0).detach().cpu().numpy().astype(np.float32)
            _, reward, done, _ = env.step(action)
            raw = make_raw_state(env)
            ep_return += reward

            prefix_actions = raw["actions"][: raw["length"]]
            prefix_obs = raw["obs"][: raw["length"]]
            ep_bank_path.append(discrete_bank_ig_from_logits(raw["posterior_logits"], env.prior_bank_logits))
            ep_spce_path.append(estimate_spce_prefix(env, prefix_actions, prefix_obs, env.theta0, spce_L))
            if snmc_L > 0:
                ep_snmc_path.append(estimate_snmc_style_upper_prefix(env, prefix_actions, prefix_obs, env.theta0, snmc_L))

        eval_returns.append(ep_return)
        bank_ig_finals.append(ep_bank_path[-1])
        spce_lower_finals.append(ep_spce_path[-1])
        if snmc_L > 0 and len(ep_snmc_path) > 0:
            snmc_upper_finals.append(ep_snmc_path[-1])
        bank_ig_paths.append(ep_bank_path)
        spce_paths.append(ep_spce_path)
        if snmc_L > 0 and len(ep_snmc_path) > 0:
            snmc_paths.append(ep_snmc_path)

    out: Dict[str, object] = {
        "train": {"episode_returns": episode_returns},
        "eval": {
            "avg_return": float(np.mean(eval_returns)),
            "std_return": float(np.std(eval_returns, ddof=1)) if len(eval_returns) > 1 else 0.0,
            "avg_bank_ig": float(np.mean(bank_ig_finals)),
            "avg_spce_lower": float(np.mean(spce_lower_finals)),
        },
        "variant": variant,
        "seed": seed,
        "paths": {
            "bank_ig_mean_path": np.mean(np.array(bank_ig_paths, dtype=np.float64), axis=0).tolist(),
            "spce_lower_mean_path": np.mean(np.array(spce_paths, dtype=np.float64), axis=0).tolist(),
        },
    }
    if snmc_L > 0 and len(snmc_upper_finals) > 0:
        out["eval"]["avg_snmc_style_upper"] = float(np.mean(snmc_upper_finals))
        out["paths"]["snmc_style_upper_mean_path"] = np.mean(np.array(snmc_paths, dtype=np.float64), axis=0).tolist()
    if belief_errs:
        out["eval"]["avg_abs_belief_mean_error"] = float(np.mean(belief_errs))

    seed_dir = os.path.join(output_dir, variant, f"seed_{seed}")
    ensure_dir(seed_dir)
    with open(os.path.join(seed_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out



def aggregate_plot_data(all_results: Dict[str, List[Dict]]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for variant, results in all_results.items():
        ep_returns = np.array([r["train"]["episode_returns"] for r in results], dtype=np.float64)
        train_mean = ep_returns.mean(axis=0)
        train_se = ep_returns.std(axis=0, ddof=1) / math.sqrt(max(len(results), 1)) if len(results) > 1 else np.zeros_like(train_mean)
        paths_bank = np.array([r["paths"]["bank_ig_mean_path"] for r in results], dtype=np.float64)
        paths_spce = np.array([r["paths"]["spce_lower_mean_path"] for r in results], dtype=np.float64)
        bank_mean = paths_bank.mean(axis=0)
        bank_se = paths_bank.std(axis=0, ddof=1) / math.sqrt(max(len(results), 1)) if len(results) > 1 else np.zeros_like(bank_mean)
        spce_mean = paths_spce.mean(axis=0)
        spce_se = paths_spce.std(axis=0, ddof=1) / math.sqrt(max(len(results), 1)) if len(results) > 1 else np.zeros_like(spce_mean)
        rec = {
            "train_mean": train_mean,
            "train_se": train_se,
            "bank_mean": bank_mean,
            "bank_se": bank_se,
            "spce_mean": spce_mean,
            "spce_se": spce_se,
        }
        if "snmc_style_upper_mean_path" in results[0]["paths"]:
            paths_snmc = np.array([r["paths"]["snmc_style_upper_mean_path"] for r in results], dtype=np.float64)
            snmc_mean = paths_snmc.mean(axis=0)
            snmc_se = paths_snmc.std(axis=0, ddof=1) / math.sqrt(max(len(results), 1)) if len(results) > 1 else np.zeros_like(snmc_mean)
            rec["snmc_mean"] = snmc_mean
            rec["snmc_se"] = snmc_se
        out[variant] = rec
    return out



def save_standard_plots(output_dir: str, all_results: Dict[str, List[Dict]], summary: Dict, horizon: int) -> None:
    plot_data = aggregate_plot_data(all_results)
    colors = {"blau_approx": "tab:blue", "ours_ebm_control": "tab:orange", "ours_ebm_cross": "tab:green"}

    plt.figure(figsize=(8, 5))
    for variant, rec in plot_data.items():
        x = np.arange(1, len(rec["train_mean"]) + 1)
        plt.plot(x, rec["train_mean"], label=variant, color=colors.get(variant))
        plt.fill_between(x, rec["train_mean"] - rec["train_se"], rec["train_mean"] + rec["train_se"], color=colors.get(variant), alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Training return")
    plt.title("Learning curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=180)
    plt.close()

    steps = np.arange(1, horizon + 1)
    for key, title, fname in [
        ("bank", "Bank information gain path", "bank_ig_paths.png"),
        ("spce", "SPCE lower path", "spce_lower_paths.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for variant, rec in plot_data.items():
            m, s = rec[f"{key}_mean"], rec[f"{key}_se"]
            plt.plot(steps, m, label=variant, color=colors.get(variant))
            plt.fill_between(steps, m - s, m + s, color=colors.get(variant), alpha=0.2)
        plt.xlabel("# Experiments")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=180)
        plt.close()

    if any("snmc_mean" in rec for rec in plot_data.values()):
        plt.figure(figsize=(8, 5))
        for variant, rec in plot_data.items():
            if "snmc_mean" not in rec:
                continue
            m, s = rec["snmc_mean"], rec["snmc_se"]
            plt.plot(steps, m, label=variant, color=colors.get(variant))
            plt.fill_between(steps, m - s, m + s, color=colors.get(variant), alpha=0.2)
        plt.xlabel("# Experiments")
        plt.ylabel("SNMC-style upper path")
        plt.title("SNMC-style upper path")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "snmc_style_upper_paths.png"), dpi=180)
        plt.close()

    def barplot(metric_keys: List[str], labels: List[str], title: str, fname: str) -> None:
        xs = np.arange(len(labels))
        vals = [summary[k]["mean"] for k in metric_keys]
        plt.figure(figsize=(8, 5))
        plt.bar(xs, vals, color=[colors.get(v, None) for v in labels])
        plt.xticks(xs, labels, rotation=15)
        plt.ylabel(title)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=180)
        plt.close()

    variants = summary["variants"]
    barplot([f"{v}_avg_return" for v in variants], variants, "Evaluation return", "eval_return_bars.png")
    if all(f"{v}_avg_bank_ig" in summary for v in variants):
        barplot([f"{v}_avg_bank_ig" for v in variants], variants, "Final bank IG", "eval_bank_ig_bars.png")
    if all(f"{v}_avg_spce_lower" in summary for v in variants):
        barplot([f"{v}_avg_spce_lower" for v in variants], variants, "Final SPCE lower", "eval_spce_lower_bars.png")
    if all(f"{v}_avg_snmc_style_upper" in summary for v in variants):
        barplot([f"{v}_avg_snmc_style_upper" for v in variants], variants, "Final SNMC-style upper", "eval_snmc_style_upper_bars.png")

    if all(f"{v}_avg_return" in summary and f"{v}_avg_spce_lower" in summary for v in variants):
        plt.figure(figsize=(6, 5))
        for v in variants:
            plt.scatter(summary[f"{v}_avg_return"]["mean"], summary[f"{v}_avg_spce_lower"]["mean"], label=v, s=90, color=colors.get(v))
        plt.xlabel("Avg return")
        plt.ylabel("Avg SPCE lower")
        plt.title("Return vs SPCE lower")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "return_vs_spce_scatter.png"), dpi=180)
        plt.close()



def run_experiment_suite(
    experiment_name: str,
    env_factory: Callable[[torch.device], GenericBankBOEDEnv],
    output_dir: str,
    train_cfg: GenericTrainConfig,
    seeds: Sequence[int],
    variants: Sequence[str],
    spce_L: int,
    snmc_L: int,
    belief_cfg: Optional[BeliefConfig] = None,
) -> Dict:
    ensure_dir(output_dir)
    all_results: Dict[str, List[Dict]] = {}
    sample_env = env_factory(torch.device(train_cfg.device if train_cfg.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")))

    for variant in variants:
        print("\n============================")
        print(f"Training {experiment_name} variant: {variant}")
        print("============================")
        variant_results = []
        for seed in seeds:
            variant_results.append(train_one_seed(variant, env_factory, train_cfg, seed, output_dir, spce_L, snmc_L, belief_cfg=belief_cfg))
        all_results[variant] = variant_results

    summary: Dict[str, object] = {}
    for variant, results in all_results.items():
        for field in [
            "avg_return",
            "avg_bank_ig",
            "avg_spce_lower",
            "avg_snmc_style_upper",
            "avg_abs_belief_mean_error",
        ]:
            vals = [r["eval"][field] for r in results if field in r["eval"]]
            if vals:
                summary[f"{variant}_{field}"] = mean_std_ci95(np.array(vals, dtype=np.float64))
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
        summary["belief_config"] = {"mode": belief_cfg.mode, "feature_mode": belief_cfg.feature_mode}

    with open(os.path.join(output_dir, "summary_multi_seed.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_standard_plots(output_dir, all_results, summary, sample_env.get_horizon())
    print("\n=== Multi-seed summary ===")
    print(json.dumps(summary, indent=2))
    return summary
