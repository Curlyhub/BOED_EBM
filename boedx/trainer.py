"""
Training loop, evaluation, and multi-seed experiment orchestration.

Entry points
------------
``train_one_seed``        — train + evaluate one (variant, seed) pair.
``run_experiment_suite``  — sweep all (variant × seed) pairs and aggregate.
``aggregate_plot_data``   — reduce per-seed results to mean/SE arrays for plotting.
``load_results_from_dir`` — reconstruct all_results from saved JSON files.
"""

from __future__ import annotations

import json
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from boedx.buffer import ReplayBuffer
from boedx.env import BeliefConfig, GenericBankBOEDEnv, GenericTrainConfig
from boedx.models import (
    ApsiHead,
    BetaContrastiveEnergyNet,
    CrossInteractionEnergyNet,
    CachedFilterBackbone,
    DiscreteCategoricalActor,
    EnergyNet,
    MixtureTanhGaussianActor,
    QCritic,
    SymmetricSourceCrossNet,
    SymmetricSourceEnergyNet,
    TanhGaussianActor,
)
from boedx.state import (
    belief_feature_dim,
    belief_features_from_probs,
    belief_kl_divergence,
    belief_l1_error,
    compute_state_from_batch,
    make_raw_state,
    posterior_probs_from_energy,
    raw_state_to_policy_state,
    variant_uses_beta_contrastive,
    variant_uses_cross_ebm,
    variant_uses_ebm,
)
from boedx.utils import (
    ensure_dir,
    logmeanexp_t,
    mean_std_ci95,
    paired_summary,
    set_seed,
    soft_update,
)


# ---------------------------------------------------------------------------
# Discrete-action helpers (prey-population env and similar)
# ---------------------------------------------------------------------------

def uses_discrete_actor(env: GenericBankBOEDEnv) -> bool:
    return getattr(env, "name", "") == "prey_population" and int(getattr(env, "action_dim", 0)) == 1


def get_discrete_action_values(
    env: GenericBankBOEDEnv, device: torch.device
) -> torch.Tensor:
    low = int(round(float(env.action_low[0].detach().cpu())))
    high = int(round(float(env.action_high[0].detach().cpu())))
    return torch.arange(low, high + 1, dtype=torch.float32, device=device).unsqueeze(-1)


def evaluate_q_over_discrete_actions(
    qnet: nn.Module, state: torch.Tensor, action_values: torch.Tensor
) -> torch.Tensor:
    B = state.shape[0]
    N = action_values.shape[0]
    state_rep = state[:, None, :].expand(B, N, -1).reshape(B * N, -1)
    action_rep = action_values[None, :, :].expand(B, N, -1).reshape(B * N, -1)
    return qnet(state_rep, action_rep).reshape(B, N)


# ---------------------------------------------------------------------------
# Information-gain and bound estimators
# ---------------------------------------------------------------------------

def discrete_bank_ig_from_logits(
    filter_logits: np.ndarray, prior_bank_logits: torch.Tensor
) -> float:
    """Discrete KL(posterior ‖ prior) as a proxy for information gain."""
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
    """SPCE lower bound on EIG for a prefix trajectory (Foster et al., 2021)."""
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
    """SNMC-style upper bound on EIG for a prefix trajectory."""
    theta0 = torch.tensor(true_theta[None], dtype=torch.float32, device=env.device)
    ll_true = env.trajectory_loglik_thetas(actions, obs, theta0)[0]
    nested = env.sample_prior_thetas(L)
    ll_nested = env.trajectory_loglik_thetas(actions, obs, nested)
    log_marg_est = logmeanexp_t(ll_nested, dim=0)
    return float((ll_true - log_marg_est).detach().cpu())


# ---------------------------------------------------------------------------
# Module construction
# ---------------------------------------------------------------------------

def build_modules(
    variant: str,
    env: GenericBankBOEDEnv,
    train_cfg: GenericTrainConfig,
    device: torch.device,
    belief_cfg: Optional[BeliefConfig] = None,
) -> Tuple:
    """Instantiate all trainable modules for a given variant.

    Returns:
        (filter_backbone, actor, q1, q2, q1_tgt, q2_tgt,
         actor_optim, critic_optim, energy_net, apsi_head, ebm_optim)
    """
    belief_cfg = belief_cfg or BeliefConfig()
    filter_backbone = CachedFilterBackbone().to(device)
    hist_dim = env.H
    aux_dim = int(env.current_aux_state().shape[0])
    quotient_base_state_dim = hist_dim + 1 + 1 + aux_dim
    raw_history_state_dim = (
        env.get_horizon() * env.action_dim + env.get_horizon() + 1 + 1 + aux_dim
    )
    minimal_base_state_dim = 1 + 1 + aux_dim
    energy_net: Optional[nn.Module] = None
    apsi_head: Optional[nn.Module] = None
    belief_dim = 0

    if variant_uses_ebm(variant) and belief_cfg.mode != "exact":
        use_cross = variant_uses_cross_ebm(variant)
        belief_dim = belief_feature_dim(
            env.theta_dim, belief_cfg.feature_mode, belief_cfg.modal_top_k
        )
        if variant_uses_beta_contrastive(variant):
            contrastive_dim = belief_feature_dim(
                env.theta_dim, belief_cfg.feature_mode, belief_cfg.modal_top_k
            )
            energy_net = BetaContrastiveEnergyNet(
                hist_dim=hist_dim,
                contrastive_dim=contrastive_dim,
                theta_dim=env.theta_dim,
                hidden=train_cfg.hidden_ebm,
                use_cross=use_cross,
                ebm_architecture=belief_cfg.ebm_architecture,
                n_sources=belief_cfg.n_sources,
                add_pairwise_dist=belief_cfg.add_pairwise_dist,
            ).to(device)
            apsi_head = nn.Identity().to(device)
        else:
            if belief_cfg.ebm_architecture == "geometric":
                source_dim = belief_cfg.source_dim or (env.theta_dim // max(belief_cfg.n_sources, 1))
                if belief_cfg.n_sources * source_dim != env.theta_dim:
                    raise ValueError(
                        f"Geometric EBM: n_sources*source_dim must equal theta_dim, "
                        f"got {belief_cfg.n_sources}*{source_dim} != {env.theta_dim}."
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
                energy_net = ebm_cls(
                    hist_dim=hist_dim, theta_dim=env.theta_dim, hidden=train_cfg.hidden_ebm
                ).to(device)
            apsi_head = ApsiHead(hist_dim=hist_dim, hidden=train_cfg.hidden_ebm).to(device)

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
        "ours_ebm_control_beta_contrastive",
        "ours_ebm_cross_beta_contrastive",
    }:
        raise ValueError(f"Unknown variant: {variant!r}")

    if variant == "blau_approx":
        state_dim = raw_history_state_dim
    elif belief_cfg.mode == "learned_only" and belief_dim > 0:
        state_dim = minimal_base_state_dim + belief_dim
    else:
        state_dim = quotient_base_state_dim + belief_dim

    if uses_discrete_actor(env):
        action_values = get_discrete_action_values(env, device)
        actor = DiscreteCategoricalActor(
            state_dim=state_dim, num_actions=action_values.shape[0], hidden=train_cfg.hidden_rl
        ).to(device)
    elif train_cfg.actor_family == "mog":
        actor = MixtureTanhGaussianActor(
            state_dim=state_dim,
            action_dim=env.action_dim,
            hidden=train_cfg.hidden_rl,
            dropout=train_cfg.actor_dropout,
            n_components=train_cfg.actor_mixture_components,
        ).to(device)
    elif train_cfg.actor_family == "gaussian":
        actor = TanhGaussianActor(
            state_dim=state_dim,
            action_dim=env.action_dim,
            hidden=train_cfg.hidden_rl,
            dropout=train_cfg.actor_dropout,
        ).to(device)
    else:
        raise ValueError(f"Unknown actor_family={train_cfg.actor_family!r}")

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
    actor_optim = torch.optim.Adam(
        actor_params, lr=train_cfg.lr_actor, weight_decay=train_cfg.actor_weight_decay
    )
    critic_optim = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()), lr=train_cfg.lr_critic
    )
    ebm_optim: Optional[torch.optim.Optimizer] = None
    if energy_net is not None and apsi_head is not None:
        ebm_optim = torch.optim.Adam(
            list(energy_net.parameters()) + list(apsi_head.parameters()), lr=train_cfg.lr_ebm
        )

    return (
        filter_backbone, actor, q1, q2, q1_tgt, q2_tgt,
        actor_optim, critic_optim, energy_net, apsi_head, ebm_optim,
    )


# ---------------------------------------------------------------------------
# Single-seed training + evaluation
# ---------------------------------------------------------------------------

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
    """Train and evaluate one (variant, seed) pair; persist results to disk.

    Args:
        variant:      Name of the policy-state variant to train.
        env_factory:  Callable that instantiates a fresh env on the given device.
        train_cfg:    Training hyper-parameters.
        seed:         Random seed.
        output_dir:   Root output directory; results are written to
                      ``<output_dir>/<variant>/seed_<seed>/result.json``.
        spce_L:       Number of contrastive samples for the SPCE estimator.
        snmc_L:       Number of nested samples for the SNMC estimator (0 to skip).
        belief_cfg:   EBM belief configuration.

    Returns:
        Result dict with keys ``train``, ``eval``, ``paths``, ``variant``, ``seed``.
    """
    set_seed(seed)
    device = torch.device(
        train_cfg.device if train_cfg.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    env = env_factory(device)
    (
        filter_backbone, actor, q1, q2, q1_tgt, q2_tgt,
        actor_optim, critic_optim, energy_net, apsi_head, ebm_optim,
    ) = build_modules(variant, env, train_cfg, device, belief_cfg=belief_cfg)
    replay = ReplayBuffer(capacity=train_cfg.replay_size)
    action_scale = env.action_scale
    action_bias = env.action_bias
    discrete_action_values = (
        get_discrete_action_values(env, device) if uses_discrete_actor(env) else None
    )
    episode_returns: List[float] = []
    update_idx = 0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for ep in range(train_cfg.episodes):
        env.reset()
        raw = make_raw_state(env)
        done = False
        ep_return = 0.0

        while not done:
            state_t, _, _, _ = raw_state_to_policy_state(
                variant, raw, filter_backbone, env, device, energy_net, apsi_head,
                belief_cfg=belief_cfg,
            )
            if ep < train_cfg.warmup_episodes:
                # Random exploration before filling the replay buffer
                if discrete_action_values is not None:
                    idx = np.random.randint(0, discrete_action_values.shape[0])
                    action = discrete_action_values[idx].detach().cpu().numpy().astype(np.float32)
                else:
                    action = np.random.uniform(
                        env.action_low.detach().cpu().numpy(),
                        env.action_high.detach().cpu().numpy(),
                    ).astype(np.float32)
            else:
                with torch.no_grad():
                    if discrete_action_values is not None:
                        act, _, _ = actor.sample(state_t, action_values=discrete_action_values)
                    else:
                        act, _, _ = actor.sample(state_t, action_scale=action_scale, action_bias=action_bias)
                action = act.squeeze(0).detach().cpu().numpy().astype(np.float32)

            _, reward, done, _ = env.step(action)
            next_raw = make_raw_state(env)
            replay.add({
                "last_obs": raw["last_obs"],
                "t_idx": raw["t_idx"],
                "aux_state": raw["aux_state"],
                "actions": raw["actions"],
                "obs": raw["obs"],
                "posterior": raw["posterior"],
                "filter_probs": raw["filter_probs"],
                "posterior_logits": raw["posterior_logits"],
                "filter_logits": raw["filter_logits"],
                "contrastive_thetas": raw["contrastive_thetas"],
                "contrastive_log_weights": raw["contrastive_log_weights"],
                "next_last_obs": next_raw["last_obs"],
                "next_t_idx": next_raw["t_idx"],
                "next_aux_state": next_raw["aux_state"],
                "next_actions": next_raw["actions"],
                "next_obs": next_raw["obs"],
                "next_posterior": next_raw["posterior"],
                "next_filter_probs": next_raw["filter_probs"],
                "next_posterior_logits": next_raw["posterior_logits"],
                "next_filter_logits": next_raw["filter_logits"],
                "next_contrastive_thetas": next_raw["contrastive_thetas"],
                "next_contrastive_log_weights": next_raw["contrastive_log_weights"],
                "reward": np.float32(reward),
                "done": np.float32(done),
                "action_taken": action.astype(np.float32),
            })
            raw = next_raw
            ep_return += reward

            if len(replay) >= train_cfg.batch_size:
                for _ in range(train_cfg.updates_per_step):
                    batch = replay.sample(train_cfg.batch_size, device=device)
                    update_idx += 1

                    # Compute current-state features once; reuse for critic and
                    # (in distilled_detached mode) actor without re-running the graph.
                    cur_state, _, cur_energy, cur_A = compute_state_from_batch(
                        variant, filter_backbone, batch, env, energy_net, apsi_head,
                        belief_cfg=belief_cfg, use_next=False,
                    )

                    # -- Critic update --
                    critic_state = cur_state.detach()
                    with torch.no_grad():
                        next_state, _, _, _ = compute_state_from_batch(
                            variant, filter_backbone, batch, env, energy_net, apsi_head,
                            belief_cfg=belief_cfg, use_next=True,
                        )
                        if discrete_action_values is not None:
                            next_probs, next_log_probs = actor.probs_and_log_probs(next_state)
                            nq1 = evaluate_q_over_discrete_actions(q1_tgt, next_state, discrete_action_values)
                            nq2 = evaluate_q_over_discrete_actions(q2_tgt, next_state, discrete_action_values)
                            next_v = (next_probs * (
                                torch.min(nq1, nq2) - train_cfg.alpha * next_log_probs
                            )).sum(dim=-1, keepdim=True)
                            target = batch["reward"] + (1.0 - batch["done"]) * train_cfg.gamma * next_v
                        else:
                            next_action, next_logp, _ = actor.sample(
                                next_state, action_scale=action_scale, action_bias=action_bias
                            )
                            next_q = torch.min(q1_tgt(next_state, next_action), q2_tgt(next_state, next_action))
                            target = batch["reward"] + (1.0 - batch["done"]) * train_cfg.gamma * (
                                next_q - train_cfg.alpha * next_logp
                            )
                    q1_pred = q1(critic_state, batch["action_taken"])
                    q2_pred = q2(critic_state, batch["action_taken"])
                    critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
                    critic_optim.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(q1.parameters()) + list(q2.parameters()), train_cfg.grad_clip
                    )
                    critic_optim.step()

                    # -- Actor update --
                    if belief_cfg is not None and belief_cfg.mode == "distilled_detached":
                        actor_state = cur_state.detach()
                    else:
                        actor_state, _, _, _ = compute_state_from_batch(
                            variant, filter_backbone, batch, env, energy_net, apsi_head,
                            belief_cfg=belief_cfg, use_next=False,
                        )
                    if discrete_action_values is not None:
                        probs, log_probs = actor.probs_and_log_probs(actor_state)
                        q1_all = evaluate_q_over_discrete_actions(q1, actor_state, discrete_action_values)
                        q2_all = evaluate_q_over_discrete_actions(q2, actor_state, discrete_action_values)
                        actor_loss = (
                            probs * (train_cfg.alpha * log_probs - torch.min(q1_all, q2_all))
                        ).sum(dim=-1).mean()
                    else:
                        new_action, logp, _ = actor.sample(
                            actor_state, action_scale=action_scale, action_bias=action_bias
                        )
                        actor_loss = (
                            train_cfg.alpha * logp
                            - torch.min(q1(actor_state, new_action), q2(actor_state, new_action))
                        ).mean()
                    actor_optim.zero_grad()
                    actor_loss.backward()
                    actor_grad_params = list(actor.parameters())
                    if (
                        energy_net is not None and apsi_head is not None
                        and belief_cfg is not None
                        and belief_cfg.mode in {"distilled_e2e", "learned_only"}
                    ):
                        actor_grad_params += list(energy_net.parameters()) + list(apsi_head.parameters())
                    nn.utils.clip_grad_norm_(actor_grad_params, train_cfg.grad_clip)
                    actor_optim.step()

                    # -- EBM update (every ebm_update_every steps) --
                    do_ebm_update = (
                        energy_net is not None
                        and apsi_head is not None
                        and ebm_optim is not None
                        and train_cfg.ebm_update_every > 0
                        and (update_idx % train_cfg.ebm_update_every == 0)
                    )
                    if do_ebm_update:
                        if belief_cfg is not None and belief_cfg.mode == "distilled_detached":
                            energy, A = cur_energy, cur_A
                        else:
                            _, _, energy, A = compute_state_from_batch(
                                variant, filter_backbone, batch, env, energy_net, apsi_head,
                                belief_cfg=belief_cfg, use_next=False,
                            )
                        if energy is not None and A is not None:
                            log_probs = F.log_softmax(-energy, dim=-1)
                            target_probs = batch["posterior"]
                            # Cross-entropy vs. exact posterior + ψ² regularisation
                            ebm_loss = (
                                -(target_probs * log_probs).sum(dim=-1).mean()
                                + train_cfg.apsi_coef * A.pow(2).mean()
                            )
                            ebm_optim.zero_grad()
                            ebm_loss.backward()
                            nn.utils.clip_grad_norm_(
                                list(energy_net.parameters()) + list(apsi_head.parameters()),
                                train_cfg.grad_clip,
                            )
                            ebm_optim.step()

                    soft_update(q1_tgt, q1, train_cfg.tau)
                    soft_update(q2_tgt, q2, train_cfg.tau)

        episode_returns.append(float(ep_return))
        if (ep + 1) % train_cfg.print_every == 0:
            recent = np.mean(episode_returns[-train_cfg.print_every:])
            print(
                f"[{env.name}:{variant}] seed={seed} "
                f"ep={ep+1}/{train_cfg.episodes} avg_return={recent:.4f}"
            )

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    eval_returns: List[float] = []
    belief_mean_errs: List[float] = []
    belief_kl_errs: List[float] = []
    belief_l1_errs: List[float] = []
    belief_map_exact_errs: List[float] = []
    belief_map_true_errs: List[float] = []
    bank_ig_finals: List[float] = []
    filter_bank_ig_finals: List[float] = []
    spce_lower_finals: List[float] = []
    snmc_upper_finals: List[float] = []
    bank_ig_paths: List[List[float]] = []
    filter_bank_ig_paths: List[List[float]] = []
    spce_paths: List[List[float]] = []
    snmc_paths: List[List[float]] = []

    for _ in range(train_cfg.eval_episodes):
        env.reset()
        raw = make_raw_state(env)
        done = False
        ep_return = 0.0
        ep_bank_path: List[float] = []
        ep_filter_bank_path: List[float] = []
        ep_spce_path: List[float] = []
        ep_snmc_path: List[float] = []

        while not done:
            state_t, _, energy_t, A_t = raw_state_to_policy_state(
                variant, raw, filter_backbone, env, device, energy_net, apsi_head,
                belief_cfg=belief_cfg,
            )
            # Belief quality metrics (EBM variants only)
            if energy_net is not None and energy_t is not None and A_t is not None:
                exact_probs_t = torch.tensor(raw["posterior"][None], dtype=torch.float32, device=device)
                pred_probs = posterior_probs_from_energy(energy_t)
                exact_mean_t = exact_probs_t @ env.hypothesis_bank
                pred_mean = pred_probs @ env.hypothesis_bank
                belief_mean_errs.append(float(torch.mean(torch.abs(pred_mean - exact_mean_t)).detach().cpu()))
                belief_kl_errs.append(float(belief_kl_divergence(exact_probs_t, pred_probs).mean().detach().cpu()))
                belief_l1_errs.append(float(belief_l1_error(exact_probs_t, pred_probs).mean().detach().cpu()))
                pred_map = env.hypothesis_bank[torch.argmax(pred_probs, dim=-1)]
                exact_map = env.hypothesis_bank[torch.argmax(exact_probs_t, dim=-1)]
                true_theta_t = torch.tensor(env.theta0[None], dtype=torch.float32, device=device)
                belief_map_exact_errs.append(float(env.belief_distance(pred_map, exact_map).mean().detach().cpu()))
                belief_map_true_errs.append(float(env.belief_distance(pred_map, true_theta_t).mean().detach().cpu()))

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
                    estimate_snmc_style_upper_prefix(env, prefix_actions, prefix_obs, env.theta0, snmc_L)
                )

        eval_returns.append(ep_return)
        bank_ig_finals.append(ep_bank_path[-1])
        filter_bank_ig_finals.append(ep_filter_bank_path[-1])
        spce_lower_finals.append(ep_spce_path[-1])
        if snmc_L > 0 and ep_snmc_path:
            snmc_upper_finals.append(ep_snmc_path[-1])
        bank_ig_paths.append(ep_bank_path)
        filter_bank_ig_paths.append(ep_filter_bank_path)
        spce_paths.append(ep_spce_path)
        if snmc_L > 0 and ep_snmc_path:
            snmc_paths.append(ep_snmc_path)

    out: Dict = {
        "train": {"episode_returns": episode_returns},
        "eval": {
            "avg_return": float(np.mean(eval_returns)),
            "std_return": float(np.std(eval_returns, ddof=1)) if len(eval_returns) > 1 else 0.0,
            "avg_bank_ig": float(np.mean(bank_ig_finals)),
            "avg_filter_bank_ig": float(np.mean(filter_bank_ig_finals)),
            "avg_spce_lower": float(np.mean(spce_lower_finals)),
        },
        "variant": variant,
        "seed": seed,
        "paths": {
            "bank_ig_mean_path": np.mean(np.array(bank_ig_paths, dtype=np.float64), axis=0).tolist(),
            "filter_bank_ig_mean_path": np.mean(np.array(filter_bank_ig_paths, dtype=np.float64), axis=0).tolist(),
            "spce_lower_mean_path": np.mean(np.array(spce_paths, dtype=np.float64), axis=0).tolist(),
        },
    }
    if snmc_L > 0 and snmc_upper_finals:
        out["eval"]["avg_snmc_style_upper"] = float(np.mean(snmc_upper_finals))
        out["paths"]["snmc_style_upper_mean_path"] = np.mean(
            np.array(snmc_paths, dtype=np.float64), axis=0
        ).tolist()
    if belief_mean_errs:
        out["eval"].update({
            "avg_abs_belief_mean_error": float(np.mean(belief_mean_errs)),
            "avg_belief_kl": float(np.mean(belief_kl_errs)),
            "avg_belief_l1": float(np.mean(belief_l1_errs)),
            "avg_belief_map_to_exact_distance": float(np.mean(belief_map_exact_errs)),
            "avg_belief_map_to_true_distance": float(np.mean(belief_map_true_errs)),
        })

    seed_dir = os.path.join(output_dir, variant, f"seed_{seed}")
    ensure_dir(seed_dir)
    with open(os.path.join(seed_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# Aggregate and summarise
# ---------------------------------------------------------------------------

def aggregate_plot_data(
    all_results: Dict[str, List[Dict]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Reduce per-seed results to mean ± SE arrays suitable for plotting."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for variant, results in all_results.items():
        n = max(len(results), 1)
        ep_returns = np.array([r["train"]["episode_returns"] for r in results], dtype=np.float64)
        train_mean = ep_returns.mean(axis=0)
        train_se = ep_returns.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros_like(train_mean)

        def _path_stats(key: str) -> Tuple[np.ndarray, np.ndarray]:
            # Older result files (e.g. prey_population) may not have every key
            if not all(key in r.get("paths", {}) for r in results):
                horizon_len = len(results[0]["paths"].get("bank_ig_mean_path", [0]))
                zeros = np.zeros(horizon_len)
                return zeros, zeros
            arr = np.array([r["paths"][key] for r in results], dtype=np.float64)
            m = arr.mean(axis=0)
            se = arr.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros_like(m)
            return m, se

        bank_mean, bank_se = _path_stats("bank_ig_mean_path")
        fbank_mean, fbank_se = _path_stats("filter_bank_ig_mean_path")
        spce_mean, spce_se = _path_stats("spce_lower_mean_path")

        rec: Dict[str, np.ndarray] = {
            "train_mean": train_mean, "train_se": train_se,
            "bank_mean": bank_mean, "bank_se": bank_se,
            "filter_bank_mean": fbank_mean, "filter_bank_se": fbank_se,
            "spce_mean": spce_mean, "spce_se": spce_se,
        }
        if "snmc_style_upper_mean_path" in results[0]["paths"]:
            snmc_mean, snmc_se = _path_stats("snmc_style_upper_mean_path")
            rec["snmc_mean"] = snmc_mean
            rec["snmc_se"] = snmc_se
        out[variant] = rec
    return out


def load_results_from_dir(
    output_dir: str,
) -> Tuple[Dict[str, List[Dict]], Dict]:
    """Load per-seed result JSONs and the summary from an experiment output directory.

    Args:
        output_dir: Path to the experiment output directory (must contain
                    ``summary_multi_seed.json`` and ``<variant>/seed_<s>/result.json``).

    Returns:
        (all_results, summary) where ``all_results[variant]`` is a list of per-seed dicts.
    """
    summary_path = os.path.join(output_dir, "summary_multi_seed.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    variants: List[str] = summary.get("variants", [])
    seeds: List[int] = summary.get("seeds", [])
    all_results: Dict[str, List[Dict]] = {}
    for variant in variants:
        results = []
        for seed in seeds:
            result_path = os.path.join(output_dir, variant, f"seed_{seed}", "result.json")
            if os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
            else:
                print(f"Warning: missing result for {variant}/seed_{seed} — skipping.")
        all_results[variant] = results
    return all_results, summary


# ---------------------------------------------------------------------------
# Full experiment suite
# ---------------------------------------------------------------------------

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
    """Run all (variant × seed) pairs and save aggregate statistics + plots.

    Args:
        experiment_name: Human-readable name embedded in summary JSON.
        env_factory:     Callable env constructor for the given torch device.
        output_dir:      Root output directory.
        train_cfg:       Shared training configuration.
        seeds:           List of random seeds.
        variants:        List of variant names to train.
        spce_L:          SPCE estimator sample count.
        snmc_L:          SNMC estimator sample count (0 to skip).
        belief_cfg:      EBM belief configuration.

    Returns:
        Summary dict (also written to ``<output_dir>/summary_multi_seed.json``).
    """
    from boedx.plotting import save_standard_plots  # local import to avoid circular

    ensure_dir(output_dir)
    device_str = train_cfg.device if train_cfg.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    sample_env = env_factory(torch.device(device_str))
    all_results: Dict[str, List[Dict]] = {}

    for variant in variants:
        print(f"\n{'='*60}\nTraining {experiment_name} — variant: {variant}\n{'='*60}")
        variant_results = []
        for seed in seeds:
            variant_results.append(
                train_one_seed(
                    variant, env_factory, train_cfg, seed, output_dir, spce_L, snmc_L,
                    belief_cfg=belief_cfg,
                )
            )
        all_results[variant] = variant_results

    # Aggregate statistics
    summary: Dict = {}
    for variant, results in all_results.items():
        for field in [
            "avg_return", "avg_bank_ig", "avg_filter_bank_ig", "avg_spce_lower",
            "avg_snmc_style_upper", "avg_abs_belief_mean_error", "avg_belief_kl",
            "avg_belief_l1", "avg_belief_map_to_exact_distance",
            "avg_belief_map_to_true_distance",
        ]:
            vals = [r["eval"][field] for r in results if field in r["eval"]]
            if vals:
                summary[f"{variant}_{field}"] = mean_std_ci95(np.array(vals, dtype=np.float64))

    # Paired comparisons vs. Blau baseline
    if "blau_approx" in all_results:
        blau = np.array([r["eval"]["avg_return"] for r in all_results["blau_approx"]], dtype=np.float64)
        for variant in variants:
            if variant == "blau_approx":
                continue
            cur = np.array([r["eval"]["avg_return"] for r in all_results[variant]], dtype=np.float64)
            summary[f"paired_return_diff_{variant}_minus_blau_approx"] = paired_summary(blau, cur)

    summary.update({
        "experiment_name": experiment_name,
        "variants": list(variants),
        "seeds": list(seeds),
        "horizon": sample_env.get_horizon(),
        "spce_L": spce_L,
        "snmc_L": snmc_L,
    })
    if belief_cfg is not None:
        summary["belief_config"] = {
            "mode": belief_cfg.mode,
            "feature_mode": belief_cfg.feature_mode,
            "ebm_architecture": belief_cfg.ebm_architecture,
            "n_sources": belief_cfg.n_sources,
            "source_dim": belief_cfg.source_dim,
            "add_pairwise_dist": belief_cfg.add_pairwise_dist,
            "modal_top_k": belief_cfg.modal_top_k,
            "beta_start": belief_cfg.beta_start,
            "beta_end": belief_cfg.beta_end,
            "beta_power": belief_cfg.beta_power,
        }

    with open(os.path.join(output_dir, "summary_multi_seed.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_standard_plots(output_dir, all_results, summary, sample_env.get_horizon())
    print("\n=== Multi-seed summary ===")
    print(json.dumps(summary, indent=2))
    return summary
