"""
Policy-state construction from raw environment observations.

The key abstraction is the *variant*:

+---------------------------------------+----------------------------------------------+
| Variant name                          | State representation                         |
+=======================================+==============================================+
| ``blau_approx``                       | Raw (flattened) action + obs history         |
+---------------------------------------+----------------------------------------------+
| ``control_filter_exact``              | Quotient filter probabilities (no EBM)       |
| ``control_posterior_exact``           | Quotient posterior probabilities (no EBM)    |
+---------------------------------------+----------------------------------------------+
| ``ours_ebm_control_posterior``        | Quotient + EBM belief (posterior-backed)     |
| ``ours_ebm_cross_posterior``          | Quotient + EBM belief (cross, posterior)     |
| ``ours_ebm_control_beta_contrastive`` | Quotient + EBM belief (beta-contrastive)     |
| ``ours_ebm_cross_beta_contrastive``   | Quotient + EBM belief (cross, beta-ctr.)     |
+---------------------------------------+----------------------------------------------+

All public functions consume batched tensors so they can be reused in both
the online rollout (batch-size 1) and the off-policy training loop.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from boedx.env import BeliefConfig, GenericBankBOEDEnv
from boedx.models import CachedFilterBackbone
from boedx.utils import sanitize


# ---------------------------------------------------------------------------
# Variant classification helpers
# ---------------------------------------------------------------------------

def variant_uses_ebm(variant: str) -> bool:
    return variant in {
        "ours_ebm_control",
        "ours_ebm_cross",
        "ours_ebm_control_filter",
        "ours_ebm_control_posterior",
        "ours_ebm_cross_filter",
        "ours_ebm_cross_posterior",
        "ours_ebm_control_beta_contrastive",
        "ours_ebm_cross_beta_contrastive",
    }


def variant_uses_cross_ebm(variant: str) -> bool:
    return variant in {
        "ours_ebm_cross",
        "ours_ebm_cross_filter",
        "ours_ebm_cross_posterior",
        "ours_ebm_cross_beta_contrastive",
    }


def variant_uses_beta_contrastive(variant: str) -> bool:
    return variant in {
        "ours_ebm_control_beta_contrastive",
        "ours_ebm_cross_beta_contrastive",
    }


# ---------------------------------------------------------------------------
# State builders
# ---------------------------------------------------------------------------

def build_base_state(
    hist_feat: torch.Tensor,
    t_idx: torch.Tensor,
    horizon: int,
    last_obs: torch.Tensor,
    aux_state: torch.Tensor,
) -> torch.Tensor:
    """Quotient base state: [filter_probs, t/T, last_obs, (aux)]."""
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
    """Minimal base state used in 'learned_only' belief mode: [t/T, last_obs, (aux)]."""
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
    """Raw flattened history state for the Blau baseline."""
    B = actions.shape[0]
    flat_actions = actions.reshape(B, -1)
    flat_obs = obs.reshape(B, -1)
    time_feat = t_idx.float().unsqueeze(-1) / float(max(horizon, 1))
    pieces = [flat_actions, flat_obs, time_feat, last_obs]
    if aux_state.shape[-1] > 0:
        pieces.append(aux_state)
    return sanitize(torch.cat(pieces, dim=-1), 1e3)


# ---------------------------------------------------------------------------
# Belief feature computation
# ---------------------------------------------------------------------------

def posterior_probs_from_energy(energy: torch.Tensor) -> torch.Tensor:
    """Convert energy matrix E to belief probabilities p ∝ exp(-E)."""
    return torch.exp(F.log_softmax(-energy, dim=-1))


def belief_kl_divergence(
    target_probs: torch.Tensor, pred_probs: torch.Tensor
) -> torch.Tensor:
    """KL(target ‖ pred) per batch element."""
    target = target_probs.clamp_min(1e-12)
    pred = pred_probs.clamp_min(1e-12)
    return (target * (torch.log(target) - torch.log(pred))).sum(dim=-1)


def belief_l1_error(
    target_probs: torch.Tensor, pred_probs: torch.Tensor
) -> torch.Tensor:
    """Total variation (L1) distance between two belief vectors."""
    return torch.abs(target_probs - pred_probs).sum(dim=-1)


def belief_feature_dim(
    theta_dim: int, feature_mode: str, modal_top_k: int = 4
) -> int:
    """Return the number of belief feature dimensions for a given mode."""
    if feature_mode == "legacy":
        return theta_dim + 2
    if feature_mode == "moments":
        upper = theta_dim * (theta_dim - 1) // 2
        return theta_dim + theta_dim + upper + 1 + 1
    if feature_mode == "modal":
        upper = theta_dim * (theta_dim - 1) // 2
        moments_dim = theta_dim + theta_dim + upper + 1 + 1
        modal_dim = modal_top_k * (theta_dim + 1)
        return moments_dim + modal_dim
    raise ValueError(f"Unknown belief feature mode: {feature_mode!r}")


def belief_features_from_probs(
    probs: torch.Tensor,
    theta_bank: torch.Tensor,
    A_scalar: Optional[torch.Tensor] = None,
    feature_mode: str = "legacy",
    modal_top_k: int = 4,
) -> torch.Tensor:
    """Extract a fixed-dimensional feature vector from a belief distribution.

    Args:
        probs:        (B, H) belief probabilities over the hypothesis bank.
        theta_bank:   (H, D) hypothesis positions.
        A_scalar:     (B, 1) optional EBM baseline scalar ψ(h).
        feature_mode: One of "legacy", "moments", "modal".
        modal_top_k:  Number of top atoms for "modal" mode.
    """
    log_probs = torch.log(probs.clamp_min(1e-12))
    mean = probs @ theta_bank                                # (B, D)
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True) # (B, 1)

    if feature_mode == "legacy":
        pieces = [mean, entropy]
        if A_scalar is not None:
            pieces.append(sanitize(A_scalar, 50.0))
        return sanitize(torch.cat(pieces, dim=-1), 1e3)

    if feature_mode not in {"moments", "modal"}:
        raise ValueError(f"Unknown belief feature mode: {feature_mode!r}")

    # Second-order statistics
    xc = theta_bank.unsqueeze(0) - mean.unsqueeze(1)         # (B, H, D)
    cov = torch.einsum("bh,bhd,bhe->bde", probs, xc, xc)    # (B, D, D)
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)             # (B, D)
    D = theta_bank.shape[-1]
    upper_terms = [cov[:, i, j:j+1] for i in range(D) for j in range(i + 1, D)]
    pieces = [mean, diag]
    if upper_terms:
        pieces.append(torch.cat(upper_terms, dim=-1))
    pieces.append(entropy)
    if A_scalar is not None:
        pieces.append(sanitize(A_scalar, 50.0))

    if feature_mode == "modal":
        # Top-K atoms by probability, sorted by bank index for a deterministic
        # and canonical representation on multi-modal posteriors (2-source case).
        B, H = probs.shape
        K_eff = min(int(modal_top_k), H)
        top_probs, top_idx = torch.topk(probs, k=K_eff, dim=-1)   # (B, K)
        top_theta = theta_bank[top_idx]                            # (B, K, D)
        sort_idx = torch.argsort(top_idx, dim=-1)
        idx_expand = sort_idx.unsqueeze(-1).expand(-1, -1, D)
        top_theta_sorted = torch.gather(top_theta, dim=1, index=idx_expand)
        top_probs_sorted = torch.gather(top_probs, dim=1, index=sort_idx)
        modal_flat = torch.cat(
            [top_theta_sorted.reshape(B, -1), top_probs_sorted.reshape(B, -1)], dim=-1
        )
        # Pad to fixed dimension if K_eff < modal_top_k
        target_dim = int(modal_top_k) * (D + 1)
        if modal_flat.shape[-1] < target_dim:
            pad = torch.zeros(
                B, target_dim - modal_flat.shape[-1],
                dtype=modal_flat.dtype, device=modal_flat.device,
            )
            modal_flat = torch.cat([modal_flat, pad], dim=-1)
        pieces.append(modal_flat)

    return sanitize(torch.cat(pieces, dim=-1), 1e3)


def contrastive_summary_features_from_particles(
    particle_thetas: torch.Tensor,
    particle_log_weights: torch.Tensor,
    feature_mode: str,
    modal_top_k: int,
) -> torch.Tensor:
    """Compute belief-style features from the running contrastive particle set.

    Used by the beta-contrastive EBM to build the contrastive input c_t.
    """
    probs = torch.exp(F.log_softmax(particle_log_weights, dim=-1))
    log_probs = torch.log(probs.clamp_min(1e-12))
    mean = torch.einsum("bk,bkd->bd", probs, particle_thetas)
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

    if feature_mode == "legacy":
        return sanitize(
            torch.cat([mean, entropy, torch.zeros_like(entropy)], dim=-1), 1e3
        )

    xc = particle_thetas - mean.unsqueeze(1)
    cov = torch.einsum("bk,bkd,bke->bde", probs, xc, xc)
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    D = particle_thetas.shape[-1]
    upper_terms = [cov[:, i, j:j+1] for i in range(D) for j in range(i + 1, D)]
    pieces = [mean, diag]
    if upper_terms:
        pieces.append(torch.cat(upper_terms, dim=-1))
    pieces.append(entropy)
    pieces.append(torch.zeros_like(entropy))  # placeholder for A_scalar

    if feature_mode == "modal":
        B, K, D = particle_thetas.shape
        K_eff = min(int(modal_top_k), K)
        top_probs, top_idx = torch.topk(probs, k=K_eff, dim=-1)
        idx_expand = top_idx.unsqueeze(-1).expand(-1, -1, D)
        top_theta = torch.gather(particle_thetas, dim=1, index=idx_expand)
        modal_flat = torch.cat(
            [top_theta.reshape(B, -1), top_probs.reshape(B, -1)], dim=-1
        )
        target_dim = int(modal_top_k) * (D + 1)
        if modal_flat.shape[-1] < target_dim:
            pad = torch.zeros(
                B, target_dim - modal_flat.shape[-1],
                dtype=modal_flat.dtype, device=modal_flat.device,
            )
            modal_flat = torch.cat([modal_flat, pad], dim=-1)
        pieces.append(modal_flat)

    return sanitize(torch.cat(pieces, dim=-1), 1e3)


def beta_schedule_from_tidx(
    t_idx: torch.Tensor, horizon: int, belief_cfg: BeliefConfig
) -> torch.Tensor:
    """Compute per-step β_t = β_end + (β_start − β_end)·(1 − t/(T−1))^p."""
    denom = float(max(horizon - 1, 1))
    frac = (t_idx.float().unsqueeze(-1) / denom).clamp(0.0, 1.0)
    beta = belief_cfg.beta_end + (belief_cfg.beta_start - belief_cfg.beta_end) * torch.pow(
        1.0 - frac, belief_cfg.beta_power
    )
    return beta.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# History logit selection
# ---------------------------------------------------------------------------

def history_logits_from_batch(
    variant: str, batch: Dict[str, torch.Tensor], prefix: str
) -> torch.Tensor:
    """Select which exact filter the variant should consume.

    Posterior-backed variants (``*_posterior``, beta-contrastive) use the
    Bayesian posterior logits.  Filter-backed variants use the control-filter
    logits (pure likelihood accumulation, prior-free).
    """
    if (
        variant == "control_posterior_exact"
        or variant.endswith("_posterior")
        or variant_uses_beta_contrastive(variant)
    ):
        return batch[f"{prefix}posterior_logits"]
    if variant == "control_filter_exact" or variant.endswith("_filter"):
        return batch[f"{prefix}filter_logits"]
    if variant in {"ours_ebm_control", "ours_ebm_cross"}:
        return batch[f"{prefix}filter_logits"]
    raise ValueError(f"Unknown variant for history selection: {variant!r}")


# ---------------------------------------------------------------------------
# Raw state snapshot
# ---------------------------------------------------------------------------

def make_raw_state(env: GenericBankBOEDEnv) -> Dict:
    """Snapshot the current environment state into a plain dict of numpy arrays."""
    horizon = env.get_horizon()
    actions = np.zeros((horizon, env.action_dim), dtype=np.float32)
    obs = np.zeros(horizon, dtype=np.float32)
    for i, (a, y) in enumerate(env.history):
        actions[i] = a
        obs[i] = y
    return {
        "actions": actions,
        "obs": obs,
        "length": len(env.history),
        "last_obs": float(env.last_obs),
        "t_idx": env.t,
        "aux_state": env.current_aux_state().astype(np.float32),
        "posterior": env.posterior_bank().detach().cpu().numpy().astype(np.float32),
        "filter_probs": env.equivalence_bank().detach().cpu().numpy().astype(np.float32),
        "posterior_logits": env._posterior_logu.detach().cpu().numpy().astype(np.float32),
        "filter_logits": env._equiv_logu.detach().cpu().numpy().astype(np.float32),
        "contrastive_thetas": env._contrastive_t.detach().cpu().numpy().astype(np.float32),
        "contrastive_log_weights": np.asarray(env.logC, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Batched state computation (training loop)
# ---------------------------------------------------------------------------

def compute_state_from_batch(
    variant: str,
    filter_backbone: CachedFilterBackbone,
    batch: Dict[str, torch.Tensor],
    env: GenericBankBOEDEnv,
    energy_net,
    apsi_head,
    belief_cfg: Optional[BeliefConfig] = None,
    use_next: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert a replay-buffer batch into policy state tensors.

    Returns:
        state:      (B, state_dim) policy input.
        hist_feat:  (B, H) filter / posterior probabilities.
        energy:     (B, H) EBM energy matrix or None.
        A:          (B, 1) EBM baseline scalar or None.
    """
    belief_cfg = belief_cfg or BeliefConfig()
    prefix = "next_" if use_next else ""
    last_obs = batch[f"{prefix}last_obs"]
    t_idx = batch[f"{prefix}t_idx"]
    aux_state = batch[f"{prefix}aux_state"]
    actions = batch[f"{prefix}actions"]
    obs = batch[f"{prefix}obs"]

    if variant == "blau_approx":
        raw_state = build_raw_history_state(
            actions, obs, t_idx, env.get_horizon(), last_obs, aux_state
        )
        return raw_state, raw_state, None, None

    selected_logits = history_logits_from_batch(variant, batch, prefix)
    hist_feat = filter_backbone.forward_from_logits(selected_logits)
    quotient_base = build_base_state(hist_feat, t_idx, env.get_horizon(), last_obs, aux_state)

    if variant in {"control_filter_exact", "control_posterior_exact"}:
        return quotient_base, hist_feat, None, None

    if energy_net is None or apsi_head is None or belief_cfg.mode == "exact":
        return quotient_base, hist_feat, None, None

    if variant_uses_beta_contrastive(variant):
        particle_thetas = batch[f"{prefix}contrastive_thetas"]
        particle_log_weights = batch[f"{prefix}contrastive_log_weights"]
        contrastive_feat = contrastive_summary_features_from_particles(
            particle_thetas=particle_thetas,
            particle_log_weights=particle_log_weights,
            feature_mode=belief_cfg.feature_mode,
            modal_top_k=belief_cfg.modal_top_k,
        )
        beta = beta_schedule_from_tidx(t_idx, env.get_horizon(), belief_cfg)
        energy, A, _ = energy_net.forward_beta(
            hist_feat, contrastive_feat, env.hypothesis_bank, beta
        )
    else:
        energy = energy_net(hist_feat, env.hypothesis_bank)
        A = apsi_head(hist_feat)

    probs = posterior_probs_from_energy(energy)
    belief = belief_features_from_probs(
        probs=probs,
        theta_bank=env.hypothesis_bank,
        A_scalar=A,
        feature_mode=belief_cfg.feature_mode,
        modal_top_k=belief_cfg.modal_top_k,
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
        raise ValueError(f"Unknown belief mode: {belief_cfg.mode!r}")

    return state, hist_feat, energy, A


@torch.no_grad()
def raw_state_to_policy_state(
    variant: str,
    raw_state: Dict,
    filter_backbone: CachedFilterBackbone,
    env: GenericBankBOEDEnv,
    device: torch.device,
    energy_net,
    apsi_head,
    belief_cfg: Optional[BeliefConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert a single-step raw state dict to a batched policy state (B=1)."""
    batch = {
        "last_obs": torch.tensor([[raw_state["last_obs"]]], dtype=torch.float32, device=device),
        "t_idx": torch.tensor([raw_state["t_idx"]], dtype=torch.long, device=device),
        "aux_state": torch.tensor(raw_state["aux_state"][None], dtype=torch.float32, device=device),
        "actions": torch.tensor(raw_state["actions"][None], dtype=torch.float32, device=device),
        "obs": torch.tensor(raw_state["obs"][None], dtype=torch.float32, device=device),
        "posterior_logits": torch.tensor(raw_state["posterior_logits"][None], dtype=torch.float32, device=device),
        "filter_probs": torch.tensor(raw_state["filter_probs"][None], dtype=torch.float32, device=device),
        "filter_logits": torch.tensor(raw_state["filter_logits"][None], dtype=torch.float32, device=device),
        "contrastive_thetas": torch.tensor(raw_state["contrastive_thetas"][None], dtype=torch.float32, device=device),
        "contrastive_log_weights": torch.tensor(raw_state["contrastive_log_weights"][None], dtype=torch.float32, device=device),
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
