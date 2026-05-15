"""
2-D two-source localisation benchmark optimised with NES (OpenAI-ES).

This module reuses the same ``SourceLocalization2DEnv`` physics as the RL
variant (``boedx.experiments.source_location``) but replaces the SAC RL
training loop with a Natural Evolution Strategy optimiser.

NES vs RL
---------
Unlike SAC, NES operates without a replay buffer or Q-critics:

  - **Population perturbation** — each generation, a population of actor
    parameters is sampled around a mean vector ``μ`` with isotropic Gaussian
    noise ``σ``.
  - **Fitness evaluation** — each candidate policy is evaluated via fresh
    episode rollouts; the cumulative SPCE return is the fitness signal.
  - **Natural gradient update** — utilities (fitness ranks or raw scores) are
    used to compute a weighted sum of the perturbations, giving a natural
    gradient estimate that drives the Adam/RMSProp update of ``μ``.
  - **Offline EBM training** — the EBM posterior surrogate is trained
    supervised against the exact filter, not via temporal-difference learning.

Compared variants
-----------------
+------------------------------------------+-----------------------------------------------+
| Variant name                             | Policy-state representation                   |
+==========================================+===============================================+
| ``blau_approx``                          | Flattened raw (action, obs) history           |
+------------------------------------------+-----------------------------------------------+
| ``control_posterior_exact``              | Quotient + exact posterior                    |
| ``control_filter_exact``                 | Quotient + exact control filter               |
| ``ours_ebm_cross_posterior``             | Quotient + EBM-cross (posterior mode)         |
| ``ours_ebm_cross_filter``                | Quotient + EBM-cross (filter mode)            |
+------------------------------------------+-----------------------------------------------+

Canonical pair symmetry
-----------------------
The two sources are unordered — swapping θ₁ ↔ θ₂ gives the same physical
configuration.  ``SourceLocalization2DEnv`` (imported from the RL experiment
module) canonicalises pairs so that the first source is lexicographically ≤
the second.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Sequence

import torch

from boedx.env import BeliefConfig, GenericTrainConfig
from boedx.experiments.source_location import SourceLocalization2DEnv, SourceLocationConfig
from boedx.nes_trainer import NESConfig, run_experiment_suite_nes


# Canonical set of compared variants for NES source-location experiments.
VARIANTS = [
    "blau_approx",
    "control_posterior_exact",
    "control_filter_exact",
    "ours_ebm_cross_posterior",
    "ours_ebm_cross_filter",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2-D source-localisation benchmark with NES (OpenAI-ES) optimisation."
    )

    # ── NES core ──────────────────────────────────────────────────────────────
    p.add_argument("--generations", type=int, default=200,
                   help="Number of NES generations (analogous to RL episodes).")
    p.add_argument("--population-size", type=int, default=48,
                   help="Number of parameter perturbations sampled per generation.")
    p.add_argument("--rollout-episodes-per-candidate", type=int, default=2,
                   help="Environment episodes used to evaluate each candidate's fitness.")
    p.add_argument("--eval-episodes", type=int, default=150,
                   help="Episodes used for periodic policy evaluation.")
    p.add_argument("--nes-lr-mu", type=float, default=0.04,
                   help="Learning rate for the NES mean (μ) update.")
    p.add_argument("--nes-lr-sigma", type=float, default=0.10,
                   help="Learning rate for the adaptive σ (if sigma_adapt_on_success).")
    p.add_argument("--nes-sigma-init", type=float, default=0.03,
                   help="Initial perturbation std σ.")
    p.add_argument("--nes-sigma-final", type=float, default=0.005,
                   help="Final perturbation std σ for annealing schedules.")
    p.add_argument("--nes-sigma-schedule", type=str, default="exp",
                   choices=["constant", "linear", "exp"],
                   help="σ annealing schedule across generations.")
    p.add_argument("--nes-mirrored-sampling", action="store_true", default=True,
                   help="Enable antithetic (mirrored) perturbation pairs.")
    p.add_argument("--no-nes-mirrored-sampling", dest="nes_mirrored_sampling",
                   action="store_false")
    p.add_argument("--nes-utility-mode", type=str, default="nes",
                   choices=["nes", "centered_ranks"],
                   help="How raw fitness scores are converted to update utilities.")
    p.add_argument("--nes-optimizer", type=str, default="adam",
                   choices=["adam", "rmsprop", "sgd"],
                   help="Meta-optimiser for the μ update.")
    p.add_argument("--nes-beta1", type=float, default=0.9)
    p.add_argument("--nes-beta2", type=float, default=0.999)
    p.add_argument("--nes-eps", type=float, default=1e-8)
    p.add_argument("--nes-sigma-adapt-on-success", action="store_true",
                   help="Adapt σ based on fraction of population beating a threshold.")
    p.add_argument("--nes-sigma-success-target", type=float, default=0.20)
    p.add_argument("--nes-sigma-adapt-rate", type=float, default=0.05)

    # ── EBM training (offline, supervised) ───────────────────────────────────
    p.add_argument("--ebm-updates-per-generation", type=int, default=10,
                   help="Gradient steps on the EBM per NES generation.")
    p.add_argument("--ebm-batch-size", type=int, default=128)
    p.add_argument("--ebm-data-episodes", type=int, default=8,
                   help="Rollout episodes used to collect EBM training data.")
    p.add_argument("--ebm-pretrain-episodes", type=int, default=0,
                   help="Episodes collected for EBM pre-training before NES starts.")
    p.add_argument("--ebm-pretrain-updates", type=int, default=0)
    p.add_argument("--ebm-update-every-generations", type=int, default=1,
                   help="Run EBM training every N generations.")
    p.add_argument("--freeze-ebm", action="store_true",
                   help="Keep EBM weights frozen throughout training.")
    p.add_argument("--ebm-freeze-after-generation", type=int, default=-1,
                   help="Freeze EBM after this generation (-1 = never).")

    # ── Variance reduction / parallelism ─────────────────────────────────────
    p.add_argument("--use-common-random-numbers", action="store_true",
                   help="Use the same random seeds across all candidates (CRN).")
    p.add_argument("--common-random-numbers-seed-stride", type=int, default=1000003)
    p.add_argument("--reevaluate-top-candidates", type=int, default=0,
                   help="Re-evaluate the top-K candidates with extra episodes.")
    p.add_argument("--reevaluate-top-episodes", type=int, default=0)
    p.add_argument("--parallel-candidates", action="store_true",
                   help="Evaluate candidates in parallel using joblib.")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--parallel-backend", type=str, default="threading",
                   choices=["threading", "loky"])

    # ── Model selection ───────────────────────────────────────────────────────
    p.add_argument("--selection-eval-episodes", type=int, default=40)
    p.add_argument("--selection-every", type=int, default=10)
    p.add_argument("--selection-start-generation", type=int, default=10)
    p.add_argument("--selection-top-k", type=int, default=3)
    p.add_argument("--selection-final-eval-episodes", type=int, default=0)
    p.add_argument("--selection-return-weight", type=float, default=1.0)
    p.add_argument("--selection-belief-kl-weight", type=float, default=0.05)
    p.add_argument("--selection-belief-map-weight", type=float, default=0.15)
    p.add_argument("--selection-belief-mean-weight", type=float, default=0.25)
    # Cross-variant EBM selection weights (active only for cross-EBM variants).
    p.add_argument("--cross-selection-bank-ig-weight", type=float, default=0.30)
    p.add_argument("--cross-selection-spce-weight", type=float, default=0.20)
    p.add_argument("--cross-selection-filter-bank-ig-weight", type=float, default=0.00)
    p.add_argument("--cross-selection-gap-penalty-weight", type=float, default=0.35)
    p.add_argument("--cross-selection-belief-kl-weight", type=float, default=0.02)
    p.add_argument("--cross-selection-belief-map-weight", type=float, default=0.04)
    p.add_argument("--cross-selection-belief-mean-weight", type=float, default=0.04)

    # ── Experiment scope ──────────────────────────────────────────────────────
    p.add_argument("--seeds", type=str, default="0,1,2",
                   help="Comma-separated random seeds.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="./outputs/source_location_nes")
    p.add_argument("--variants", type=str,
                   default=",".join(["blau_approx", "ours_ebm_cross_posterior"]),
                   help="Comma-separated variants to run.")

    # ── Environment ───────────────────────────────────────────────────────────
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--prior-std", type=float, default=1.0)
    p.add_argument("--design-min", type=float, default=-4.0)
    p.add_argument("--design-max", type=float, default=4.0)
    p.add_argument("--bank-grid-size", type=int, default=7)
    p.add_argument("--bank-grid-min", type=float, default=-3.0)
    p.add_argument("--bank-grid-max", type=float, default=3.0)
    p.add_argument("--sigma", type=float, default=0.35,
                   help="Observation noise std.")
    p.add_argument("--n-contrastive", type=int, default=128,
                   help="Number of contrastive particles for SPCE reward.")
    p.add_argument("--exact-filter", type=str, default="likelihood",
                   choices=["likelihood", "posterior"])
    # Budget-constrained mode (optional — disabled by default).
    p.add_argument("--enable-aux-state", action="store_true",
                   help="Expose remaining budget as auxiliary policy input.")
    p.add_argument("--budget-total", type=float, default=24.0)
    p.add_argument("--move-cost-coef", type=float, default=1.0)
    p.add_argument("--probe-cost", type=float, default=0.35)
    p.add_argument("--budget-violation-penalty", type=float, default=-2.0)

    # ── Networks ──────────────────────────────────────────────────────────────
    p.add_argument("--hidden-rl", type=int, default=256,
                   help="Hidden size of the actor MLP.")
    p.add_argument("--hidden-ebm", type=int, default=256,
                   help="Hidden size of the EBM network.")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Discount factor (typically 1.0 for BOED).")
    p.add_argument("--actor-family", type=str, default="mog",
                   choices=["gaussian", "mog", "transformer"],
                   help="Actor architecture family.")
    p.add_argument("--actor-mixture-components", type=int, default=4,
                   help="Number of components for mixture-of-Gaussians actors.")
    # EBM-actor overrides: the cross-variant actor can use a different family.
    p.add_argument("--ebm-actor-family", type=str, default="",
                   choices=["", "gaussian", "mog", "transformer"])
    p.add_argument("--ebm-hidden-rl", type=int, default=0)
    p.add_argument("--ebm-actor-mixture-components", type=int, default=0)
    p.add_argument("--ebm-dual-branch-actor", action="store_true",
                   help="Separate base/belief encoder branches in the EBM actor.")
    # Transformer actor hyper-parameters.
    p.add_argument("--transformer-d-model", type=int, default=64)
    p.add_argument("--transformer-nhead", type=int, default=4)
    p.add_argument("--transformer-layers", type=int, default=2)
    p.add_argument("--transformer-ff", type=int, default=128)
    # Phase-adaptive late-horizon residual head.
    p.add_argument("--phase-adaptive-actor", action="store_true")
    p.add_argument("--phase-start-frac", type=float, default=0.6)
    p.add_argument("--phase-strength", type=float, default=1.0)
    p.add_argument("--late-std-scale", type=float, default=0.5)
    p.add_argument("--late-mix-temp", type=float, default=0.75)

    # ── Belief / EBM configuration ────────────────────────────────────────────
    p.add_argument("--belief-mode", type=str, default="distilled_detached",
                   choices=["exact", "distilled_detached", "distilled_e2e", "learned_only"])
    p.add_argument("--belief-feature-mode", type=str, default="modal",
                   choices=["legacy", "moments", "modal"])
    p.add_argument("--ebm-architecture", type=str, default="geometric",
                   choices=["standard", "geometric"],
                   help="'geometric' uses permutation-invariant Deep Sets EBM.")
    p.add_argument("--n-sources", type=int, default=2)
    p.add_argument("--source-dim", type=int, default=2)
    p.add_argument("--modal-top-k", type=int, default=4)
    p.add_argument("--nes-actor-feature-mode", type=str, default="",
                   choices=["", "legacy", "moments", "modal"],
                   help="Override belief feature mode seen by the NES actor only.")
    p.add_argument("--nes-actor-modal-top-k", type=int, default=0)
    p.add_argument("--nes-cross-compact-belief", action="store_true",
                   help="Cap actor top-K at 2 for cross-variant actors.")
    p.add_argument("--add-pairwise-dist", action="store_true", default=True,
                   help="Append pairwise source distances to Deep Sets EBM features.")
    p.add_argument("--no-pairwise-dist", dest="add_pairwise_dist", action="store_false")

    # ── Evaluation ────────────────────────────────────────────────────────────
    p.add_argument("--spce-L", type=int, default=1024,
                   help="SPCE contrastive particle count for final evaluation.")
    p.add_argument("--snmc-L", type=int, default=512,
                   help="SNMC particle count for upper-bound evaluation.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Environment config ────────────────────────────────────────────────────
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

    # ── Training config (shared network sizes / device) ───────────────────────
    train_cfg = GenericTrainConfig(
        episodes=args.generations,   # NES maps "generations" → "episodes" slot
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
    # The Transformer actor needs the horizon to size its sequence buffers.
    train_cfg.sequence_horizon = args.horizon  # type: ignore[attr-defined]

    # ── NES config (all evolution-strategy hyper-parameters) ──────────────────
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

    # ── Belief / EBM config ───────────────────────────────────────────────────
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
        # Automatically enable raw history when a Transformer actor is in use.
        include_raw_history_for_ebm_actor=(
            args.ebm_actor_family == "transformer"
            or args.actor_family == "transformer"
        ),
    )

    seeds: Sequence[int] = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

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
    with open(
        os.path.join(args.output_dir, "run_config_and_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
