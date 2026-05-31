# Architecture

This document describes the main internal components of BOEDX and how data moves through an experiment.

## Core Abstractions

### `GenericBankBOEDEnv`

`boedx.env.GenericBankBOEDEnv` is the base class for sequential BOED environments that maintain a finite hypothesis bank. Concrete environments implement domain-specific sampling, likelihoods, action bounds, and bank construction.

Required methods:

- `get_horizon()`: number of design steps in an episode.
- `get_action_low()` and `get_action_high()`: continuous or discrete action bounds.
- `build_hypothesis_bank()`: tensor of candidate latent parameters with shape `(H, theta_dim)`.
- `build_prior_bank_logits()`: log prior weights over the bank.
- `sample_theta()`: draw the true latent parameter for an episode.
- `sample_prior_thetas(n)`: draw prior particles for SPCE/SNMC estimates.
- `clip_action(action)`: enforce domain-level action bounds.
- `sample_observation(theta, action)`: simulate one observation.
- `loglik_scalar(obs, theta, action)`: scalar log likelihood for one observation.
- `trajectory_loglik_thetas(actions, obs, thetas)`: batch trajectory likelihood.
- `bank_loglik_single(obs_t, action_t)`: likelihood of one step under every bank atom.

Optional hooks:

- `current_aux_state()`: append auxiliary features, such as budget, to the policy state.
- `before_episode()`: reset domain-specific counters before a new episode.
- `after_step_update_aux(prev_action, action, obs)`: update budgets or costs after each step.
- `observation_to_feature_scalar(obs)`: normalize observations before adding them to policy state.
- `project_equivalence_logits(logits)`: project filters under symmetry or quotient structure.
- `belief_distance(theta_a, theta_b)`: custom distance for MAP-error metrics.

### Configuration Dataclasses

`GenericTrainConfig` controls standard trainer hyperparameters: number of episodes, replay and batch sizes, learning rates, network widths, selection settings, actor family, and regularization.

`BeliefConfig` controls how belief features are constructed and coupled to the policy. It supports exact filters, detached distillation, end-to-end distillation, learned-only belief features, modal features, geometric EBMs, and NES-specific actor-side overrides.

`NESConfig` in `boedx.nes_trainer` controls evolution-strategy training: generations, population size, perturbation schedule, utility transform, meta-optimizer, EBM update cadence, candidate reevaluation, common random numbers, and model selection.

`HomeostaticConfig` in `boedx.homeostatic` controls optional action admissibility filtering. See `homeostatic_filter.md`.

## Episode Data Flow

A normal environment step performs this sequence:

1. The trainer builds the policy state from history, filter state, belief features, and optional auxiliary state.
2. The actor proposes a raw design action.
3. If enabled, the homeostatic filter projects the action to an admissible alternative.
4. `env.step(action)` clips the action, samples an observation, and computes the SPCE reward increment.
5. The environment updates posterior and equivalence-filter logits.
6. Optional environment hooks update budgets or other domain state.
7. The trainer stores the transition, updates networks, and records diagnostics.

The environment tracks two bank distributions:

- Posterior bank: initialized from the prior and updated with likelihood increments.
- Equivalence/control filter: initialized either as pure likelihood logits or posterior logits depending on `exact_filter`.

## State and Belief Features

`boedx.state` constructs policy inputs from the trajectory and belief summaries. The supported belief feature modes are:

- `legacy`: posterior mean and entropy.
- `moments`: posterior mean, diagonal covariance, upper-triangular covariance terms, and entropy.
- `modal`: moment features plus top-probability bank atoms.

For source-location problems, `BeliefConfig` can use geometric/permutation-aware representations with `n_sources`, `source_dim`, `modal_top_k`, and `add_pairwise_dist`.

## Trainers

### Standard Trainer

`boedx.trainer.run_experiment_suite` runs multi-seed experiments with SAC-style actor/critic updates and optional EBM belief learning. It handles:

- Variant creation.
- Per-seed training.
- Periodic model selection.
- Evaluation rollouts.
- SPCE/SNMC/bank-IG diagnostics.
- JSON summaries and standard plots.

### NES Trainer

`boedx.nes_trainer.run_experiment_suite_nes` optimizes actor parameters with Natural Evolution Strategies. It samples perturbations around a mean parameter vector, evaluates candidate policies by rollout, converts returns to utilities, and updates the mean with Adam/RMSProp/SGD-style optimizers.

NES can train or freeze the EBM belief model, use mirrored sampling, common random numbers, top-candidate reevaluation, parallel candidate evaluation, transformer actors, and phase-adaptive late-horizon actor heads.

## Plotting

`boedx.plotting` provides two layers:

- `save_standard_plots(output_dir, all_results, summary, horizon)`: lightweight plots created during experiment runs.
- `generate_scientific_plots(output_dir)`: loads saved JSON files and creates publication-ready PNG/PDF figures in `graph_<experiment_name>/`.

The scientific plot set includes SPCE trajectories, bank information gain, SNMC bounds when available, per-seed fan plots, paired-difference bars, belief-quality panels, and return-vs-SPCE scatter plots.
