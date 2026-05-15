"""
Backward-compatibility shim.

The code previously in this file has been reorganised into the ``boedx``
installable package::

    boedx/utils.py      — set_seed, sanitize, gaussian_logpdf, …
    boedx/models.py     — EnergyNet, BetaContrastiveEnergyNet, actors, critics, …
    boedx/buffer.py     — ReplayBuffer
    boedx/env.py        — GenericBankBOEDEnv, GenericTrainConfig, BeliefConfig
    boedx/state.py      — build_*_state, compute_state_from_batch, …
    boedx/trainer.py    — train_one_seed, run_experiment_suite, …
    boedx/plotting.py   — save_standard_plots, generate_scientific_plots

All public names that were previously importable from this module are
re-exported here so that existing code continues to work without changes.
"""

# ruff: noqa: F401

from boedx.env import BeliefConfig, GenericBankBOEDEnv, GenericTrainConfig
from boedx.models import (
    ApsiHead,
    BetaContrastiveEnergyNet,
    CachedFilterBackbone,
    CrossInteractionEnergyNet,
    DiscreteCategoricalActor,
    EnergyNet,
    MixtureTanhGaussianActor,
    QCritic,
    SymmetricSourceCrossNet,
    SymmetricSourceEnergyNet,
    TanhGaussianActor,
)
from boedx.buffer import ReplayBuffer
from boedx.plotting import save_standard_plots
from boedx.state import (
    belief_feature_dim,
    belief_features_from_probs,
    belief_kl_divergence,
    belief_l1_error,
    build_base_state,
    build_minimal_state,
    build_raw_history_state,
    compute_state_from_batch,
    contrastive_summary_features_from_particles,
    beta_schedule_from_tidx,
    history_logits_from_batch,
    make_raw_state,
    posterior_probs_from_energy,
    raw_state_to_policy_state,
    variant_uses_beta_contrastive,
    variant_uses_cross_ebm,
    variant_uses_ebm,
)
from boedx.trainer import (
    aggregate_plot_data,
    build_modules,
    discrete_bank_ig_from_logits,
    estimate_snmc_style_upper_prefix,
    estimate_spce_prefix,
    evaluate_q_over_discrete_actions,
    get_discrete_action_values,
    run_experiment_suite,
    train_one_seed,
    uses_discrete_actor,
)
from boedx.utils import (
    ensure_dir,
    gaussian_logpdf,
    logmeanexp_t,
    mean_std_ci95,
    paired_summary,
    sanitize,
    set_seed,
    soft_update,
)
