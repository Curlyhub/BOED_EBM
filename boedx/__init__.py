"""
BOEDX — Bayesian Optimal Experimental Design with Expressive Policies.

Public API re-exported from sub-modules for convenience::

    from boedx import (
        GenericBankBOEDEnv, GenericTrainConfig, BeliefConfig,
        run_experiment_suite, generate_scientific_plots,
        # NES (Natural Evolution Strategy) path:
        NESConfig, run_experiment_suite_nes, train_one_seed_nes,
    )
"""

from boedx.env import BeliefConfig, GenericBankBOEDEnv, GenericTrainConfig
from boedx.homeostatic import (
    HomeostaticAdmissibility,
    HomeostaticConfig,
    HomeostaticContext,
    build_homeostatic_admissibility,
    enforce_homeostatic_admissibility,
    filter_action,
)
from boedx.nes_trainer import NESConfig, run_experiment_suite_nes, train_one_seed_nes
from boedx.plotting import generate_scientific_plots, save_standard_plots
from boedx.trainer import run_experiment_suite, train_one_seed
from boedx.utils import ensure_dir, set_seed

__version__ = "0.1.0"

__all__ = [
    # Core environment / config
    "BeliefConfig",
    "GenericBankBOEDEnv",
    "GenericTrainConfig",
    "HomeostaticAdmissibility",
    "HomeostaticConfig",
    "HomeostaticContext",
    "build_homeostatic_admissibility",
    "enforce_homeostatic_admissibility",
    "filter_action",
    # RL training path (SAC)
    "run_experiment_suite",
    "train_one_seed",
    # NES training path (OpenAI-ES)
    "NESConfig",
    "run_experiment_suite_nes",
    "train_one_seed_nes",
    # Plotting
    "generate_scientific_plots",
    "save_standard_plots",
    # Utilities
    "ensure_dir",
    "set_seed",
]
