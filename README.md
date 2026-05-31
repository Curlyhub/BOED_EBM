# BOEDX

BOEDX is a research codebase for Bayesian Optimal Experimental Design with expressive policies. It provides bank-based sequential BOED environments, policy-gradient and Natural Evolution Strategy trainers, EBM belief models, scientific plotting utilities, and optional homeostatic action filtering for safety or viability constraints.

The package is organized around reusable environment and trainer abstractions in `boedx/`, plus runnable benchmark scripts for source localization and prey-population experiments.

## Main Features

- Bank-based sequential BOED environment interface with exact posterior/filter tracking.
- SPCE-style dense rewards for sequential design policies.
- SAC-style policy-gradient trainer for continuous and discrete benchmarks.
- NES/OpenAI-ES trainer for direct policy parameter search.
- EBM belief features with exact, distilled, end-to-end, and learned-only modes.
- Source-location and prey-population benchmark experiments.
- Optional homeostatic action filter for movement, risk, budget, and population-viability constraints.
- Multi-seed result summaries and publication-oriented plots.

## Repository Layout

```text
boedx/
  env.py                     Generic BOED environment interface and config dataclasses
  trainer.py                 SAC-style training/evaluation suite
  nes_trainer.py             NES/OpenAI-ES training/evaluation suite
  homeostatic.py             Optional action admissibility filter
  models.py                  Actor, critic, and EBM network modules
  state.py                   State and belief feature construction
  plotting.py                Standard and publication-quality plotting
  experiments/
    source_location.py       Policy-gradient source-location benchmark
    source_location_nes.py   NES source-location benchmark
    prey_population.py       Prey-population benchmark
tests/
  test_homeostatic_filter.py Homeostatic filter regression and smoke tests
docs/
  architecture.md            Internal architecture and data flow
  experiments.md             CLI usage and experiment recipes
  homeostatic_filter.md      Homeostatic filtering reference
  development.md             Testing and extension notes
```

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e '.[dev]'
```

Core dependencies are `torch`, `numpy`, and `matplotlib`. The development extra adds `pytest` and `ruff`.

If you already have an environment with the dependencies installed, install the package in editable mode:

```bash
python3 -m pip install -e .
```

## Quick Start

Run the targeted regression tests:

```bash
pytest -vv tests/test_homeostatic_filter.py
```

Run a small source-location policy-gradient experiment:

```bash
boedx-source-location \
  --episodes 20 \
  --eval-episodes 5 \
  --seeds 0 \
  --output-dir ./outputs/source_location_smoke
```

Run a small NES source-location experiment:

```bash
boedx-source-location-nes \
  --generations 10 \
  --population-size 8 \
  --rollout-episodes-per-candidate 1 \
  --eval-episodes 5 \
  --seeds 0 \
  --output-dir ./outputs/source_location_nes_smoke
```

Run a small prey-population experiment:

```bash
boedx-prey-population \
  --episodes 20 \
  --eval-episodes 5 \
  --seeds 0 \
  --output-dir ./outputs/prey_population_smoke
```

Generate scientific plots from an experiment output directory:

```bash
boedx-plot ./outputs/source_location_smoke
```

The plotting command expects an experiment output directory containing `summary_multi_seed.json` and per-seed result JSON files.

## Console Scripts

The package defines these scripts in `pyproject.toml`:

- `boedx-source-location`: policy-gradient two-source localization benchmark.
- `boedx-source-location-nes`: NES/OpenAI-ES source-localization benchmark.
- `boedx-prey-population`: discrete prey-population benchmark.
- `boedx-plot`: generate publication-quality figures from saved output.

Each experiment script accepts `--help` for the full argument list.

## Experiment Variants

The trainers compare baseline and EBM-enhanced design policies. Common variant names include:

- `blau_approx`: baseline approximation.
- `control_filter_exact`: exact control-side filter ablation.
- `control_posterior_exact`: exact posterior-control ablation.
- `ours_ebm_control`: EBM belief used for control-side features.
- `ours_ebm_cross`: EBM cross-interaction variant.
- `ours_ebm_control_beta_contrastive` and `ours_ebm_cross_beta_contrastive`: beta-contrastive schedules.

Available defaults vary by experiment. Pass `--variants name1,name2` to run a subset.

## Homeostatic Filtering

Homeostatic filtering is disabled by default. Enable it with:

```bash
boedx-source-location \
  --homeo-enabled \
  --homeo-mode source_location \
  --homeo-max-step-norm 0.5 \
  --homeo-danger-radius 0.25 \
  --homeo-danger-prob-max 0.10
```

For prey population viability:

```bash
boedx-prey-population \
  --homeo-enabled \
  --homeo-mode prey_population \
  --homeo-prey-min 5 \
  --homeo-extinction-prob-max 0.05
```

See `docs/homeostatic_filter.md` for the full behavior and diagnostic fields.

## Documentation

- `docs/architecture.md`: package architecture, state flow, environment interface, and trainer responsibilities.
- `docs/experiments.md`: CLI recipes, output layout, plotting, and common options.
- `docs/homeostatic_filter.md`: detailed homeostatic action filtering reference.
- `docs/development.md`: testing, warning policy, and how to add new environments or variants.

## Output Files

Experiment suites write JSON summaries and plots into the selected `--output-dir`. The plotting module can also create a sibling `graph_<experiment_name>/` directory containing PNG and PDF figures.

Typical outputs include multi-seed summaries, per-variant/per-seed records, SPCE and bank-IG trajectories, SNMC estimates when requested, belief diagnostics, and homeostatic diagnostics when the filter is enabled.

## Current Test Status

The focused homeostatic regression suite passes cleanly:

```text
6 passed
```

The pytest configuration filters known third-party `PyparsingDeprecationWarning` noise emitted through `matplotlib`/`pyparsing`, while keeping project warnings visible.
