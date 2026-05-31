# Experiments

BOEDX ships three main experiment entry points and one plotting entry point.

## Installation Check

From the repository root, verify the package imports and tests run:

```bash
python3 -m pip install -e '.[dev]'
pytest -vv tests/test_homeostatic_filter.py
```

## Source Location: Policy-Gradient Trainer

Entry point:

```bash
boedx-source-location --help
```

Small smoke run:

```bash
boedx-source-location \
  --episodes 20 \
  --eval-episodes 5 \
  --seeds 0 \
  --variants blau_approx,ours_ebm_cross \
  --output-dir ./outputs/source_location_smoke
```

Useful options:

- `--episodes`: training budget.
- `--eval-episodes`: evaluation episodes per evaluation call.
- `--seeds`: comma-separated seeds.
- `--device`: `auto`, `cpu`, or a PyTorch device string.
- `--variants`: comma-separated variant names.
- `--horizon`: BOED horizon.
- `--bank-grid-size`, `--bank-grid-min`, `--bank-grid-max`: source-location bank discretization.
- `--sigma`: observation noise standard deviation.
- `--belief-mode`: `exact`, `distilled_detached`, `distilled_e2e`, or `learned_only`.
- `--belief-feature-mode`: `legacy`, `moments`, or `modal`.
- `--ebm-architecture`: `standard` or `geometric`.
- `--actor-family`: `gaussian` or `mog`.
- `--enable-aux-state`: expose environment auxiliary state to the actor.
- `--homeo-*`: homeostatic action-filter settings.

## Source Location: NES Trainer

Entry point:

```bash
boedx-source-location-nes --help
```

Small smoke run:

```bash
boedx-source-location-nes \
  --generations 10 \
  --population-size 8 \
  --rollout-episodes-per-candidate 1 \
  --eval-episodes 5 \
  --seeds 0 \
  --variants blau_approx,ours_ebm_cross_posterior \
  --output-dir ./outputs/source_location_nes_smoke
```

Useful NES options:

- `--generations`: number of NES generations.
- `--population-size`: number of perturbations per generation.
- `--rollout-episodes-per-candidate`: fitness rollouts per candidate.
- `--nes-lr-mu`: learning rate for the mean parameter vector.
- `--nes-sigma-init` and `--nes-sigma-final`: perturbation scale schedule endpoints.
- `--nes-sigma-schedule`: `constant`, `linear`, or `exp`.
- `--nes-mirrored-sampling` / `--no-nes-mirrored-sampling`: antithetic sampling.
- `--nes-utility-mode`: `nes` or `centered_ranks`.
- `--nes-optimizer`: `adam`, `rmsprop`, or `sgd`.
- `--parallel-candidates`, `--n-jobs`, `--parallel-backend`: parallel candidate evaluation.
- `--freeze-ebm` or `--ebm-freeze-after-generation`: EBM training control.
- `--actor-family`: `gaussian`, `mog`, or `transformer`.
- `--phase-adaptive-actor`: enable late-horizon actor correction.

## Prey Population

Entry point:

```bash
boedx-prey-population --help
```

Small smoke run:

```bash
boedx-prey-population \
  --episodes 20 \
  --eval-episodes 5 \
  --seeds 0 \
  --variants blau_approx,ours_ebm_cross \
  --output-dir ./outputs/prey_population_smoke
```

Useful options:

- `--bank-size`: number of prey-population hypotheses.
- `--horizon`: number of sequential interventions.
- `--belief-mode`: belief coupling mode.
- `--belief-feature-mode`: `legacy` or `moments`.
- `--spce-L` and `--snmc-L`: final evaluation particle counts.
- `--homeo-*`: homeostatic viability filter settings.

## Plotting

Generate scientific plots from a completed output directory:

```bash
boedx-plot ./outputs/source_location_smoke
```

The plotting command expects:

- `summary_multi_seed.json` in the output directory.
- Per-seed JSON result files written by the experiment suite.

It creates a sibling graph directory named like `graph_source_location_smoke/` containing PNG and PDF figures.

## Output Layout

Experiment outputs are JSON-first so downstream analysis scripts can inspect them directly. Exact filenames depend on the trainer path, but typical outputs include:

- `summary_multi_seed.json`: aggregate metrics and configuration.
- Per-seed/per-variant result JSON files.
- Standard summary plots saved during the run.
- Scientific plots after `boedx-plot`.

Important metric families:

- Evaluation return.
- SPCE lower-bound trajectories.
- SNMC upper-bound estimates when enabled.
- Posterior bank information gain.
- Control/filter bank information gain.
- Belief diagnostics such as KL, L1, MAP-to-exact, and MAP-to-true.
- Homeostatic diagnostics when filtering is enabled.

## Reproducibility Notes

- Use explicit `--seeds` for reproducible multi-seed comparisons.
- Use fixed `--variants` lists when comparing methods.
- Keep `--spce-L` and `--snmc-L` stable across comparisons.
- For NES, `--use-common-random-numbers` can reduce candidate-evaluation noise.
- Record the full command line in experiment notes because many research knobs materially affect the results.
