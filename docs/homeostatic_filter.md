# Homeostatic Action Filter

The homeostatic filter is an optional admissibility layer between a policy action and `env.step`. It is implemented in `boedx.homeostatic` and is disabled by default.

The filter does not replace the environment's own `clip_action`; it adds constraints that depend on previous actions, posterior risk, budgets, or population viability.

## Core Objects

### `HomeostaticConfig`

Main fields:

- `enabled`: activates filtering.
- `mode`: `none`, `source_location`, or `prey_population`.
- `projection_mode`: currently `nearest`.
- `max_projection_candidates`: maximum number of candidate replacement actions.
- `max_step_norm`: maximum allowed movement from the previous action.
- `danger_radius`: source-location risk radius around posterior particles.
- `danger_prob_max`: maximum posterior probability mass allowed inside the danger radius.
- `use_ebm_posterior_for_risk`: use EBM posterior particles/weights for risk when available.
- `prey_min`, `prey_max`, `predator_max`: prey-population viability thresholds.
- `extinction_prob_max`: maximum probability of prey falling below `prey_min`.
- `explosion_prob_max`: maximum probability of prey or predator exceeding upper thresholds.
- `initial_budget`: optional homeostatic budget.
- `obs_cost`: fixed per-observation budget cost.
- `movement_cost`: budget cost multiplier for movement distance.

### `HomeostaticContext`

The trainer builds a context for each action containing:

- `env`: current environment instance.
- `step_index`: current time index.
- `prev_action`: previous design action, if any.
- `budget`: remaining homeostatic budget, if configured.
- `posterior_particles` and `posterior_weights`: exact/bank posterior risk source.
- `ebm_particles` and `ebm_weights`: learned posterior risk source when available.
- `raw_state`: trainer-specific raw state metadata.

### `HomeostaticStats`

Aggregates filter diagnostics over steps and returns summary fields such as:

- `homeo_enabled`
- `homeo_was_filtered`
- `homeo_raw_action_feasible`
- `homeo_num_feasible_candidates`
- `homeo_rejection_rate`
- `homeo_mean_raw_filtered_distance`
- `homeo_mean_risk_prob`
- `homeo_mean_movement_cost`
- `homeo_budget_exhaustion_rate`
- `homeo_violation_rate`

## Filtering Algorithm

For each raw action:

1. Convert the raw action to a NumPy float32 array.
2. Return immediately if filtering is disabled or `mode == "none"`.
3. Clip the raw action with the environment's `clip_action`.
4. Check feasibility against budget, movement, and mode-specific risk constraints.
5. If clipped/raw action is feasible, return it.
6. Generate candidate projected actions near the raw action.
7. Keep feasible candidates and choose the one with minimum Euclidean distance to the raw action.
8. If no feasible candidate exists, fall back to movement projection or clipping and mark the diagnostic as a violation if still infeasible.

Only `nearest` projection is currently implemented.

## Source-Location Mode

Use `mode = source_location` for design points in source-localization tasks.

Constraints:

- `max_step_norm`: limits movement between consecutive design locations.
- `danger_radius`: defines a ball around each posterior source particle.
- `danger_prob_max`: rejects actions whose posterior probability mass inside the danger radius is too large.
- `initial_budget`, `obs_cost`, `movement_cost`: optionally limit total experiment cost.

Example:

```bash
boedx-source-location \
  --homeo-enabled \
  --homeo-mode source_location \
  --homeo-max-step-norm 0.5 \
  --homeo-danger-radius 0.25 \
  --homeo-danger-prob-max 0.10 \
  --homeo-initial-budget 10 \
  --homeo-obs-cost 0.1 \
  --homeo-movement-cost 1.0
```

Risk probability is computed from posterior particles. If a theta particle contains multiple sources, the action is considered risky if it is within `danger_radius` of any source in that particle.

## Prey-Population Mode

Use `mode = prey_population` for intervention choices where the next predicted population should remain viable.

Constraints:

- `prey_min`: minimum acceptable predicted prey population.
- `prey_max`: maximum acceptable predicted prey population.
- `predator_max`: maximum acceptable predicted predator population, if the environment predicts predator state.
- `extinction_prob_max`: maximum posterior probability of violating `prey_min`.
- `explosion_prob_max`: maximum posterior probability of exceeding `prey_max` or `predator_max`.

Example:

```bash
boedx-prey-population \
  --homeo-enabled \
  --homeo-mode prey_population \
  --homeo-prey-min 5 \
  --homeo-prey-max 500 \
  --homeo-extinction-prob-max 0.05 \
  --homeo-explosion-prob-max 0.05
```

For prey-population viability, the environment must provide `predict_population_next(particles, action)`. If that method is not available, the diagnostic marks the violation as not computable.

## Budget Handling

`update_budget_after_action` subtracts:

```text
obs_cost + movement_cost * ||action - prev_action||
```

If there is no previous action, only `obs_cost` is charged. If the remaining budget is lower than the proposed step cost, the action is infeasible and `budget_exhausted` is set in diagnostics.

## Testing

The regression tests in `tests/test_homeostatic_filter.py` cover:

- Disabled filter returns the action unchanged.
- Feasible source-location movement remains unchanged.
- Infeasible movement is projected.
- Source danger risk rejects unsafe actions.
- Prey viability rejects extinction risk.
- Existing experiment paths still run when homeostatic filtering is disabled.
