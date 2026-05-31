# Development Guide

## Running Tests

Focused homeostatic regression suite:

```bash
pytest -vv tests/test_homeostatic_filter.py
```

Compact form:

```bash
pytest -q tests/test_homeostatic_filter.py
```

The current pytest configuration suppresses known third-party `PyparsingDeprecationWarning` noise emitted from `matplotlib`/`pyparsing`. Project warnings are not globally disabled.

## Code Style

The project uses standard Python modules with dataclasses for configuration and PyTorch modules for trainable components. Keep new code consistent with existing patterns:

- Put reusable environment-independent logic in `boedx/`.
- Put benchmark-specific CLI wiring in `boedx/experiments/`.
- Keep configuration in dataclasses when the values are consumed by trainers.
- Prefer explicit NumPy/Torch conversions at environment boundaries.
- Add focused tests for new safety filters, state features, or trainer diagnostics.

## Adding a New Environment

Create a subclass of `GenericBankBOEDEnv` and implement the required methods:

```python
class MyEnv(GenericBankBOEDEnv):
    name = "my_env"
    action_dim = 1

    def get_horizon(self) -> int: ...
    def get_action_low(self) -> np.ndarray: ...
    def get_action_high(self) -> np.ndarray: ...
    def build_hypothesis_bank(self) -> torch.Tensor: ...
    def build_prior_bank_logits(self) -> torch.Tensor: ...
    def sample_theta(self) -> np.ndarray: ...
    def sample_prior_thetas(self, n: int) -> torch.Tensor: ...
    def clip_action(self, action: np.ndarray) -> np.ndarray: ...
    def sample_observation(self, theta: np.ndarray, action: np.ndarray) -> float: ...
    def loglik_scalar(self, obs: float, theta: np.ndarray, action: np.ndarray) -> float: ...
    def trajectory_loglik_thetas(self, actions, obs, thetas) -> torch.Tensor: ...
    def bank_loglik_single(self, obs_t, action_t) -> torch.Tensor: ...
```

Then create a CLI module under `boedx/experiments/` that:

1. Defines an environment-specific config dataclass.
2. Parses CLI arguments.
3. Builds `GenericTrainConfig`, `BeliefConfig`, and optionally `HomeostaticConfig`.
4. Defines an `env_factory(device)` function.
5. Calls `run_experiment_suite` or `run_experiment_suite_nes`.

Add the script to `[project.scripts]` in `pyproject.toml` if it should be installed as a command.

## Adding a Homeostatic Constraint

Prefer adding mode-specific logic inside `boedx.homeostatic` while keeping `filter_action` as the public entry point.

A new constraint should provide:

- A config field in `HomeostaticConfig`.
- A feasibility check in `_is_feasible` or a mode-specific helper.
- Diagnostics that can be accumulated by `HomeostaticStats`.
- Unit tests covering feasible, infeasible, and fallback behavior.

Do not bypass environment `clip_action`; filtering should work alongside the environment's native bounds.

## Adding a New Variant

Trainer variants are selected by string names. When adding a variant:

- Update the variant construction path in the relevant trainer.
- Make the state/belief/actor settings explicit.
- Add display names and colors in `boedx.plotting` if the variant should appear in figures.
- Run at least one short smoke experiment and the relevant tests.

## Plotting Notes

`boedx.plotting.generate_scientific_plots` loads saved JSON outputs. It is intended for completed experiment directories, not as a help-style CLI. Use:

```bash
boedx-plot ./outputs/my_experiment
```

The plotting module writes both PNG and PDF figures for publication workflows.

## Warning Policy

Warnings from project code should stay visible during development. The only configured warning filters are scoped to third-party pyparsing deprecations raised through installed plotting dependencies. If new warnings appear, prefer fixing the source or adding a narrow filter over globally disabling warning classes.
