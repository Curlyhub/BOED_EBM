import json
import sys

import numpy as np
import pytest
import torch

import boedx.experiments.prey_population as prey_population_mod
import boedx.trainer as trainer_mod
from boedx.buffer import ReplayBuffer
from boedx.env import BeliefConfig, GenericTrainConfig
from boedx.experiments.prey_population import PreyConfig, PreyPopulationEnv
from boedx.experiments.source_location import SourceLocationConfig, SourceLocalization2DEnv
from boedx.homeostatic import (
    HomeostaticConfig,
    HomeostaticContext,
    apply_discrete_action_mask,
    build_homeostatic_admissibility,
    enforce_homeostatic_admissibility,
    get_homeostatic_feature_spec,
    filter_action,
)
from boedx.models import DiscreteMoECategoricalActor
from boedx.state import compute_state_from_batch, make_raw_state, raw_state_to_policy_state
from boedx.trainer import build_modules, run_experiment_suite


def test_noop_filter_returns_action_unchanged_when_disabled():
    action = np.array([3.0, -2.0], dtype=np.float32)
    cfg = HomeostaticConfig(enabled=False, mode="source_location", max_step_norm=0.1)
    ctx = HomeostaticContext(env=None, step_index=0)

    filtered, diag = filter_action(action, cfg, ctx)

    np.testing.assert_allclose(filtered, action)
    assert diag["was_filtered"] is False
    assert diag["enabled"] is False


def test_source_movement_constraint_keeps_feasible_action_unchanged():
    env = SourceLocalization2DEnv(SourceLocationConfig(horizon=2), torch.device("cpu"))
    raw = np.array([0.4, 0.3], dtype=np.float32)
    cfg = HomeostaticConfig(enabled=True, mode="source_location", max_step_norm=1.0)
    ctx = HomeostaticContext(env=env, step_index=1, prev_action=np.array([0.0, 0.0], dtype=np.float32))

    filtered, diag = filter_action(raw, cfg, ctx)

    np.testing.assert_allclose(filtered, raw)
    assert diag["raw_action_feasible"] is True
    assert diag["was_filtered"] is False


def test_source_movement_constraint_projects_infeasible_action():
    env = SourceLocalization2DEnv(SourceLocationConfig(horizon=2), torch.device("cpu"))
    raw = np.array([3.0, 4.0], dtype=np.float32)
    cfg = HomeostaticConfig(enabled=True, mode="source_location", max_step_norm=1.0)
    prev = np.array([0.0, 0.0], dtype=np.float32)
    ctx = HomeostaticContext(env=env, step_index=1, prev_action=prev)

    filtered, diag = filter_action(raw, cfg, ctx)

    assert diag["was_filtered"] is True
    assert np.linalg.norm(filtered - prev) <= 1.0 + 1e-5


def test_source_danger_constraint_rejects_action_close_to_posterior_particles():
    env = SourceLocalization2DEnv(SourceLocationConfig(horizon=2), torch.device("cpu"))
    raw = np.array([0.0, 0.0], dtype=np.float32)
    cfg = HomeostaticConfig(
        enabled=True,
        mode="source_location",
        danger_radius=0.5,
        danger_prob_max=0.1,
        max_projection_candidates=64,
    )
    particles = np.array([[0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 3.0, 3.0]], dtype=np.float32)
    weights = np.array([0.9, 0.1], dtype=np.float32)
    ctx = HomeostaticContext(
        env=env,
        step_index=0,
        posterior_particles=particles,
        posterior_weights=weights,
    )

    filtered, diag = filter_action(raw, cfg, ctx)

    assert diag["was_filtered"] is True
    assert diag["raw_action_feasible"] is False
    assert np.linalg.norm(filtered - raw) > 0.25


def test_prey_viability_constraint_rejects_extinction_risk():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    raw = np.array([1.0], dtype=np.float32)
    cfg = HomeostaticConfig(
        enabled=True,
        mode="prey_population",
        prey_min=1.5,
        extinction_prob_max=0.0,
        max_projection_candidates=32,
    )
    particles = np.array([[3.0, -1.4], [2.5, -1.4]], dtype=np.float32)
    weights = np.array([0.5, 0.5], dtype=np.float32)
    ctx = HomeostaticContext(
        env=env,
        step_index=0,
        posterior_particles=particles,
        posterior_weights=weights,
    )

    filtered, diag = filter_action(raw, cfg, ctx)

    assert diag["raw_action_feasible"] is False
    assert diag["was_filtered"] is True
    assert diag["violation_computable"] is True


def test_prey_discrete_admissibility_mask_covers_full_action_range():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()
    raw = {
        "t_idx": 0,
        "posterior": np.full(env.hypothesis_bank.shape[0], 1.0 / env.hypothesis_bank.shape[0], dtype=np.float32),
    }
    cfg = HomeostaticConfig(enabled=True, mode="prey_population")

    adm = build_homeostatic_admissibility(env, raw, cfg, prev_action=env.last_action)

    assert adm.action_mask is not None
    assert adm.action_mask.dtype == np.bool_
    assert adm.action_mask.shape == (300,)
    assert adm.candidate_actions.shape == (300, 1)


def test_posterior_risk_source_ignores_ebm_weights_for_source_location():
    env = SourceLocalization2DEnv(SourceLocationConfig(horizon=2), torch.device("cpu"))
    particles = np.array([[0.0, 0.0], [3.0, 3.0]], dtype=np.float32)
    raw = {
        "t_idx": 0,
        "posterior_particles": particles,
        "posterior": np.array([1.0, 0.0], dtype=np.float32),
        "ebm_weights": np.array([0.0, 1.0], dtype=np.float32),
    }
    cfg = HomeostaticConfig(
        enabled=True, mode="source_location", risk_source="posterior",
        danger_radius=0.5, danger_prob_max=0.1, max_projection_candidates=16,
    )

    adm_a = build_homeostatic_admissibility(env, raw, cfg)
    raw_changed_ebm = dict(raw)
    raw_changed_ebm["ebm_weights"] = np.array([0.5, 0.5], dtype=np.float32)
    adm_b = build_homeostatic_admissibility(env, raw_changed_ebm, cfg)

    np.testing.assert_array_equal(adm_a.diagnostics["candidate_violation"], adm_b.diagnostics["candidate_violation"])


def test_no_admissible_prey_uses_least_violating_fallback():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()
    raw = {
        "t_idx": 0,
        "posterior": np.full(env.hypothesis_bank.shape[0], 1.0 / env.hypothesis_bank.shape[0], dtype=np.float32),
    }
    cfg = HomeostaticConfig(
        enabled=True, mode="prey_population", prey_min=1e9, extinction_prob_max=0.0,
    )
    adm = build_homeostatic_admissibility(env, raw, cfg)

    action, diag = enforce_homeostatic_admissibility(np.array([300.0], dtype=np.float32), adm, cfg)

    assert adm.action_mask is not None and not adm.action_mask.any()
    assert diag["least_violating_fallback_used"] is True
    fallback_idx = int(adm.diagnostics["fallback_index"])
    np.testing.assert_allclose(action, adm.candidate_actions[fallback_idx])


def test_smoke_old_experiment_path_runs_with_homeostatic_disabled(tmp_path):
    cfg = PreyConfig(horizon=1, bank_size=8, n_contrastive=2)
    train_cfg = GenericTrainConfig(
        episodes=1,
        eval_episodes=1,
        warmup_episodes=1,
        batch_size=64,
        device="cpu",
        seeds="0",
        hidden_rl=8,
        hidden_ebm=8,
        print_every=10,
    )

    def factory(device):
        return PreyPopulationEnv(cfg=cfg, device=device)

    summary = run_experiment_suite(
        experiment_name="smoke_prey",
        env_factory=factory,
        output_dir=str(tmp_path),
        train_cfg=train_cfg,
        seeds=[0],
        variants=["blau_approx"],
        spce_L=2,
        snmc_L=0,
        belief_cfg=BeliefConfig(mode="exact"),
        homeostatic_cfg=HomeostaticConfig(enabled=False),
    )

    assert summary["experiment_name"] == "smoke_prey"
    assert (tmp_path / "blau_approx" / "seed_0" / "result.json").exists()


def _uniform_prey_raw(env: PreyPopulationEnv) -> dict:
    raw = make_raw_state(env)
    raw["posterior"] = np.full(env.hypothesis_bank.shape[0], 1.0 / env.hypothesis_bank.shape[0], dtype=np.float32)
    return raw

def _posterior_raw_with_particles(env: PreyPopulationEnv, weights: np.ndarray) -> dict:
    return {
        "t_idx": 0,
        "posterior_particles": np.zeros((len(weights), env.theta_dim), dtype=np.float32),
        "posterior": np.asarray(weights, dtype=np.float32),
    }


def test_prey_relative_survival_constraint():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()

    def fake_predict_population_next_batch(thetas, actions):
        acts = np.asarray(actions, dtype=np.float32).reshape(-1)
        prey = np.stack([
            0.4 * acts,
            np.where(acts <= 10.0, 0.1 * acts, 0.35 * acts),
        ], axis=1)
        return prey.astype(np.float32), None

    env.predict_population_next_batch = fake_predict_population_next_batch
    raw = _posterior_raw_with_particles(env, np.array([0.5, 0.5], dtype=np.float32))
    cfg = HomeostaticConfig(
        enabled=True,
        mode="prey_population",
        survival_fraction_min=0.3,
        survival_fraction_prob_max=0.4,
    )

    adm = build_homeostatic_admissibility(env, raw, cfg)

    idx_10 = 9
    idx_20 = 19
    np.testing.assert_allclose(adm.diagnostics["candidate_survival_fraction_risk"][idx_10], 0.5)
    np.testing.assert_allclose(adm.diagnostics["candidate_survival_fraction_risk"][idx_20], 0.0)
    assert bool(adm.action_mask[idx_10]) is False
    assert bool(adm.action_mask[idx_20]) is True


def test_prey_relative_consumption_constraint():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()

    def fake_predict_population_next_batch(thetas, actions):
        acts = np.asarray(actions, dtype=np.float32).reshape(-1)
        prey = np.stack([
            0.4 * acts,
            np.where(acts <= 10.0, 0.1 * acts, 0.35 * acts),
        ], axis=1)
        return prey.astype(np.float32), None

    env.predict_population_next_batch = fake_predict_population_next_batch
    raw = _posterior_raw_with_particles(env, np.array([0.5, 0.5], dtype=np.float32))
    cfg = HomeostaticConfig(
        enabled=True,
        mode="prey_population",
        consumption_fraction_max=0.7,
        consumption_fraction_prob_max=0.4,
    )

    adm = build_homeostatic_admissibility(env, raw, cfg)

    idx_10 = 9
    idx_20 = 19
    np.testing.assert_allclose(adm.diagnostics["candidate_consumption_fraction_risk"][idx_10], 0.5)
    np.testing.assert_allclose(adm.diagnostics["candidate_consumption_fraction_risk"][idx_20], 0.0)
    assert bool(adm.action_mask[idx_10]) is False
    assert bool(adm.action_mask[idx_20]) is True


def test_absolute_and_relative_constraints_combine_by_and():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()

    def fake_predict_population_next_batch(thetas, actions):
        acts = np.asarray(actions, dtype=np.float32).reshape(-1)
        prey = np.stack([
            np.where(acts <= 10.0, 4.0, 8.0),
            np.where(acts <= 10.0, 4.0, 2.0),
        ], axis=1)
        return prey.astype(np.float32), None

    env.predict_population_next_batch = fake_predict_population_next_batch
    raw = _posterior_raw_with_particles(env, np.array([0.5, 0.5], dtype=np.float32))
    cfg = HomeostaticConfig(
        enabled=True,
        mode="prey_population",
        prey_min=5.0,
        extinction_prob_max=0.25,
        survival_fraction_min=0.3,
        survival_fraction_prob_max=0.25,
    )

    adm = build_homeostatic_admissibility(env, raw, cfg)

    idx_10 = 9
    idx_20 = 19
    assert bool(adm.action_mask[idx_10]) is False
    assert bool(adm.action_mask[idx_20]) is False
    np.testing.assert_allclose(adm.diagnostics["candidate_extinction_prob"][idx_10], 1.0)
    np.testing.assert_allclose(adm.diagnostics["candidate_survival_fraction_risk"][idx_20], 0.5)


def test_relative_constraint_avoids_artificial_small_d_rejection():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()

    def fake_predict_population_next_batch(thetas, actions):
        acts = np.asarray(actions, dtype=np.float32).reshape(-1)
        prey = np.full((acts.shape[0], 2), 4.0, dtype=np.float32)
        return prey, None

    env.predict_population_next_batch = fake_predict_population_next_batch
    raw = _posterior_raw_with_particles(env, np.array([0.5, 0.5], dtype=np.float32))
    abs_cfg = HomeostaticConfig(enabled=True, mode="prey_population", prey_min=5.0, extinction_prob_max=0.0)
    rel_cfg = HomeostaticConfig(enabled=True, mode="prey_population", survival_fraction_min=0.1, survival_fraction_prob_max=0.0)

    abs_adm = build_homeostatic_admissibility(env, raw, abs_cfg)
    rel_adm = build_homeostatic_admissibility(env, raw, rel_cfg)

    idx_10 = 9
    assert bool(abs_adm.action_mask[idx_10]) is False
    assert bool(rel_adm.action_mask[idx_10]) is True
    np.testing.assert_allclose(rel_adm.diagnostics["candidate_mean_survival_fraction"][idx_10], 0.4)


def test_homeostatic_features_prey_shape():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()
    raw = _uniform_prey_raw(env)
    cfg = HomeostaticConfig(
        enabled=True,
        mode="prey_population",
        include_features=True,
        feature_mode="basic",
        prey_min=5.0,
        prey_max=500.0,
        extinction_prob_max=0.2,
        explosion_prob_max=0.05,
    )

    adm = build_homeostatic_admissibility(env, raw, cfg, prev_action=env.last_action)
    dim, names = get_homeostatic_feature_spec(env, cfg)

    assert adm.features is not None
    assert adm.feature_names == names
    assert adm.features.shape == (dim,)
    assert len(adm.feature_names) == dim


def test_homeostatic_features_disabled():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    train_cfg = GenericTrainConfig(episodes=1, eval_episodes=1, hidden_rl=8, hidden_ebm=8, device="cpu")
    cfg_off = HomeostaticConfig(enabled=False, mode="prey_population")
    cfg_on = HomeostaticConfig(enabled=True, mode="prey_population", include_features=True, feature_mode="basic")

    _, actor_off, *_ = build_modules("blau_approx", env, train_cfg, torch.device("cpu"), belief_cfg=BeliefConfig(mode="exact"), homeostatic_cfg=cfg_off)
    _, actor_on, *_ = build_modules("blau_approx", env, train_cfg, torch.device("cpu"), belief_cfg=BeliefConfig(mode="exact"), homeostatic_cfg=cfg_on)

    assert getattr(actor_off, "homeo_feature_dim", 0) == 0
    assert getattr(actor_on, "homeo_feature_dim", 0) == 10


def test_homeostatic_features_variant_invariance():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()
    raw = _uniform_prey_raw(env)
    cfg = HomeostaticConfig(
        enabled=True,
        mode="prey_population",
        include_features=True,
        feature_mode="basic",
        prey_min=5.0,
        prey_max=500.0,
        extinction_prob_max=0.2,
        explosion_prob_max=0.05,
    )
    adm = build_homeostatic_admissibility(env, raw, cfg, prev_action=env.last_action)
    raw["homeo_features"] = adm.features.astype(np.float32)
    train_cfg = GenericTrainConfig(episodes=1, eval_episodes=1, hidden_rl=8, hidden_ebm=8, device="cpu")

    states = {}
    for variant, belief_mode in [
        ("blau_approx", "exact"),
        ("ours_ebm_control", "learned_only"),
        ("ours_ebm_cross", "learned_only"),
    ]:
        filter_backbone, actor, q1, q2, q1_tgt, q2_tgt, actor_optim, critic_optim, energy_net, apsi_head, ebm_optim = build_modules(
            variant, env, train_cfg, torch.device("cpu"), belief_cfg=BeliefConfig(mode=belief_mode, feature_mode="moments"), homeostatic_cfg=cfg
        )
        state_t, _, _, _ = raw_state_to_policy_state(
            variant, raw, filter_backbone, env, torch.device("cpu"), energy_net, apsi_head,
            belief_cfg=BeliefConfig(mode=belief_mode, feature_mode="moments"),
        )
        states[variant] = state_t.squeeze(0).detach().cpu().numpy()
    np.testing.assert_allclose(states["blau_approx"][-10:], adm.features)
    np.testing.assert_allclose(states["ours_ebm_control"][-10:], adm.features)
    np.testing.assert_allclose(states["ours_ebm_cross"][-10:], adm.features)


def test_homeostatic_features_no_future_leakage():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()
    raw = _uniform_prey_raw(env)
    cfg = HomeostaticConfig(
        enabled=True,
        mode="prey_population",
        include_features=True,
        feature_mode="basic",
        prey_min=5.0,
        prey_max=500.0,
        extinction_prob_max=0.2,
        explosion_prob_max=0.05,
    )
    adm = build_homeostatic_admissibility(env, raw, cfg, prev_action=env.last_action)
    action_a, _ = enforce_homeostatic_admissibility(np.array([1.0], dtype=np.float32), adm, cfg)
    action_b, _ = enforce_homeostatic_admissibility(np.array([300.0], dtype=np.float32), adm, cfg)

    assert adm.feature_names is not None
    assert "raw_action_extinction_prob" not in adm.feature_names
    assert "selected_action_extinction_prob" not in adm.feature_names
    np.testing.assert_allclose(adm.features, adm.features)
    assert action_a.shape == action_b.shape

def test_no_future_leakage_relative_features():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()
    raw = _uniform_prey_raw(env)
    cfg = HomeostaticConfig(
        enabled=True,
        mode="prey_population",
        include_features=True,
        feature_mode="relative",
        survival_fraction_min=0.1,
        survival_fraction_prob_max=0.2,
    )
    adm = build_homeostatic_admissibility(env, raw, cfg, prev_action=env.last_action)
    action_a, _ = enforce_homeostatic_admissibility(np.array([1.0], dtype=np.float32), adm, cfg)
    action_b, _ = enforce_homeostatic_admissibility(np.array([300.0], dtype=np.float32), adm, cfg)

    assert adm.feature_names is not None
    assert "raw_action_survival_fraction_risk" not in adm.feature_names
    assert "selected_survival_fraction_risk" not in adm.feature_names
    assert "raw_consumption_fraction_risk" not in adm.feature_names
    assert "selected_consumption_fraction_risk" not in adm.feature_names
    np.testing.assert_allclose(adm.features, adm.features)
    assert action_a.shape == action_b.shape


def test_relative_feature_shape():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()
    raw = _uniform_prey_raw(env)
    cfg = HomeostaticConfig(
        enabled=True,
        mode="prey_population",
        include_features=True,
        feature_mode="relative",
        survival_fraction_min=0.1,
        survival_fraction_prob_max=0.2,
    )
    adm = build_homeostatic_admissibility(env, raw, cfg, prev_action=env.last_action)
    dim, names = get_homeostatic_feature_spec(env, cfg)

    assert dim == 10
    assert names == [
        "num_admissible_norm",
        "min_admissible_action_norm",
        "max_admissible_action_norm",
        "mean_admissible_action_norm",
        "std_admissible_action_norm",
        "mean_survival_risk_adm",
        "min_survival_risk_all",
        "mean_survival_risk_all",
        "no_admissible_action_flag",
        "fallback_required_flag",
    ]
    assert adm.features is not None
    assert adm.features.shape == (10,)
    assert adm.feature_names == names


def test_fallback_uses_relative_violation():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()

    def fake_predict_population_next_batch(thetas, actions):
        acts = np.asarray(actions, dtype=np.float32).reshape(-1)
        frac_1 = np.full_like(acts, 0.05)
        frac_2 = np.clip(0.08 + 0.0002 * acts, 0.0, 1.0)
        frac_3 = np.clip(0.09 + 0.0004 * acts, 0.0, 1.0)
        prey = np.stack([frac_1 * acts, frac_2 * acts, frac_3 * acts], axis=1)
        return prey.astype(np.float32), None

    env.predict_population_next_batch = fake_predict_population_next_batch
    raw = _posterior_raw_with_particles(env, np.array([0.2, 0.3, 0.5], dtype=np.float32))
    cfg = HomeostaticConfig(
        enabled=True,
        mode="prey_population",
        survival_fraction_min=0.1,
        survival_fraction_prob_max=0.1,
    )

    adm = build_homeostatic_admissibility(env, raw, cfg)
    action, diag = enforce_homeostatic_admissibility(np.array([1.0], dtype=np.float32), adm, cfg)

    assert adm.action_mask is not None and not adm.action_mask.any()
    assert diag["least_violating_fallback_used"] is True
    fallback_idx = int(adm.diagnostics["fallback_index"])
    np.testing.assert_allclose(action, adm.candidate_actions[fallback_idx])
    expected_idx = int(np.argmin(adm.diagnostics["candidate_violation"]))
    assert fallback_idx == expected_idx
    np.testing.assert_allclose(adm.diagnostics["candidate_survival_fraction_risk"][fallback_idx], 0.2)


def test_prey_mask_monotonicity():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()
    raw = _uniform_prey_raw(env)
    cfg_lo = HomeostaticConfig(enabled=True, mode="prey_population", prey_min=5.0, prey_max=500.0, extinction_prob_max=0.2, explosion_prob_max=0.05)
    cfg_hi = HomeostaticConfig(enabled=True, mode="prey_population", prey_min=5.0, prey_max=500.0, extinction_prob_max=0.3, explosion_prob_max=0.05)

    adm_lo = build_homeostatic_admissibility(env, raw, cfg_lo, prev_action=env.last_action)
    adm_hi = build_homeostatic_admissibility(env, raw, cfg_hi, prev_action=env.last_action)

    assert adm_lo.action_mask is not None and adm_hi.action_mask is not None
    assert int(adm_hi.action_mask.sum()) >= int(adm_lo.action_mask.sum())


def test_no_admissible_rate_semantics():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    env.reset()
    raw = _uniform_prey_raw(env)
    cfg_none = HomeostaticConfig(enabled=True, mode="prey_population", prey_min=1e9, extinction_prob_max=0.0)
    cfg_some = HomeostaticConfig(enabled=True, mode="prey_population", prey_min=0.0, prey_max=1e9, extinction_prob_max=1.0, explosion_prob_max=1.0)

    adm_none = build_homeostatic_admissibility(env, raw, cfg_none, prev_action=env.last_action)
    adm_some = build_homeostatic_admissibility(env, raw, cfg_some, prev_action=env.last_action)

    assert bool(adm_none.diagnostics["homeo_no_admissible_action"]) is True
    assert bool(adm_some.diagnostics["homeo_no_admissible_action"]) is False


def test_sac_batch_homeo_feature_shapes():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    train_cfg = GenericTrainConfig(episodes=1, eval_episodes=1, hidden_rl=8, hidden_ebm=8, device="cpu")
    cfg = HomeostaticConfig(enabled=True, mode="prey_population", include_features=True, feature_mode="basic")
    filter_backbone, actor, q1, q2, q1_tgt, q2_tgt, actor_optim, critic_optim, energy_net, apsi_head, ebm_optim = build_modules(
        "ours_ebm_control", env, train_cfg, torch.device("cpu"), belief_cfg=BeliefConfig(mode="learned_only", feature_mode="moments"), homeostatic_cfg=cfg
    )
    replay = ReplayBuffer(capacity=4)
    env.reset()
    raw = make_raw_state(env)
    raw["homeo_features"] = np.zeros(10, dtype=np.float32)
    next_raw = make_raw_state(env)
    next_raw["homeo_features"] = np.ones(10, dtype=np.float32)
    item = {
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
        "reward": np.float32(0.0),
        "done": np.float32(0.0),
        "action_taken": np.array([1.0], dtype=np.float32),
        "homeo_features": raw["homeo_features"],
        "next_homeo_features": next_raw["homeo_features"],
        "homeo_action_mask": np.ones(300, dtype=np.float32),
        "next_homeo_action_mask": np.ones(300, dtype=np.float32),
        "homeo_num_admissible_actions": np.float32(300),
        "next_homeo_num_admissible_actions": np.float32(300),
        "homeo_no_admissible_action": np.float32(0.0),
        "next_homeo_no_admissible_action": np.float32(0.0),
    }
    replay.add(item)
    replay.add(item)
    batch = replay.sample(2, device=torch.device("cpu"))

    assert batch["homeo_features"].shape == (2, 10)
    assert batch["next_homeo_features"].shape == (2, 10)
    state_t, _, _, _ = compute_state_from_batch(
        "ours_ebm_control", filter_backbone, batch, env, energy_net, apsi_head,
        belief_cfg=BeliefConfig(mode="learned_only", feature_mode="moments"), use_next=False,
    )
    next_state_t, _, _, _ = compute_state_from_batch(
        "ours_ebm_control", filter_backbone, batch, env, energy_net, apsi_head,
        belief_cfg=BeliefConfig(mode="learned_only", feature_mode="moments"), use_next=True,
    )
    assert state_t.shape[0] == 2
    assert next_state_t.shape[0] == 2


def test_tiny_prey_homeo_feature_smoke(tmp_path):
    cfg = PreyConfig(horizon=2, bank_size=8, n_contrastive=2)
    train_cfg = GenericTrainConfig(
        episodes=2,
        eval_episodes=1,
        warmup_episodes=1,
        batch_size=2,
        replay_size=8,
        device="cpu",
        seeds="0",
        hidden_rl=8,
        hidden_ebm=8,
        print_every=10,
    )

    def factory(device):
        return PreyPopulationEnv(cfg=cfg, device=device)

    summary = run_experiment_suite(
        experiment_name="smoke_prey_homeo_features",
        env_factory=factory,
        output_dir=str(tmp_path),
        train_cfg=train_cfg,
        seeds=[0],
        variants=["blau_approx"],
        spce_L=2,
        snmc_L=0,
        belief_cfg=BeliefConfig(mode="exact"),
        homeostatic_cfg=HomeostaticConfig(
            enabled=True,
            mode="prey_population",
            include_features=True,
            feature_mode="basic",
            prey_min=5.0,
            prey_max=500.0,
            extinction_prob_max=0.2,
            explosion_prob_max=0.05,
        ),
    )

    assert summary["blau_approx_homeo_features_enabled"]["mean"] == 1.0



def test_categorical_moe_output_shape_and_normalization():
    actor = DiscreteMoECategoricalActor(state_dim=7, num_actions=11, hidden=16, n_experts=3)
    state = torch.randn(5, 7)

    log_probs = actor(state)
    probs, stable_log_probs = actor.probs_and_log_probs(state)

    assert log_probs.shape == (5, 11)
    assert probs.shape == (5, 11)
    assert stable_log_probs.shape == (5, 11)
    assert torch.allclose(log_probs, stable_log_probs, atol=1e-6)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-6)


def test_categorical_moe_masked_policy_uses_fallback_when_all_actions_masked():
    probs = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    log_probs = torch.log(probs)
    mask = torch.zeros_like(probs, dtype=torch.bool)

    masked_probs, masked_log_probs = apply_discrete_action_mask(probs, log_probs, mask)

    assert torch.allclose(masked_probs, probs, atol=1e-7)
    assert torch.allclose(masked_log_probs, log_probs, atol=1e-7)


def test_categorical_moe_builds_and_masks_for_prey_homeostasis():
    env = PreyPopulationEnv(PreyConfig(horizon=2, bank_size=8), torch.device("cpu"))
    train_cfg = GenericTrainConfig(
        episodes=1,
        eval_episodes=1,
        hidden_rl=8,
        hidden_ebm=8,
        actor_family="categorical_moe",
        actor_mixture_components=3,
        device="cpu",
    )
    cfg = HomeostaticConfig(enabled=True, mode="prey_population", include_features=True, feature_mode="basic")

    _, actor, *_ = build_modules(
        "blau_approx",
        env,
        train_cfg,
        torch.device("cpu"),
        belief_cfg=BeliefConfig(mode="exact"),
        homeostatic_cfg=cfg,
    )
    assert isinstance(actor, DiscreteMoECategoricalActor)

    env.reset()
    raw = _uniform_prey_raw(env)
    adm = build_homeostatic_admissibility(env, raw, cfg, prev_action=env.last_action)
    raw["homeo_features"] = adm.features.astype(np.float32)
    state_t, _, _, _ = raw_state_to_policy_state(
        "blau_approx", raw, None, env, torch.device("cpu"), None, None,
        belief_cfg=BeliefConfig(mode="exact"),
    )
    probs, log_probs = actor.probs_and_log_probs(state_t)
    masked_probs, masked_log_probs = apply_discrete_action_mask(probs, log_probs, adm.action_mask)

    assert masked_probs.shape == (1, 300)
    assert masked_log_probs.shape == (1, 300)
    assert torch.allclose(masked_probs.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_tiny_prey_homeo_feature_smoke_categorical_moe(tmp_path):
    cfg = PreyConfig(horizon=2, bank_size=8, n_contrastive=2)
    train_cfg = GenericTrainConfig(
        episodes=2,
        eval_episodes=1,
        warmup_episodes=1,
        batch_size=2,
        replay_size=8,
        device="cpu",
        seeds="0",
        hidden_rl=8,
        hidden_ebm=8,
        actor_family="categorical_moe",
        actor_mixture_components=3,
        print_every=10,
    )

    def factory(device):
        return PreyPopulationEnv(cfg=cfg, device=device)

    summary = run_experiment_suite(
        experiment_name="smoke_prey_homeo_features_categorical_moe",
        env_factory=factory,
        output_dir=str(tmp_path),
        train_cfg=train_cfg,
        seeds=[0],
        variants=["blau_approx"],
        spce_L=2,
        snmc_L=0,
        belief_cfg=BeliefConfig(mode="exact"),
        homeostatic_cfg=HomeostaticConfig(
            enabled=True,
            mode="prey_population",
            include_features=True,
            feature_mode="basic",
            prey_min=5.0,
            prey_max=500.0,
            extinction_prob_max=0.2,
            explosion_prob_max=0.05,
        ),
    )

    assert summary["experiment_name"] == "smoke_prey_homeo_features_categorical_moe"



def test_prey_help_includes_selection_flags(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["boedx-prey-population", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        prey_population_mod.parse_args()
    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for flag in [
        "--selection-start-episode",
        "--selection-every",
        "--selection-eval-episodes",
        "--selection-return-weight",
        "--selection-bank-ig-weight",
        "--selection-spce-weight",
        "--selection-survival-risk-weight",
        "--selection-fallback-weight",
        "--selection-belief-kl-weight",
        "--selection-belief-mean-weight",
        "--selection-belief-map-weight",
    ]:
        assert flag in help_text


def test_tiny_prey_selection_disabled_preserves_last_checkpoint_behavior(tmp_path):
    cfg = PreyConfig(horizon=2, bank_size=8, n_contrastive=2)
    train_cfg = GenericTrainConfig(
        episodes=2,
        eval_episodes=1,
        warmup_episodes=1,
        batch_size=2,
        replay_size=8,
        device="cpu",
        seeds="0",
        hidden_rl=8,
        hidden_ebm=8,
        actor_family="categorical",
        print_every=10,
        selection_start_episode=0,
        selection_every=0,
        selection_eval_episodes=0,
    )

    def factory(device):
        return PreyPopulationEnv(cfg=cfg, device=device)

    summary = run_experiment_suite(
        experiment_name="tiny_prey_selection_disabled",
        env_factory=factory,
        output_dir=str(tmp_path),
        train_cfg=train_cfg,
        seeds=[0],
        variants=["blau_approx"],
        spce_L=2,
        snmc_L=0,
        belief_cfg=BeliefConfig(mode="exact"),
    )
    result = json.loads((tmp_path / "blau_approx" / "seed_0" / "result.json").read_text())

    assert result["selection_enabled"] is False
    assert result["selection_best_episode"] is None
    assert result["selection_best_score"] is None
    assert result["selection_best_preview_eval"] is None
    assert result["selection_num_candidates"] == 0
    assert result["eval"] == result["eval_last"] == result["eval_selected"]
    assert summary["blau_approx_avg_return"]["mean"] == pytest.approx(result["eval"]["avg_return"])


def test_tiny_prey_selection_can_choose_checkpoint_before_last(tmp_path, monkeypatch):
    cfg = PreyConfig(horizon=2, bank_size=8, n_contrastive=2)
    train_cfg = GenericTrainConfig(
        episodes=2,
        eval_episodes=1,
        warmup_episodes=1,
        batch_size=2,
        replay_size=8,
        device="cpu",
        seeds="0",
        hidden_rl=8,
        hidden_ebm=8,
        actor_family="categorical",
        print_every=10,
        selection_start_episode=0,
        selection_every=1,
        selection_eval_episodes=1,
        selection_return_weight=1.0,
    )

    def factory(device):
        return PreyPopulationEnv(cfg=cfg, device=device)

    def make_bundle(avg_return: float) -> dict:
        return {
            "eval": {
                "avg_return": avg_return,
                "std_return": 0.0,
                "avg_bank_ig": 0.0,
                "avg_filter_bank_ig": 0.0,
                "avg_spce_lower": 0.0,
            },
            "belief_snapshot": None,
            "paths": {
                "bank_ig_mean_path": [0.0, 0.0],
                "filter_bank_ig_mean_path": [0.0, 0.0],
                "spce_lower_mean_path": [0.0, 0.0],
            },
        }

    scripted_bundles = iter([
        make_bundle(5.0),
        make_bundle(1.0),
        make_bundle(1.0),
        make_bundle(5.0),
    ])

    def fake_evaluate_policy_bundle(*args, **kwargs):
        return next(scripted_bundles)

    monkeypatch.setattr(trainer_mod, "_evaluate_policy_bundle", fake_evaluate_policy_bundle)

    run_experiment_suite(
        experiment_name="tiny_prey_selection_prefers_earlier_checkpoint",
        env_factory=factory,
        output_dir=str(tmp_path),
        train_cfg=train_cfg,
        seeds=[0],
        variants=["blau_approx"],
        spce_L=2,
        snmc_L=0,
        belief_cfg=BeliefConfig(mode="exact"),
    )
    result = json.loads((tmp_path / "blau_approx" / "seed_0" / "result.json").read_text())

    assert result["selection_enabled"] is True
    assert result["selection_num_candidates"] == 2
    assert result["selection_best_episode"] == 1
    assert result["selection_best_score"] == pytest.approx(5.0)
    assert result["selection_best_preview_eval"]["avg_return"] == pytest.approx(5.0)
    assert result["eval_last"]["avg_return"] == pytest.approx(1.0)
    assert result["eval_selected"]["avg_return"] == pytest.approx(5.0)
    assert result["eval"]["avg_return"] == pytest.approx(5.0)


def test_tiny_prey_homeo_selection_smoke_categorical_moe(tmp_path):
    cfg = PreyConfig(horizon=2, bank_size=8, n_contrastive=2)
    train_cfg = GenericTrainConfig(
        episodes=2,
        eval_episodes=1,
        warmup_episodes=1,
        batch_size=2,
        replay_size=8,
        device="cpu",
        seeds="0",
        hidden_rl=8,
        hidden_ebm=8,
        actor_family="categorical_moe",
        actor_mixture_components=3,
        print_every=10,
        selection_start_episode=0,
        selection_every=1,
        selection_eval_episodes=1,
    )

    def factory(device):
        return PreyPopulationEnv(cfg=cfg, device=device)

    summary = run_experiment_suite(
        experiment_name="smoke_prey_homeo_selection_categorical_moe",
        env_factory=factory,
        output_dir=str(tmp_path),
        train_cfg=train_cfg,
        seeds=[0],
        variants=["blau_approx"],
        spce_L=2,
        snmc_L=0,
        belief_cfg=BeliefConfig(mode="exact"),
        homeostatic_cfg=HomeostaticConfig(
            enabled=True,
            mode="prey_population",
            include_features=True,
            feature_mode="basic",
            prey_min=5.0,
            prey_max=500.0,
            extinction_prob_max=0.2,
            explosion_prob_max=0.05,
        ),
    )
    result = json.loads((tmp_path / "blau_approx" / "seed_0" / "result.json").read_text())

    assert summary["experiment_name"] == "smoke_prey_homeo_selection_categorical_moe"
    assert result["selection_enabled"] is True
    assert result["selection_num_candidates"] == 2
    assert result["selection_best_preview_eval"] is not None
    assert "avg_return" in result["eval_selected"]
