"""Raw-history homeostatic admissibility for BOED policies.

Homeostasis is a causal admissibility layer.  The admissible action set is
constructed from the adapted raw history before any policy-state quotient,
posterior compression, EBM tilt, or actor proposal.  The actor may still use a
derived state to propose an action, but enforcement uses only this raw-history
admissibility object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class HomeostaticConfig:
    enabled: bool = False
    mode: str = "none"  # none | source_location | prey_population
    include_features: bool = True
    feature_mode: str = "basic"  # basic | none
    projection_mode: str = "nearest"
    max_projection_candidates: int = 256

    max_step_norm: float | None = None

    danger_radius: float | None = None
    danger_prob_max: float | None = None
    risk_source: str = "posterior"  # posterior | ebm_ablation
    use_ebm_posterior_for_risk: bool = False  # deprecated compatibility alias

    prey_min: float | None = None
    prey_max: float | None = None
    predator_max: float | None = None
    extinction_prob_max: float | None = None
    explosion_prob_max: float | None = None
    survival_fraction_min: float | None = None
    survival_fraction_prob_max: float | None = None
    consumption_fraction_max: float | None = None
    consumption_fraction_prob_max: float | None = None

    initial_budget: float | None = None
    obs_cost: float = 0.0
    movement_cost: float = 0.0


@dataclass
class HomeostaticContext:
    """Deprecated compatibility context for ``filter_action`` callers."""

    env: Any
    step_index: int
    prev_action: np.ndarray | None = None
    budget: float | None = None
    posterior_particles: np.ndarray | None = None
    posterior_weights: np.ndarray | None = None
    raw_state: Any | None = None


@dataclass
class HomeostaticAdmissibility:
    enabled: bool
    mode: str
    step_index: int
    prev_action: np.ndarray | None
    budget: float | None
    action_mask: np.ndarray | None
    candidate_actions: np.ndarray | None
    posterior_particles: np.ndarray | None
    posterior_weights: np.ndarray | None
    features: np.ndarray | None
    feature_names: list[str] | None
    diagnostics: dict


class HomeostaticStats:
    """Accumulate additive homeostatic diagnostics over steps."""

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)
        self.count = 0
        self.filtered = 0
        self.raw_feasible = 0
        self.selected_feasible = 0
        self.feasible_candidates = 0
        self.no_admissible = 0
        self.least_violating = 0
        self.masked_policy = 0
        self.projection = 0
        self.distances = []
        self.risk_probs = []
        self.movement_costs = []
        self.budget_exhausted = 0
        self.violations = 0
        self.violation_known = 0
        self.raw_extinction = []
        self.selected_extinction = []
        self.raw_explosion = []
        self.selected_explosion = []
        self.raw_survival_risk = []
        self.selected_survival_risk = []
        self.raw_consumption_risk = []
        self.selected_consumption_risk = []
        self.selected_mean_survival = []
        self.selected_mean_consumption = []
        self.feature_names: list[str] | None = None
        self.feature_sums: np.ndarray | None = None
        self.ext_all_min = []
        self.ext_all_mean = []
        self.survival_risk_all_min = []
        self.survival_risk_all_mean = []
        self.consumption_risk_all_min = []
        self.consumption_risk_all_mean = []
        self.survival_mean_adm = []
        self.consumption_mean_adm = []
        self.fallback_ext_min = []
        self.fallback_ext_mean = []
        self.fallback_survival_risk_min = []
        self.fallback_survival_risk_mean = []
        self.fallback_consumption_risk_min = []
        self.fallback_consumption_risk_mean = []
        self.fallback_violation_min = []
        self.risk_source = "posterior"

    def add(self, diag: Dict[str, Any]) -> None:
        self.count += 1
        self.filtered += int(bool(diag.get("was_filtered", False)))
        self.raw_feasible += int(bool(diag.get("raw_action_feasible", False)))
        self.selected_feasible += int(bool(diag.get("selected_action_feasible", diag.get("raw_action_feasible", True))))
        self.feasible_candidates += int(diag.get("num_feasible_candidates", diag.get("homeo_num_admissible_actions", 0)))
        self.no_admissible += int(bool(diag.get("no_admissible_action", diag.get("homeo_no_admissible_action", False))))
        self.least_violating += int(bool(diag.get("least_violating_fallback_used", False)))
        self.masked_policy += int(bool(diag.get("masked_policy_used", False)))
        self.projection += int(bool(diag.get("projection_used", diag.get("homeo_projection_used", False))))
        self.distances.append(float(diag.get("raw_filtered_distance", diag.get("homeo_mean_raw_filtered_distance", 0.0))))
        risk = diag.get("risk_prob")
        if risk is None:
            risk = diag.get("selected_risk_prob")
        if risk is not None:
            self.risk_probs.append(float(risk))
        self.movement_costs.append(float(diag.get("movement_cost", 0.0)))
        self.budget_exhausted += int(bool(diag.get("budget_exhausted", False)))
        if diag.get("violation_computable", True):
            self.violation_known += 1
            self.violations += int(bool(diag.get("violation", not diag.get("selected_action_feasible", True))))
        for key, store in [
            ("raw_extinction_prob", self.raw_extinction),
            ("selected_extinction_prob", self.selected_extinction),
            ("raw_explosion_prob", self.raw_explosion),
            ("selected_explosion_prob", self.selected_explosion),
            ("raw_survival_fraction_risk", self.raw_survival_risk),
            ("selected_survival_fraction_risk", self.selected_survival_risk),
            ("raw_consumption_fraction_risk", self.raw_consumption_risk),
            ("selected_consumption_fraction_risk", self.selected_consumption_risk),
            ("selected_mean_survival_fraction", self.selected_mean_survival),
            ("selected_mean_consumption_fraction", self.selected_mean_consumption),
        ]:
            val = diag.get(key)
            if val is not None:
                store.append(float(val))
        feat = diag.get("homeo_features")
        feat_names = diag.get("homeo_feature_names")
        if feat is not None:
            feat_arr = np.asarray(feat, dtype=np.float64).reshape(-1)
            if self.feature_sums is None:
                self.feature_sums = np.zeros_like(feat_arr)
                self.feature_names = list(feat_names) if feat_names is not None else [
                    f"feature_{i}" for i in range(feat_arr.shape[0])
                ]
            self.feature_sums += feat_arr
        if diag.get("min_extinction_prob_all") is not None:
            self.ext_all_min.append(float(diag["min_extinction_prob_all"]))
        if diag.get("mean_extinction_prob_all") is not None:
            self.ext_all_mean.append(float(diag["mean_extinction_prob_all"]))
        if diag.get("min_survival_fraction_risk_all") is not None:
            self.survival_risk_all_min.append(float(diag["min_survival_fraction_risk_all"]))
        if diag.get("mean_survival_fraction_risk_all") is not None:
            self.survival_risk_all_mean.append(float(diag["mean_survival_fraction_risk_all"]))
        if diag.get("min_consumption_fraction_risk_all") is not None:
            self.consumption_risk_all_min.append(float(diag["min_consumption_fraction_risk_all"]))
        if diag.get("mean_consumption_fraction_risk_all") is not None:
            self.consumption_risk_all_mean.append(float(diag["mean_consumption_fraction_risk_all"]))
        if diag.get("mean_survival_fraction_admissible") is not None:
            self.survival_mean_adm.append(float(diag["mean_survival_fraction_admissible"]))
        if diag.get("mean_consumption_fraction_admissible") is not None:
            self.consumption_mean_adm.append(float(diag["mean_consumption_fraction_admissible"]))
        if bool(diag.get("least_violating_fallback_used", False)):
            if diag.get("min_extinction_prob_all") is not None:
                self.fallback_ext_min.append(float(diag["min_extinction_prob_all"]))
            if diag.get("mean_extinction_prob_all") is not None:
                self.fallback_ext_mean.append(float(diag["mean_extinction_prob_all"]))
            if diag.get("min_survival_fraction_risk_all") is not None:
                self.fallback_survival_risk_min.append(float(diag["min_survival_fraction_risk_all"]))
            if diag.get("mean_survival_fraction_risk_all") is not None:
                self.fallback_survival_risk_mean.append(float(diag["mean_survival_fraction_risk_all"]))
            if diag.get("min_consumption_fraction_risk_all") is not None:
                self.fallback_consumption_risk_min.append(float(diag["min_consumption_fraction_risk_all"]))
            if diag.get("mean_consumption_fraction_risk_all") is not None:
                self.fallback_consumption_risk_mean.append(float(diag["mean_consumption_fraction_risk_all"]))
            if diag.get("min_violation_score") is not None:
                self.fallback_violation_min.append(float(diag["min_violation_score"]))
        self.risk_source = str(diag.get("homeo_risk_source", self.risk_source))

    def summary(self) -> Dict[str, float | bool | str]:
        denom = max(self.count, 1)
        out: Dict[str, float | bool | str | list[str]] = {
            "homeo_enabled": self.enabled,
            "homeo_risk_source": self.risk_source,
            "homeo_admissibility_built_before_policy_state": True,
            "homeo_was_filtered": bool(self.filtered > 0),
            "homeo_raw_action_feasible": bool(self.raw_feasible == self.count) if self.count else True,
            "homeo_selected_action_feasible": bool(self.selected_feasible == self.count) if self.count else True,
            "homeo_num_feasible_candidates": float(self.feasible_candidates / denom),
            "homeo_num_admissible_actions": float(self.feasible_candidates / denom),
            "homeo_no_admissible_action_rate": float(self.no_admissible / denom),
            "homeo_least_violating_fallback_rate": float(self.least_violating / denom),
            "homeo_masked_policy_rate": float(self.masked_policy / denom),
            "homeo_projection_rate": float(self.projection / denom),
            "homeo_rejection_rate": 0.0,
            "homeo_mean_raw_filtered_distance": float(np.mean(self.distances)) if self.distances else 0.0,
            "homeo_mean_risk_prob": float(np.mean(self.risk_probs)) if self.risk_probs else 0.0,
            "homeo_mean_movement_cost": float(np.mean(self.movement_costs)) if self.movement_costs else 0.0,
            "homeo_budget_exhaustion_rate": float(self.budget_exhausted / denom),
            "homeo_violation_rate": float(self.violations / max(self.violation_known, 1)) if self.violation_known else 0.0,
            "homeo_mean_selected_extinction_prob": float(np.mean(self.selected_extinction)) if self.selected_extinction else 0.0,
            "homeo_mean_raw_extinction_prob": float(np.mean(self.raw_extinction)) if self.raw_extinction else 0.0,
            "homeo_mean_selected_explosion_prob": float(np.mean(self.selected_explosion)) if self.selected_explosion else 0.0,
            "homeo_mean_raw_explosion_prob": float(np.mean(self.raw_explosion)) if self.raw_explosion else 0.0,
            "homeo_min_extinction_prob_all": float(np.mean(self.ext_all_min)) if self.ext_all_min else 0.0,
            "homeo_mean_extinction_prob_all": float(np.mean(self.ext_all_mean)) if self.ext_all_mean else 0.0,
            "homeo_mean_selected_survival_fraction_risk": float(np.mean(self.selected_survival_risk)) if self.selected_survival_risk else 0.0,
            "homeo_mean_raw_survival_fraction_risk": float(np.mean(self.raw_survival_risk)) if self.raw_survival_risk else 0.0,
            "homeo_mean_selected_consumption_fraction_risk": float(np.mean(self.selected_consumption_risk)) if self.selected_consumption_risk else 0.0,
            "homeo_mean_raw_consumption_fraction_risk": float(np.mean(self.raw_consumption_risk)) if self.raw_consumption_risk else 0.0,
            "homeo_mean_selected_mean_survival_fraction": float(np.mean(self.selected_mean_survival)) if self.selected_mean_survival else 0.0,
            "homeo_mean_selected_mean_consumption_fraction": float(np.mean(self.selected_mean_consumption)) if self.selected_mean_consumption else 0.0,
            "homeo_min_survival_fraction_risk": float(np.mean(self.survival_risk_all_min)) if self.survival_risk_all_min else 0.0,
            "homeo_mean_survival_fraction_risk": float(np.mean(self.survival_risk_all_mean)) if self.survival_risk_all_mean else 0.0,
            "homeo_min_consumption_fraction_risk": float(np.mean(self.consumption_risk_all_min)) if self.consumption_risk_all_min else 0.0,
            "homeo_mean_consumption_fraction_risk": float(np.mean(self.consumption_risk_all_mean)) if self.consumption_risk_all_mean else 0.0,
            "homeo_mean_survival_fraction_admissible": float(np.mean(self.survival_mean_adm)) if self.survival_mean_adm else 0.0,
            "homeo_mean_consumption_fraction_admissible": float(np.mean(self.consumption_mean_adm)) if self.consumption_mean_adm else 0.0,
            "fallback_min_extinction_prob": float(np.mean(self.fallback_ext_min)) if self.fallback_ext_min else 0.0,
            "fallback_mean_extinction_prob": float(np.mean(self.fallback_ext_mean)) if self.fallback_ext_mean else 0.0,
            "fallback_min_survival_fraction_risk": float(np.mean(self.fallback_survival_risk_min)) if self.fallback_survival_risk_min else 0.0,
            "fallback_mean_survival_fraction_risk": float(np.mean(self.fallback_survival_risk_mean)) if self.fallback_survival_risk_mean else 0.0,
            "fallback_min_consumption_fraction_risk": float(np.mean(self.fallback_consumption_risk_min)) if self.fallback_consumption_risk_min else 0.0,
            "fallback_mean_consumption_fraction_risk": float(np.mean(self.fallback_consumption_risk_mean)) if self.fallback_consumption_risk_mean else 0.0,
            "fallback_min_violation_score": float(np.mean(self.fallback_violation_min)) if self.fallback_violation_min else 0.0,
        }
        if self.feature_sums is not None and self.feature_names is not None:
            feat_means = self.feature_sums / float(denom)
            out["homeo_features_enabled"] = True
            out["homeo_feature_dim"] = int(feat_means.shape[0])
            out["homeo_feature_names"] = list(self.feature_names)
            for name, val in zip(self.feature_names, feat_means.tolist()):
                out[f"homeo_mean_{name}"] = float(val)
        else:
            out["homeo_features_enabled"] = False
            out["homeo_feature_dim"] = 0
            out["homeo_feature_names"] = []
        return out


def config_to_dict(cfg: HomeostaticConfig) -> Dict[str, Any]:
    return asdict(cfg)


def update_budget_after_action(
    budget: float | None,
    cfg: HomeostaticConfig,
    prev_action: np.ndarray | None,
    action: np.ndarray,
) -> float | None:
    if budget is None:
        return None
    return float(budget - _step_cost(cfg, prev_action, action))


def build_homeostatic_admissibility(
    env: Any,
    raw_state: dict,
    cfg: HomeostaticConfig,
    budget: float | None = None,
    prev_action: np.ndarray | None = None,
) -> HomeostaticAdmissibility:
    if cfg.risk_source not in {"posterior", "ebm_ablation"}:
        raise ValueError(f"Unsupported homeostatic risk_source={cfg.risk_source!r}")

    particles, weights = _posterior_from_raw(env, raw_state, cfg)
    step_index = int(raw_state.get("t_idx", raw_state.get("length", 0)))
    prev = None if prev_action is None else np.asarray(prev_action, dtype=np.float32).copy()
    diag: Dict[str, Any] = {
        "enabled": bool(cfg.enabled),
        "mode": cfg.mode,
        "homeo_risk_source": cfg.risk_source,
        "homeo_admissibility_built_before_policy_state": True,
        "homeo_mask_available": False,
        "homeo_num_admissible_actions": 0,
        "homeo_no_admissible_action": False,
        "homeo_projection_used": False,
        "homeo_raw_action_feasible": True,
        "homeo_selected_action_feasible": True,
        "homeo_violation_rate": 0.0,
        "homeo_mean_raw_filtered_distance": 0.0,
        "homeo_mean_risk_prob": 0.0,
        "homeo_least_violating_fallback_used": False,
        "homeo_features": None,
        "homeo_feature_names": [],
    }
    if not cfg.enabled or cfg.mode == "none":
        return HomeostaticAdmissibility(False, cfg.mode, step_index, prev, budget, None, None, particles, weights, None, None, diag)

    ctx = HomeostaticContext(env, step_index, prev, budget, particles, weights, raw_state)
    if cfg.mode == "prey_population":
        candidates = _all_discrete_actions(env)
        mask, ext, exp, move_costs, violation, risk = _evaluate_candidates(candidates, cfg, ctx)
        features, feature_names = _build_homeo_features(cfg, env, ctx, candidates, mask, ext, exp, violation, risk)
        risk_prob = _aggregate_primary_risk(ext, exp, risk)
        diag.update({
            "homeo_mask_available": True,
            "homeo_num_admissible_actions": int(mask.sum()),
            "homeo_no_admissible_action": bool(mask.sum() == 0),
            "num_feasible_candidates": int(mask.sum()),
            "no_admissible_action": bool(mask.sum() == 0),
            "candidate_extinction_prob": ext,
            "candidate_explosion_prob": exp,
            "candidate_survival_fraction_risk": risk["survival_fraction_risk"],
            "candidate_consumption_fraction_risk": risk["consumption_fraction_risk"],
            "candidate_mean_survival_fraction": risk["mean_survival_fraction"],
            "candidate_mean_consumption_fraction": risk["mean_consumption_fraction"],
            "candidate_violation": violation,
            "fallback_index": int(np.argmin(violation)) if len(violation) else None,
            "fallback_action": float(candidates[int(np.argmin(violation)), 0]) if len(violation) else 0.0,
            "min_extinction_prob_all": float(np.min(ext)) if len(ext) else 0.0,
            "max_extinction_prob_all": float(np.max(ext)) if len(ext) else 0.0,
            "mean_extinction_prob_all": float(np.mean(ext)) if len(ext) else 0.0,
            "min_survival_fraction_risk_all": float(np.min(risk["survival_fraction_risk"])) if len(risk["survival_fraction_risk"]) else 0.0,
            "mean_survival_fraction_risk_all": float(np.mean(risk["survival_fraction_risk"])) if len(risk["survival_fraction_risk"]) else 0.0,
            "min_consumption_fraction_risk_all": float(np.min(risk["consumption_fraction_risk"])) if len(risk["consumption_fraction_risk"]) else 0.0,
            "mean_consumption_fraction_risk_all": float(np.mean(risk["consumption_fraction_risk"])) if len(risk["consumption_fraction_risk"]) else 0.0,
            "mean_survival_fraction_admissible": _masked_mean(risk["mean_survival_fraction"], mask),
            "mean_consumption_fraction_admissible": _masked_mean(risk["mean_consumption_fraction"], mask),
            "min_violation_score": float(np.min(violation)) if len(violation) else 0.0,
            "risk_prob": risk_prob,
            "homeo_mean_risk_prob": risk_prob,
            "homeo_features": None if features is None else features.tolist(),
            "homeo_feature_names": feature_names or [],
        })
        return HomeostaticAdmissibility(True, cfg.mode, step_index, prev, budget, mask, candidates, particles, weights, features, feature_names, diag)

    raw_shape = np.asarray(getattr(env, "action_low", np.zeros(1))).reshape(-1).shape
    seed = np.zeros(raw_shape, dtype=np.float32)
    candidates = np.asarray(_candidate_actions(seed, cfg, ctx), dtype=np.float32)
    mask, ext, exp, move_costs, violation, risk = _evaluate_candidates(candidates, cfg, ctx)
    features, feature_names = _build_homeo_features(cfg, env, ctx, candidates, mask, ext, exp, violation, risk)
    diag.update({
        "homeo_num_admissible_actions": int(mask.sum()),
        "homeo_no_admissible_action": bool(mask.sum() == 0),
        "num_feasible_candidates": int(mask.sum()),
        "no_admissible_action": bool(mask.sum() == 0),
        "candidate_violation": violation,
        "fallback_index": int(np.argmin(violation)) if len(violation) else None,
        "min_extinction_prob_all": float(np.min(ext)) if len(ext) else 0.0,
        "max_extinction_prob_all": float(np.max(ext)) if len(ext) else 0.0,
        "mean_extinction_prob_all": float(np.mean(ext)) if len(ext) else 0.0,
        "min_violation_score": float(np.min(violation)) if len(violation) else 0.0,
        "risk_prob": _aggregate_primary_risk(ext, exp, risk),
        "homeo_mean_risk_prob": _aggregate_primary_risk(ext, exp, risk),
        "homeo_features": None if features is None else features.tolist(),
        "homeo_feature_names": feature_names or [],
    })
    return HomeostaticAdmissibility(True, cfg.mode, step_index, prev, budget, None, candidates, particles, weights, features, feature_names, diag)


def enforce_homeostatic_admissibility(
    raw_action: np.ndarray,
    adm: HomeostaticAdmissibility,
    cfg: HomeostaticConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    diag = dict(adm.diagnostics)
    diag.update({
        "was_filtered": False,
        "projection_used": False,
        "least_violating_fallback_used": False,
        "raw_filtered_distance": 0.0,
        "violation_computable": True,
    })
    if not adm.enabled or cfg.mode == "none":
        diag.update({"raw_action_feasible": True, "selected_action_feasible": True, "violation": False})
        return raw.astype(np.float32), diag
    if cfg.projection_mode != "nearest":
        raise ValueError(f"Unsupported homeostatic projection_mode={cfg.projection_mode!r}")
    if adm.candidate_actions is None or len(adm.candidate_actions) == 0:
        diag.update({"raw_action_feasible": True, "selected_action_feasible": True, "violation": False})
        return raw.astype(np.float32), diag

    candidates = np.asarray(adm.candidate_actions, dtype=np.float32)
    ctx = HomeostaticContext(None, adm.step_index, adm.prev_action, adm.budget, adm.posterior_particles, adm.posterior_weights, None)
    env = diag.get("env")
    raw_eval = raw
    if cfg.mode == "prey_population" and adm.action_mask is not None:
        values = candidates.reshape(candidates.shape[0], -1)[:, 0]
        idx = int(np.argmin(np.abs(values - float(raw[0]))))
        raw_feasible = bool(adm.action_mask[idx])
        if raw_feasible:
            selected_idx = idx
        elif adm.action_mask.any():
            feasible_idx = np.where(adm.action_mask)[0]
            selected_idx = int(feasible_idx[np.argmin(np.abs(values[feasible_idx] - float(raw[0])) )])
        else:
            selected_idx = int(diag.get("fallback_index") or 0)
            diag["least_violating_fallback_used"] = True
        selected = candidates[selected_idx].astype(np.float32)
        diag.update(_selected_candidate_diag(
            raw, selected, raw_feasible, bool(adm.action_mask[selected_idx]), diag, selected_idx, raw_idx=idx
        ))
        return selected, diag

    mask, ext, exp, move_costs, violation, risk = _evaluate_candidates(candidates, cfg, ctx)
    # For continuous actions, test a candidate set that includes the raw action.
    raw_ctx = ctx
    raw_ok, raw_details = _candidate_feasibility(raw_eval, cfg, raw_ctx)
    diag.update(raw_details)
    if raw_ok:
        selected = raw_eval.astype(np.float32)
        diag.update(_selected_candidate_diag(raw, selected, True, True, diag, None, raw_idx=None))
        return selected, diag
    if mask.any():
        feasible_idx = np.where(mask)[0]
        dists = np.linalg.norm(candidates[feasible_idx] - raw[None, :], axis=1)
        selected_idx = int(feasible_idx[np.argmin(dists)])
    else:
        selected_idx = int(np.argmin(violation))
        diag["least_violating_fallback_used"] = True
    selected = candidates[selected_idx].astype(np.float32)
    selected_ok = bool(mask[selected_idx])
    diag.update({
        "selected_extinction_prob": float(ext[selected_idx]) if len(ext) else 0.0,
        "selected_explosion_prob": float(exp[selected_idx]) if len(exp) else 0.0,
        "selected_survival_fraction_risk": float(risk["survival_fraction_risk"][selected_idx]) if len(risk["survival_fraction_risk"]) else 0.0,
        "selected_consumption_fraction_risk": float(risk["consumption_fraction_risk"][selected_idx]) if len(risk["consumption_fraction_risk"]) else 0.0,
        "selected_mean_survival_fraction": float(risk["mean_survival_fraction"][selected_idx]) if len(risk["mean_survival_fraction"]) else 0.0,
        "selected_mean_consumption_fraction": float(risk["mean_consumption_fraction"][selected_idx]) if len(risk["mean_consumption_fraction"]) else 0.0,
        "movement_cost": float(move_costs[selected_idx]) if len(move_costs) else 0.0,
    })
    diag.update(_selected_candidate_diag(raw, selected, False, selected_ok, diag, selected_idx, raw_idx=None))
    return selected, diag


def filter_action(raw_action: np.ndarray, cfg: HomeostaticConfig, ctx: HomeostaticContext) -> Tuple[np.ndarray, Dict[str, Any]]:
    raw_state = ctx.raw_state if isinstance(ctx.raw_state, dict) else {"t_idx": ctx.step_index}
    if ctx.posterior_weights is not None or ctx.posterior_particles is not None:
        raw_state = dict(raw_state)
        if ctx.posterior_weights is not None:
            raw_state.setdefault("posterior", ctx.posterior_weights)
        if ctx.posterior_particles is not None:
            raw_state.setdefault("posterior_particles", ctx.posterior_particles)
    env = ctx.env
    adm = build_homeostatic_admissibility(env, raw_state, cfg, ctx.budget, ctx.prev_action)
    return enforce_homeostatic_admissibility(raw_action, adm, cfg)


def apply_discrete_action_mask(
    probs: torch.Tensor, log_probs: torch.Tensor, action_mask
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=probs.device)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0).expand_as(probs)
    has_mask = mask.any(dim=-1, keepdim=True)
    masked_probs = probs.masked_fill(~mask, 0.0)
    denom = masked_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    masked_probs = torch.where(has_mask, masked_probs / denom, probs)
    masked_log_probs = torch.log(masked_probs.clamp_min(1e-12))
    return masked_probs, masked_log_probs


def masked_discrete_sample(actor, state_t, action_values, action_mask, deterministic: bool = False):
    probs, log_probs = actor.probs_and_log_probs(state_t)
    probs, log_probs = apply_discrete_action_mask(probs, log_probs, action_mask)
    idx = torch.argmax(log_probs, dim=-1) if deterministic else torch.distributions.Categorical(probs=probs).sample()
    action = action_values[idx]
    log_prob = log_probs.gather(-1, idx.unsqueeze(-1))
    det_idx = torch.argmax(log_probs, dim=-1)
    deterministic_action = action_values[det_idx]
    return action, log_prob, deterministic_action, probs, log_probs


def _selected_candidate_diag(raw, selected, raw_feasible, selected_feasible, diag, selected_idx, raw_idx=None):
    dist = float(np.linalg.norm(np.asarray(selected, dtype=np.float32) - np.asarray(raw, dtype=np.float32)))
    filtered = dist > 1e-6 or not raw_feasible
    out = {
        "was_filtered": bool(filtered),
        "projection_used": bool(filtered),
        "homeo_projection_used": bool(filtered),
        "raw_action_feasible": bool(raw_feasible),
        "homeo_raw_action_feasible": bool(raw_feasible),
        "selected_action_feasible": bool(selected_feasible),
        "homeo_selected_action_feasible": bool(selected_feasible),
        "raw_filtered_distance": dist,
        "homeo_mean_raw_filtered_distance": dist,
        "violation": not bool(selected_feasible),
        "homeo_violation_rate": 0.0 if selected_feasible else 1.0,
    }
    if selected_idx is not None:
        ext = diag.get("candidate_extinction_prob")
        exp = diag.get("candidate_explosion_prob")
        surv = diag.get("candidate_survival_fraction_risk")
        cons = diag.get("candidate_consumption_fraction_risk")
        mean_surv = diag.get("candidate_mean_survival_fraction")
        mean_cons = diag.get("candidate_mean_consumption_fraction")
        if ext is not None:
            out["selected_extinction_prob"] = float(ext[selected_idx])
        if exp is not None:
            out["selected_explosion_prob"] = float(exp[selected_idx])
        if surv is not None:
            out["selected_survival_fraction_risk"] = float(surv[selected_idx])
        if cons is not None:
            out["selected_consumption_fraction_risk"] = float(cons[selected_idx])
        if mean_surv is not None:
            out["selected_mean_survival_fraction"] = float(mean_surv[selected_idx])
        if mean_cons is not None:
            out["selected_mean_consumption_fraction"] = float(mean_cons[selected_idx])
    if raw_idx is not None:
        ext = diag.get("candidate_extinction_prob")
        exp = diag.get("candidate_explosion_prob")
        surv = diag.get("candidate_survival_fraction_risk")
        cons = diag.get("candidate_consumption_fraction_risk")
        if ext is not None:
            out.setdefault("raw_extinction_prob", float(ext[raw_idx]))
        if exp is not None:
            out.setdefault("raw_explosion_prob", float(exp[raw_idx]))
        if surv is not None:
            out.setdefault("raw_survival_fraction_risk", float(surv[raw_idx]))
        if cons is not None:
            out.setdefault("raw_consumption_fraction_risk", float(cons[raw_idx]))
    out["least_violating_fallback_used"] = bool(diag.get("least_violating_fallback_used", False))
    out["homeo_least_violating_fallback_used"] = out["least_violating_fallback_used"]
    return out


def _posterior_from_raw(env: Any, raw_state: dict, cfg: HomeostaticConfig) -> Tuple[np.ndarray | None, np.ndarray | None]:
    bank = raw_state.get("posterior_particles")
    if bank is not None:
        bank = np.asarray(bank, dtype=np.float32)
    else:
        bank = getattr(env, "_cached_bank_np", None)
    if bank is None and hasattr(env, "hypothesis_bank"):
        hb = env.hypothesis_bank
        bank = hb.detach().cpu().numpy().astype(np.float32) if hasattr(hb, "detach") else np.asarray(hb, dtype=np.float32)
        env._cached_bank_np = bank
    if cfg.risk_source == "ebm_ablation":
        weights = raw_state.get("ebm_weights", raw_state.get("ebm_posterior"))
    else:
        weights = raw_state.get("posterior")
    if weights is None and "posterior_logits" in raw_state:
        logits = np.asarray(raw_state["posterior_logits"], dtype=np.float64)
        logits = logits - np.max(logits)
        weights = np.exp(logits) / np.exp(logits).sum()
    if weights is None and bank is not None:
        weights = np.full(len(bank), 1.0 / max(len(bank), 1), dtype=np.float32)
    return bank, None if weights is None else np.asarray(weights, dtype=np.float32)


def _all_discrete_actions(env: Any) -> np.ndarray:
    low, high = _bounds(env, np.zeros(int(getattr(env, "action_dim", 1)), dtype=np.float32))
    lo = int(round(float(low[0])))
    hi = int(round(float(high[0])))
    return np.arange(lo, hi + 1, dtype=np.float32).reshape(-1, 1)


def homeostatic_features_enabled(cfg: Optional[HomeostaticConfig]) -> bool:
    return bool(
        cfg is not None
        and cfg.enabled
        and cfg.mode != "none"
        and cfg.include_features
        and cfg.feature_mode != "none"
    )


def get_homeostatic_feature_spec(env: Any, cfg: Optional[HomeostaticConfig]) -> Tuple[int, list[str]]:
    del env
    if not homeostatic_features_enabled(cfg):
        return 0, []
    if cfg.mode == "prey_population":
        if cfg.feature_mode == "relative":
            names = [
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
        else:
            names = [
                "num_admissible_norm",
                "min_admissible_action_norm",
                "max_admissible_action_norm",
                "mean_admissible_action_norm",
                "std_admissible_action_norm",
                "mean_extinction_prob_adm",
                "min_extinction_prob_all",
                "mean_extinction_prob_all",
                "no_admissible_action_flag",
                "fallback_required_flag",
            ]
        return len(names), names
    if cfg.mode == "source_location":
        names = [
            "has_prev_action_flag",
            "normalized_remaining_movement_radius",
            "posterior_danger_mass_at_prev_action",
            "danger_radius_norm",
            "danger_prob_max",
            "budget_norm",
            "budget_enabled_flag",
            "projection_context_available_flag",
        ]
        return len(names), names
    return 0, []


def _build_homeo_features(
    cfg: HomeostaticConfig,
    env: Any,
    ctx: HomeostaticContext,
    candidates: np.ndarray,
    mask: np.ndarray,
    ext: np.ndarray,
    exp: np.ndarray,
    violation: np.ndarray,
    risk: Dict[str, np.ndarray],
) -> Tuple[np.ndarray | None, list[str] | None]:
    del exp
    dim, names = get_homeostatic_feature_spec(env, cfg)
    if dim <= 0:
        return None, None
    if cfg.mode == "prey_population":
        return _prey_homeo_features(env, cfg, candidates, mask, ext, violation, risk, names), names
    if cfg.mode == "source_location":
        return _source_homeo_features(env, cfg, ctx, names), names
    return None, None


def _prey_homeo_features(
    env: Any,
    cfg: HomeostaticConfig,
    candidates: np.ndarray,
    mask: np.ndarray,
    ext: np.ndarray,
    violation: np.ndarray,
    risk: Dict[str, np.ndarray],
    names: list[str],
) -> np.ndarray:
    actions = np.asarray(candidates, dtype=np.float32).reshape(-1)
    ext = np.asarray(ext, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    violation = np.asarray(violation, dtype=np.float32).reshape(-1)
    survival_risk = np.asarray(risk["survival_fraction_risk"], dtype=np.float32).reshape(-1)
    num_actions = max(actions.shape[0], 1)
    design_max = float(getattr(getattr(env, "cfg", None), "design_max", np.max(actions) if actions.size else 1.0))
    design_max = max(design_max, 1.0)
    if mask.any():
        adm_actions = actions[mask]
        ext_adm = ext[mask]
        min_adm = float(np.min(adm_actions)) / design_max
        max_adm = float(np.max(adm_actions)) / design_max
        mean_adm = float(np.mean(adm_actions)) / design_max
        std_adm = float(np.std(adm_actions)) / design_max
        mean_ext_adm = float(np.mean(ext_adm))
        mean_survival_risk_adm = float(np.mean(survival_risk[mask]))
    else:
        min_adm = max_adm = mean_adm = std_adm = 0.0
        mean_ext_adm = float(np.mean(ext)) if ext.size else 0.0
        mean_survival_risk_adm = float(np.mean(survival_risk)) if survival_risk.size else 0.0
    if cfg.feature_mode == "relative":
        feat = np.array([
            float(mask.sum()) / float(num_actions),
            min_adm,
            max_adm,
            mean_adm,
            std_adm,
            mean_survival_risk_adm,
            float(np.min(survival_risk)) if survival_risk.size else 0.0,
            float(np.mean(survival_risk)) if survival_risk.size else 0.0,
            float(not mask.any()),
            float(np.all(violation > 1e-12)) if violation.size else 0.0,
        ], dtype=np.float32)
    else:
        feat = np.array([
            float(mask.sum()) / float(num_actions),
            min_adm,
            max_adm,
            mean_adm,
            std_adm,
            mean_ext_adm,
            float(np.min(ext)) if ext.size else 0.0,
            float(np.mean(ext)) if ext.size else 0.0,
            float(not mask.any()),
            float(np.all(violation > 1e-12)) if violation.size else 0.0,
        ], dtype=np.float32)
    if feat.shape[0] != len(names):
        raise ValueError("Prey homeostatic feature dimension mismatch.")
    return feat


def _source_homeo_features(env: Any, cfg: HomeostaticConfig, ctx: HomeostaticContext, names: list[str]) -> np.ndarray:
    action_low = getattr(env, "action_low", None)
    action_high = getattr(env, "action_high", None)
    if action_low is None or action_high is None:
        action_span = 1.0
    else:
        low = action_low.detach().cpu().numpy() if hasattr(action_low, "detach") else np.asarray(action_low)
        high = action_high.detach().cpu().numpy() if hasattr(action_high, "detach") else np.asarray(action_high)
        action_span = float(np.linalg.norm(np.asarray(high, dtype=np.float32) - np.asarray(low, dtype=np.float32)))
    action_span = max(action_span, 1e-6)
    prev = None if ctx.prev_action is None else np.asarray(ctx.prev_action, dtype=np.float32).reshape(-1)
    risk_at_prev = 0.0
    if prev is not None and cfg.danger_radius is not None and ctx.posterior_particles is not None:
        risk_at_prev = float(_source_danger_probability_batch(prev[None, :], cfg, ctx)[0])
    if cfg.max_step_norm is not None:
        rem_radius = float(cfg.max_step_norm) / action_span
    else:
        rem_radius = 0.0
    if cfg.initial_budget is not None and cfg.initial_budget > 0 and ctx.budget is not None:
        budget_norm = float(ctx.budget) / float(cfg.initial_budget)
    else:
        budget_norm = 0.0
    feat = np.array([
        float(prev is not None),
        rem_radius,
        risk_at_prev,
        float(cfg.danger_radius or 0.0) / action_span,
        float(cfg.danger_prob_max or 0.0),
        budget_norm,
        float(ctx.budget is not None),
        float(prev is not None or ctx.posterior_particles is not None),
    ], dtype=np.float32)
    if feat.shape[0] != len(names):
        raise ValueError("Source homeostatic feature dimension mismatch.")
    return feat


def _evaluate_candidates(candidates: np.ndarray, cfg: HomeostaticConfig, ctx: HomeostaticContext):
    K = len(candidates)
    mask = np.ones(K, dtype=bool)
    ext = np.zeros(K, dtype=np.float32)
    exp = np.zeros(K, dtype=np.float32)
    risk = _empty_prey_risk(K)
    move = np.full(K, float(cfg.obs_cost), dtype=np.float32)
    violation = np.zeros(K, dtype=np.float32)
    if ctx.prev_action is not None:
        deltas = np.linalg.norm(candidates - np.asarray(ctx.prev_action, dtype=np.float32)[None, :], axis=1)
        move += float(cfg.movement_cost) * deltas.astype(np.float32)
        if cfg.max_step_norm is not None:
            bad = deltas > float(cfg.max_step_norm) + 1e-8
            mask &= ~bad
            violation += np.maximum(deltas - float(cfg.max_step_norm), 0.0).astype(np.float32)
    if ctx.budget is not None:
        bad = move > float(ctx.budget) + 1e-8
        mask &= ~bad
        violation += np.maximum(move - float(ctx.budget), 0.0).astype(np.float32)
    if cfg.mode == "prey_population":
        vmask, ext, exp, risk = _prey_viability_batch(candidates, cfg, ctx)
        mask &= vmask
        if cfg.extinction_prob_max is not None:
            violation += np.maximum(ext - float(cfg.extinction_prob_max), 0.0)
        if cfg.explosion_prob_max is not None:
            violation += np.maximum(exp - float(cfg.explosion_prob_max), 0.0)
        if cfg.survival_fraction_min is not None and cfg.survival_fraction_prob_max is not None:
            violation += np.maximum(risk["survival_fraction_risk"] - float(cfg.survival_fraction_prob_max), 0.0)
        if cfg.consumption_fraction_max is not None and cfg.consumption_fraction_prob_max is not None:
            violation += np.maximum(risk["consumption_fraction_risk"] - float(cfg.consumption_fraction_prob_max), 0.0)
    elif cfg.mode == "source_location":
        source_risk = _source_danger_probability_batch(candidates, cfg, ctx)
        ext = source_risk.astype(np.float32)
        if cfg.danger_prob_max is not None:
            bad = source_risk > float(cfg.danger_prob_max) + 1e-12
            mask &= ~bad
            violation += np.maximum(source_risk - float(cfg.danger_prob_max), 0.0).astype(np.float32)
    return mask, ext, exp, move, violation, risk


def _candidate_feasibility(action: np.ndarray, cfg: HomeostaticConfig, ctx: HomeostaticContext):
    candidates = np.asarray(action, dtype=np.float32).reshape(1, -1)
    mask, ext, exp, move, violation, risk = _evaluate_candidates(candidates, cfg, ctx)
    return bool(mask[0]), {
        "movement_cost": float(move[0]),
        "raw_extinction_prob": float(ext[0]),
        "raw_explosion_prob": float(exp[0]),
        "raw_survival_fraction_risk": float(risk["survival_fraction_risk"][0]) if len(risk["survival_fraction_risk"]) else 0.0,
        "raw_consumption_fraction_risk": float(risk["consumption_fraction_risk"][0]) if len(risk["consumption_fraction_risk"]) else 0.0,
        "raw_mean_survival_fraction": float(risk["mean_survival_fraction"][0]) if len(risk["mean_survival_fraction"]) else 0.0,
        "raw_mean_consumption_fraction": float(risk["mean_consumption_fraction"][0]) if len(risk["mean_consumption_fraction"]) else 0.0,
        "risk_prob": _aggregate_primary_risk(ext, exp, risk),
    }


def _step_cost(cfg: HomeostaticConfig, prev_action: np.ndarray | None, action: np.ndarray) -> float:
    cost = float(cfg.obs_cost)
    if prev_action is not None:
        cost += float(cfg.movement_cost) * float(np.linalg.norm(np.asarray(action, dtype=np.float32) - np.asarray(prev_action, dtype=np.float32)))
    return cost


def _bounds(env: Any, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    low = getattr(env, "action_low", None)
    high = getattr(env, "action_high", None)
    if low is not None:
        low = low.detach().cpu().numpy() if hasattr(low, "detach") else np.asarray(low)
    else:
        low = np.full_like(raw, -np.inf, dtype=np.float32)
    if high is not None:
        high = high.detach().cpu().numpy() if hasattr(high, "detach") else np.asarray(high)
    else:
        high = np.full_like(raw, np.inf, dtype=np.float32)
    return np.asarray(low, dtype=np.float32).reshape(-1), np.asarray(high, dtype=np.float32).reshape(-1)


def _clip(env: Any, action: np.ndarray) -> np.ndarray:
    if env is not None and hasattr(env, "clip_action"):
        return np.asarray(env.clip_action(action), dtype=np.float32)
    return np.asarray(action, dtype=np.float32)


def _source_danger_probability_batch(candidates: np.ndarray, cfg: HomeostaticConfig, ctx: HomeostaticContext) -> np.ndarray:
    K = len(candidates)
    zeros = np.zeros(K, dtype=np.float32)
    if cfg.danger_radius is None or ctx.posterior_particles is None:
        return zeros
    parts = np.asarray(ctx.posterior_particles, dtype=np.float32)
    acts = np.asarray(candidates, dtype=np.float32)
    if parts.ndim != 2:
        parts = parts.reshape(parts.shape[0], -1)
    w = _normalise_weights(ctx.posterior_weights, parts.shape[0])
    D = acts.shape[1]
    if parts.shape[1] == D:
        dkp = np.linalg.norm(acts[:, None, :] - parts[None, :, :], axis=-1)
    elif parts.shape[1] % D == 0:
        n_src = parts.shape[1] // D
        src = parts.reshape(parts.shape[0], n_src, D)
        dkp = np.linalg.norm(acts[:, None, :] - src[None, :, 0, :], axis=-1)
        for s in range(1, n_src):
            np.minimum(dkp, np.linalg.norm(acts[:, None, :] - src[None, :, s, :], axis=-1), out=dkp)
    else:
        return zeros
    return ((dkp < float(cfg.danger_radius)).astype(np.float64) @ w).astype(np.float32)


def _normalise_weights(weights: Optional[np.ndarray], n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != n:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    return w / total


def _empty_prey_risk(K: int) -> Dict[str, np.ndarray]:
    zeros = np.zeros(K, dtype=np.float32)
    return {
        "absolute_extinction_prob": zeros.copy(),
        "absolute_explosion_prob": zeros.copy(),
        "survival_fraction_risk": zeros.copy(),
        "consumption_fraction_risk": zeros.copy(),
        "mean_survival_fraction": zeros.copy(),
        "mean_consumption_fraction": zeros.copy(),
    }


def _prey_viability_batch(candidates: np.ndarray, cfg: HomeostaticConfig, ctx: HomeostaticContext):
    K = len(candidates)
    zeros = np.zeros(K, dtype=np.float32)
    true_mask = np.ones(K, dtype=bool)
    risk = _empty_prey_risk(K)
    if (
        cfg.extinction_prob_max is None
        and cfg.explosion_prob_max is None
        and not (cfg.survival_fraction_min is not None and cfg.survival_fraction_prob_max is not None)
        and not (cfg.consumption_fraction_max is not None and cfg.consumption_fraction_prob_max is not None)
    ):
        return true_mask, zeros, zeros, risk
    env = ctx.env
    particles = ctx.posterior_particles
    if env is None or particles is None or len(particles) == 0:
        return true_mask, zeros, zeros, risk
    if hasattr(env, "predict_population_next_batch"):
        prey_batch, pred_batch = env.predict_population_next_batch(np.asarray(particles, dtype=np.float32), candidates)
        prey = np.asarray(prey_batch, dtype=np.float64)
        P = prey.shape[1]
        w = _normalise_weights(ctx.posterior_weights, P)
        ext = zeros.astype(np.float64)
        exp = zeros.astype(np.float64)
        cand0 = np.asarray(candidates, dtype=np.float64).reshape(K, -1)[:, :1]
        denom = np.maximum(cand0, 1e-8)
        survival_fraction = prey / denom
        consumption_fraction = np.clip((cand0 - prey) / denom, 0.0, np.inf)
        surv_risk = zeros.astype(np.float64)
        cons_risk = zeros.astype(np.float64)
        if cfg.prey_min is not None:
            ext = (prey < float(cfg.prey_min)).astype(np.float64) @ w
        if cfg.prey_max is not None:
            exp = np.maximum(exp, (prey > float(cfg.prey_max)).astype(np.float64) @ w)
        if cfg.predator_max is not None and pred_batch is not None:
            pred = np.asarray(pred_batch, dtype=np.float64)
            if pred.shape == prey.shape:
                exp = np.maximum(exp, (pred > float(cfg.predator_max)).astype(np.float64) @ w)
        if cfg.survival_fraction_min is not None and cfg.survival_fraction_prob_max is not None:
            surv_risk = (survival_fraction < float(cfg.survival_fraction_min)).astype(np.float64) @ w
        if cfg.consumption_fraction_max is not None and cfg.consumption_fraction_prob_max is not None:
            cons_risk = (consumption_fraction > float(cfg.consumption_fraction_max)).astype(np.float64) @ w
        risk["mean_survival_fraction"] = (survival_fraction @ w).astype(np.float32)
        risk["mean_consumption_fraction"] = (consumption_fraction @ w).astype(np.float32)
        risk["survival_fraction_risk"] = surv_risk.astype(np.float32)
        risk["consumption_fraction_risk"] = cons_risk.astype(np.float32)
    elif hasattr(env, "predict_population_next"):
        ext = zeros.astype(np.float64)
        exp = zeros.astype(np.float64)
        w = _normalise_weights(ctx.posterior_weights, len(particles))
        surv_risk = zeros.astype(np.float64)
        cons_risk = zeros.astype(np.float64)
        mean_survival = zeros.astype(np.float64)
        mean_consumption = zeros.astype(np.float64)
        for i, cand in enumerate(candidates):
            pred = env.predict_population_next(np.asarray(particles, dtype=np.float32), cand)
            prey_next, pred_next = pred if isinstance(pred, tuple) else (pred, None)
            prey = np.asarray(prey_next, dtype=np.float64).reshape(-1)
            denom = max(float(np.asarray(cand, dtype=np.float64).reshape(-1)[0]), 1e-8)
            survival_fraction = prey / denom
            consumption_fraction = np.clip((denom - prey) / denom, 0.0, np.inf)
            if cfg.prey_min is not None:
                ext[i] = float(w[prey < float(cfg.prey_min)].sum())
            if cfg.prey_max is not None:
                exp[i] = max(exp[i], float(w[prey > float(cfg.prey_max)].sum()))
            if cfg.predator_max is not None and pred_next is not None:
                parr = np.asarray(pred_next, dtype=np.float64).reshape(-1)
                if parr.shape[0] == w.shape[0]:
                    exp[i] = max(exp[i], float(w[parr > float(cfg.predator_max)].sum()))
            if cfg.survival_fraction_min is not None and cfg.survival_fraction_prob_max is not None:
                surv_risk[i] = float(w[survival_fraction < float(cfg.survival_fraction_min)].sum())
            if cfg.consumption_fraction_max is not None and cfg.consumption_fraction_prob_max is not None:
                cons_risk[i] = float(w[consumption_fraction > float(cfg.consumption_fraction_max)].sum())
            mean_survival[i] = float(survival_fraction @ w)
            mean_consumption[i] = float(consumption_fraction @ w)
        risk["mean_survival_fraction"] = mean_survival.astype(np.float32)
        risk["mean_consumption_fraction"] = mean_consumption.astype(np.float32)
        risk["survival_fraction_risk"] = surv_risk.astype(np.float32)
        risk["consumption_fraction_risk"] = cons_risk.astype(np.float32)
    else:
        return true_mask, zeros, zeros, risk
    risk["absolute_extinction_prob"] = ext.astype(np.float32)
    risk["absolute_explosion_prob"] = exp.astype(np.float32)
    mask = np.ones(K, dtype=bool)
    if cfg.extinction_prob_max is not None:
        mask &= ext <= float(cfg.extinction_prob_max) + 1e-12
    if cfg.explosion_prob_max is not None:
        mask &= exp <= float(cfg.explosion_prob_max) + 1e-12
    if cfg.survival_fraction_min is not None and cfg.survival_fraction_prob_max is not None:
        mask &= risk["survival_fraction_risk"] <= float(cfg.survival_fraction_prob_max) + 1e-12
    if cfg.consumption_fraction_max is not None and cfg.consumption_fraction_prob_max is not None:
        mask &= risk["consumption_fraction_risk"] <= float(cfg.consumption_fraction_prob_max) + 1e-12
    return mask, ext.astype(np.float32), exp.astype(np.float32), risk


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return 0.0
    if mask.any():
        return float(np.mean(values[mask]))
    return float(np.mean(values))


def _aggregate_primary_risk(ext: np.ndarray, exp: np.ndarray, risk: Dict[str, np.ndarray]) -> float:
    arrays = [
        np.asarray(ext, dtype=np.float32).reshape(-1),
        np.asarray(exp, dtype=np.float32).reshape(-1),
        np.asarray(risk.get("survival_fraction_risk", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1),
        np.asarray(risk.get("consumption_fraction_risk", np.zeros(0, dtype=np.float32)), dtype=np.float32).reshape(-1),
    ]
    arrays = [arr for arr in arrays if arr.size]
    if not arrays:
        return 0.0
    stack = np.stack(arrays, axis=0)
    return float(np.mean(np.max(stack, axis=0)))


def _candidate_actions(raw: np.ndarray, cfg: HomeostaticConfig, ctx: HomeostaticContext) -> list[np.ndarray]:
    max_n = max(int(cfg.max_projection_candidates), 1)
    env = ctx.env
    low, high = _bounds(env, raw) if env is not None else (np.full_like(raw, -1.0), np.full_like(raw, 1.0))
    center = np.asarray(ctx.prev_action, dtype=np.float32) if ctx.prev_action is not None and cfg.max_step_norm is not None else np.clip(raw, low, high)
    radius = float(cfg.max_step_norm) if cfg.max_step_norm is not None else float(np.linalg.norm(high - low) / 4.0)
    candidates: list[np.ndarray] = []
    def add(x):
        y = _clip(env, np.clip(np.asarray(x, dtype=np.float32), low, high))
        if not any(np.allclose(y, old, atol=1e-6) for old in candidates):
            candidates.append(y.astype(np.float32))
    add(raw)
    if raw.size == 2:
        n_angles = max(8, min(max_n, 64))
        n_radii = max(2, int(np.ceil(max_n / n_angles)))
        for r in np.linspace(0.0, radius, n_radii, dtype=np.float32):
            for a in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False, dtype=np.float32):
                add(center + r * np.array([np.cos(a), np.sin(a)], dtype=np.float32))
                if len(candidates) >= max_n:
                    return candidates
    else:
        for frac in np.linspace(0.0, 1.0, max_n, dtype=np.float32):
            add(raw + frac * (_clip(env, raw) - raw))
    return candidates[:max_n]
