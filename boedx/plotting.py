"""
Publication-quality plotting for BOEDX experiments.

Two entry points
----------------
``save_standard_plots(output_dir, all_results, summary, horizon)``
    Lightweight plots saved directly into the experiment output directory.
    Called automatically by ``run_experiment_suite``.

``generate_scientific_plots(output_dir)``
    Loads saved per-seed JSON results and generates a full set of
    publication-ready figures in ``graph_<experiment_name>/`` alongside
    the output directory.  All figures are saved as high-resolution PNG
    and as vector PDF for direct inclusion in LaTeX.

CLI::

    boedx-plot ./outputs/source_location_beta_contrastive_h50
    # → ./graph_source_location_beta_contrastive_h50/

Style conventions
-----------------
- All trajectory and bar plots carry a full Cartesian grid (both axes).
- Lines are thin (lw ≈ 1.2–1.4) for trajectories; the mean line in
  per-seed fan plots is slightly thicker (lw 1.8) to stand out.
- Color palette: cool blue-gray spectrum.  Warm amber/orange is reserved
  exclusively for beta-contrastive milestone variants.
- Markers: unified as filled circles (``"o"``) throughout.  Scatter points
  share the line color, white-edged for separation from the CI band.

Scientific plot catalogue
-------------------------
Trajectory plots (mean ± 95 % CI band):
  - SPCE lower bound vs. experiment step
  - Posterior bank IG vs. step
  - Control/filter bank IG vs. step
  - SNMC-style upper bound vs. step  (if available)

Per-seed fan plots (all seed traces + mean):
  - SPCE lower bound (per-seed fan)
  - Bank IG (per-seed fan)

Information-gain gap plot:
  - (Variant − Blau baseline) IG difference per step

Belief–performance scatter:
  - EBM belief KL vs. SPCE lower bound  (EBM variants only)
  - EBM belief MAP-to-exact vs. SPCE lower bound

Paired-difference bar chart:
  - Paired return gains vs. baseline with per-seed dots

Bar comparisons (horizontal, individual seed points overlaid):
  - Evaluation return
  - Final SPCE lower
  - Final SNMC upper  (if available)
  - Final posterior bank IG
  - Final control/filter bank IG

Belief quality (EBM variants, multi-panel bar):
  - KL(exact ‖ EBM), L1 total variation, MAP-to-exact, MAP-to-true

Summary overview (combined panel for main results section)

Pairwise scatter: eval return vs. final SPCE
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

# Blue-gray spectrum for the main palette.
# Warm amber/orange is reserved for beta-contrastive milestone variants.
_PALETTE: Dict[str, str] = {
    "blau_approx":                       "#8B9BB5",  # steel-gray-blue (baseline)
    "control_filter_exact":              "#4A6B9A",  # medium slate blue
    "control_posterior_exact":           "#6A88B5",  # lighter slate
    "ours_ebm_control":                  "#2D5FA8",  # vivid medium blue
    "ours_ebm_control_filter":           "#2D5FA8",
    "ours_ebm_control_posterior":        "#2D5FA8",
    "ours_ebm_cross":                    "#1A3B7D",  # deep navy (best method)
    "ours_ebm_cross_filter":             "#1A3B7D",
    "ours_ebm_cross_posterior":          "#1A3B7D",
    "ours_ebm_control_beta_contrastive": "#D4860A",  # warm amber  (milestone)
    "ours_ebm_cross_beta_contrastive":   "#B84C00",  # warm sienna (milestone)
}

# Linestyle: baseline solid-gray, exact ablations dashed, our methods solid,
# beta-contrastive dotted (warm colour makes them stand out already).
_LINESTYLES: Dict[str, str] = {
    "blau_approx":                       "-",
    "control_filter_exact":              "--",
    "control_posterior_exact":           "-.",
    "ours_ebm_control":                  "-",
    "ours_ebm_control_filter":           "-",
    "ours_ebm_control_posterior":        "-",
    "ours_ebm_cross":                    "-",
    "ours_ebm_cross_filter":             "-",
    "ours_ebm_cross_posterior":          "-",
    "ours_ebm_control_beta_contrastive": ":",
    "ours_ebm_cross_beta_contrastive":   ":",
}

_DISPLAY_NAMES: Dict[str, str] = {
    "blau_approx":                        "Blau et al.",
    "control_filter_exact":               "Exact control (filter)",
    "control_posterior_exact":            "Exact control (posterior)",
    "ours_ebm_control":                   "EBM-control",
    "ours_ebm_cross":                     "EBM-cross",
    "ours_ebm_control_filter":            "EBM-control (filter)",
    "ours_ebm_control_posterior":         "EBM-control (posterior)",
    "ours_ebm_cross_filter":              "EBM-cross (filter)",
    "ours_ebm_cross_posterior":           "EBM-cross (posterior)",
    "ours_ebm_control_beta_contrastive":  r"EBM-control ($\beta$-ctr.)",
    "ours_ebm_cross_beta_contrastive":    r"EBM-cross ($\beta$-ctr.)",
}

# Fallback cycle for unseen variant names — stays within the blue-gray range.
_FALLBACK_CYCLE = [
    "#3D6BC4", "#5A82C8", "#7A99CC", "#9AAFD0",
    "#B0BFD5", "#6B7CA8", "#4A5C80",
]


def _color(v: str) -> str:
    if v in _PALETTE:
        return _PALETTE[v]
    return _FALLBACK_CYCLE[hash(v) % len(_FALLBACK_CYCLE)]


def _ls(v: str) -> str:
    return _LINESTYLES.get(v, "-")


def _dname(v: str) -> str:
    return _DISPLAY_NAMES.get(v, v)


# ---------------------------------------------------------------------------
# rcParams for publication quality
# ---------------------------------------------------------------------------

_PAPER_RC = {
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "legend.fontsize":   8.5,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.linewidth":    0.7,
    "lines.linewidth":   1.2,
    "lines.markersize":  5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
}


def _apply_paper_style() -> None:
    plt.rcParams.update(_PAPER_RC)


def _grid(ax: plt.Axes, *, alpha: float = 0.30) -> None:
    """Full Cartesian grid on both axes, thin dashed lines."""
    ax.grid(True, which="major", linestyle="--", linewidth=0.45,
            alpha=alpha, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: plt.Figure, path_no_ext: str) -> None:
    """Save as both PNG (300 dpi) and PDF (vector)."""
    fig.savefig(path_no_ext + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trajectory (path) plots — mean ± CI band
# ---------------------------------------------------------------------------

def _plot_trajectory(
    plot_data: Dict[str, Dict[str, np.ndarray]],
    ordered_variants: List[str],
    key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
    horizon: int,
) -> None:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    steps = np.arange(1, horizon + 1)

    for v in ordered_variants:
        if v not in plot_data:
            continue
        rec = plot_data[v]
        if f"{key}_mean" not in rec:
            continue
        m, s = rec[f"{key}_mean"], rec[f"{key}_se"]
        c = _color(v)
        label = f"{_dname(v)}  [{m[-1]:.3f}]"
        ax.plot(steps, m, label=label, color=c, linestyle=_ls(v), linewidth=1.2)
        ax.fill_between(steps, m - 1.96 * s, m + 1.96 * s, color=c, alpha=0.14)
        ax.scatter([steps[-1]], [m[-1]], color=c, s=22, zorder=5, clip_on=False,
                   marker="o", edgecolors="white", linewidths=0.6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    _grid(ax)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="#cccccc", loc="best", ncol=1)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Per-seed fan plots — all seed traces (thin) + mean (bold)
# ---------------------------------------------------------------------------

def _plot_per_seed_trajectories(
    all_results: Dict[str, List[Dict]],
    ordered_variants: List[str],
    path_key: str,
    ylabel: str,
    title: str,
    save_path: str,
    horizon: int,
) -> None:
    """Individual seed traces (thin, translucent) overlaid with the mean line.

    This exposes seed-to-seed variability that a CI band can hide.
    """
    _apply_paper_style()
    n_variants = sum(1 for v in ordered_variants if v in all_results)
    if n_variants == 0:
        return

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    steps = np.arange(1, horizon + 1)

    for v in ordered_variants:
        if v not in all_results:
            continue
        c = _color(v)
        seed_paths = []
        for r in all_results[v]:
            path = r.get("paths", {}).get(path_key)
            if path is not None and len(path) == horizon:
                arr = np.array(path, dtype=float)
                seed_paths.append(arr)
                ax.plot(steps, arr, color=c, linewidth=0.5, alpha=0.28, linestyle="-")

        if not seed_paths:
            continue
        mean_path = np.mean(seed_paths, axis=0)
        ax.plot(steps, mean_path, label=f"{_dname(v)}  [{mean_path[-1]:.3f}]",
                color=c, linewidth=1.8, linestyle=_ls(v))
        ax.scatter([steps[-1]], [mean_path[-1]], color=c, s=24, zorder=6,
                   marker="o", edgecolors="white", linewidths=0.6)

    ax.set_xlabel("Experiment step $t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    _grid(ax)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="#cccccc", loc="best", ncol=1)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Information-gain gap plot — (variant − baseline) per step
# ---------------------------------------------------------------------------

def _plot_ig_gap(
    all_results: Dict[str, List[Dict]],
    ordered_variants: List[str],
    baseline_variant: str,
    path_key: str,
    ylabel: str,
    title: str,
    save_path: str,
    horizon: int,
) -> None:
    """Per-step IG difference relative to the baseline (positive = better).

    Shows the compounding advantage of derived control states as the
    experiment sequence progresses.  Each non-baseline variant is shown as
    a mean ± CI gap curve; the zero line marks the baseline.
    """
    _apply_paper_style()
    if baseline_variant not in all_results:
        return

    # Build per-seed baseline paths
    base_paths = []
    for r in all_results[baseline_variant]:
        path = r.get("paths", {}).get(path_key)
        if path is not None and len(path) == horizon:
            base_paths.append(np.array(path, dtype=float))
    if not base_paths:
        return
    base_arr = np.stack(base_paths, axis=0)  # (n_seeds, T)
    base_mean = base_arr.mean(axis=0)

    non_baseline = [v for v in ordered_variants if v != baseline_variant]
    if not non_baseline:
        return

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    steps = np.arange(1, horizon + 1)

    # Shaded zero band (baseline ± 1 SE)
    base_se = base_arr.std(axis=0) / max(math.sqrt(len(base_paths)), 1.0)
    ax.fill_between(steps, -1.96 * base_se, 1.96 * base_se,
                    color="#8B9BB5", alpha=0.12, zorder=0, label="Blau ±95 % CI")
    ax.axhline(0.0, color="#8B9BB5", linewidth=0.9, linestyle="-", alpha=0.65)

    for v in non_baseline:
        if v not in all_results:
            continue
        c = _color(v)
        gap_seeds = []
        for i, r in enumerate(all_results[v]):
            path = r.get("paths", {}).get(path_key)
            if path is not None and len(path) == horizon:
                # Align seeds: use baseline seed at the same index if available
                b = base_arr[i] if i < len(base_arr) else base_mean
                gap_seeds.append(np.array(path, dtype=float) - b)
        if not gap_seeds:
            continue
        gap = np.stack(gap_seeds, axis=0)
        m = gap.mean(axis=0)
        se = gap.std(axis=0) / max(math.sqrt(len(gap_seeds)), 1.0)
        ax.plot(steps, m, label=f"{_dname(v)}  [Δ={m[-1]:+.3f}]",
                color=c, linewidth=1.2, linestyle=_ls(v))
        ax.fill_between(steps, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.14)
        ax.scatter([steps[-1]], [m[-1]], color=c, s=22, zorder=5,
                   marker="o", edgecolors="white", linewidths=0.6)

    ax.set_xlabel("Experiment step $t$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    _grid(ax)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="#cccccc", loc="best", ncol=1)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Belief quality vs. performance scatter (per-seed dots)
# ---------------------------------------------------------------------------

def _plot_belief_performance_scatter(
    all_results: Dict[str, List[Dict]],
    ordered_variants: List[str],
    belief_metric: str,
    belief_label: str,
    perf_metric: str,
    perf_label: str,
    title: str,
    save_path: str,
    belief_lower_better: bool = True,
) -> None:
    """Scatter of EBM belief quality vs. policy performance, one dot per seed.

    Exposes whether tighter belief surrogates (lower KL / MAP-to-exact)
    translate into better design policies (higher SPCE lower).  Each seed
    is a dot; variant means are marked with a larger circle and labelled.
    """
    _apply_paper_style()
    ebm_variants = [
        v for v in ordered_variants
        if v in all_results
        and any(belief_metric in r.get("eval", {}) for r in all_results[v])
        and any(perf_metric in r.get("eval", {}) for r in all_results[v])
    ]
    if not ebm_variants:
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.4))

    for v in ebm_variants:
        c = _color(v)
        xs, ys = [], []
        for r in all_results[v]:
            ev = r.get("eval", {})
            if belief_metric in ev and perf_metric in ev:
                xs.append(float(ev[belief_metric]))
                ys.append(float(ev[perf_metric]))
        if not xs:
            continue
        # Individual seed dots
        ax.scatter(xs, ys, color=c, s=28, alpha=0.75, zorder=4,
                   marker="o", edgecolors="white", linewidths=0.5)
        # Variant mean marker (larger, labelled)
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        ax.scatter([mx], [my], color=c, s=90, zorder=6,
                   marker="o", edgecolors=c, linewidths=1.4, label=_dname(v))
        ax.annotate(
            _dname(v), (mx, my),
            xytext=(7, 4), textcoords="offset points",
            fontsize=7.5, color=c,
        )

    hint = "←  better" if belief_lower_better else "→  better"
    ax.set_xlabel(f"{belief_label}  ({hint})")
    ax.set_ylabel(f"{perf_label}  (→  higher is better)")
    ax.set_title(title, pad=6)
    _grid(ax)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="#cccccc",
              loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Paired-difference bar chart
# ---------------------------------------------------------------------------

def _plot_paired_differences(
    all_results: Dict[str, List[Dict]],
    summary: Dict,
    ordered_variants: List[str],
    baseline_variant: str,
    metric: str,
    xlabel: str,
    title: str,
    save_path: str,
) -> None:
    """Paired gain over the baseline for each variant, one dot per seed pair.

    Shows the distribution of seed-level improvements rather than just the
    aggregate mean ± CI.  Dots to the right of zero mean the variant beat
    the baseline on that seed.
    """
    _apply_paper_style()
    if baseline_variant not in all_results:
        return

    non_baseline = [
        v for v in ordered_variants
        if v != baseline_variant
        and v in all_results
        and any(metric in r.get("eval", {}) for r in all_results[v])
    ]
    if not non_baseline:
        return

    # Build baseline seed values
    base_vals = [
        float(r["eval"][metric])
        for r in all_results[baseline_variant]
        if metric in r.get("eval", {})
    ]

    n = len(non_baseline)
    fig, ax = plt.subplots(figsize=(6.0, 0.7 * n + 1.4))
    y_pos = np.arange(n)[::-1]
    bar_height = 0.50

    for i, v in enumerate(non_baseline):
        c = _color(v)
        yi = y_pos[i]
        var_vals = [
            float(r["eval"][metric])
            for r in all_results[v]
            if metric in r.get("eval", {})
        ]
        n_pairs = min(len(base_vals), len(var_vals))
        diffs = [var_vals[j] - base_vals[j] for j in range(n_pairs)]
        if not diffs:
            continue

        mean_diff = float(np.mean(diffs))
        # 95 % CI from summary if available, else empirical
        diff_key = f"paired_return_diff_{v}_minus_{baseline_variant}"
        if diff_key in summary:
            ci_lo = summary[diff_key].get("ci95_low", mean_diff)
            ci_hi = summary[diff_key].get("ci95_high", mean_diff)
        else:
            se = float(np.std(diffs)) / max(math.sqrt(n_pairs), 1.0)
            ci_lo, ci_hi = mean_diff - 1.96 * se, mean_diff + 1.96 * se

        ci_half_lo = abs(mean_diff - ci_lo)
        ci_half_hi = abs(ci_hi - mean_diff)

        ax.barh(yi, mean_diff, height=bar_height, color=c, alpha=0.78, zorder=2)
        ax.errorbar(mean_diff, yi, xerr=[[ci_half_lo], [ci_half_hi]],
                    fmt="none", color="#333333", linewidth=1.0, capsize=3.5, zorder=4)

        # Per-seed paired diff dots
        jitter = np.linspace(-bar_height * 0.28, bar_height * 0.28, len(diffs))
        ax.scatter(diffs, [yi + j for j in jitter],
                   color=c, s=22, zorder=5, marker="o",
                   edgecolors="white", linewidths=0.5, alpha=0.85)

        # Significance star
        if ci_lo > 0.0:
            ax.text(ci_hi + 0.002, yi + bar_height * 0.52, "*",
                    ha="left", va="bottom", fontsize=11, color="#333333", fontweight="bold")

        # Mean label
        ax.text(ci_hi + 0.006, yi, f"{mean_diff:+.3f}",
                va="center", ha="left", fontsize=8, color="#333333")

    ax.axvline(0.0, color="#8B9BB5", linewidth=0.9, linestyle="--", alpha=0.7,
               label=f"{_dname(baseline_variant)} (Δ = 0)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_dname(v) for v in non_baseline])
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=6)
    _grid(ax)
    ax.yaxis.grid(False)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="#cccccc", fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Bar plots — horizontal bars with seed-point overlay
# ---------------------------------------------------------------------------

def _plot_bar_comparison(
    summary: Dict,
    all_results: Dict[str, List[Dict]],
    ordered_variants: List[str],
    metric_field: str,
    ylabel: str,
    title: str,
    save_path: str,
    higher_is_better: bool = True,
    baseline_variant: str = "blau_approx",
) -> None:
    _apply_paper_style()
    variants = [v for v in ordered_variants if f"{v}_{metric_field}" in summary]
    if not variants:
        return

    n = len(variants)
    fig, ax = plt.subplots(figsize=(6.5, 0.65 * n + 1.2))
    y_pos = np.arange(n)[::-1]
    bar_height = 0.52

    blau_mean = summary.get(f"{baseline_variant}_{metric_field}", {}).get("mean")

    for i, v in enumerate(variants):
        stats = summary[f"{v}_{metric_field}"]
        mean = stats["mean"]
        ci_half = max(0.0, stats["ci95_high"] - mean)
        c = _color(v)
        yi = y_pos[i]

        ax.barh(yi, mean, height=bar_height, color=c, alpha=0.82, zorder=2)
        ax.errorbar(mean, yi, xerr=ci_half,
                    fmt="none", color="#333333", linewidth=1.0, capsize=3.5, zorder=4)
        offset = ci_half + 0.004 * abs(mean) + 1e-4
        ax.text(mean + offset, yi,
                f"{mean:.3f} ±{ci_half:.3f}",
                va="center", ha="left", fontsize=8, color="#333333")

        if v in all_results:
            seed_vals = [r["eval"].get(metric_field)
                         for r in all_results[v] if metric_field in r.get("eval", {})]
            if seed_vals:
                jitter = np.linspace(-bar_height * 0.28, bar_height * 0.28, len(seed_vals))
                ax.scatter(seed_vals, [yi + j for j in jitter],
                           color=c, edgecolors="white", linewidths=0.5,
                           s=24, zorder=5, marker="o", alpha=0.90)

        # Significance star from paired summary
        diff_key = f"paired_return_diff_{v}_minus_{baseline_variant}"
        if v != baseline_variant and diff_key in summary:
            d = summary[diff_key]
            sig = (d.get("ci95_low", 0.0) > 0.0) if higher_is_better else \
                  (d.get("ci95_high", 0.0) < 0.0)
            if sig:
                ax.text(mean, yi + bar_height * 0.56, "*",
                        ha="center", va="bottom", fontsize=11,
                        color="#333333", fontweight="bold")

    if blau_mean is not None and baseline_variant in variants:
        ax.axvline(blau_mean, color=_color(baseline_variant), linestyle=":",
                   linewidth=1.0, alpha=0.60,
                   label=f"{_dname(baseline_variant)} mean")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_dname(v) for v in variants])
    ax.set_xlabel(ylabel)
    ax.set_title(title, pad=6)
    _grid(ax)
    ax.yaxis.grid(False)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Belief quality multi-panel bar
# ---------------------------------------------------------------------------

def _plot_belief_quality(
    summary: Dict,
    all_results: Dict[str, List[Dict]],
    ordered_variants: List[str],
    save_path: str,
) -> None:
    _apply_paper_style()

    metrics = [
        ("avg_belief_kl",                    r"KL(exact $\|$ EBM)",   False),
        ("avg_belief_l1",                    "L1 total variation",    False),
        ("avg_belief_map_to_exact_distance", "MAP-to-exact dist.",    False),
        ("avg_belief_map_to_true_distance",  "MAP-to-true dist.",     False),
    ]
    ebm_variants = [
        v for v in ordered_variants
        if any(f"{v}_{m}" in summary for m, _, _ in metrics)
    ]
    if not ebm_variants:
        return

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.2), sharey=True)
    y_pos = np.arange(len(ebm_variants))[::-1]
    bar_height = 0.55

    for ax, (field, label, higher_better) in zip(axes, metrics):
        for i, v in enumerate(ebm_variants):
            key = f"{v}_{field}"
            if key not in summary:
                continue
            stats = summary[key]
            mean = stats["mean"]
            ci_half = max(0.0, stats["ci95_high"] - mean)
            c = _color(v)
            yi = y_pos[i]
            ax.barh(yi, mean, height=bar_height, color=c, alpha=0.82)
            ax.errorbar(mean, yi, xerr=ci_half,
                        fmt="none", color="#333333", linewidth=0.9, capsize=3.0)
            if v in all_results:
                seed_vals = [r["eval"].get(field)
                             for r in all_results[v] if field in r.get("eval", {})]
                if seed_vals:
                    jitter = np.linspace(-bar_height * 0.28, bar_height * 0.28,
                                         max(len(seed_vals), 1))
                    ax.scatter(seed_vals, [yi + j for j in jitter],
                               color=c, edgecolors="white", linewidths=0.4,
                               s=20, zorder=5, marker="o")

        ax.set_xlabel(label, fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_dname(v) for v in ebm_variants])
        _grid(ax)
        ax.yaxis.grid(False)

    fig.suptitle("EBM Belief Quality", y=1.01, fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Training learning curves
# ---------------------------------------------------------------------------

def _plot_learning_curves(
    plot_data: Dict[str, Dict[str, np.ndarray]],
    ordered_variants: List[str],
    save_path: str,
) -> None:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    for v in ordered_variants:
        if v not in plot_data:
            continue
        rec = plot_data[v]
        x = np.arange(1, len(rec["train_mean"]) + 1)
        m, s = rec["train_mean"], rec["train_se"]
        c = _color(v)
        ax.plot(x, m, label=_dname(v), color=c, linestyle=_ls(v), linewidth=1.2)
        ax.fill_between(x, m - 1.96 * s, m + 1.96 * s, color=c, alpha=0.13)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Cumulative return")
    ax.set_title("Training learning curves", pad=6)
    _grid(ax)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="#cccccc",
              loc="lower right", ncol=1)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Scatter: return vs. SPCE
# ---------------------------------------------------------------------------

def _plot_return_vs_spce(
    summary: Dict,
    all_results: Dict[str, List[Dict]],
    ordered_variants: List[str],
    save_path: str,
) -> None:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))

    for v in ordered_variants:
        ret_key = f"{v}_avg_return"
        spce_key = f"{v}_avg_spce_lower"
        if ret_key not in summary or spce_key not in summary:
            continue
        c = _color(v)
        xr = summary[ret_key]["mean"]
        yr = summary[spce_key]["mean"]
        xe = max(0.0, summary[ret_key]["ci95_high"] - xr)
        ye = max(0.0, summary[spce_key]["ci95_high"] - yr)

        # Per-seed cloud (small dots)
        if v in all_results:
            xs = [r["eval"].get("avg_return") for r in all_results[v]
                  if "avg_return" in r.get("eval", {})]
            ys = [r["eval"].get("avg_spce_lower") for r in all_results[v]
                  if "avg_spce_lower" in r.get("eval", {})]
            ax.scatter(xs, ys, color=c, s=16, alpha=0.45, zorder=3,
                       marker="o", edgecolors="none")

        ax.errorbar(xr, yr, xerr=xe, yerr=ye,
                    fmt="o", color=c, markersize=9, capsize=3.5, linewidth=0.9,
                    zorder=5, label=_dname(v), markeredgecolor="white",
                    markeredgewidth=0.7)
        ax.annotate(
            _dname(v), (xr, yr),
            xytext=(7, 5), textcoords="offset points",
            fontsize=7.5, color=c,
        )

    ax.set_xlabel("Avg. evaluation return")
    ax.set_ylabel("Avg. SPCE lower bound")
    ax.set_title("Return vs. SPCE lower bound", pad=6)
    _grid(ax)
    ax.legend(frameon=True, framealpha=0.92, edgecolor="#cccccc",
              loc="best", fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Combined overview panel (for main paper figure)
# ---------------------------------------------------------------------------

def _plot_combined_overview(
    plot_data: Dict[str, Dict[str, np.ndarray]],
    summary: Dict,
    ordered_variants: List[str],
    save_path: str,
    horizon: int,
) -> None:
    """Three-panel figure: SPCE path | return bar | return-vs-SPCE scatter."""
    _apply_paper_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.0, 3.8))
    steps = np.arange(1, horizon + 1)

    # Panel 1: SPCE trajectory
    for v in ordered_variants:
        if v not in plot_data or "spce_mean" not in plot_data[v]:
            continue
        rec = plot_data[v]
        m, s = rec["spce_mean"], rec["spce_se"]
        c = _color(v)
        ax1.plot(steps, m, label=_dname(v), color=c,
                 linestyle=_ls(v), linewidth=1.2)
        ax1.fill_between(steps, m - 1.96 * s, m + 1.96 * s, color=c, alpha=0.13)
        ax1.scatter([steps[-1]], [m[-1]], color=c, s=22, zorder=5,
                    marker="o", edgecolors="white", linewidths=0.6)
    ax1.set_xlabel("Experiment step $t$")
    ax1.set_ylabel("SPCE lower bound (nats)")
    ax1.set_title("(a) SPCE lower bound vs. step", pad=6)
    _grid(ax1)

    # Panel 2: Eval return bars
    variants_with_return = [v for v in ordered_variants if f"{v}_avg_return" in summary]
    y_pos2 = np.arange(len(variants_with_return))[::-1]
    for i, v in enumerate(variants_with_return):
        stats = summary[f"{v}_avg_return"]
        mean = stats["mean"]
        ci_half = max(0.0, stats["ci95_high"] - mean)
        c = _color(v)
        yi = y_pos2[i]
        ax2.barh(yi, mean, height=0.52, color=c, alpha=0.82)
        ax2.errorbar(mean, yi, xerr=ci_half,
                     fmt="none", color="#333333", linewidth=0.9, capsize=3.0)
        ax2.text(mean + ci_half + 0.001, yi,
                 f"{mean:.3f}", va="center", ha="left", fontsize=8)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels([_dname(v) for v in variants_with_return])
    ax2.set_xlabel("Avg. evaluation return")
    ax2.set_title("(b) Evaluation return", pad=6)
    _grid(ax2)
    ax2.yaxis.grid(False)

    # Panel 3: Return vs. SPCE scatter
    for v in ordered_variants:
        rk, sk = f"{v}_avg_return", f"{v}_avg_spce_lower"
        if rk not in summary or sk not in summary:
            continue
        xr, yr = summary[rk]["mean"], summary[sk]["mean"]
        c = _color(v)
        ax3.scatter([xr], [yr], color=c, marker="o", s=60, zorder=4,
                    edgecolors="white", linewidths=0.7, label=_dname(v))
        ax3.annotate(_dname(v), (xr, yr),
                     xytext=(5, 4), textcoords="offset points",
                     fontsize=7.5, color=c)
    ax3.set_xlabel("Avg. evaluation return")
    ax3.set_ylabel("Avg. SPCE lower bound")
    ax3.set_title("(c) Return vs. SPCE", pad=6)
    _grid(ax3)

    handles = [
        mpatches.Patch(color=_color(v), label=_dname(v))
        for v in ordered_variants if v in plot_data
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 5), fontsize=8,
               frameon=True, framealpha=0.92, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Standard plots (lightweight, called by run_experiment_suite)
# ---------------------------------------------------------------------------

def save_standard_plots(
    output_dir: str,
    all_results: Dict[str, List[Dict]],
    summary: Dict,
    horizon: int,
) -> None:
    """Quick overview plots in the experiment output directory (PNG only, 220 dpi)."""
    from boedx.trainer import aggregate_plot_data  # avoid circular import

    plot_data = aggregate_plot_data(all_results)
    ordered_variants: List[str] = list(summary.get("variants", list(plot_data.keys())))

    colors = {v: _color(v) for v in ordered_variants}

    def dname(v: str) -> str:
        return _DISPLAY_NAMES.get(v, v)

    def styled_axes(ax: plt.Axes) -> None:
        ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.45)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)

    # Training curves
    plt.figure(figsize=(10.5, 6.2))
    ax = plt.gca()
    for v in ordered_variants:
        if v not in plot_data:
            continue
        rec = plot_data[v]
        x = np.arange(1, len(rec["train_mean"]) + 1)
        c = colors[v]
        ax.plot(x, rec["train_mean"], label=dname(v), color=c,
                linewidth=1.4, linestyle=_ls(v))
        ax.fill_between(x, rec["train_mean"] - rec["train_se"],
                        rec["train_mean"] + rec["train_se"], color=c, alpha=0.15)
        ax.scatter([x[-1]], [rec["train_mean"][-1]], color=c,
                   s=22, zorder=5, marker="o", edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Training return")
    ax.set_title("Learning curves")
    styled_axes(ax)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=220)
    plt.close()

    steps = np.arange(1, horizon + 1)
    for key, title, fname, ylabel in [
        ("bank",        "Posterior bank IG path",      "bank_ig_paths.png",        "Posterior bank IG"),
        ("filter_bank", "Control/filter bank IG path", "filter_bank_ig_paths.png", "Control/filter bank IG"),
        ("spce",        "SPCE lower path",             "spce_lower_paths.png",     "SPCE lower"),
    ]:
        plt.figure(figsize=(10.5, 6.2))
        ax = plt.gca()
        for v in ordered_variants:
            if v not in plot_data:
                continue
            rec = plot_data[v]
            if f"{key}_mean" not in rec:
                continue
            m, s = rec[f"{key}_mean"], rec[f"{key}_se"]
            c = colors[v]
            ax.plot(steps, m, label=dname(v), color=c,
                    linewidth=1.4, linestyle=_ls(v))
            ax.fill_between(steps, m - s, m + s, color=c, alpha=0.15)
            ax.scatter([steps[-1]], [m[-1]], color=c, s=22, zorder=5,
                       marker="o", edgecolors="white", linewidths=0.5)
        ax.set_xlabel("# Experiments")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        styled_axes(ax)
        ax.legend(frameon=True, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=220)
        plt.close()

    if any("snmc_mean" in rec for rec in plot_data.values()):
        plt.figure(figsize=(10.5, 6.2))
        ax = plt.gca()
        for v in ordered_variants:
            if v not in plot_data or "snmc_mean" not in plot_data[v]:
                continue
            rec = plot_data[v]
            m, s = rec["snmc_mean"], rec["snmc_se"]
            c = colors[v]
            ax.plot(steps, m, label=dname(v), color=c,
                    linewidth=1.4, linestyle=_ls(v))
            ax.fill_between(steps, m - s, m + s, color=c, alpha=0.15)
            ax.scatter([steps[-1]], [m[-1]], color=c, s=22, zorder=5,
                       marker="o", edgecolors="white", linewidths=0.5)
        ax.set_xlabel("# Experiments")
        ax.set_ylabel("SNMC-style upper")
        ax.set_title("SNMC-style upper path")
        styled_axes(ax)
        ax.legend(frameon=True, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "snmc_style_upper_paths.png"), dpi=220)
        plt.close()

    def barplot(metric_keys: List[str], labels: List[str], title: str, fname: str) -> None:
        xs = np.arange(len(labels))
        vals = [summary[k]["mean"] for k in metric_keys]
        errs = [max(0.0, summary[k]["ci95_high"] - summary[k]["mean"])
                for k in metric_keys]
        plt.figure(figsize=(max(8.5, 1.55 * len(labels)), 6.0))
        ax = plt.gca()
        bars = ax.bar(xs, vals, yerr=errs, capsize=4,
                      color=[colors.get(v, "#8B9BB5") for v in labels], alpha=0.88)
        ax.set_xticks(xs)
        ax.set_xticklabels([dname(v) for v in labels], rotation=18, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title)
        styled_axes(ax)
        for bar, val in zip(bars, vals):
            ax.annotate(
                f"{val:.3f}",
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=220)
        plt.close()

    vlist = ordered_variants
    barplot([f"{v}_avg_return" for v in vlist], vlist,
            "Evaluation return", "eval_return_bars.png")
    if all(f"{v}_avg_bank_ig" in summary for v in vlist):
        barplot([f"{v}_avg_bank_ig" for v in vlist], vlist,
                "Final posterior bank IG", "eval_bank_ig_bars.png")
    if all(f"{v}_avg_filter_bank_ig" in summary for v in vlist):
        barplot([f"{v}_avg_filter_bank_ig" for v in vlist], vlist,
                "Final control/filter bank IG", "eval_filter_bank_ig_bars.png")
    if all(f"{v}_avg_spce_lower" in summary for v in vlist):
        barplot([f"{v}_avg_spce_lower" for v in vlist], vlist,
                "Final SPCE lower", "eval_spce_lower_bars.png")
    if all(f"{v}_avg_snmc_style_upper" in summary for v in vlist):
        barplot([f"{v}_avg_snmc_style_upper" for v in vlist], vlist,
                "Final SNMC-style upper", "eval_snmc_style_upper_bars.png")

    if all((f"{v}_avg_return" in summary and f"{v}_avg_spce_lower" in summary)
           for v in vlist):
        plt.figure(figsize=(7.0, 5.8))
        ax = plt.gca()
        for v in vlist:
            xr = summary[f"{v}_avg_return"]["mean"]
            yr = summary[f"{v}_avg_spce_lower"]["mean"]
            ax.scatter(xr, yr, label=dname(v), s=110,
                       color=colors.get(v), edgecolor="white", linewidth=0.7,
                       marker="o")
            ax.annotate(f"{dname(v)}\n({xr:.3f}, {yr:.3f})", (xr, yr),
                        xytext=(8, 8), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Avg return")
        ax.set_ylabel("Avg SPCE lower")
        ax.set_title("Return vs SPCE lower")
        styled_axes(ax)
        ax.legend(frameon=True, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "return_vs_spce_scatter.png"), dpi=220)
        plt.close()


# ---------------------------------------------------------------------------
# Main public function: generate_scientific_plots
# ---------------------------------------------------------------------------

def generate_scientific_plots(
    output_dir: str,
    graph_dir: Optional[str] = None,
) -> str:
    """Load saved results and produce publication-ready figures.

    Reads all ``<variant>/seed_<s>/result.json`` files and the
    ``summary_multi_seed.json`` from *output_dir*, then writes a full set of
    figures to *graph_dir* (PNG at 300 dpi + vector PDF).

    New figures beyond the original catalogue:
      - ``per_seed_spce_paths``      — individual seed SPCE traces + mean
      - ``per_seed_bank_ig_paths``   — individual seed bank-IG traces + mean
      - ``ig_gap_spce``              — SPCE gap relative to blau_approx per step
      - ``ig_gap_bank_ig``           — bank-IG gap relative to blau_approx
      - ``belief_vs_spce``           — EBM belief KL vs. SPCE lower (per seed)
      - ``belief_map_vs_spce``       — EBM MAP-to-exact vs. SPCE lower (per seed)
      - ``paired_diff_return``       — paired return gain over baseline
    """
    from boedx.trainer import aggregate_plot_data, load_results_from_dir

    output_dir = os.path.abspath(output_dir)
    if graph_dir is None:
        base_name = os.path.basename(output_dir.rstrip("/\\"))
        parent = os.path.dirname(output_dir)
        graph_dir = os.path.join(parent, f"graph_{base_name}")

    os.makedirs(graph_dir, exist_ok=True)
    print(f"[generate_scientific_plots] output_dir = {output_dir}")
    print(f"[generate_scientific_plots] graph_dir  = {graph_dir}")

    all_results, summary = load_results_from_dir(output_dir)
    plot_data = aggregate_plot_data(all_results)
    ordered_variants: List[str] = list(summary.get("variants", list(plot_data.keys())))
    horizon: int = int(
        summary.get("horizon",
                    len(next(iter(plot_data.values()))["spce_mean"]))
    )
    baseline = "blau_approx"

    gp = os.path.join  # shortcut

    # 1. Training learning curves
    _plot_learning_curves(
        plot_data, ordered_variants,
        save_path=gp(graph_dir, "learning_curves"),
    )

    # 2. Trajectory paths (mean ± CI)
    _plot_trajectory(
        plot_data, ordered_variants,
        key="spce", xlabel="Experiment step $t$",
        ylabel="SPCE lower bound (nats)",
        title="SPCE lower bound vs. experiment step",
        save_path=gp(graph_dir, "spce_lower_paths"), horizon=horizon,
    )
    _plot_trajectory(
        plot_data, ordered_variants,
        key="bank", xlabel="Experiment step $t$",
        ylabel="Posterior bank IG (nats)",
        title="Posterior bank information gain vs. step",
        save_path=gp(graph_dir, "bank_ig_paths"), horizon=horizon,
    )
    _plot_trajectory(
        plot_data, ordered_variants,
        key="filter_bank", xlabel="Experiment step $t$",
        ylabel="Control/filter bank IG (nats)",
        title="Control/filter bank IG vs. step",
        save_path=gp(graph_dir, "filter_bank_ig_paths"), horizon=horizon,
    )
    if any("snmc_mean" in rec for rec in plot_data.values()):
        _plot_trajectory(
            plot_data, ordered_variants,
            key="snmc", xlabel="Experiment step $t$",
            ylabel="SNMC-style upper bound (nats)",
            title="SNMC-style upper bound vs. step",
            save_path=gp(graph_dir, "snmc_upper_paths"), horizon=horizon,
        )

    # 3. Per-seed fan plots — individual seed traces + mean
    _plot_per_seed_trajectories(
        all_results, ordered_variants,
        path_key="spce_lower_mean_path",
        ylabel="SPCE lower bound (nats)",
        title="SPCE lower — per-seed traces",
        save_path=gp(graph_dir, "per_seed_spce_paths"), horizon=horizon,
    )
    _plot_per_seed_trajectories(
        all_results, ordered_variants,
        path_key="bank_ig_mean_path",
        ylabel="Posterior bank IG (nats)",
        title="Posterior bank IG — per-seed traces",
        save_path=gp(graph_dir, "per_seed_bank_ig_paths"), horizon=horizon,
    )

    # 4. IG gap plots — (variant − baseline) per step
    _plot_ig_gap(
        all_results, ordered_variants, baseline_variant=baseline,
        path_key="spce_lower_mean_path",
        ylabel="ΔSPCE lower bound vs. Blau (nats)",
        title="SPCE lower: gain over baseline per step",
        save_path=gp(graph_dir, "ig_gap_spce"), horizon=horizon,
    )
    _plot_ig_gap(
        all_results, ordered_variants, baseline_variant=baseline,
        path_key="bank_ig_mean_path",
        ylabel="ΔBank IG vs. Blau (nats)",
        title="Posterior bank IG: gain over baseline per step",
        save_path=gp(graph_dir, "ig_gap_bank_ig"), horizon=horizon,
    )

    # 5. Belief quality vs. performance scatter
    _plot_belief_performance_scatter(
        all_results, ordered_variants,
        belief_metric="avg_belief_kl",
        belief_label=r"EBM belief KL (exact $\|$ EBM)",
        perf_metric="avg_spce_lower",
        perf_label="SPCE lower bound (nats)",
        title="Belief quality vs. information gain",
        save_path=gp(graph_dir, "belief_vs_spce"),
        belief_lower_better=True,
    )
    _plot_belief_performance_scatter(
        all_results, ordered_variants,
        belief_metric="avg_belief_map_to_exact_distance",
        belief_label="EBM MAP-to-exact distance",
        perf_metric="avg_spce_lower",
        perf_label="SPCE lower bound (nats)",
        title="EBM MAP accuracy vs. information gain",
        save_path=gp(graph_dir, "belief_map_vs_spce"),
        belief_lower_better=True,
    )

    # 6. Paired-difference bar chart
    _plot_paired_differences(
        all_results, summary, ordered_variants,
        baseline_variant=baseline,
        metric="avg_return",
        xlabel="Paired Δ return vs. Blau (nats)",
        title="Paired return gain over baseline (per seed)",
        save_path=gp(graph_dir, "paired_diff_return"),
    )

    # 7. Bar comparisons
    _plot_bar_comparison(
        summary, all_results, ordered_variants,
        metric_field="avg_return",
        ylabel="Avg. evaluation return",
        title="Evaluation return (mean ± 95 % CI)",
        save_path=gp(graph_dir, "eval_return_bars"),
    )
    if all(f"{v}_avg_spce_lower" in summary for v in ordered_variants):
        _plot_bar_comparison(
            summary, all_results, ordered_variants,
            metric_field="avg_spce_lower",
            ylabel="Avg. final SPCE lower bound (nats)",
            title="Final SPCE lower bound (mean ± 95 % CI)",
            save_path=gp(graph_dir, "eval_spce_lower_bars"),
        )
    if any(f"{v}_avg_snmc_style_upper" in summary for v in ordered_variants):
        snmc_vars = [v for v in ordered_variants
                     if f"{v}_avg_snmc_style_upper" in summary]
        _plot_bar_comparison(
            summary, all_results, snmc_vars,
            metric_field="avg_snmc_style_upper",
            ylabel="Avg. final SNMC-style upper (nats)",
            title="Final SNMC-style upper (mean ± 95 % CI)",
            save_path=gp(graph_dir, "eval_snmc_upper_bars"),
        )
    if any(f"{v}_avg_bank_ig" in summary for v in ordered_variants):
        _plot_bar_comparison(
            summary, all_results, ordered_variants,
            metric_field="avg_bank_ig",
            ylabel="Avg. final posterior bank IG (nats)",
            title="Final posterior bank IG (mean ± 95 % CI)",
            save_path=gp(graph_dir, "eval_bank_ig_bars"),
        )
    if any(f"{v}_avg_filter_bank_ig" in summary for v in ordered_variants):
        _plot_bar_comparison(
            summary, all_results, ordered_variants,
            metric_field="avg_filter_bank_ig",
            ylabel="Avg. final control/filter bank IG (nats)",
            title="Final control/filter bank IG (mean ± 95 % CI)",
            save_path=gp(graph_dir, "eval_filter_bank_ig_bars"),
        )

    # 8. Belief quality multi-panel
    _plot_belief_quality(
        summary, all_results, ordered_variants,
        save_path=gp(graph_dir, "belief_quality"),
    )

    # 9. Scatter: return vs. SPCE (with per-seed cloud)
    _plot_return_vs_spce(
        summary, all_results, ordered_variants,
        save_path=gp(graph_dir, "return_vs_spce"),
    )

    # 10. Combined overview panel
    _plot_combined_overview(
        plot_data, summary, ordered_variants,
        save_path=gp(graph_dir, "overview_combined"), horizon=horizon,
    )

    n_files = len(os.listdir(graph_dir))
    print(f"[generate_scientific_plots] Done — {n_files} files written.")
    return graph_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_generate_plots() -> None:
    """``boedx-plot <output_dir> [graph_dir]`` — generate scientific plots."""
    if len(sys.argv) < 2:
        print("Usage: boedx-plot <output_dir> [graph_dir]", file=sys.stderr)
        sys.exit(1)
    output_dir = sys.argv[1]
    graph_dir = sys.argv[2] if len(sys.argv) > 2 else None
    generate_scientific_plots(output_dir, graph_dir)


if __name__ == "__main__":
    _cli_generate_plots()
