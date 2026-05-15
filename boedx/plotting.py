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

Scientific plot catalogue
-------------------------
Trajectory plots (mean ± 95 % CI band, colour + linestyle):
  - SPCE lower bound vs. experiment step
  - Posterior bank information gain vs. step
  - Control/filter bank information gain vs. step
  - SNMC-style upper bound vs. step  (if available)

Bar comparisons (horizontal, individual seed points overlaid):
  - Evaluation return
  - Final SPCE lower
  - Final SNMC upper  (if available)
  - Final posterior bank IG
  - Final control/filter bank IG

Belief quality (EBM variants, multi-panel bar):
  - KL(posterior ‖ EBM belief)
  - L1 total variation
  - MAP-to-exact distance
  - MAP-to-true distance

Summary overview (combined panel for main results section)
  - Training return curves

Pairwise scatter:
  - Eval return vs. final SPCE
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

# Okabe–Ito 8-colour palette — designed for colourblind accessibility.
# See: https://jfly.uni-koeln.de/color/
_OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow (use sparingly)
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# Canonical variant-level style assignments.
_VARIANT_STYLES: Dict[str, Dict] = {
    "blau_approx":                        {"color": "#0072B2", "ls": "-",         "marker": "o"},
    "control_filter_exact":               {"color": "#000000", "ls": "--",        "marker": "x"},
    "control_posterior_exact":            {"color": "#000000", "ls": "-.",        "marker": "+"},
    "ours_ebm_control":                   {"color": "#E69F00", "ls": "--",        "marker": "s"},
    "ours_ebm_cross":                     {"color": "#009E73", "ls": "--",        "marker": "D"},
    "ours_ebm_control_filter":            {"color": "#E69F00", "ls": "-.",        "marker": "s"},
    "ours_ebm_control_posterior":         {"color": "#E69F00", "ls": "--",        "marker": "s"},
    "ours_ebm_cross_filter":              {"color": "#009E73", "ls": "-.",        "marker": "D"},
    "ours_ebm_cross_posterior":           {"color": "#009E73", "ls": "-",         "marker": "D"},
    "ours_ebm_control_beta_contrastive":  {"color": "#D55E00", "ls": (0, (5, 2)), "marker": "^"},
    "ours_ebm_cross_beta_contrastive":    {"color": "#CC79A7", "ls": (0, (3, 1, 1, 1)), "marker": "v"},
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


def _dname(v: str) -> str:
    return _DISPLAY_NAMES.get(v, v)


def _vstyle(v: str) -> Dict:
    return _VARIANT_STYLES.get(
        v,
        {"color": _OKABE_ITO[hash(v) % len(_OKABE_ITO)], "ls": "-", "marker": "o"},
    )


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
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.8,
    "lines.markersize":  5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "pdf.fonttype":      42,   # embed fonts as Type 1 (required by many journals)
    "ps.fonttype":       42,
}


def _apply_paper_style() -> None:
    plt.rcParams.update(_PAPER_RC)


def _clean_axes(ax: plt.Axes) -> None:
    """Remove top/right spines, add faint horizontal grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path_no_ext: str) -> None:
    """Save as both PNG (300 dpi) and PDF (vector)."""
    fig.savefig(path_no_ext + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trajectory (path) plots — scientific version
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
    """Line plot with 95 % CI band for one trajectory metric across variants."""
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    steps = np.arange(1, horizon + 1)

    for v in ordered_variants:
        if v not in plot_data:
            continue
        rec = plot_data[v]
        m, s = rec[f"{key}_mean"], rec[f"{key}_se"]
        sty = _vstyle(v)
        label = f"{_dname(v)}  [{m[-1]:.3f}]"
        ax.plot(steps, m, label=label, color=sty["color"], linestyle=sty["ls"], linewidth=1.8)
        ax.fill_between(steps, m - 1.96 * s, m + 1.96 * s, color=sty["color"], alpha=0.13)
        # Final-step dot
        ax.scatter([steps[-1]], [m[-1]], color=sty["color"], s=28, zorder=5, clip_on=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    _clean_axes(ax)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc",
              loc="best", ncol=1)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Bar plots — scientific version with seed-point overlay
# ---------------------------------------------------------------------------

def _plot_bar_comparison(
    summary: Dict,
    all_results: Dict[str, List[Dict]],
    ordered_variants: List[str],
    metric_field: str,   # e.g. "avg_return"
    ylabel: str,
    title: str,
    save_path: str,
    higher_is_better: bool = True,
    baseline_variant: str = "blau_approx",
) -> None:
    """Horizontal bar chart with 95 % CI error bars and per-seed scatter.

    Visual encoding:
      - Bar length:  cross-seed mean.
      - Error bar:   ±(95 % CI half-width).
      - Dots:        individual seed values (jittered vertically).
      - Star (*):    paired difference vs. Blau baseline is significant at
                     p ≈ 0.05 (CI does not include zero).
    """
    _apply_paper_style()

    # Filter to variants that have this metric
    variants = [v for v in ordered_variants if f"{v}_{metric_field}" in summary]
    if not variants:
        return

    n = len(variants)
    fig, ax = plt.subplots(figsize=(6.5, 0.62 * n + 1.2))

    y_pos = np.arange(n)[::-1]  # top variant at top
    bar_height = 0.52

    blau_mean = summary.get(f"{baseline_variant}_{metric_field}", {}).get("mean")

    for i, v in enumerate(variants):
        key = f"{v}_{metric_field}"
        stats = summary[key]
        mean = stats["mean"]
        ci_half = max(0.0, stats["ci95_high"] - mean)
        sty = _vstyle(v)
        yi = y_pos[i]

        # Bar
        ax.barh(
            yi, mean, height=bar_height,
            color=sty["color"], alpha=0.80, zorder=2,
        )
        # Error bar (95 % CI)
        ax.errorbar(
            mean, yi, xerr=ci_half,
            fmt="none", color="black", linewidth=1.2, capsize=3.5, zorder=4,
        )
        # Mean value label
        offset = ci_half + 0.004 * abs(mean) + 1e-4
        ax.text(
            mean + offset, yi,
            f"{mean:.3f} ±{ci_half:.3f}",
            va="center", ha="left", fontsize=8, color="#333333",
        )

        # Individual seed dots (jittered vertically within the bar)
        if v in all_results:
            seed_vals = [r["eval"].get(metric_field) for r in all_results[v] if metric_field in r["eval"]]
            jitter = np.linspace(-bar_height * 0.28, bar_height * 0.28, len(seed_vals))
            ax.scatter(
                seed_vals, [yi + j for j in jitter],
                color=sty["color"], edgecolors="white", linewidths=0.5,
                s=28, zorder=5, alpha=0.9,
            )

        # Significance marker vs. baseline
        diff_key = f"paired_return_diff_{v}_minus_{baseline_variant}"
        if v != baseline_variant and diff_key in summary:
            diff_stats = summary[diff_key]
            ci_lo = diff_stats.get("ci95_low", 0.0)
            ci_hi = diff_stats.get("ci95_high", 0.0)
            sig = (ci_lo > 0.0) if higher_is_better else (ci_hi < 0.0)
            if sig:
                ax.text(
                    mean, yi + bar_height * 0.55, "*",
                    ha="center", va="bottom", fontsize=11, color="black", fontweight="bold",
                )

    # Baseline reference line
    if blau_mean is not None and baseline_variant in variants:
        ax.axvline(blau_mean, color="#0072B2", linestyle=":", linewidth=1.0,
                   alpha=0.55, label=f"{_dname(baseline_variant)} mean")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([_dname(v) for v in variants])
    ax.set_xlabel(ylabel)
    ax.set_title(title, pad=6)
    _clean_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
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
    """Four-panel comparison of EBM belief quality metrics (EBM variants only)."""
    _apply_paper_style()

    metrics = [
        ("avg_belief_kl",                    r"KL(exact $\|$ EBM)",   False),
        ("avg_belief_l1",                    "L1 total variation",    False),
        ("avg_belief_map_to_exact_distance", "MAP-to-exact dist.",    False),
        ("avg_belief_map_to_true_distance",  "MAP-to-true dist.",     False),
    ]
    # EBM variants only (Blau has no EBM belief)
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
            sty = _vstyle(v)
            yi = y_pos[i]
            ax.barh(yi, mean, height=bar_height, color=sty["color"], alpha=0.80)
            ax.errorbar(mean, yi, xerr=ci_half, fmt="none", color="black",
                        linewidth=1.0, capsize=3.0)
            # Seed dots
            if v in all_results:
                seed_vals = [r["eval"].get(field) for r in all_results[v] if field in r["eval"]]
                jitter = np.linspace(-bar_height * 0.28, bar_height * 0.28, max(len(seed_vals), 1))
                ax.scatter(seed_vals, [yi + j for j in jitter],
                           color=sty["color"], edgecolors="white", linewidths=0.4, s=22, zorder=5)

        ax.set_xlabel(label, fontsize=9)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_dname(v) for v in ebm_variants])
        _clean_axes(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

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
    """Training episode-return curves with 95 % CI band."""
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    for v in ordered_variants:
        if v not in plot_data:
            continue
        rec = plot_data[v]
        x = np.arange(1, len(rec["train_mean"]) + 1)
        m, s = rec["train_mean"], rec["train_se"]
        sty = _vstyle(v)
        ax.plot(x, m, label=_dname(v), color=sty["color"], linestyle=sty["ls"], linewidth=1.6)
        ax.fill_between(x, m - 1.96 * s, m + 1.96 * s, color=sty["color"], alpha=0.12)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Cumulative return")
    ax.set_title("Training learning curves", pad=6)
    _clean_axes(ax)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc",
              loc="lower right", ncol=1)
    fig.tight_layout()
    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Scatter: return vs. SPCE
# ---------------------------------------------------------------------------

def _plot_return_vs_spce(
    summary: Dict,
    ordered_variants: List[str],
    save_path: str,
) -> None:
    """Scatter plot of eval return vs. final SPCE lower bound with error bars."""
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    for v in ordered_variants:
        ret_key = f"{v}_avg_return"
        spce_key = f"{v}_avg_spce_lower"
        if ret_key not in summary or spce_key not in summary:
            continue
        xr = summary[ret_key]["mean"]
        yr = summary[spce_key]["mean"]
        xe = max(0.0, summary[ret_key]["ci95_high"] - xr)
        ye = max(0.0, summary[spce_key]["ci95_high"] - yr)
        sty = _vstyle(v)
        ax.errorbar(
            xr, yr, xerr=xe, yerr=ye,
            fmt=sty["marker"], color=sty["color"],
            markersize=8, capsize=3.5, linewidth=1.0,
            label=_dname(v),
        )
        ax.annotate(
            _dname(v),
            (xr, yr),
            xytext=(7, 5), textcoords="offset points",
            fontsize=7.5, color=sty["color"],
        )

    ax.set_xlabel("Avg. evaluation return")
    ax.set_ylabel("Avg. SPCE lower bound")
    ax.set_title("Return vs. SPCE lower bound", pad=6)
    _clean_axes(ax)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc",
              loc="best", fontsize=8, ncol=1)
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

    # -- Panel 1: SPCE trajectory --
    for v in ordered_variants:
        if v not in plot_data or "spce_mean" not in plot_data[v]:
            continue
        rec = plot_data[v]
        m, s = rec["spce_mean"], rec["spce_se"]
        sty = _vstyle(v)
        ax1.plot(steps, m, label=_dname(v), color=sty["color"], linestyle=sty["ls"], linewidth=1.8)
        ax1.fill_between(steps, m - 1.96 * s, m + 1.96 * s, color=sty["color"], alpha=0.12)
    ax1.set_xlabel("Experiment step")
    ax1.set_ylabel("SPCE lower bound")
    ax1.set_title("(a) SPCE lower bound vs. step", pad=6)
    _clean_axes(ax1)

    # -- Panel 2: Eval return bars (horizontal) --
    variants_with_return = [v for v in ordered_variants if f"{v}_avg_return" in summary]
    y_pos2 = np.arange(len(variants_with_return))[::-1]
    for i, v in enumerate(variants_with_return):
        stats = summary[f"{v}_avg_return"]
        mean = stats["mean"]
        ci_half = max(0.0, stats["ci95_high"] - mean)
        sty = _vstyle(v)
        yi = y_pos2[i]
        ax2.barh(yi, mean, height=0.52, color=sty["color"], alpha=0.80)
        ax2.errorbar(mean, yi, xerr=ci_half, fmt="none",
                     color="black", linewidth=1.0, capsize=3.0)
        ax2.text(mean + ci_half + 0.001, yi,
                 f"{mean:.3f}", va="center", ha="left", fontsize=8)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels([_dname(v) for v in variants_with_return])
    ax2.set_xlabel("Avg. evaluation return")
    ax2.set_title("(b) Evaluation return", pad=6)
    _clean_axes(ax2)
    ax2.yaxis.grid(False)
    ax2.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    # -- Panel 3: Return vs. SPCE scatter --
    for v in ordered_variants:
        rk = f"{v}_avg_return"
        sk = f"{v}_avg_spce_lower"
        if rk not in summary or sk not in summary:
            continue
        xr, yr = summary[rk]["mean"], summary[sk]["mean"]
        sty = _vstyle(v)
        ax3.scatter([xr], [yr], color=sty["color"], marker=sty["marker"],
                    s=60, zorder=4, label=_dname(v))
        ax3.annotate(_dname(v), (xr, yr),
                     xytext=(5, 4), textcoords="offset points", fontsize=7.5,
                     color=sty["color"])
    ax3.set_xlabel("Avg. evaluation return")
    ax3.set_ylabel("Avg. SPCE lower bound")
    ax3.set_title("(c) Return vs. SPCE", pad=6)
    _clean_axes(ax3)

    handles = [
        mpatches.Patch(color=_vstyle(v)["color"], label=_dname(v))
        for v in ordered_variants if v in plot_data
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 5), fontsize=8,
               frameon=True, framealpha=0.9, edgecolor="#cccccc",
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
    """Generate quick overview plots in the experiment output directory.

    These are lower-resolution (220 dpi PNG only) plots intended for fast
    inspection during or after training, not for paper inclusion.
    """
    from boedx.trainer import aggregate_plot_data  # avoid circular import

    plot_data = aggregate_plot_data(all_results)
    ordered_variants: List[str] = list(summary.get("variants", list(plot_data.keys())))

    # -- Colour map (fall back to Okabe–Ito cycle) --
    colors = {v: _vstyle(v)["color"] for v in ordered_variants}

    def dname(v: str) -> str:
        return _DISPLAY_NAMES.get(v, v)

    def annotate_last(ax: plt.Axes, x: np.ndarray, y: np.ndarray, label: str, color: str) -> None:
        ax.scatter([x[-1]], [y[-1]], color=color, s=28, zorder=5)
        ax.annotate(
            f"{label}: {y[-1]:.3f}",
            xy=(x[-1], y[-1]),
            xytext=(8, 0), textcoords="offset points",
            color=color, fontsize=8, va="center",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, alpha=0.75),
        )

    def styled_axes(ax: plt.Axes) -> None:
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
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
        ax.plot(x, rec["train_mean"], label=dname(v), color=c, linewidth=2.0)
        ax.fill_between(x, rec["train_mean"] - rec["train_se"],
                        rec["train_mean"] + rec["train_se"], color=c, alpha=0.16)
        annotate_last(ax, x, rec["train_mean"], dname(v), c)
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
            m, s = rec[f"{key}_mean"], rec[f"{key}_se"]
            c = colors[v]
            ax.plot(steps, m, label=dname(v), color=c, linewidth=2.1)
            ax.fill_between(steps, m - s, m + s, color=c, alpha=0.16)
            annotate_last(ax, steps, m, dname(v), c)
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
            ax.plot(steps, m, label=dname(v), color=c, linewidth=2.1)
            ax.fill_between(steps, m - s, m + s, color=c, alpha=0.16)
            annotate_last(ax, steps, m, dname(v), c)
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
        errs = [max(0.0, summary[k]["ci95_high"] - summary[k]["mean"]) for k in metric_keys]
        plt.figure(figsize=(max(8.5, 1.55 * len(labels)), 6.0))
        ax = plt.gca()
        bars = ax.bar(xs, vals, yerr=errs, capsize=4,
                      color=[colors.get(v, "#999999") for v in labels], alpha=0.9)
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
    barplot([f"{v}_avg_return" for v in vlist], vlist, "Evaluation return", "eval_return_bars.png")
    if all(f"{v}_avg_bank_ig" in summary for v in vlist):
        barplot([f"{v}_avg_bank_ig" for v in vlist], vlist, "Final posterior bank IG", "eval_bank_ig_bars.png")
    if all(f"{v}_avg_filter_bank_ig" in summary for v in vlist):
        barplot([f"{v}_avg_filter_bank_ig" for v in vlist], vlist, "Final control/filter bank IG", "eval_filter_bank_ig_bars.png")
    if all(f"{v}_avg_spce_lower" in summary for v in vlist):
        barplot([f"{v}_avg_spce_lower" for v in vlist], vlist, "Final SPCE lower", "eval_spce_lower_bars.png")
    if all(f"{v}_avg_snmc_style_upper" in summary for v in vlist):
        barplot([f"{v}_avg_snmc_style_upper" for v in vlist], vlist, "Final SNMC-style upper", "eval_snmc_style_upper_bars.png")

    if all((f"{v}_avg_return" in summary and f"{v}_avg_spce_lower" in summary) for v in vlist):
        plt.figure(figsize=(7.0, 5.8))
        ax = plt.gca()
        for v in vlist:
            xr = summary[f"{v}_avg_return"]["mean"]
            yr = summary[f"{v}_avg_spce_lower"]["mean"]
            ax.scatter(xr, yr, label=dname(v), s=110, color=colors.get(v),
                       edgecolor="black", linewidth=0.5)
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
    figures to *graph_dir* (both PNG at 300 dpi and vector PDF).

    Args:
        output_dir: Path to the experiment output directory, e.g.
                    ``./outputs/source_location_beta_contrastive_h50``.
        graph_dir:  Optional explicit output path.  If omitted, the graph
                    directory is created as ``graph_<basename(output_dir)>``
                    as a sibling of *output_dir*.

    Returns:
        Absolute path to the graph directory.

    Example::

        generate_scientific_plots("./outputs/source_location_beta_contrastive_h50")
        # writes to: ./graph_source_location_beta_contrastive_h50/
    """
    from boedx.trainer import aggregate_plot_data, load_results_from_dir  # avoid circular

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
    horizon: int = int(summary.get("horizon", len(next(iter(plot_data.values()))["spce_mean"])))

    # ---- 1. Training learning curves ----
    _plot_learning_curves(
        plot_data, ordered_variants,
        save_path=os.path.join(graph_dir, "learning_curves"),
    )

    # ---- 2. Trajectory path plots ----
    _plot_trajectory(
        plot_data, ordered_variants,
        key="spce", xlabel="Experiment step $t$",
        ylabel="SPCE lower bound (nats)",
        title="SPCE lower bound vs. experiment step",
        save_path=os.path.join(graph_dir, "spce_lower_paths"),
        horizon=horizon,
    )
    _plot_trajectory(
        plot_data, ordered_variants,
        key="bank", xlabel="Experiment step $t$",
        ylabel="Posterior bank IG (nats)",
        title="Posterior bank information gain vs. step",
        save_path=os.path.join(graph_dir, "bank_ig_paths"),
        horizon=horizon,
    )
    _plot_trajectory(
        plot_data, ordered_variants,
        key="filter_bank", xlabel="Experiment step $t$",
        ylabel="Control/filter bank IG (nats)",
        title="Control/filter bank information gain vs. step",
        save_path=os.path.join(graph_dir, "filter_bank_ig_paths"),
        horizon=horizon,
    )
    if any("snmc_mean" in rec for rec in plot_data.values()):
        _plot_trajectory(
            plot_data, ordered_variants,
            key="snmc", xlabel="Experiment step $t$",
            ylabel="SNMC-style upper bound (nats)",
            title="SNMC-style upper bound vs. step",
            save_path=os.path.join(graph_dir, "snmc_upper_paths"),
            horizon=horizon,
        )

    # ---- 3. Bar comparisons ----
    _plot_bar_comparison(
        summary, all_results, ordered_variants,
        metric_field="avg_return",
        ylabel="Avg. evaluation return",
        title="Evaluation return (mean ± 95 % CI)",
        save_path=os.path.join(graph_dir, "eval_return_bars"),
        higher_is_better=True,
    )
    if all(f"{v}_avg_spce_lower" in summary for v in ordered_variants):
        _plot_bar_comparison(
            summary, all_results, ordered_variants,
            metric_field="avg_spce_lower",
            ylabel="Avg. final SPCE lower bound (nats)",
            title="Final SPCE lower bound (mean ± 95 % CI)",
            save_path=os.path.join(graph_dir, "eval_spce_lower_bars"),
            higher_is_better=True,
        )
    if any(f"{v}_avg_snmc_style_upper" in summary for v in ordered_variants):
        snmc_variants = [v for v in ordered_variants if f"{v}_avg_snmc_style_upper" in summary]
        _plot_bar_comparison(
            summary, all_results, snmc_variants,
            metric_field="avg_snmc_style_upper",
            ylabel="Avg. final SNMC-style upper bound (nats)",
            title="Final SNMC-style upper bound (mean ± 95 % CI)",
            save_path=os.path.join(graph_dir, "eval_snmc_upper_bars"),
            higher_is_better=True,
        )
    if any(f"{v}_avg_bank_ig" in summary for v in ordered_variants):
        _plot_bar_comparison(
            summary, all_results, ordered_variants,
            metric_field="avg_bank_ig",
            ylabel="Avg. final posterior bank IG (nats)",
            title="Final posterior bank IG (mean ± 95 % CI)",
            save_path=os.path.join(graph_dir, "eval_bank_ig_bars"),
            higher_is_better=True,
        )
    if any(f"{v}_avg_filter_bank_ig" in summary for v in ordered_variants):
        _plot_bar_comparison(
            summary, all_results, ordered_variants,
            metric_field="avg_filter_bank_ig",
            ylabel="Avg. final control/filter bank IG (nats)",
            title="Final control/filter bank IG (mean ± 95 % CI)",
            save_path=os.path.join(graph_dir, "eval_filter_bank_ig_bars"),
            higher_is_better=True,
        )

    # ---- 4. Belief quality ----
    _plot_belief_quality(
        summary, all_results, ordered_variants,
        save_path=os.path.join(graph_dir, "belief_quality"),
    )

    # ---- 5. Scatter: return vs. SPCE ----
    _plot_return_vs_spce(
        summary, ordered_variants,
        save_path=os.path.join(graph_dir, "return_vs_spce"),
    )

    # ---- 6. Combined overview (paper main figure) ----
    _plot_combined_overview(
        plot_data, summary, ordered_variants,
        save_path=os.path.join(graph_dir, "overview_combined"),
        horizon=horizon,
    )

    print(f"[generate_scientific_plots] Done — {len(os.listdir(graph_dir))} files written.")
    return graph_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli_generate_plots() -> None:
    """``boedx-plot <output_dir>`` — generate scientific plots from saved results."""
    if len(sys.argv) < 2:
        print("Usage: boedx-plot <output_dir> [graph_dir]", file=sys.stderr)
        sys.exit(1)
    output_dir = sys.argv[1]
    graph_dir = sys.argv[2] if len(sys.argv) > 2 else None
    generate_scientific_plots(output_dir, graph_dir)


if __name__ == "__main__":
    _cli_generate_plots()
