#!/usr/bin/env python3
"""Emit the Bellman-fidelity DATA fragment for prey_population_benchmark.html.

Reads only saved experiment artefacts. No training, no re-simulation.

    raw experiment JSON  ->  bellman_html_data.json  ->  single-file HTML

Per-seed arrays are emitted with their seed IDs so the page can align paired
comparisons by ID rather than by array position.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Artefact variant name -> page-facing identity. Names follow the repository.
METHODS = [
    {"id": "exact_state", "artifact": "posterior_direct",
     "name": "Exact state — Bellman reference", "short": "Exact S",
     "hex": "#0F2A44", "role": "reference"},
    {"id": "blau_history", "artifact": "blau_approx",
     "name": "Blau — raw flattened history", "short": "Blau history",
     "hex": "#8B9BB5", "role": "baseline"},
    {"id": "bellman_h1", "artifact": "posterior_bottleneck_bellman_h1",
     "name": "Bellman-H1 bottleneck", "short": "Bellman-H1",
     "hex": "#2D5FA8", "role": "bellman"},
]

# Page metric key -> path inside bellman_probe_error.by_direction.<depth>
METRICS = {
    "epsilon_q_inf": ("epsilon_Q_inf", "mean"),
    "mae_q": ("bellman_mae",),
    "advantage": ("epsilon_adv", "mean"),
    "selection_regret": ("selection_regret", "mean"),
}
DEPTHS = {"h1": "depth1", "h2": "depth2"}


def dig(node, path):
    for p in path:
        node = node[p]
    return node


def build(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "run_config_and_summary.json").read_text())
    cfg, rows = summary["config"], summary["results"]

    by_variant = {}
    for r in rows:
        by_variant.setdefault(r["variant"], {})[r["seed"]] = r

    present = [m for m in METHODS if m["artifact"] in by_variant]
    missing = [m["artifact"] for m in METHODS if m["artifact"] not in by_variant]
    if missing:
        raise SystemExit(f"BELLMAN_HTML_DATA_BLOCKED: variants absent from artefacts: {missing}")

    # Depths actually present in the artefacts, in order.
    sample = next(iter(by_variant[present[0]["artifact"]].values()))
    directions = sample["bellman_probe_error"]["by_direction"]
    depths = {k: v for k, v in DEPTHS.items() if v in directions}

    payload = {
        "experiment": {
            "name": run_dir.name,
            "artifact_path": f"outputs/{run_dir.name}/",
            "source_file": "run_config_and_summary.json",
            "commit": summary.get("git_sha"),
            "horizon": cfg["horizon"],
            "bank_size": cfg["bank_size"],
            "latent_dim": int(str(cfg["latent_dims"]).split(",")[0]),
            "train_episodes": cfg["episodes"],
            "eval_episodes": cfg["eval_episodes"],
            "seeds": sorted({r["seed"] for r in rows}),
            "reference": "exact_state",
            "train_depth": "H1",
            "eval_depths": [d.upper() for d in depths],
            "h3_available": "depth3" in directions,
            "probe": "held-out Bellman probe on the teacher held-out split",
            "reward_mode": cfg["reward_mode"],
            "rng_mode": cfg["rng_mode"],
            "band_mode": cfg["bellman_band_mode"],
        },
        "methods": [{k: m[k] for k in ("id", "name", "short", "hex", "role")} for m in present],
        "depths": {},
    }

    for dkey, dname in depths.items():
        block = {}
        for mkey, path in METRICS.items():
            per_method = {}
            for m in present:
                seeds = sorted(by_variant[m["artifact"]])
                vals, ids = [], []
                for s in seeds:
                    node = by_variant[m["artifact"]][s]["bellman_probe_error"]["by_direction"][dname]
                    vals.append(round(float(dig(node, path)), 6))
                    ids.append(s)
                per_method[m["id"]] = {"seeds": ids, "v": vals}
            block[mkey] = per_method
        payload["depths"][dkey] = block

    # Realised utility, for the explicitly-not-established correlation note.
    payload["utility"] = {
        m["id"]: {"seeds": sorted(by_variant[m["artifact"]]),
                  "v": [round(float(by_variant[m["artifact"]][s]["bank_eig_return"]), 6)
                        for s in sorted(by_variant[m["artifact"]])]}
        for m in present
    }
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", default="outputs/prey_bellman_30seed")
    ap.add_argument("-o", "--out", default="outputs/_report/bellman_html_data.json")
    a = ap.parse_args()
    payload = build(Path(a.run_dir))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    e = payload["experiment"]
    print(f"wrote {out}")
    print(f"  experiment {e['name']} · {len(e['seeds'])} seeds · depths {e['eval_depths']} "
          f"· H3 available: {e['h3_available']}")
    print(f"  methods: {[m['id'] for m in payload['methods']]}")


if __name__ == "__main__":
    main()
