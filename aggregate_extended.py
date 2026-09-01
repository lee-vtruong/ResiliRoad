from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resilience.osm import load_preprocessed_network


DISPLAY = {
    "direct_full": "Direct GCN",
    "direct_no_fiedler": "GCN without Fiedler",
    "residual_full": "Residual spectral-GCN",
    "spectral": "Spectral first order",
}
COLORS = {
    "direct_full": "#1f77b4", "direct_no_fiedler": "#7f7f7f",
    "residual_full": "#2ca02c", "spectral": "#ff7f0e",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/extended"))
    parser.add_argument("--output", type=Path, default=Path("outputs/extended_summary"))
    parser.add_argument("--osm-graph", type=Path, default=Path("data/osm/hcmus_650m_drive.graphml"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], []
    for directory in sorted(args.input.glob("seed_*")):
        payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        seed = payload["metadata"]["seed"]
        for model, values in payload["models"].items():
            for domain in ("synthetic", "osm"):
                rows.append({"seed": seed, "model": model, "domain": domain, **values[domain]})
        for domain in ("synthetic", "osm"):
            rows.append({
                "seed": seed, "model": "spectral", "domain": domain,
                **payload["spectral_reference"][domain],
            })
        frame = pd.read_csv(directory / "predictions.csv")
        frame["seed"] = seed
        predictions.append(frame)
    raw = pd.DataFrame(rows)
    all_predictions = pd.concat(predictions, ignore_index=True)
    raw.to_csv(args.output / "metrics_by_seed.csv", index=False)
    all_predictions.to_csv(args.output / "predictions.csv", index=False)
    summary = raw.groupby(["domain", "model"])[["mae", "rmse", "r2", "spearman"]].agg(["mean", "std"])
    summary.to_csv(args.output / "metrics_summary.csv")

    paired = []
    for seed in sorted(raw.seed.unique()):
        for domain in ("synthetic", "osm"):
            part = raw[(raw.seed == seed) & (raw.domain == domain)].set_index("model")
            paired.append({
                "seed": seed, "domain": domain,
                "mae_no_fiedler_minus_full": part.loc["direct_no_fiedler", "mae"] - part.loc["direct_full", "mae"],
                "mae_residual_minus_full": part.loc["residual_full", "mae"] - part.loc["direct_full", "mae"],
                "spearman_residual_minus_full": part.loc["residual_full", "spearman"] - part.loc["direct_full", "spearman"],
            })
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(args.output / "paired_differences.csv", index=False)

    methods = ["direct_full", "direct_no_fiedler", "residual_full", "spectral"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    width = .19
    x = np.arange(2)
    for k, model in enumerate(methods):
        means_mae = [summary.loc[(d, model), ("mae", "mean")] for d in ("synthetic", "osm")]
        std_mae = [summary.loc[(d, model), ("mae", "std")] for d in ("synthetic", "osm")]
        means_rho = [summary.loc[(d, model), ("spearman", "mean")] for d in ("synthetic", "osm")]
        std_rho = [summary.loc[(d, model), ("spearman", "std")] for d in ("synthetic", "osm")]
        offset = (k - 1.5) * width
        axes[0].bar(x + offset, means_mae, width, yerr=std_mae, capsize=2,
                    label=DISPLAY[model], color=COLORS[model])
        axes[1].bar(x + offset, means_rho, width, yerr=std_rho, capsize=2,
                    color=COLORS[model])
    for axis, title, ylabel in [
        (axes[0], "Absolute-error performance", "MAE (lower is better)"),
        (axes[1], "Ranking performance", "Spearman $\\rho$ (higher is better)"),
    ]:
        axis.set_xticks(x, ["Synthetic", "OSM transfer"])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output / "extended_comparison.pdf")
    plt.savefig(args.output / "extended_comparison.png", dpi=240)
    plt.close()

    graph = load_preprocessed_network(args.osm_graph)
    pos = {n: (graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in graph.nodes()}
    plt.figure(figsize=(6.2, 5.2))
    import networkx as nx
    nx.draw_networkx_edges(graph, pos, width=.65, edge_color="#4c78a8", alpha=.8)
    nx.draw_networkx_nodes(graph, pos, node_size=5, node_color="#d62728")
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.output / "osm_network.pdf", bbox_inches="tight")
    plt.savefig(args.output / "osm_network.png", dpi=240, bbox_inches="tight")
    plt.close()

    readable = {}
    for domain in ("synthetic", "osm"):
        readable[domain] = {}
        for model in methods:
            readable[domain][model] = {
                metric: {
                    "mean": float(summary.loc[(domain, model), (metric, "mean")]),
                    "std": float(summary.loc[(domain, model), (metric, "std")]),
                } for metric in ("mae", "rmse", "r2", "spearman")
            }
    readable["paired_mean"] = paired_frame.groupby("domain").mean(numeric_only=True).drop(columns="seed").to_dict("index")
    (args.output / "summary.json").write_text(json.dumps(readable, indent=2), encoding="utf-8")
    print(json.dumps(readable, indent=2))


if __name__ == "__main__":
    main()
