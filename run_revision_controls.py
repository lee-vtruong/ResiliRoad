"""Reviewer controls: transport proxies and size-only spectral sensitivity."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from resilience.data import _sample_failed_edges
from resilience.osm import load_preprocessed_network
from resilience.spectral import algebraic_connectivity, edge_sensitivity, fiedler_data
from run_large_scaling_benchmark import road_like_graph


MODES = ("independent", "spatial_cluster", "targeted")
SEEDS = (11, 22, 33, 44, 55)


def hierarchical_interval(frame: pd.DataFrame, rng, draws: int = 10_000):
    matrix = frame.pivot_table(index="area", columns="seed", values="mae").to_numpy()
    a = rng.integers(0, matrix.shape[0], size=(draws, matrix.shape[0]))
    s = rng.integers(0, matrix.shape[1], size=(draws, matrix.shape[0]))
    return np.quantile(matrix[a, s].mean(axis=1), [0.025, 0.975])


def sampled_efficiency(graph: nx.Graph, pairs: list[tuple[int, int]]) -> float:
    values = []
    for source, target in pairs:
        try:
            distance = nx.shortest_path_length(graph, source, target, weight="length")
            values.append(1.0 / max(float(distance), 1e-12))
        except nx.NetworkXNoPath:
            values.append(0.0)
    return float(np.mean(values))


def transport_controls(manifest_path: Path, raw_root: Path, output: Path, pairs_n: int):
    cached = output / "transport_proxy_predictions.csv"
    if cached.exists():
        frame = pd.read_csv(cached)
        area = frame.groupby(["area", "country", "seed", "failure_mode", "model"], observed=True).abs_error.mean().reset_index(name="mae")
        rng = np.random.default_rng(20260904)
        summary_rows = []
        for (mode, model), group in area.groupby(["failure_mode", "model"], observed=True):
            lo, hi = hierarchical_interval(group, rng)
            summary_rows.append({"failure_mode": mode, "model": model,
                "area_mean_mae": group.groupby("area").mae.mean().mean(), "ci_low": lo,
                "ci_high": hi, "areas": group.area.nunique(), "seeds": group.seed.nunique()})
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(output / "transport_proxy_summary.csv", index=False)
        return summary
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for seed in SEEDS:
        raw = pd.read_csv(raw_root / f"seed_{seed}" / "predictions.csv")
        raw = raw[(raw.domain == "osm") & (raw.model == "spectral")]
        for area_index, site in enumerate(manifest):
            graph = nx.convert_node_labels_to_integers(
                load_preprocessed_network(Path(site["path"])).copy())
            for u, v in graph.edges:
                graph[u][v]["length"] = 1.0 / max(float(graph[u][v]["weight"]), 1e-12)
            nodes = list(graph.nodes())
            pair_rng = np.random.default_rng(880_000 + area_index)
            pairs = []
            while len(pairs) < min(pairs_n, len(nodes) * (len(nodes) - 1) // 2):
                a, b = pair_rng.choice(nodes, 2, replace=False).tolist()
                pair = (min(a, b), max(a, b))
                if pair not in pairs:
                    pairs.append(pair)
            base_efficiency = sampled_efficiency(graph, pairs)
            edges = list(graph.edges())
            for mode_index, mode in enumerate(MODES):
                subset = raw[(raw.area == site["name"]) & (raw.failure_mode == mode)].reset_index(drop=True)
                rng = np.random.default_rng(seed + 1000 + 100 * area_index + 10 * mode_index)
                py_rng = random.Random(seed + 1000 + 100 * area_index + 10 * mode_index)
                for scenario_index, record in subset.iterrows():
                    failed_count = int(rng.integers(1, max(1, min(8, len(edges) // 8)) + 1))
                    if failed_count != int(record.failed_count):
                        raise RuntimeError("Scenario replay mismatch")
                    failed = _sample_failed_edges(graph, edges, failed_count, rng, py_rng, mode)
                    damaged = graph.copy(); damaged.remove_edges_from(failed)
                    largest = len(max(nx.connected_components(damaged), key=len))
                    lcc_loss = 1.0 - largest / graph.number_of_nodes()
                    efficiency_loss = 1.0 - sampled_efficiency(damaged, pairs) / base_efficiency
                    rows.extend([
                        {"seed": seed, "area": site["name"], "country": site["country"],
                         "failure_mode": mode, "scenario": scenario_index,
                         "model": "largest-component loss", "target": record.target,
                         "prediction": float(np.clip(lcc_loss, 0, 1))},
                        {"seed": seed, "area": site["name"], "country": site["country"],
                         "failure_mode": mode, "scenario": scenario_index,
                         "model": f"OD-efficiency loss ({len(pairs)} pairs)", "target": record.target,
                         "prediction": float(np.clip(efficiency_loss, 0, 1))},
                    ])
    frame = pd.DataFrame(rows); frame["abs_error"] = abs(frame.target - frame.prediction)
    frame.to_csv(output / "transport_proxy_predictions.csv", index=False)
    area = frame.groupby(["area", "country", "seed", "failure_mode", "model"], observed=True).abs_error.mean().reset_index(name="mae")
    rng = np.random.default_rng(20260904)
    summary_rows = []
    for (mode, model), group in area.groupby(["failure_mode", "model"], observed=True):
        lo, hi = hierarchical_interval(group, rng)
        summary_rows.append({"failure_mode": mode, "model": model,
            "area_mean_mae": group.groupby("area").mae.mean().mean(), "ci_low": lo,
            "ci_high": hi, "areas": group.area.nunique(), "seeds": group.seed.nunique()})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "transport_proxy_summary.csv", index=False)
    return summary


def size_control(output: Path, graphs_per_size: int, scenarios: int):
    cached = output / "size_control_predictions.csv"
    if cached.exists():
        frame = pd.read_csv(cached)
        summary = frame.groupby(["nodes", "failure_mode"], observed=True).agg(
            spectral_mae=("abs_error", "mean"), disconnected_rate=("target", lambda x: np.mean(x >= 1 - 1e-9)),
            scenarios=("target", "size")).reset_index()
        summary.to_csv(output / "size_control_summary.csv", index=False)
        return summary
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed + 990_000)
        py_rng = random.Random(seed + 990_000)
        for n in (50, 100, 200, 400, 800, 1200):
            for replicate in range(graphs_per_size):
                graph = road_like_graph(n, rng)
                base_lambda2, fiedler = fiedler_data(graph)
                edges = list(graph.edges())
                for mode in MODES:
                    for scenario in range(scenarios):
                        count = int(rng.integers(1, min(8, len(edges) // 8) + 1))
                        failed = _sample_failed_edges(graph, edges, count, rng, py_rng, mode)
                        damaged = graph.copy(); damaged.remove_edges_from(failed)
                        target = 1.0 - algebraic_connectivity(damaged) / base_lambda2
                        prior = sum(graph[u][v]["weight"] * edge_sensitivity(fiedler, (u, v))
                                    for u, v in failed) / base_lambda2
                        rows.append({"seed": seed, "nodes": n, "replicate": replicate,
                                     "failure_mode": mode, "scenario": scenario,
                                     "target": np.clip(target, 0, 1), "spectral": np.clip(prior, 0, 1)})
    frame = pd.DataFrame(rows); frame["abs_error"] = abs(frame.target - frame.spectral)
    frame.to_csv(output / "size_control_predictions.csv", index=False)
    summary = frame.groupby(["nodes", "failure_mode"], observed=True).agg(
        spectral_mae=("abs_error", "mean"), disconnected_rate=("target", lambda x: np.mean(x >= 1 - 1e-9)),
        scenarios=("target", "size")).reset_index()
    summary.to_csv(output / "size_control_summary.csv", index=False)
    return summary


def plot_controls(transport: pd.DataFrame, size: pd.DataFrame, output: Path):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    pivot = transport.pivot(index="failure_mode", columns="model", values="area_mean_mae")
    pivot.plot.bar(ax=axes[0], color=["#2878b5", "#d9534f"])
    for container, model in zip(axes[0].containers, pivot.columns):
        for rect, mode in zip(container, pivot.index):
            row = transport[(transport.failure_mode == mode) & (transport.model == model)].iloc[0]
            mean = float(row.area_mean_mae)
            axes[0].errorbar(rect.get_x() + rect.get_width() / 2, mean,
                yerr=np.array([[mean - float(row.ci_low)], [float(row.ci_high) - mean]]),
                fmt="none", color="black", capsize=2, lw=.9)
    axes[0].set(title="Transport-topology proxies", ylabel="MAE against $\\lambda_2$ loss",
                xlabel="Failure regime")
    axes[0].tick_params(axis="x", rotation=20); axes[0].legend(title="")
    for mode, marker in zip(MODES, ("o", "s", "^")):
        part = size[size.failure_mode == mode]
        axes[1].plot(part.nodes, part.spectral_mae, marker=marker,
                     label=mode.replace("_", " "))
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set(title="Size-only control within one graph family",
                xlabel="Nodes", ylabel="First-order spectral MAE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "revision_controls.pdf", bbox_inches="tight")
    fig.savefig(output / "revision_controls.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/osm/manifest.json"))
    parser.add_argument("--raw-root", type=Path, default=Path("outputs/jcn2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/revision_controls"))
    parser.add_argument("--od-pairs", type=int, default=64)
    parser.add_argument("--graphs-per-size", type=int, default=2)
    parser.add_argument("--scenarios", type=int, default=6)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    transport = transport_controls(args.manifest, args.raw_root, args.output, args.od_pairs)
    size = size_control(args.output, args.graphs_per_size, args.scenarios)
    plot_controls(transport, size, args.output)
    print(transport.to_string(index=False)); print(size.to_string(index=False))


if __name__ == "__main__":
    main()
