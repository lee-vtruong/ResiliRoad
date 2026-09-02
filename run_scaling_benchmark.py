"""Benchmark exact and amortized first-order spectral evaluation by graph size."""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
import torch

from resilience.model import ScenarioGCN


def spectrum(graph):
    adjacency = nx.to_scipy_sparse_array(graph, weight="weight", format="csr", dtype=float)
    laplacian = csgraph.laplacian(adjacency, normed=False)
    # Explicit start vector makes ARPACK timing/accuracy runs reproducible.
    v0 = np.linspace(1.0, 2.0, graph.number_of_nodes(), dtype=float)
    values, vectors = eigsh(laplacian, k=3, which="SM", tol=1e-5, v0=v0)
    order = np.argsort(values)
    return float(max(values[order[1]], 0.0)), vectors[:, order[1]]


def graph_for_size(n, rng):
    radius = min(0.5, 1.8 * np.sqrt(np.log(n) / n))
    for _ in range(20):
        graph = nx.random_geometric_graph(n, radius, seed=int(rng.integers(2**31 - 1)))
        if nx.is_connected(graph):
            break
        radius *= 1.08
    graph = nx.convert_node_labels_to_integers(graph)
    for u, v in graph.edges:
        p, q = np.asarray(graph.nodes[u]["pos"]), np.asarray(graph.nodes[v]["pos"])
        graph[u][v]["weight"] = 1.0 / max(np.linalg.norm(p - q), 1e-6)
    return graph


def normalized_adjacency(graph):
    adjacency = nx.to_numpy_array(graph, weight="weight", dtype=np.float32)
    adjacency += np.eye(graph.number_of_nodes(), dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.maximum(degree, 1e-12) ** -0.5
    return torch.from_numpy(inv_sqrt[:, None] * adjacency * inv_sqrt[None, :])


def main():
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/journal_summary"))
    parser.add_argument("--graphs", type=int, default=3)
    parser.add_argument("--scenarios", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260903)
    torch.manual_seed(20260903)
    torch.set_num_threads(1)
    model = ScenarioGCN(input_dim=5, residual=True).eval()
    rows = []
    for n in (50, 100, 200, 400, 800):
        for replicate in range(args.graphs):
            graph = graph_for_size(n, rng)
            start = perf_counter(); base_l2, u2 = spectrum(graph); setup = perf_counter() - start
            edges = list(graph.edges)
            for scenario in range(args.scenarios):
                failed = [edges[i] for i in rng.choice(len(edges), size=min(8, len(edges)), replace=False)]
                damaged = graph.copy(); damaged.remove_edges_from(failed)
                start = perf_counter(); damaged_l2, _ = spectrum(damaged); exact = perf_counter() - start
                start = perf_counter()
                raw = sum(graph[u][v]["weight"] * (u2[u] - u2[v]) ** 2 for u, v in failed) / base_l2
                prediction = float(np.clip(raw, 0, 1)); update = perf_counter() - start
                target = float(np.clip(1 - damaged_l2 / base_l2, 0, 1))
                positions = nx.get_node_attributes(graph, "pos")
                degrees = dict(graph.degree())
                failed_incident = {node: 0 for node in graph.nodes}
                for u, v in failed:
                    failed_incident[u] += 1; failed_incident[v] += 1
                u2_scale = max(float(np.max(np.abs(u2))), 1e-12)
                x = torch.tensor([
                    [degrees[node] / max(degrees.values()),
                     failed_incident[node] / max(degrees[node], 1),
                     abs(u2[node]) / u2_scale, *positions[node]]
                    for node in sorted(graph.nodes)
                ], dtype=torch.float32)
                adjacency = normalized_adjacency(damaged)
                spectral_tensor = torch.tensor(prediction, dtype=torch.float32)
                with torch.no_grad():
                    model(x, adjacency, spectral_tensor)  # untimed warm-up
                    start = perf_counter()
                    for _ in range(10):
                        model(x, adjacency, spectral_tensor)
                    gnn = (perf_counter() - start) / 10
                rows.append(dict(nodes=n, edges=graph.number_of_edges(), replicate=replicate,
                                 scenario=scenario, exact_seconds=exact,
                                 spectral_setup_seconds=setup,
                                 spectral_update_seconds=update,
                                 spectral_amortized_seconds=setup / args.scenarios + update,
                                 gnn_inference_seconds=gnn,
                                 target=target, spectral_prediction=prediction))
    data = pd.DataFrame(rows); data.to_csv(args.output / "scaling_raw.csv", index=False)
    summary = data.groupby("nodes").agg(edges=("edges", "mean"),
        exact_ms=("exact_seconds", lambda x: 1000*x.mean()),
        update_ms=("spectral_update_seconds", lambda x: 1000*x.mean()),
        amortized_ms=("spectral_amortized_seconds", lambda x: 1000*x.mean()),
        gnn_ms=("gnn_inference_seconds", lambda x: 1000*x.mean()),
        spectral_mae=("target", lambda x: 0.0)).reset_index()
    for n in summary.nodes:
        part=data[data.nodes.eq(n)]
        summary.loc[summary.nodes.eq(n),"spectral_mae"] = np.mean(np.abs(part.target-part.spectral_prediction))
    summary.to_csv(args.output / "scaling_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5,4.2))
    ax.plot(summary.nodes, summary.exact_ms, "o-", label="Exact eigensolve")
    ax.plot(summary.nodes, summary.amortized_ms, "s-", label="Spectral, setup amortized")
    ax.plot(summary.nodes, summary.update_ms, "^-", label="Spectral update only")
    ax.plot(summary.nodes, summary.gnn_ms, "D-", label="Residual GCN inference")
    ax.set(xlabel="Number of nodes", ylabel="Milliseconds per scenario", yscale="log")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.output / "scaling_runtime.pdf"); fig.savefig(args.output / "scaling_runtime.png", dpi=240)
    print(summary.to_string(index=False))

if __name__ == "__main__": main()
