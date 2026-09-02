"""City-scale sparse benchmark on connected planar, road-like lattice graphs."""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

from resilience.model import ScenarioGCN


def road_like_graph(n, rng):
    width = int(np.ceil(np.sqrt(n)))
    graph = nx.Graph()
    for node in range(n):
        row, column = divmod(node, width)
        graph.add_node(node, pos=(column / max(width - 1, 1), row / max(width - 1, 1)))
        if column and node - 1 >= 0:
            graph.add_edge(node, node - 1)
        if row and node - width >= 0:
            graph.add_edge(node, node - width)
        if row and column and rng.random() < .12:
            graph.add_edge(node, node - width - 1)
    for u, v in graph.edges:
        p, q = np.asarray(graph.nodes[u]["pos"]), np.asarray(graph.nodes[v]["pos"])
        graph[u][v]["weight"] = float(1.0 / max(np.linalg.norm(p - q), 1e-9))
    return graph


def spectrum(graph):
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(graph.number_of_nodes()),
                                         weight="weight", format="csr", dtype=float)
    laplacian = csgraph.laplacian(adjacency, normed=False)
    v0 = np.linspace(1.0, 2.0, graph.number_of_nodes())
    values, vectors = eigsh(laplacian, k=2, which="SM", tol=1e-4, v0=v0)
    order = np.argsort(values)
    return float(max(values[order[1]], 0)), vectors[:, order[1]]


def torch_sparse_adjacency(graph):
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(graph.number_of_nodes()),
                                         weight="weight", format="coo", dtype=np.float32)
    adjacency = adjacency.tocsr(); adjacency.setdiag(adjacency.diagonal() + 1)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv = np.maximum(degree, 1e-12) ** -.5
    normalized = adjacency.multiply(inv[:, None]).multiply(inv[None, :]).tocoo()
    indices = torch.tensor(np.vstack([normalized.row, normalized.col]), dtype=torch.long)
    values = torch.tensor(normalized.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, normalized.shape).coalesce()


def main():
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/journal_summary"))
    parser.add_argument("--graphs", type=int, default=2)
    parser.add_argument("--scenarios", type=int, default=3)
    parser.add_argument("--screening-batch", type=int, default=1000,
                        help="Number of scenarios over which intact eigensolver setup is amortized")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260904)
    torch.manual_seed(20260904); torch.set_num_threads(1)
    model = ScenarioGCN(input_dim=5, residual=True).eval()
    rows = []
    for n in (1000, 2000, 5000, 10000, 20000):
        for replicate in range(args.graphs):
            graph = road_like_graph(n, rng)
            start = perf_counter(); base_l2, u2 = spectrum(graph); setup = perf_counter() - start
            edges = list(graph.edges)
            for scenario in range(args.scenarios):
                failed = [edges[i] for i in rng.choice(len(edges), 8, replace=False)]
                damaged = graph.copy(); damaged.remove_edges_from(failed)
                start = perf_counter(); damaged_l2, _ = spectrum(damaged); exact = perf_counter() - start
                start = perf_counter()
                raw = sum(graph[u][v]["weight"] * (u2[u] - u2[v]) ** 2 for u, v in failed) / base_l2
                spectral = float(np.clip(raw, 0, 1)); update = perf_counter() - start
                failed_incident = np.zeros(n, dtype=np.float32)
                for u, v in failed: failed_incident[u] += 1; failed_incident[v] += 1
                degree = np.array([graph.degree[node] for node in range(n)], dtype=np.float32)
                position = np.array([graph.nodes[node]["pos"] for node in range(n)], dtype=np.float32)
                x = torch.tensor(np.column_stack([
                    degree / degree.max(), failed_incident / np.maximum(degree, 1),
                    np.abs(u2) / max(np.max(np.abs(u2)), 1e-12), position
                ]), dtype=torch.float32)
                adjacency = torch_sparse_adjacency(damaged)
                prior = torch.tensor(spectral, dtype=torch.float32)
                with torch.no_grad():
                    model(x, adjacency, prior)
                    start = perf_counter()
                    for _ in range(10): model(x, adjacency, prior)
                    gnn = (perf_counter() - start) / 10
                rows.append({"nodes": n, "edges": graph.number_of_edges(),
                             "replicate": replicate, "scenario": scenario,
                             "exact_seconds": exact, "setup_seconds": setup,
                             "update_seconds": update,
                             "amortized_seconds": setup / args.screening_batch + update,
                             "gnn_seconds": gnn,
                             "spectral_error": abs((1 - damaged_l2 / base_l2) - spectral)})
    raw = pd.DataFrame(rows); raw.to_csv(args.output / "large_scaling_raw.csv", index=False)
    summary = raw.groupby("nodes").agg(edges=("edges", "mean"),
        exact_ms=("exact_seconds", lambda x: 1000*x.mean()),
        setup_ms=("setup_seconds", lambda x: 1000*x.mean()),
        amortized_ms=("amortized_seconds", lambda x: 1000*x.mean()),
        update_ms=("update_seconds", lambda x: 1000*x.mean()),
        gnn_ms=("gnn_seconds", lambda x: 1000*x.mean()),
        spectral_mae=("spectral_error", "mean")).reset_index()
    summary["hybrid_ms"] = summary.amortized_ms + summary.gnn_ms
    summary.to_csv(args.output / "large_scaling_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
