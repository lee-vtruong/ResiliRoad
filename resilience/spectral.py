from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


def laplacian_eigensystem(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted eigenvalues/eigenvectors of the combinatorial Laplacian."""
    nodes = sorted(graph.nodes())
    if len(nodes) > 400:
        adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight",
                                             dtype=float, format="csr")
        laplacian = csgraph.laplacian(adjacency, normed=False)
        k = min(12, len(nodes) - 1)
        values, vectors = eigsh(laplacian, k=k, which="SM", tol=1e-7,
                                v0=np.linspace(1.0, 2.0, len(nodes)))
        order = np.argsort(values)
        return values[order], vectors[:, order]
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight", dtype=float)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    values, vectors = np.linalg.eigh(laplacian)
    return values, vectors


def algebraic_connectivity(graph: nx.Graph, tolerance: float = 1e-10) -> float:
    if graph.number_of_nodes() < 2:
        return 0.0
    if not nx.is_connected(graph):
        return 0.0
    values, _ = laplacian_eigensystem(graph)
    value = float(values[1])
    return 0.0 if value < tolerance else value


def fiedler_data(graph: nx.Graph) -> tuple[float, dict[int, float]]:
    values, vectors = laplacian_eigensystem(graph)
    nodes = sorted(graph.nodes())
    return float(max(values[1], 0.0)), dict(zip(nodes, vectors[:, 1].tolist()))


def spectral_prior_data(graph: nx.Graph, max_modes: int = 12):
    """Return low-frequency spectrum, Fiedler map, and relative eigengap."""
    values, vectors = laplacian_eigensystem(graph)
    keep = min(len(values), max_modes)
    values, vectors = values[:keep], vectors[:, :keep]
    nodes = sorted(graph.nodes())
    lambda2 = float(max(values[1], 0.0))
    eigengap = float((values[2] - values[1]) / lambda2) if keep > 2 and lambda2 > 0 else np.nan
    return values, vectors, dict(zip(nodes, vectors[:, 1].tolist())), eigengap


def second_order_relative_loss(graph, failed_edges, values, vectors) -> float:
    """Truncated second-order perturbation estimate of relative lambda_2 loss."""
    nodes = sorted(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    u2 = vectors[:, 1]
    first = -sum(graph[u][v]["weight"] *
                 (u2[index[u]] - u2[index[v]]) ** 2 for u, v in failed_edges)
    second = 0.0
    for k in range(vectors.shape[1]):
        if k == 1 or abs(values[1] - values[k]) < 1e-12:
            continue
        uk = vectors[:, k]
        coupling = -sum(graph[u][v]["weight"] *
                        (uk[index[u]] - uk[index[v]]) *
                        (u2[index[u]] - u2[index[v]]) for u, v in failed_edges)
        second += coupling * coupling / float(values[1] - values[k])
    return float(np.clip(-(first + second) / max(float(values[1]), 1e-12), 0.0, 1.0))


def edge_sensitivity(fiedler: dict[int, float], edge: tuple[int, int]) -> float:
    """First-order derivative of lambda_2 with respect to edge weight."""
    u, v = edge
    return float((fiedler[u] - fiedler[v]) ** 2)


def relative_drop(base_lambda2: float, damaged_lambda2: float) -> float:
    if base_lambda2 <= 0:
        raise ValueError("Base graph must be connected with positive lambda_2")
    return float(np.clip(1.0 - damaged_lambda2 / base_lambda2, 0.0, 1.0))
