from __future__ import annotations

import networkx as nx
import numpy as np


def laplacian_eigensystem(graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted eigenvalues/eigenvectors of the combinatorial Laplacian."""
    nodes = sorted(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight="weight", dtype=float)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    values, vectors = np.linalg.eigh(laplacian)
    return values, vectors


def algebraic_connectivity(graph: nx.Graph, tolerance: float = 1e-10) -> float:
    if graph.number_of_nodes() < 2:
        return 0.0
    values, _ = laplacian_eigensystem(graph)
    value = float(values[1])
    return 0.0 if value < tolerance else value


def fiedler_data(graph: nx.Graph) -> tuple[float, dict[int, float]]:
    values, vectors = laplacian_eigensystem(graph)
    nodes = sorted(graph.nodes())
    return float(max(values[1], 0.0)), dict(zip(nodes, vectors[:, 1].tolist()))


def edge_sensitivity(fiedler: dict[int, float], edge: tuple[int, int]) -> float:
    """First-order derivative of lambda_2 with respect to edge weight."""
    u, v = edge
    return float((fiedler[u] - fiedler[v]) ** 2)


def relative_drop(base_lambda2: float, damaged_lambda2: float) -> float:
    if base_lambda2 <= 0:
        raise ValueError("Base graph must be connected with positive lambda_2")
    return float(np.clip(1.0 - damaged_lambda2 / base_lambda2, 0.0, 1.0))

