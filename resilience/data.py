from __future__ import annotations

from dataclasses import dataclass
import random

import networkx as nx
import numpy as np
import torch

from .spectral import algebraic_connectivity, edge_sensitivity, fiedler_data, relative_drop


@dataclass
class Scenario:
    x: torch.Tensor
    adjacency: torch.Tensor
    target: float
    spectral_prediction: float
    graph_id: int
    failed_count: int


def _connected_geometric_graph(rng: np.random.Generator, n: int) -> nx.Graph:
    # Radius grows only when needed, which preserves road-like spatial locality.
    for radius in np.linspace(0.18, 0.42, 9):
        seed = int(rng.integers(0, 2**31 - 1))
        graph = nx.random_geometric_graph(n, float(radius), seed=seed)
        if nx.is_connected(graph):
            break
    if not nx.is_connected(graph):
        largest = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest).copy()
    graph = nx.convert_node_labels_to_integers(graph)
    for u, v in graph.edges():
        x1, y1 = graph.nodes[u]["pos"]
        x2, y2 = graph.nodes[v]["pos"]
        # Conductance: short links receive larger weights.
        distance = max(np.hypot(x1 - x2, y1 - y2), 1e-3)
        graph[u][v]["weight"] = float(1.0 / distance)
    return graph


def _normalized_adjacency(graph: nx.Graph) -> torch.Tensor:
    nodes = sorted(graph.nodes())
    a = nx.to_numpy_array(graph, nodelist=nodes, weight="weight", dtype=np.float32)
    a += np.eye(len(nodes), dtype=np.float32)
    degree = a.sum(axis=1)
    inv_sqrt = np.power(np.maximum(degree, 1e-12), -0.5)
    normalized = inv_sqrt[:, None] * a * inv_sqrt[None, :]
    return torch.from_numpy(normalized)


def generate_dataset(
    samples: int,
    seed: int = 42,
    min_nodes: int = 35,
    max_nodes: int = 65,
    scenarios_per_graph: int = 20,
) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    scenarios: list[Scenario] = []
    graph_id = 0
    while len(scenarios) < samples:
        n = int(rng.integers(min_nodes, max_nodes + 1))
        graph = _connected_geometric_graph(rng, n)
        base_lambda2, fiedler = fiedler_data(graph)
        if base_lambda2 <= 1e-8:
            continue
        degrees = dict(graph.degree())
        max_degree = max(degrees.values())
        edges = list(graph.edges())
        positions = nx.get_node_attributes(graph, "pos")

        for _ in range(scenarios_per_graph):
            if len(scenarios) >= samples:
                break
            max_failures = max(1, min(8, len(edges) // 8))
            failed_count = int(rng.integers(1, max_failures + 1))
            failed = py_rng.sample(edges, failed_count)
            damaged = graph.copy()
            damaged.remove_edges_from(failed)
            damaged_lambda2 = algebraic_connectivity(damaged)
            target = relative_drop(base_lambda2, damaged_lambda2)
            first_order_loss = sum(
                graph[u][v]["weight"] * edge_sensitivity(fiedler, (u, v))
                for u, v in failed
            )
            spectral_prediction = float(np.clip(first_order_loss / base_lambda2, 0.0, 1.0))

            failed_incident = {node: 0 for node in graph.nodes()}
            for u, v in failed:
                failed_incident[u] += 1
                failed_incident[v] += 1
            features = []
            fiedler_scale = max(max(abs(value) for value in fiedler.values()), 1e-8)
            for node in sorted(graph.nodes()):
                px, py = positions[node]
                features.append([
                    degrees[node] / max_degree,
                    failed_incident[node] / max(degrees[node], 1),
                    abs(fiedler[node]) / fiedler_scale,
                    float(px),
                    float(py),
                ])
            scenarios.append(Scenario(
                x=torch.tensor(features, dtype=torch.float32),
                adjacency=_normalized_adjacency(damaged),
                target=target,
                spectral_prediction=spectral_prediction,
                graph_id=graph_id,
                failed_count=failed_count,
            ))
        graph_id += 1
    return scenarios


def generate_scenarios_for_graph(
    graph: nx.Graph,
    samples: int,
    seed: int = 42,
    graph_id: int = 0,
) -> list[Scenario]:
    """Create disruption scenarios for one preprocessed real or synthetic graph."""
    if not nx.is_connected(graph):
        raise ValueError("Input graph must be connected")
    graph = nx.convert_node_labels_to_integers(graph.copy())
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    base_lambda2, fiedler = fiedler_data(graph)
    if base_lambda2 <= 1e-8:
        raise ValueError("Input graph has numerically zero algebraic connectivity")
    degrees = dict(graph.degree())
    max_degree = max(degrees.values())
    edges = list(graph.edges())
    positions = nx.get_node_attributes(graph, "pos")
    if len(positions) != graph.number_of_nodes():
        raise ValueError("Every node must have a normalized 'pos' attribute")
    fiedler_scale = max(max(abs(value) for value in fiedler.values()), 1e-8)
    scenarios = []
    max_failures = max(1, min(8, len(edges) // 8))
    for _ in range(samples):
        failed_count = int(rng.integers(1, max_failures + 1))
        failed = py_rng.sample(edges, failed_count)
        damaged = graph.copy()
        damaged.remove_edges_from(failed)
        target = relative_drop(base_lambda2, algebraic_connectivity(damaged))
        first_order_loss = sum(
            graph[u][v]["weight"] * edge_sensitivity(fiedler, (u, v))
            for u, v in failed
        )
        failed_incident = {node: 0 for node in graph.nodes()}
        for u, v in failed:
            failed_incident[u] += 1
            failed_incident[v] += 1
        features = []
        for node in sorted(graph.nodes()):
            px, py = positions[node]
            features.append([
                degrees[node] / max_degree,
                failed_incident[node] / max(degrees[node], 1),
                abs(fiedler[node]) / fiedler_scale,
                float(px), float(py),
            ])
        scenarios.append(Scenario(
            x=torch.tensor(features, dtype=torch.float32),
            adjacency=_normalized_adjacency(damaged),
            target=target,
            spectral_prediction=float(np.clip(first_order_loss / base_lambda2, 0.0, 1.0)),
            graph_id=graph_id,
            failed_count=failed_count,
        ))
    return scenarios
