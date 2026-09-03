from __future__ import annotations

from dataclasses import dataclass
import random
from time import perf_counter

import networkx as nx
import numpy as np
import torch

from .spectral import (algebraic_connectivity, edge_sensitivity, relative_drop,
                       second_order_relative_loss, spectral_prior_data)


@dataclass
class Scenario:
    x: torch.Tensor
    adjacency: torch.Tensor
    target: float
    spectral_prediction: float
    graph_id: int
    failed_count: int
    failure_mode: str = "independent"
    damaged_connected: bool = True
    base_density: float = 0.0
    spectral_clipped: bool = False
    area: str = "synthetic"
    exact_seconds: float = 0.0
    spectral_seconds: float = 0.0
    second_order_prediction: float = 0.0
    relative_eigengap: float = float("nan")
    edge_index: torch.Tensor | None = None
    edge_attr: torch.Tensor | None = None


def _sample_failed_edges(graph, edges, count, rng, py_rng, mode):
    if mode == "independent":
        return py_rng.sample(edges, count)
    if mode == "targeted":
        centrality = graph.graph.get("_edge_betweenness")
        if centrality is None:
            centrality = nx.edge_betweenness_centrality(graph, normalized=True, weight=None)
            graph.graph["_edge_betweenness"] = centrality
        scores = np.array([centrality.get(edge, centrality.get((edge[1], edge[0]), 0.0)) for edge in edges])
        probabilities = (scores + 1e-12) / (scores.sum() + 1e-12 * len(scores))
        return [edges[i] for i in rng.choice(len(edges), count, replace=False, p=probabilities)]
    if mode != "spatial_cluster":
        raise ValueError(f"Unknown failure mode: {mode}")
    positions = nx.get_node_attributes(graph, "pos")
    epicenter = edges[int(rng.integers(0, len(edges)))]
    eu, ev = epicenter
    center = (np.asarray(positions[eu]) + np.asarray(positions[ev])) / 2
    scored = []
    for edge in edges:
        u, v = edge
        midpoint = (np.asarray(positions[u]) + np.asarray(positions[v])) / 2
        distance = float(np.linalg.norm(midpoint - center))
        scored.append((distance + float(rng.uniform(0, 1e-9)), edge))
    scored.sort(key=lambda item: item[0])
    return [edge for _, edge in scored[:count]]


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
    if len(nodes) > 400:
        adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight="weight",
                                             dtype=np.float32, format="csr")
        adjacency.setdiag(adjacency.diagonal() + 1)
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inv = np.maximum(degree, 1e-12) ** -0.5
        normalized = adjacency.multiply(inv[:, None]).multiply(inv[None, :]).tocoo()
        indices = torch.tensor(np.vstack([normalized.row, normalized.col]), dtype=torch.long)
        values = torch.tensor(normalized.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices, values, normalized.shape, check_invariants=False
        ).coalesce()
    a = nx.to_numpy_array(graph, nodelist=nodes, weight="weight", dtype=np.float32)
    a += np.eye(len(nodes), dtype=np.float32)
    degree = a.sum(axis=1)
    inv_sqrt = np.power(np.maximum(degree, 1e-12), -0.5)
    normalized = inv_sqrt[:, None] * a * inv_sqrt[None, :]
    return torch.from_numpy(normalized)


def _edge_tensors(graph, failed, fiedler):
    failed_set = {tuple(sorted(edge)) for edge in failed}
    weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()], dtype=float)
    max_weight = max(float(weights.max()), 1e-12)
    lengths = 1.0 / np.maximum(weights, 1e-12)
    max_length = max(float(lengths.max()), 1e-12)
    sensitivities = np.array([(fiedler[u] - fiedler[v]) ** 2 for u, v in graph.edges()])
    max_sensitivity = max(float(sensitivities.max()), 1e-12)
    indices, attributes = [], []
    for (u, v), weight, length, sensitivity in zip(graph.edges(), weights, lengths, sensitivities):
        attr = [weight / max_weight, length / max_length,
                float(tuple(sorted((u, v))) in failed_set), sensitivity / max_sensitivity]
        indices.extend([(u, v), (v, u)]); attributes.extend([attr, attr])
    return (torch.tensor(indices, dtype=torch.long).T.contiguous(),
            torch.tensor(attributes, dtype=torch.float32))


def generate_dataset(
    samples: int,
    seed: int = 42,
    min_nodes: int = 35,
    max_nodes: int = 65,
    scenarios_per_graph: int = 20,
    failure_mode: str = "independent",
) -> list[Scenario]:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    scenarios: list[Scenario] = []
    graph_id = 0
    while len(scenarios) < samples:
        n = int(rng.integers(min_nodes, max_nodes + 1))
        graph = _connected_geometric_graph(rng, n)
        values, vectors, fiedler, eigengap = spectral_prior_data(graph)
        base_lambda2 = float(values[1])
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
            failed = _sample_failed_edges(graph, edges, failed_count, rng, py_rng, failure_mode)
            damaged = graph.copy()
            damaged.remove_edges_from(failed)
            exact_started = perf_counter()
            damaged_lambda2 = algebraic_connectivity(damaged)
            exact_seconds = perf_counter() - exact_started
            target = relative_drop(base_lambda2, damaged_lambda2)
            spectral_started = perf_counter()
            first_order_loss = sum(
                graph[u][v]["weight"] * edge_sensitivity(fiedler, (u, v))
                for u, v in failed
            )
            spectral_seconds = perf_counter() - spectral_started
            spectral_prediction = float(np.clip(first_order_loss / base_lambda2, 0.0, 1.0))
            second_order = second_order_relative_loss(graph, failed, values, vectors)
            edge_index, edge_attr = _edge_tensors(graph, failed, fiedler)

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
                failure_mode=failure_mode,
                damaged_connected=nx.is_connected(damaged),
                base_density=nx.density(graph),
                spectral_clipped=first_order_loss / base_lambda2 >= 1.0,
                exact_seconds=exact_seconds,
                spectral_seconds=spectral_seconds,
                second_order_prediction=second_order, relative_eigengap=eigengap,
                edge_index=edge_index, edge_attr=edge_attr,
            ))
        graph_id += 1
    return scenarios


def generate_scenarios_for_graph(
    graph: nx.Graph,
    samples: int,
    seed: int = 42,
    graph_id: int = 0,
    failure_mode: str = "independent",
    area: str = "osm",
) -> list[Scenario]:
    """Create disruption scenarios for one preprocessed real or synthetic graph."""
    if not nx.is_connected(graph):
        raise ValueError("Input graph must be connected")
    graph = nx.convert_node_labels_to_integers(graph.copy())
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    values, vectors, fiedler, eigengap = spectral_prior_data(graph)
    base_lambda2 = float(values[1])
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
        failed = _sample_failed_edges(graph, edges, failed_count, rng, py_rng, failure_mode)
        damaged = graph.copy()
        damaged.remove_edges_from(failed)
        exact_started = perf_counter()
        target = relative_drop(base_lambda2, algebraic_connectivity(damaged))
        exact_seconds = perf_counter() - exact_started
        spectral_started = perf_counter()
        first_order_loss = sum(
            graph[u][v]["weight"] * edge_sensitivity(fiedler, (u, v))
            for u, v in failed
        )
        spectral_seconds = perf_counter() - spectral_started
        second_order = second_order_relative_loss(graph, failed, values, vectors)
        edge_index, edge_attr = _edge_tensors(graph, failed, fiedler)
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
            failure_mode=failure_mode,
            damaged_connected=nx.is_connected(damaged),
            base_density=nx.density(graph),
            spectral_clipped=first_order_loss / base_lambda2 >= 1.0,
            area=area,
            exact_seconds=exact_seconds,
            spectral_seconds=spectral_seconds,
            second_order_prediction=second_order, relative_eigengap=eigengap,
            edge_index=edge_index, edge_attr=edge_attr,
        ))
    return scenarios
