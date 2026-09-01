from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np


def download_drive_network(
    output: Path,
    latitude: float = 10.7626,
    longitude: float = 106.6822,
    distance_m: int = 650,
) -> None:
    """Download and cache a small drivable OSM network around HCMUS."""
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.log_console = True
    graph = ox.graph_from_point(
        (latitude, longitude), dist=distance_m, network_type="drive",
        simplify=True, retain_all=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, output)


def load_preprocessed_network(path: Path) -> nx.Graph:
    """Load OSMnx GraphML and convert it to a weighted simple connected graph."""
    import osmnx as ox

    multi = ox.load_graphml(path)
    undirected = ox.convert.to_undirected(multi)
    simple = nx.Graph()
    for node, data in undirected.nodes(data=True):
        simple.add_node(node, x=float(data["x"]), y=float(data["y"]))
    for u, v, data in undirected.edges(data=True):
        length = float(data.get("length", 1.0))
        weight = 100.0 / max(length, 1.0)
        if simple.has_edge(u, v):
            # Parallel carriageways: retain the strongest conductance.
            simple[u][v]["weight"] = max(simple[u][v]["weight"], weight)
            simple[u][v]["length"] = min(simple[u][v]["length"], length)
        else:
            simple.add_edge(u, v, weight=weight, length=length)
    component = max(nx.connected_components(simple), key=len)
    simple = simple.subgraph(component).copy()
    simple.remove_edges_from(nx.selfloop_edges(simple))
    simple = nx.convert_node_labels_to_integers(simple)
    xs = np.array([simple.nodes[n]["x"] for n in simple.nodes()])
    ys = np.array([simple.nodes[n]["y"] for n in simple.nodes()])
    xscale = max(float(xs.max() - xs.min()), 1e-12)
    yscale = max(float(ys.max() - ys.min()), 1e-12)
    for node in simple.nodes():
        simple.nodes[node]["pos"] = (
            (simple.nodes[node]["x"] - float(xs.min())) / xscale,
            (simple.nodes[node]["y"] - float(ys.min())) / yscale,
        )
    return simple
