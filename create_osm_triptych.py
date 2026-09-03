"""Create the three-area OSM morphology panel used in the journal paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from resilience.osm import load_preprocessed_network


PANELS = [
    ("HCMUS, Vietnam", Path("data/osm/hcmus_650m_drive.graphml")),
    ("Singapore", Path("data/osm/singapore_1400m_drive.graphml")),
    ("Kyoto, Japan", Path("data/osm/kyoto_1500m_drive.graphml")),
]
OUTPUT = Path("figures")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.45))
    for (label, path), axis in zip(PANELS, axes):
        graph = load_preprocessed_network(path)
        positions = {
            node: (graph.nodes[node]["x"], graph.nodes[node]["y"])
            for node in graph.nodes
        }
        nx.draw_networkx_edges(
            graph, positions, ax=axis, width=0.48, edge_color="#78a6d2", alpha=0.78
        )
        nx.draw_networkx_nodes(
            graph, positions, ax=axis, node_size=2.4, node_color="#df3232", linewidths=0
        )
        axis.set_title(
            f"{label}\n{graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges",
            fontsize=10,
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.axis("off")
    figure.tight_layout(w_pad=1.0)
    for suffix in ("pdf", "png"):
        figure.savefig(
            OUTPUT / f"osm_network_triptych.{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
