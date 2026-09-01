from pathlib import Path
from resilience.osm import download_drive_network, load_preprocessed_network


if __name__ == "__main__":
    destination = Path("data/osm/hcmus_650m_drive.graphml")
    download_drive_network(destination)
    graph = load_preprocessed_network(destination)
    print({"path": str(destination), "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()})
