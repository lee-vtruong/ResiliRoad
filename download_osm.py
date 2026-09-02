from pathlib import Path
import json

from resilience.osm import download_drive_network, load_preprocessed_network


SITES = {
    "hcmus": (10.7626, 106.6822),
    "viasm": (21.0156, 105.8019),
    "danang": (16.0678, 108.2208),
    "cantho": (10.0340, 105.7880),
    "dalat": (11.9404, 108.4583),
}


if __name__ == "__main__":
    records = []
    for name, (latitude, longitude) in SITES.items():
        destination = Path(f"data/osm/{name}_650m_drive.graphml")
        if not destination.exists():
            download_drive_network(destination, latitude, longitude, 650)
        graph = load_preprocessed_network(destination)
        records.append({
            "name": name, "path": str(destination), "latitude": latitude,
            "longitude": longitude, "radius_m": 650,
            "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
            "density": 2 * graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1)),
        })
    Path("data/osm/manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))
