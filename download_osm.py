from pathlib import Path
import json

from resilience.osm import download_drive_network, load_preprocessed_network


SITES = {
    "hcmus": (10.7626, 106.6822, 650, "Vietnam"),
    "viasm": (21.0156, 105.8019, 650, "Vietnam"),
    "danang": (16.0678, 108.2208, 650, "Vietnam"),
    "cantho": (10.0340, 105.7880, 650, "Vietnam"),
    "dalat": (11.9404, 108.4583, 650, "Vietnam"),
    "singapore": (1.3008, 103.8515, 1400, "Singapore"),
    "kuala_lumpur": (3.1478, 101.6953, 1500, "Malaysia"),
    "george_town": (5.4141, 100.3288, 1600, "Malaysia"),
    "bangkok": (13.7563, 100.5018, 1400, "Thailand"),
    "chiang_mai": (18.7883, 98.9853, 1600, "Thailand"),
    "taipei": (25.0478, 121.5319, 1300, "Taiwan"),
    "kyoto": (35.0116, 135.7681, 1500, "Japan"),
    "tokyo": (35.6896, 139.7006, 1200, "Japan"),
}


if __name__ == "__main__":
    records = []
    for name, (latitude, longitude, radius, country) in SITES.items():
        destination = Path(f"data/osm/{name}_{radius}m_drive.graphml")
        legacy = Path(f"data/osm/{name}_650m_drive.graphml")
        if radius == 650 and legacy.exists():
            destination = legacy
        if not destination.exists():
            download_drive_network(destination, latitude, longitude, radius)
        graph = load_preprocessed_network(destination)
        records.append({
            "name": name, "path": str(destination), "latitude": latitude,
            "longitude": longitude, "radius_m": radius,
            "country": country,
            "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
            "density": 2 * graph.number_of_edges() / (graph.number_of_nodes() * (graph.number_of_nodes() - 1)),
        })
    Path("data/osm/manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))
