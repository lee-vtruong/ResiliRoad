# OpenStreetMap validation networks

Five 650 m `drive` networks were downloaded with OSMnx 2.1.1 on 2026-09-02,
simplified, converted to undirected simple graphs, and restricted to the largest
connected component.

| Key | Area | Center (lat, lon) | Nodes | Edges | Density |
|---|---|---:|---:|---:|---:|
| `hcmus` | HCMUS, Ho Chi Minh City | 10.7626, 106.6822 | 163 | 221 | 0.0167 |
| `viasm` | VIASM, Hanoi | 21.0156, 105.8019 | 206 | 301 | 0.0143 |
| `danang` | Da Nang | 16.0678, 108.2208 | 142 | 203 | 0.0203 |
| `cantho` | Can Tho | 10.0340, 105.7880 | 114 | 178 | 0.0276 |
| `dalat` | Da Lat | 11.9404, 108.4583 | 48 | 55 | 0.0488 |

Exact metadata are in `manifest.json`. Cached GraphML files make validation
repeatable without new Overpass requests.

Data copyright [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
available under the Open Database License. These networks support structural
disruption scenarios only and contain no flood observations.
