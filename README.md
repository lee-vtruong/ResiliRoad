# ResiliRoad

Reproducible code and artefacts for **Residual Spectral Graph Learning for
Connectivity-Loss Estimation under Multi-Edge Disruptions**.

ResiliRoad estimates the relative loss of algebraic connectivity after multiple
road-network edges fail. It combines an interpretable first-order Fiedler
sensitivity with a graph neural network that learns only the residual
correction.

![ResiliRoad method overview](figures/method_overview.png)

## Main findings

The study uses five random seeds, graph-disjoint synthetic splits, independent
and spatially clustered disruptions, zero-shot transfer to five Vietnamese
OpenStreetMap networks, and geographically blocked leave-one-area-out transfer.

| OSM failure process | Direct GCN MAE | Residual GCN MAE | Paired improvement [95% CI] |
|---|---:|---:|---:|
| Independent | 0.106 | 0.097 | 0.0085 [0.0039, 0.0131] |
| Spatial cluster | 0.102 | 0.090 | 0.0121 [0.0026, 0.0247] |

Bootstrap intervals resample seed-level metrics rather than treating correlated
scenarios as independent observations. Fiedler-feature and coordinate ablations
are inconclusive with five seeds; the stable improvement comes from using the
analytical estimate as an explicit residual prior.

This is a structural-disruption study. It is not a flood prediction, traffic
assignment, or city-wide resilience model.

The journal extension also finds an important boundary condition. When models
are trained on three OSM areas, validated on a fourth, and tested on a fifth,
the direct GCN is more stable than the residual GCN (MAE 0.135 vs 0.161 for
independent and 0.109 vs 0.144 for spatial failures). The repository therefore
does not claim that the spectral residual is universally superior. The paired
five-seed intervals include zero, so this reversal is a caution about stability
and domain shift rather than a confirmed direct-GCN advantage.

## Methods compared

- exact post-disruption eigendecomposition as ground truth;
- first-order Fiedler spectral approximation;
- direct and residual GCNs;
- GCN ablations without the Fiedler node feature or coordinates;
- Deep Sets and graph-summary MLP baselines.

The OSM evaluation covers HCMUS (Ho Chi Minh City), VIASM (Hanoi), Da Nang,
Can Tho, and Da Lat. Their cached graphs range from 48 to 206 nodes and densities
from 0.0143 to 0.0488.

## Reproduce the paper

Python 3.11+ is recommended. A GPU is not required; the reported experiment was
run entirely on CPU with one PyTorch compute thread.

```powershell
pip install -r requirements.txt
.\reproduce_all.ps1
```

Individual stages can also be run separately:

```powershell
python download_osm.py
python run_paper_benchmark.py --seed 11 --output outputs/paper/seed_11
python analyze_paper_results.py --input outputs/paper --output outputs/paper_summary
python run_scaling_benchmark.py
python run_geographic_transfer.py --seed 11 --scenarios-per-site-mode 40 --epochs 25 --output outputs/geographic_transfer/seed_11
python analyze_journal_results.py
python collect_environment.py
python create_method_overview.py
```

Repeat the benchmark for seeds `11`, `22`, `33`, `44`, and `55`. The wrapper
script performs this loop and compiles the paper.

## Repository structure

- [`journal/`](journal/) - Journal of Complex Networks manuscript, cover letter,
  and submission checklist.
- [`report/`](report/) - conference-report LaTeX source and PDF.
- [`poster/`](poster/) - final VMS60 poster source and PDF.
- [`resilience/`](resilience/) - scenario generation, models, and training.
- [`data/osm/`](data/osm/) - cached OSM graphs and download manifest.
- [`outputs/paper_summary/`](outputs/paper_summary/) - bootstrap metrics, paired
  differences, error analysis, runtime, and environment metadata.
- [`docs/EXPERIMENT_MATRIX.md`](docs/EXPERIMENT_MATRIX.md) - preregistered
  experiment and claim matrix.

## Paper and poster

- [Journal of Complex Networks manuscript](journal/output/pdf/ResiliRoad_JCN_Manuscript.pdf)
- [JCN cover letter](journal/output/pdf/ResiliRoad_JCN_Cover_Letter.pdf)
- [Final paper PDF](report/output/pdf/ResiliRoad_Final_Paper.pdf)
- [Final VMS60 poster PDF](poster/output/pdf/ResiliRoad_VMS60_Poster.pdf)
- [Method overview (SVG)](figures/method_overview.svg)

## Author

**Van-Truong Le**  
Faculty of Information Technology, University of Science, Viet Nam National
University Ho Chi Minh City, Viet Nam  
23120181@student.hcmus.edu.vn · lvtruong@selab.hcmus.edu.vn

ORCID: [0009-0008-7015-7392](https://orcid.org/0009-0008-7015-7392)

## Data licence

OpenStreetMap data are copyright OpenStreetMap contributors and available under
the Open Database License. Cached networks are used only for structural
disruption experiments.
