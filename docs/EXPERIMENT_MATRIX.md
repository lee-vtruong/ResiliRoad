# Preregistered extended experiment matrix

This document fixes the comparison before inspecting the new multi-area and
spatial-failure results.

## Domains

- Synthetic random geometric graphs: 35--65 nodes.
- Five cached OSM drive networks: HCMUS, VIASM, Da Nang, Can Tho, Da Lat.
- Five independent seeds: 11, 22, 33, 44, 55.

## Failure mechanisms

1. `independent`: uniformly sample 1--8 edges.
2. `spatial_cluster`: choose a random epicenter edge and remove its nearest
   1--8 edges by midpoint distance, with small random tie-breaking noise.

The primary target remains relative algebraic-connectivity loss. Results for
the two mechanisms are never pooled without a mechanism label.

## Models

- spectral first-order approximation;
- direct GCN with all features;
- direct GCN without Fiedler node coordinate;
- direct GCN without spatial coordinates;
- residual spectral-GCN with all features;
- residual spectral-GCN without Fiedler node coordinate;
- residual spectral-GCN without coordinates;
- Deep Sets baseline over node features;
- MLP baseline over fixed graph/scenario summary features.

Primary learned comparison: residual full versus direct full. Primary metric:
MAE. Secondary metrics: RMSE, R2, Spearman correlation, and runtime.

## Statistics

- Unit of independent replication: seed.
- Report mean and sample SD over five paired seeds.
- Report paired per-seed differences.
- Bootstrap 95% confidence intervals by resampling seeds, not scenarios.
- Scenario-level bootstrap is not used for inferential claims.

## Error analysis

- connected versus disconnected damaged graphs;
- failure-count groups: 1, 2--3, 4--5, 6+;
- base-graph density quartiles for synthetic data and per-area OSM density;
- scenarios where the spectral estimate is clipped to one.

## Interpretation guardrails

- OSM scenarios are zero-shot tests and never tune hyperparameters.
- Random or clustered edge deletion is not called flood prediction.
- Results from one geographic area are not generalized city-wide.
- Negative or null ablations are retained.

