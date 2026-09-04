from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def hierarchical_ci(frame, value, rng, draws=10_000):
    matrix = frame.pivot_table(index="area", columns="seed", values=value, aggfunc="mean").to_numpy()
    n_areas, n_seeds = matrix.shape
    area_draws = rng.integers(0, n_areas, size=(draws, n_areas))
    seed_draws = rng.integers(0, n_seeds, size=(draws, n_areas))
    samples = matrix[area_draws, seed_draws].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975])


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/jcn2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/jcn2_summary"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    for directory in sorted(args.input.glob("seed_*")):
        path = directory / "predictions.csv"
        if path.exists():
            frame = pd.read_csv(path); frame["seed"] = int(directory.name.split("_")[-1]); frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["abs_error"] = abs(data.target - data.prediction)
    manifest = pd.read_json("data/osm/manifest.json")[["name", "country", "nodes", "edges"]]
    data = data.merge(manifest, how="left", left_on="area", right_on="name")
    data.to_csv(args.output / "all_predictions.csv", index=False)
    osm = data[data.domain == "osm"].copy()
    area = osm.groupby(["area", "country", "nodes", "edges", "seed", "failure_mode", "model"], observed=True).agg(
        mae=("abs_error", "mean"), n=("target", "size"), eigengap=("relative_eigengap", "first")
    ).reset_index()
    area.to_csv(args.output / "area_seed_metrics.csv", index=False)
    rng = np.random.default_rng(20260903)
    summary = []
    for (mode, model), group in area.groupby(["failure_mode", "model"]):
        lo, hi = hierarchical_ci(group, "mae", rng)
        summary.append({"failure_mode": mode, "model": model, "area_mean_mae": group.groupby("area").mae.mean().mean(),
                        "hierarchical_ci_low": lo, "hierarchical_ci_high": hi,
                        "areas": group.area.nunique(), "seeds": group.seed.nunique()})
    summary = pd.DataFrame(summary); summary.to_csv(args.output / "hierarchical_metrics.csv", index=False)

    paired = []
    pairs = [("GCN", "direct_full", "residual_full"), ("GraphSAGE", "direct_sage", "residual_sage"),
             ("Edge-MPNN", "direct_edge_mpnn", "residual_edge_mpnn")]
    wide = area.pivot(index=["area", "country", "seed", "failure_mode"], columns="model", values="mae").reset_index()
    for backbone, direct, residual in pairs:
        for mode, group in wide.groupby("failure_mode"):
            part = group.copy(); part["delta"] = part[direct] - part[residual]
            lo, hi = hierarchical_ci(part, "delta", rng)
            paired.append({"backbone": backbone, "failure_mode": mode,
                           "direct_minus_residual": part.groupby("area").delta.mean().mean(),
                           "ci_low": lo, "ci_high": hi})
    paired = pd.DataFrame(paired); paired.to_csv(args.output / "hierarchical_paired.csv", index=False)
    analytical = []
    for mode, group in wide.groupby("failure_mode"):
        # Analytical rows are obtained separately because the architecture pivot excludes no models.
        part = area[area.failure_mode == mode].pivot(index=["area", "country", "seed"], columns="model", values="mae").reset_index()
        part["delta"] = part["spectral"] - part["spectral_second_order"]
        lo, hi = hierarchical_ci(part, "delta", rng)
        analytical.append({"failure_mode": mode, "first_minus_second_order": part.delta.mean(),
                           "ci_low": lo, "ci_high": hi})
    pd.DataFrame(analytical).to_csv(args.output / "second_order_paired.csv", index=False)

    correction_rows = []
    residuals = ["residual_full", "residual_sage", "residual_edge_mpnn"]
    for keys, group in data[data.model.isin(residuals)].groupby(
            ["domain", "failure_mode", "model", "area", "seed"], observed=True):
        needed = group.target - group.spectral
        learned = group.prediction - group.spectral
        slope = np.nan if np.var(needed) < 1e-12 else np.cov(needed, learned, ddof=0)[0, 1] / np.var(needed)
        rho = np.nan if np.std(needed) < 1e-12 or np.std(learned) < 1e-12 else spearmanr(needed, learned).statistic
        correction_rows.append(dict(zip(["domain", "failure_mode", "model", "area", "seed"], keys),
                                    slope=slope, rho=rho, correction_mae=np.mean(abs(needed-learned))))
    correction = pd.DataFrame(correction_rows); correction.to_csv(args.output / "correction_diagnostics.csv", index=False)
    correction_summary = correction.groupby(["domain", "failure_mode", "model"])[["slope", "rho", "correction_mae"]].mean().reset_index()
    correction_summary.to_csv(args.output / "correction_summary.csv", index=False)

    graph_metrics = data.groupby(["domain", "area", "seed", "graph_id", "failure_mode", "model"], observed=True).agg(
        eigengap=("relative_eigengap", "first"), mae=("abs_error", "mean")
    ).reset_index()
    bases = graph_metrics[["domain", "area", "seed", "graph_id", "failure_mode", "eigengap"]].drop_duplicates()
    bases["eigengap_bin"] = bases.groupby("domain")["eigengap"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["low", "medium", "high"]))
    graph_metrics = graph_metrics.merge(bases.drop(columns="eigengap"),
        on=["domain", "area", "seed", "graph_id", "failure_mode"], how="left")
    graph_metrics.to_csv(args.output / "eigengap_metrics.csv", index=False)

    # Descriptive graph-level diagnostic: average GCN residual gain versus
    # three pre-disruption graph properties. These correlations are not causal.
    gcn = area.pivot_table(
        index=["area", "country", "nodes", "edges", "seed", "failure_mode", "eigengap"],
        columns="model", values="mae"
    ).reset_index()
    gcn["delta_mae"] = gcn["direct_full"] - gcn["residual_full"]
    gcn["density"] = 2 * gcn.edges / (gcn.nodes * (gcn.nodes - 1))
    structural = gcn.groupby(["area", "country", "nodes", "edges"], observed=True).agg(
        delta_mae=("delta_mae", "mean"), density=("density", "first"),
        eigengap=("eigengap", "mean")
    ).reset_index()
    structural.to_csv(args.output / "structural_gain_diagnostics.csv", index=False)
    fig_s, axes_s = plt.subplots(1, 3, figsize=(11.8, 3.45))
    variables = [("nodes", "Number of nodes", True), ("density", "Density", False),
                 ("eigengap", "Relative eigengap", False)]
    for axis, (column, label, log_x) in zip(axes_s, variables):
        x = structural[column].to_numpy(); y = structural.delta_mae.to_numpy()
        rho = spearmanr(x, y).statistic
        axis.scatter(x, y, s=36, color="#2878b5", alpha=.85)
        axis.axhline(0, color="black", lw=.8)
        if log_x:
            axis.set_xscale("log")
        axis.set(xlabel=label, ylabel="Direct MAE - residual MAE",
                 title=f"Spearman $\\rho$={rho:.2f}")
    fig_s.tight_layout()
    fig_s.savefig(args.output / "structural_gain_diagnostics.pdf", bbox_inches="tight")
    fig_s.savefig(args.output / "structural_gain_diagnostics.png", dpi=240, bbox_inches="tight")
    plt.close(fig_s)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    focus = summary[summary.model.isin(["spectral", "spectral_second_order", "direct_edge_mpnn", "residual_edge_mpnn"])]
    for model, marker in [("spectral", "o"), ("spectral_second_order", "s")]:
        part = graph_metrics[(graph_metrics.domain == "osm") & (graph_metrics.model == model)]
        values = part.groupby("eigengap_bin", observed=True).mae.mean().reindex(["low", "medium", "high"])
        axes[0].plot(values.index, values.values, marker=marker, label=model.replace("_", " "))
    axes[0].set(title="Spectral error by relative eigengap", ylabel="MAE"); axes[0].legend(fontsize=8)
    p = paired[paired.failure_mode == "targeted"]
    axes[1].bar(p.backbone, p.direct_minus_residual, color="#2878b5")
    axes[1].errorbar(p.backbone, p.direct_minus_residual,
                     yerr=[p.direct_minus_residual-p.ci_low, p.ci_high-p.direct_minus_residual], fmt="none", color="black")
    axes[1].axhline(0, color="black", lw=.8); axes[1].set(title="Targeted disruption", ylabel="Direct MAE - residual MAE")
    c = correction[(correction.domain == "osm") & (correction.model == "residual_edge_mpnn")]
    c.groupby("failure_mode").slope.mean().plot.bar(ax=axes[2], color="#d9534f")
    axes[2].axhline(1, color="black", ls="--", lw=.8); axes[2].set(title="Learned vs needed correction", ylabel="Regression slope", xlabel="")
    plt.tight_layout(); plt.savefig(args.output / "jcn2_diagnostics.pdf"); plt.savefig(args.output / "jcn2_diagnostics.png", dpi=240); plt.close()
    print(summary.to_string(index=False)); print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
