from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MODELS = [
    "direct_full", "direct_no_fiedler", "direct_no_coordinates",
    "residual_full", "residual_no_fiedler", "residual_no_coordinates",
    "deepsets", "summary_mlp", "spectral",
]
LABELS = {
    "direct_full": "Direct GCN", "direct_no_fiedler": "Direct -Fiedler",
    "direct_no_coordinates": "Direct -coords", "residual_full": "Residual GCN",
    "residual_no_fiedler": "Residual -Fiedler",
    "residual_no_coordinates": "Residual -coords", "deepsets": "Deep Sets",
    "summary_mlp": "Summary MLP", "spectral": "Spectral",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paper_summary"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def scores(frame):
    target = frame.target.to_numpy()
    pred = frame.prediction.to_numpy()
    rho = 0.0 if np.std(pred) < 1e-12 or np.std(target) < 1e-12 else float(spearmanr(target, pred).statistic)
    target_ss = np.sum((target - target.mean()) ** 2)
    r2 = np.nan if target_ss < 1e-12 else float(1 - np.sum((target - pred) ** 2) / target_ss)
    return pd.Series({
        "mae": float(np.mean(np.abs(target - pred))),
        "rmse": float(np.sqrt(np.mean((target - pred) ** 2))),
        "r2": r2,
        "spearman": rho,
    })


def bootstrap_seed_mean(values, draws, rng):
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975])


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions, runtime_rows = [], []
    for directory in sorted(args.input.glob("seed_*")):
        seed = int(directory.name.split("_")[-1])
        frame = pd.read_csv(directory / "predictions.csv")
        frame["seed"] = seed
        predictions.append(frame)
        payload = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        runtime_rows.extend([
            {"seed": seed, "method": "exact", "domain": "synthetic", "seconds": payload["reference_runtime"]["synthetic_exact_mean_seconds"]},
            {"seed": seed, "method": "spectral", "domain": "synthetic", "seconds": payload["reference_runtime"]["synthetic_spectral_mean_seconds"]},
            {"seed": seed, "method": "exact", "domain": "osm", "seconds": payload["reference_runtime"]["osm_exact_mean_seconds"]},
            {"seed": seed, "method": "spectral", "domain": "osm", "seconds": payload["reference_runtime"]["osm_spectral_mean_seconds"]},
        ])
        for model, values in payload["models"].items():
            syn_times = [v["inference_seconds_per_scenario"] for v in values["synthetic"].values()]
            osm_times = [v["inference_seconds_per_scenario"] for v in values["osm"].values()]
            runtime_rows.extend([
                {"seed": seed, "method": model, "domain": "synthetic", "seconds": float(np.mean(syn_times)), "train_seconds": values["train_seconds"]},
                {"seed": seed, "method": model, "domain": "osm", "seconds": float(np.mean(osm_times)), "train_seconds": values["train_seconds"]},
            ])
    data = pd.concat(predictions, ignore_index=True)
    data.to_csv(args.output / "all_predictions.csv", index=False)

    per_seed = data.groupby(["seed", "domain", "failure_mode", "model"], sort=False).apply(scores, include_groups=False).reset_index()
    per_seed.to_csv(args.output / "metrics_by_seed_mode.csv", index=False)
    rng = np.random.default_rng(20260902)
    summary_rows = []
    for keys, group in per_seed.groupby(["domain", "failure_mode", "model"]):
        row = dict(zip(["domain", "failure_mode", "model"], keys))
        for metric in ("mae", "rmse", "r2", "spearman"):
            low, high = bootstrap_seed_mean(group[metric], args.bootstrap, rng)
            row.update({f"{metric}_mean": group[metric].mean(), f"{metric}_sd": group[metric].std(),
                        f"{metric}_ci_low": low, f"{metric}_ci_high": high})
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "metrics_summary_bootstrap.csv", index=False)

    paired_rows = []
    for domain in ("synthetic", "osm"):
        for mode in ("independent", "spatial_cluster"):
            part = per_seed[(per_seed.domain == domain) & (per_seed.failure_mode == mode)]
            pivot = part.pivot(index="seed", columns="model", values=["mae", "spearman"])
            for comparison in ("direct_full", "residual_no_fiedler", "residual_no_coordinates"):
                for metric in ("mae", "spearman"):
                    delta = pivot[metric][comparison] - pivot[metric]["residual_full"]
                    low, high = bootstrap_seed_mean(delta, args.bootstrap, rng)
                    paired_rows.append({"domain": domain, "failure_mode": mode,
                        "comparison_minus_residual": comparison, "metric": metric,
                        "mean_difference": delta.mean(), "ci_low": low, "ci_high": high})
    pd.DataFrame(paired_rows).to_csv(args.output / "paired_bootstrap_differences.csv", index=False)

    # Error analyses use per-seed MAE first, then summarize seeds.
    analysis_frames = []
    enriched = data.copy()
    enriched["connectivity_group"] = np.where(enriched.connected, "connected", "disconnected")
    enriched["failure_count_group"] = pd.cut(enriched.failed_count, [0, 1, 3, 5, np.inf], labels=["1", "2-3", "4-5", "6+"])
    enriched["density_quartile"] = enriched.groupby(["seed", "domain"])["base_density"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    )
    enriched["clip_group"] = np.where(enriched.spectral_clipped, "spectral_clipped", "not_clipped")
    for column in ("connectivity_group", "failure_count_group", "density_quartile", "clip_group", "area"):
        grouped = enriched.groupby(["seed", "domain", "model", column], observed=True).apply(scores, include_groups=False).reset_index()
        grouped["analysis"] = column
        grouped = grouped.rename(columns={column: "group"})
        analysis_frames.append(grouped)
    analysis = pd.concat(analysis_frames, ignore_index=True)
    analysis.to_csv(args.output / "error_analysis_by_seed.csv", index=False)
    analysis.groupby(["analysis", "domain", "model", "group"], observed=True)[["mae", "rmse", "r2", "spearman"]].agg(["mean", "std"]).to_csv(args.output / "error_analysis_summary.csv")

    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(args.output / "runtime_by_seed.csv", index=False)
    runtime.groupby(["domain", "method"])["seconds"].agg(["mean", "std"]).to_csv(args.output / "runtime_summary.csv")

    # Compact paper figures.
    focus = summary[(summary.failure_mode == "spatial_cluster") & summary.model.isin(["direct_full", "residual_full", "deepsets", "summary_mlp", "spectral"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, domain in zip(axes, ["synthetic", "osm"]):
        part = focus[focus.domain == domain].set_index("model").loc[["direct_full", "residual_full", "deepsets", "summary_mlp", "spectral"]]
        x = np.arange(len(part))
        ax.bar(x, part.mae_mean, color=["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#ff7f0e"])
        ax.errorbar(x, part.mae_mean, yerr=[part.mae_mean-part.mae_ci_low, part.mae_ci_high-part.mae_mean], fmt="none", color="black", capsize=3)
        ax.set_xticks(x, [LABELS[m] for m in part.index], rotation=25, ha="right")
        ax.set_title(f"{domain.title()}: spatial clusters")
        ax.set_ylabel("MAE with seed-bootstrap 95% CI")
    plt.tight_layout()
    plt.savefig(args.output / "paper_primary_results.pdf")
    plt.savefig(args.output / "paper_primary_results.png", dpi=240)
    plt.close()

    print(summary[(summary.model.isin(["direct_full", "residual_full", "deepsets", "summary_mlp", "spectral"]))].to_string(index=False))


if __name__ == "__main__":
    main()
