from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEEDS = (11, 22, 33, 44, 55)


def bootstrap(values, rng, draws=10_000):
    values = np.asarray(values, dtype=float)
    means = np.mean(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
    return float(np.mean(values)), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def load_predictions(root, seeds):
    frames = []
    for seed in seeds:
        frame = pd.read_csv(root / f"seed_{seed}" / "predictions.csv")
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_loao(frame, rng):
    rows = []
    for (seed, model, mode), group in frame.groupby(["seed", "model", "failure_mode"]):
        rows.append({"seed": seed, "model": model, "failure_mode": mode,
                     "mae": mean_absolute_error(group.target, group.prediction),
                     "rmse": mean_squared_error(group.target, group.prediction) ** .5,
                     "r2": r2_score(group.target, group.prediction)})
    by_seed = pd.DataFrame(rows)
    summary = []
    for (model, mode), group in by_seed.groupby(["model", "failure_mode"]):
        for metric in ("mae", "rmse", "r2"):
            mean, low, high = bootstrap(group[metric], rng)
            summary.append({"model": model, "failure_mode": mode, "metric": metric,
                            "mean": mean, "ci_low": low, "ci_high": high})
    return by_seed, pd.DataFrame(summary)


def paired_loao(by_seed, rng):
    wide = by_seed.pivot(index=["seed", "failure_mode"], columns="model", values="mae").reset_index()
    wide["delta_residual_minus_direct"] = wide.residual_full - wide.direct_full
    rows = []
    for mode, group in wide.groupby("failure_mode"):
        mean, low, high = bootstrap(group.delta_residual_minus_direct, rng)
        rows.append({"failure_mode": mode, "mean_delta_residual_minus_direct": mean,
                     "ci_low": low, "ci_high": high})
    return wide, pd.DataFrame(rows)


def calibration(frame):
    rows = []
    for keys, group in frame.groupby(["seed", "model", "failure_mode"]):
        fit = LinearRegression().fit(group[["prediction"]], group.target)
        rows.append({"seed": keys[0], "model": keys[1], "failure_mode": keys[2],
                     "calibration_intercept": float(fit.intercept_),
                     "calibration_slope": float(fit.coef_[0]),
                     "mean_error": float(np.mean(group.prediction - group.target))})
    return pd.DataFrame(rows)


def reliability_comparison(paper, aware, rng):
    rows = []
    for seed in SEEDS:
        for domain in ("synthetic", "osm"):
            for mode in ("independent", "spatial_cluster"):
                full = paper[(paper.seed == seed) & (paper.domain == domain) &
                             (paper.failure_mode == mode) & (paper.model == "residual_full")]
                rel = aware[(aware.seed == seed) & (aware.domain == domain) &
                            (aware.failure_mode == mode) & (aware.model == "residual_reliability")]
                rows.append({"seed": seed, "domain": domain, "failure_mode": mode,
                             "residual_full_mae": mean_absolute_error(full.target, full.prediction),
                             "reliability_aware_mae": mean_absolute_error(rel.target, rel.prediction)})
    by_seed = pd.DataFrame(rows)
    by_seed["delta_aware_minus_full"] = (by_seed.reliability_aware_mae - by_seed.residual_full_mae)
    summary = []
    for (domain, mode), group in by_seed.groupby(["domain", "failure_mode"]):
        mean, low, high = bootstrap(group.delta_aware_minus_full, rng)
        summary.append({"domain": domain, "failure_mode": mode,
                        "mean_delta_aware_minus_full": mean, "ci_low": low, "ci_high": high})
    return by_seed, pd.DataFrame(summary)


def architecture_comparison(paper, architecture, rng):
    core = paper[paper.model.isin(["direct_full", "residual_full"])]
    data = pd.concat([core, architecture[architecture.model.isin(["direct_sage", "residual_sage"])]],
                     ignore_index=True)
    rows = []
    for (seed, domain, mode, model), group in data.groupby(
            ["seed", "domain", "failure_mode", "model"]):
        rows.append({"seed": seed, "domain": domain, "failure_mode": mode, "model": model,
                     "mae": mean_absolute_error(group.target, group.prediction),
                     "rmse": mean_squared_error(group.target, group.prediction) ** .5,
                     "r2": r2_score(group.target, group.prediction)})
    by_seed = pd.DataFrame(rows)
    summary = []
    for (domain, mode, model), group in by_seed.groupby(["domain", "failure_mode", "model"]):
        mean, low, high = bootstrap(group.mae, rng)
        summary.append({"domain": domain, "failure_mode": mode, "model": model,
                        "mae_mean": mean, "mae_ci_low": low, "mae_ci_high": high})
    return by_seed, pd.DataFrame(summary)


def architecture_paired(by_seed, rng):
    rows = []
    for (domain, mode), group in by_seed.groupby(["domain", "failure_mode"]):
        wide = group.pivot(index="seed", columns="model", values="mae")
        for backbone, direct, residual in (("GCN", "direct_full", "residual_full"),
                                           ("GraphSAGE", "direct_sage", "residual_sage")):
            delta = wide[direct] - wide[residual]
            mean, low, high = bootstrap(delta, rng)
            rows.append({"domain": domain, "failure_mode": mode, "backbone": backbone,
                         "mean_direct_minus_residual": mean, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def make_figure(loao_summary, reliability, scaling, output):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax = axes[0]
    for label, column, marker in (("Exact eigensolve", "exact_ms", "o"),
                                   ("Spectral, amortized", "amortized_ms", "s"),
                                   ("Update only", "update_ms", "^"),
                                   ("Residual GCN", "gnn_ms", "D")):
        ax.plot(scaling.nodes, scaling[column], marker=marker, label=label)
    ax.set(xlabel="Nodes", ylabel="Time per scenario (ms)", yscale="log", title="(a) Controlled scaling")
    ax.legend(fontsize=8)

    ax = axes[1]
    mae = loao_summary[loao_summary.metric == "mae"].copy()
    x = np.arange(2)
    for i, model in enumerate(("direct_full", "residual_full")):
        part = mae[mae.model == model].set_index("failure_mode").loc[["independent", "spatial_cluster"]]
        ax.bar(x + (i - .5) * .34, part["mean"], .34,
               yerr=[part["mean"] - part.ci_low, part.ci_high - part["mean"]],
               capsize=3, label=model.replace("_", " "))
    ax.set_xticks(x, ["Independent", "Spatial"])
    ax.set(ylabel="MAE", title="(b) Leave-one-area-out")
    ax.legend(fontsize=8)

    ax = axes[2]
    labels, vals, lows, highs, colors = [], [], [], [], []
    for _, row in reliability.iterrows():
        labels.append(f"{row.domain[:3]}.\n{row.failure_mode[:3]}.")
        vals.append(row.mean_delta_aware_minus_full)
        lows.append(row.mean_delta_aware_minus_full - row.ci_low)
        highs.append(row.ci_high - row.mean_delta_aware_minus_full)
        colors.append("#b2182b" if row.mean_delta_aware_minus_full > 0 else "#2166ac")
    ax.bar(np.arange(len(vals)), vals, color=colors,
           yerr=[lows, highs], capsize=3)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(np.arange(len(vals)), labels)
    ax.set(ylabel=r"MAE change (aware $-$ full)", title="(c) Reliability-context ablation")
    fig.tight_layout()
    fig.savefig(output / "journal_extension_results.pdf", bbox_inches="tight")
    fig.savefig(output / "journal_extension_results.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_journal_figures(loao_summary, reliability, architecture, large, output):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, column, marker in (("Exact sparse eigensolve", "exact_ms", "o"),
                                  ("Intact setup", "setup_ms", "v"),
                                  ("Hybrid: amortized prior + GCN", "hybrid_ms", "D"),
                                  ("Residual GCN inference", "gnn_ms", "s"),
                                  ("Spectral update only", "update_ms", "^")):
        ax.plot(large.nodes, large[column], marker=marker, linewidth=1.8, label=label)
    ax.set(xlabel="Number of nodes", ylabel="Milliseconds per scenario", xscale="log",
           yscale="log")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(output / "large_scale_runtime.pdf", bbox_inches="tight")
    fig.savefig(output / "large_scale_runtime.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    ax = axes[0]; mae = loao_summary[loao_summary.metric == "mae"]
    for i, model in enumerate(("direct_full", "residual_full")):
        part = mae[mae.model == model].set_index("failure_mode").loc[["independent", "spatial_cluster"]]
        ax.bar(np.arange(2) + (i - .5) * .34, part["mean"], .34,
               yerr=[part["mean"]-part.ci_low, part.ci_high-part["mean"]], capsize=3,
               label=model.replace("_", " "))
    ax.set_xticks(np.arange(2), ["Independent", "Spatial cluster"])
    ax.set(ylabel="MAE", title="(a) Leave-one-area-out OSM"); ax.legend(fontsize=8)
    ax = axes[1]
    labels = [f"{r.domain.title()}\n{r.failure_mode.replace('_',' ')}" for _, r in reliability.iterrows()]
    values = reliability.mean_delta_aware_minus_full.to_numpy()
    ax.bar(np.arange(4), values, color=["#b2182b" if x > 0 else "#2166ac" for x in values],
           yerr=[values-reliability.ci_low, reliability.ci_high-values], capsize=3)
    ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(np.arange(4), labels, fontsize=8)
    ax.set(ylabel="MAE change (context minus full)", title="(b) Reliability-context ablation")
    fig.tight_layout(); fig.savefig(output / "transfer_and_ablation.pdf", bbox_inches="tight")
    fig.savefig(output / "transfer_and_ablation.png", dpi=240, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, default=Path("outputs/paper"))
    parser.add_argument("--aware-root", type=Path, default=Path("outputs/journal"))
    parser.add_argument("--loao-root", type=Path, default=Path("outputs/geographic_transfer"))
    parser.add_argument("--arch-root", type=Path, default=Path("outputs/journal_arch"))
    parser.add_argument("--output", type=Path, default=Path("outputs/journal_summary"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260902)
    loao = load_predictions(args.loao_root, SEEDS)
    paper = load_predictions(args.paper_root, SEEDS)
    aware = load_predictions(args.aware_root, SEEDS)
    architecture = load_predictions(args.arch_root, SEEDS)
    loao_by_seed, loao_summary = summarize_loao(loao, rng)
    loao_paired_by_seed, loao_paired_summary = paired_loao(loao_by_seed, rng)
    calibration_table = calibration(loao)
    reliability_by_seed, reliability_summary = reliability_comparison(paper, aware, rng)
    architecture_by_seed, architecture_summary = architecture_comparison(paper, architecture, rng)
    architecture_paired_summary = architecture_paired(architecture_by_seed, rng)
    worst = loao.assign(abs_error=lambda x: abs(x.prediction - x.target)).nlargest(30, "abs_error")
    for name, table in (("loao_metrics_by_seed.csv", loao_by_seed),
                        ("loao_metrics_bootstrap.csv", loao_summary),
                        ("loao_paired_by_seed.csv", loao_paired_by_seed),
                        ("loao_paired_bootstrap.csv", loao_paired_summary),
                        ("loao_calibration.csv", calibration_table),
                        ("reliability_ablation_by_seed.csv", reliability_by_seed),
                        ("reliability_ablation_bootstrap.csv", reliability_summary),
                        ("architecture_metrics_by_seed.csv", architecture_by_seed),
                        ("architecture_metrics_bootstrap.csv", architecture_summary),
                        ("architecture_paired_bootstrap.csv", architecture_paired_summary),
                        ("loao_worst_cases.csv", worst)):
        table.to_csv(args.output / name, index=False)
    scaling = pd.read_csv(args.output / "scaling_summary.csv")
    make_figure(loao_summary, reliability_summary, scaling, args.output)
    large = pd.read_csv(args.output / "large_scaling_summary.csv")
    make_journal_figures(loao_summary, reliability_summary, architecture_summary, large, args.output)
    print(loao_summary.to_string(index=False))
    print(reliability_summary.to_string(index=False))


if __name__ == "__main__":
    main()
