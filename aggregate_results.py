from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ("mae", "rmse", "r2", "spearman")
METHODS = ("gcn", "spectral_first_order", "train_mean_constant")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate independent benchmark seeds")
    parser.add_argument("--input", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("outputs/summary"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    grouped_rows = []
    prediction_frames = []
    for directory in sorted(args.input.glob("seed_*")):
        seed = int(directory.name.split("_")[-1])
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        for method in METHODS:
            rows.append({"seed": seed, "method": method, **metrics[method]})
        for group, values in metrics.get("by_failed_count", {}).items():
            for method in ("gcn", "spectral_first_order"):
                grouped_rows.append({
                    "seed": seed, "failed_group": group, "n": values["n"],
                    "method": method, **values[method],
                })
        predictions = pd.read_csv(directory / "predictions.csv")
        predictions["seed"] = seed
        prediction_frames.append(predictions)

    raw = pd.DataFrame(rows)
    grouped_raw = pd.DataFrame(grouped_rows)
    raw.to_csv(args.output / "metrics_by_seed.csv", index=False)
    grouped_raw.to_csv(args.output / "metrics_by_failure_group_and_seed.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(args.output / "all_test_predictions.csv", index=False)

    summary = raw.groupby("method")[list(METRICS)].agg(["mean", "std"])
    summary.to_csv(args.output / "metrics_summary.csv")
    group_summary = grouped_raw.groupby(["failed_group", "method"])[list(METRICS)].agg(["mean", "std"])
    group_summary.to_csv(args.output / "failure_group_summary.csv")

    order = ["1", "2-3", "4-5", "6+"]
    mae = grouped_raw.groupby(["failed_group", "method"])["mae"].agg(["mean", "std"])
    x = np.arange(len(order))
    width = 0.36
    plt.figure(figsize=(7.2, 4.3))
    for offset, method, label, color in [
        (-width / 2, "gcn", "ScenarioGCN", "#1f77b4"),
        (width / 2, "spectral_first_order", "Spectral first order", "#ff7f0e"),
    ]:
        means = [mae.loc[(group, method), "mean"] for group in order]
        stds = [mae.loc[(group, method), "std"] for group in order]
        plt.bar(x + offset, means, width, yerr=stds, capsize=3, label=label, color=color)
    plt.xticks(x, order)
    plt.xlabel("Number of failed edges")
    plt.ylabel("MAE (mean $\\pm$ SD over 5 seeds)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.output / "mae_by_failure_group.pdf")
    plt.savefig(args.output / "mae_by_failure_group.png", dpi=240)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    axes[0].scatter(predictions["target"], predictions["spectral"], s=8, alpha=.35, color="#ff7f0e")
    axes[0].set_title("Spectral first order")
    axes[1].scatter(predictions["target"], predictions["gcn"], s=8, alpha=.35, color="#1f77b4")
    axes[1].set_title("ScenarioGCN")
    for axis in axes:
        axis.plot([0, 1], [0, 1], "k--", linewidth=.8)
        axis.set_xlim(-.02, 1.02)
        axis.set_ylim(-.02, 1.02)
        axis.set_xlabel("True relative loss")
    axes[0].set_ylabel("Predicted relative loss")
    plt.tight_layout()
    plt.savefig(args.output / "pooled_predictions.pdf")
    plt.savefig(args.output / "pooled_predictions.png", dpi=240)
    plt.close()

    readable = {}
    for method in METHODS:
        readable[method] = {
            metric: {
                "mean": float(summary.loc[method, (metric, "mean")]),
                "std": float(summary.loc[method, (metric, "std")]),
            } for metric in METRICS
        }
    (args.output / "summary.json").write_text(json.dumps(readable, indent=2), encoding="utf-8")
    print(json.dumps(readable, indent=2))


if __name__ == "__main__":
    main()
