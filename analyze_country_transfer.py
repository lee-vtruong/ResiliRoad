import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clustered_interval(frame, column, rng, draws=10_000):
    matrix = frame.pivot_table(index="test_country", columns="seed", values=column).to_numpy()
    n_country, n_seed = matrix.shape
    country_draw = rng.integers(0, n_country, (draws, n_country))
    seed_draw = rng.integers(0, n_seed, (draws, n_country))
    values = matrix[country_draw, seed_draw].mean(axis=1)
    return np.quantile(values, [0.025, 0.975])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/country_transfer_per_mode"))
    parser.add_argument("--output", type=Path, default=Path("outputs/country_transfer_per_mode_summary"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(path) for path in sorted(args.input.glob("seed_*/predictions.csv"))]
    data = pd.concat(frames, ignore_index=True)
    data["abs_error"] = abs(data.target - data.prediction)
    units = data.groupby(["test_country", "seed", "failure_mode", "model"], observed=True).agg(
        mae=("abs_error", "mean"), areas=("area", "nunique"), n=("target", "size")
    ).reset_index()
    units.to_csv(args.output / "country_seed_metrics.csv", index=False)
    wide = units.pivot(index=["test_country", "seed", "failure_mode"], columns="model", values="mae").reset_index()
    wide["direct_minus_residual"] = wide.direct_full - wide.residual_full
    rng = np.random.default_rng(20260903)
    rows = []
    for mode, group in wide.groupby("failure_mode"):
        lo, hi = clustered_interval(group, "direct_minus_residual", rng)
        rows.append({"failure_mode": mode,
                     "direct_mae": group.direct_full.mean(),
                     "residual_mae": group.residual_full.mean(),
                     "direct_minus_residual": group.direct_minus_residual.mean(),
                     "ci_low": lo, "ci_high": hi,
                     "countries": group.test_country.nunique(), "seeds": group.seed.nunique()})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "loco_summary.csv", index=False)
    per_country = wide.groupby(["test_country", "failure_mode"], observed=True).agg(
        direct_mae=("direct_full", "mean"), residual_mae=("residual_full", "mean"),
        direct_minus_residual=("direct_minus_residual", "mean")
    ).reset_index()
    per_country.to_csv(args.output / "loco_by_country.csv", index=False)

    fig, axis = plt.subplots(figsize=(8.4, 4.2))
    pivot = per_country.pivot(index="test_country", columns="failure_mode", values="direct_minus_residual")
    pivot = pivot.rename(columns={"independent": "Independent",
                                  "spatial_cluster": "Spatial cluster",
                                  "targeted": "Targeted"})
    pivot.plot.bar(ax=axis, color=["#2878b5", "#65a765", "#d9534f"])
    axis.axhline(0, color="black", lw=.8)
    axis.set(ylabel="Direct MAE - residual MAE", xlabel="Held-out country",
             title="Leave-one-country-out OSM transfer")
    axis.legend(title="Failure mode", frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "loco_by_country.pdf", bbox_inches="tight")
    fig.savefig(args.output / "loco_by_country.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
