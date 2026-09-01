from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from resilience.data import generate_dataset
from resilience.train import evaluate, regression_metrics, split_by_graph, train_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/default"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = generate_dataset(args.samples, seed=args.seed)
    train, validation, test = split_by_graph(data, seed=args.seed)
    model, history, device = train_model(train, validation, epochs=args.epochs, seed=args.seed)
    metrics, targets, gcn, spectral = evaluate(model, train, test, device)
    failed_counts = np.array([item.failed_count for item in test])
    bins = {
        "1": failed_counts == 1,
        "2-3": (failed_counts >= 2) & (failed_counts <= 3),
        "4-5": (failed_counts >= 4) & (failed_counts <= 5),
        "6+": failed_counts >= 6,
    }
    metrics["by_failed_count"] = {
        name: {
            "n": int(mask.sum()),
            "gcn": regression_metrics(targets[mask], gcn[mask]),
            "spectral_first_order": regression_metrics(targets[mask], spectral[mask]),
        }
        for name, mask in bins.items() if mask.sum() >= 2
    }
    metadata = {
        "seed": args.seed,
        "samples": len(data),
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
        "device": str(device),
        "split_unit": "base_graph",
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "dataset_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame({
        "target": targets,
        "gcn": gcn,
        "spectral": spectral,
        "failed_count": failed_counts,
        "graph_id": [item.graph_id for item in test],
    }).to_csv(
        args.output / "predictions.csv", index=False
    )
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    
    limit = max(float(targets.max()), float(gcn.max()), float(spectral.max()), 0.05)
    plt.figure(figsize=(6, 5))
    plt.scatter(targets, spectral, alpha=0.65, label="Spectral first order")
    plt.scatter(targets, gcn, alpha=0.65, label="ScenarioGCN")
    plt.plot([0, limit], [0, limit], "k--", linewidth=1)
    plt.xlabel("True relative drop")
    plt.ylabel("Predicted relative drop")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / "prediction_scatter.png", dpi=180)
    plt.close()

    frame = pd.DataFrame(history)
    plt.figure(figsize=(6, 4))
    plt.plot(frame["epoch"], frame["train_mse"], label="train")
    plt.plot(frame["epoch"], frame["val_mse"], label="validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output / "learning_curve.png", dpi=180)
    plt.close()
    print(json.dumps({"metadata": metadata, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
