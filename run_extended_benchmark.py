from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from resilience.data import generate_dataset, generate_scenarios_for_graph
from resilience.osm import load_preprocessed_network
from resilience.train import evaluate, regression_metrics, split_by_graph, train_model


CONFIGS = {
    "direct_full": {"use_fiedler": True, "residual": False},
    "direct_no_fiedler": {"use_fiedler": False, "residual": False},
    "residual_full": {"use_fiedler": True, "residual": True},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--osm-samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--osm-graph", type=Path, default=Path("data/osm/hcmus_650m_drive.graphml"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scenario_frame(data, predictions, model_name, split):
    return pd.DataFrame({
        "target": [item.target for item in data],
        "prediction": predictions,
        "spectral": [item.spectral_prediction for item in data],
        "failed_count": [item.failed_count for item in data],
        "graph_id": [item.graph_id for item in data],
        "model": model_name,
        "split": split,
    })


def grouped_metrics(data, predictions):
    target = np.array([item.target for item in data])
    failed = np.array([item.failed_count for item in data])
    result = {}
    for name, mask in {
        "1": failed == 1,
        "2-3": (failed >= 2) & (failed <= 3),
        "4-5": (failed >= 4) & (failed <= 5),
        "6+": failed >= 6,
    }.items():
        if mask.sum() >= 2:
            result[name] = {"n": int(mask.sum()), **regression_metrics(target[mask], predictions[mask])}
    return result


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    synthetic = generate_dataset(args.samples, seed=args.seed)
    train, validation, test = split_by_graph(synthetic, seed=args.seed)
    osm_graph = load_preprocessed_network(args.osm_graph)
    osm_test = generate_scenarios_for_graph(
        osm_graph, args.osm_samples, seed=args.seed + 1000, graph_id=1_000_000 + args.seed
    )
    results = {
        "metadata": {
            "seed": args.seed, "synthetic_samples": len(synthetic),
            "train": len(train), "validation": len(validation), "synthetic_test": len(test),
            "osm_test": len(osm_test), "osm_nodes": osm_graph.number_of_nodes(),
            "osm_edges": osm_graph.number_of_edges(), "epochs": args.epochs,
            "torch_threads": 1,
        },
        "models": {},
    }
    frames = []
    for name, config in CONFIGS.items():
        started = perf_counter()
        model, history, device = train_model(
            list(train), list(validation), epochs=args.epochs, seed=args.seed, **config
        )
        train_seconds = perf_counter() - started
        syn_metrics, syn_target, syn_prediction, spectral = evaluate(
            model, train, test, device, use_fiedler=config["use_fiedler"]
        )
        osm_metrics, osm_target, osm_prediction, osm_spectral = evaluate(
            model, train, osm_test, device, use_fiedler=config["use_fiedler"]
        )
        results["models"][name] = {
            "config": config,
            "train_seconds": train_seconds,
            "synthetic": syn_metrics["gcn"],
            "synthetic_by_failed_count": grouped_metrics(test, syn_prediction),
            "osm": osm_metrics["gcn"],
            "osm_by_failed_count": grouped_metrics(osm_test, osm_prediction),
        }
        frames.append(scenario_frame(test, syn_prediction, name, "synthetic"))
        frames.append(scenario_frame(osm_test, osm_prediction, name, "osm"))
        pd.DataFrame(history).to_csv(args.output / f"history_{name}.csv", index=False)
    # Analytical reference is model independent.
    results["spectral_reference"] = {
        "synthetic": regression_metrics(
            np.array([x.target for x in test]), np.array([x.spectral_prediction for x in test])
        ),
        "osm": regression_metrics(
            np.array([x.target for x in osm_test]), np.array([x.spectral_prediction for x in osm_test])
        ),
    }
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
