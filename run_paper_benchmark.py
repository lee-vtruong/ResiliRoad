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
    "direct_full": dict(model_kind="gcn", use_fiedler=True, use_coordinates=True, residual=False),
    "direct_no_fiedler": dict(model_kind="gcn", use_fiedler=False, use_coordinates=True, residual=False),
    "direct_no_coordinates": dict(model_kind="gcn", use_fiedler=True, use_coordinates=False, residual=False),
    "residual_full": dict(model_kind="gcn", use_fiedler=True, use_coordinates=True, residual=True),
    "residual_reliability": dict(model_kind="gcn", use_fiedler=True, use_coordinates=True,
                                 residual=True, reliability_aware=True),
    "residual_no_fiedler": dict(model_kind="gcn", use_fiedler=False, use_coordinates=True, residual=True),
    "residual_no_coordinates": dict(model_kind="gcn", use_fiedler=True, use_coordinates=False, residual=True),
    "deepsets": dict(model_kind="deepsets", use_fiedler=True, use_coordinates=True, residual=False),
    "summary_mlp": dict(model_kind="mlp", use_fiedler=True, use_coordinates=True, residual=False),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--synthetic-per-mode", type=int, default=800)
    parser.add_argument("--osm-per-site-mode", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--osm-manifest", type=Path, default=Path("data/osm/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(CONFIGS),
                        default=None, help="Optional subset of model configurations")
    return parser.parse_args()


def metrics(target, prediction):
    return regression_metrics(np.asarray(target), np.asarray(prediction))


def scenario_frame(data, prediction, model, domain):
    return pd.DataFrame({
        "target": [x.target for x in data], "prediction": prediction,
        "spectral": [x.spectral_prediction for x in data],
        "failed_count": [x.failed_count for x in data],
        "failure_mode": [x.failure_mode for x in data],
        "connected": [x.damaged_connected for x in data],
        "base_density": [x.base_density for x in data],
        "spectral_clipped": [x.spectral_clipped for x in data],
        "area": [x.area for x in data], "model": model, "domain": domain,
    })


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    modes = ("independent", "spatial_cluster")
    synthetic_parts = [
        generate_dataset(args.synthetic_per_mode, seed=args.seed, failure_mode=mode)
        for mode in modes
    ]
    synthetic = synthetic_parts[0] + synthetic_parts[1]
    train, validation, synthetic_test = split_by_graph(synthetic, seed=args.seed)

    manifest = json.loads(args.osm_manifest.read_text(encoding="utf-8"))
    osm_sets = {}
    for area_index, site in enumerate(manifest):
        graph = load_preprocessed_network(Path(site["path"]))
        for mode_index, mode in enumerate(modes):
            key = f"{site['name']}::{mode}"
            osm_sets[key] = generate_scenarios_for_graph(
                graph, args.osm_per_site_mode,
                seed=args.seed + 1000 + 100 * area_index + 10 * mode_index,
                graph_id=1_000_000 + 100 * area_index + mode_index,
                failure_mode=mode, area=site["name"],
            )
    osm_test = [item for values in osm_sets.values() for item in values]

    result = {
        "metadata": {
            "seed": args.seed, "epochs": args.epochs,
            "synthetic_train": len(train), "synthetic_validation": len(validation),
            "synthetic_test": len(synthetic_test), "osm_test": len(osm_test),
            "sites": [site["name"] for site in manifest], "failure_modes": list(modes),
            "torch_threads": 1,
        },
        "models": {},
        "reference_runtime": {
            "synthetic_exact_mean_seconds": float(np.mean([x.exact_seconds for x in synthetic_test])),
            "synthetic_spectral_mean_seconds": float(np.mean([x.spectral_seconds for x in synthetic_test])),
            "osm_exact_mean_seconds": float(np.mean([x.exact_seconds for x in osm_test])),
            "osm_spectral_mean_seconds": float(np.mean([x.spectral_seconds for x in osm_test])),
        },
    }
    frames = []
    selected = args.models or list(CONFIGS)
    for name in selected:
        config = CONFIGS[name]
        started = perf_counter()
        model, history, device = train_model(
            list(train), list(validation), epochs=args.epochs, seed=args.seed, **config
        )
        train_seconds = perf_counter() - started
        model_result = {"config": config, "train_seconds": train_seconds, "synthetic": {}, "osm": {}}
        for mode in modes:
            subset = [x for x in synthetic_test if x.failure_mode == mode]
            infer_started = perf_counter()
            evaluation, target, prediction, _ = evaluate(
                model, train, subset, device,
                use_fiedler=config["use_fiedler"], use_coordinates=config["use_coordinates"],
            )
            infer_seconds = (perf_counter() - infer_started) / len(subset)
            model_result["synthetic"][mode] = {**evaluation["gcn"], "inference_seconds_per_scenario": infer_seconds}
            frames.append(scenario_frame(subset, prediction, name, "synthetic"))
        for key, subset in osm_sets.items():
            infer_started = perf_counter()
            evaluation, target, prediction, _ = evaluate(
                model, train, subset, device,
                use_fiedler=config["use_fiedler"], use_coordinates=config["use_coordinates"],
            )
            infer_seconds = (perf_counter() - infer_started) / len(subset)
            model_result["osm"][key] = {**evaluation["gcn"], "inference_seconds_per_scenario": infer_seconds}
            frames.append(scenario_frame(subset, prediction, name, "osm"))
        result["models"][name] = model_result
        pd.DataFrame(history).to_csv(args.output / f"history_{name}.csv", index=False)

    for domain, dataset in (("synthetic", synthetic_test), ("osm", osm_test)):
        target = np.array([x.target for x in dataset])
        prediction = np.array([x.spectral_prediction for x in dataset])
        result[f"spectral_{domain}"] = metrics(target, prediction)
        frames.append(scenario_frame(dataset, prediction, "spectral", domain))

    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_csv(args.output / "predictions.csv", index=False)
    print(json.dumps(result["metadata"], indent=2))
    print(json.dumps(result["reference_runtime"], indent=2))


if __name__ == "__main__":
    main()
