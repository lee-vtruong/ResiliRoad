from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from resilience.data import generate_scenarios_for_graph
from resilience.osm import load_preprocessed_network
from resilience.train import evaluate, train_model


CONFIGS = {
    "direct_full": dict(model_kind="gcn", use_fiedler=True, use_coordinates=True, residual=False),
    "residual_full": dict(model_kind="gcn", use_fiedler=True, use_coordinates=True, residual=True),
}
MODES = ("independent", "spatial_cluster", "targeted")


def parse_args():
    parser = argparse.ArgumentParser(description="Leave-one-country-out OSM transfer")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenarios-per-area-mode", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--training-protocol", choices=("joint", "per_mode"),
                        default="per_mode")
    parser.add_argument("--osm-manifest", type=Path, default=Path("data/osm/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.osm_manifest.read_text(encoding="utf-8"))
    by_country: dict[str, list] = {}
    for area_index, site in enumerate(manifest):
        graph = load_preprocessed_network(Path(site["path"]))
        for mode_index, mode in enumerate(MODES):
            scenarios = generate_scenarios_for_graph(
                graph, args.scenarios_per_area_mode,
                seed=args.seed + 20_000 + area_index * 100 + mode_index * 10,
                graph_id=area_index, failure_mode=mode, area=site["name"],
            )
            by_country.setdefault(site["country"], []).extend(scenarios)

    countries = list(by_country)
    rows, folds = [], []
    for fold_index, test_country in enumerate(countries):
        validation_country = countries[(fold_index + 1) % len(countries)]
        training_countries = [c for c in countries if c not in {test_country, validation_country}]
        train = [item for country in training_countries for item in by_country[country]]
        validation = list(by_country[validation_country])
        test = list(by_country[test_country])
        folds.append({"test_country": test_country, "validation_country": validation_country,
                      "training_countries": training_countries})
        training_modes = ("joint",) if args.training_protocol == "joint" else MODES
        for training_mode in training_modes:
            mode_train = train if training_mode == "joint" else [x for x in train if x.failure_mode == training_mode]
            mode_validation = validation if training_mode == "joint" else [x for x in validation if x.failure_mode == training_mode]
            evaluation_modes = MODES if training_mode == "joint" else (training_mode,)
            for model_name, config in CONFIGS.items():
                started = perf_counter()
                model, history, device = train_model(
                    list(mode_train), list(mode_validation), epochs=args.epochs,
                    seed=args.seed + fold_index + (
                        0 if training_mode == "joint" else 100 * MODES.index(training_mode)
                    ),
                    **config,
                )
                train_seconds = perf_counter() - started
                pd.DataFrame(history).to_csv(
                    args.output / f"history_{test_country}_{training_mode}_{model_name}.csv", index=False
                )
                for mode in evaluation_modes:
                    subset = [item for item in test if item.failure_mode == mode]
                    _, target, prediction, spectral = evaluate(
                        model, mode_train, subset, device,
                        use_fiedler=config["use_fiedler"],
                        use_coordinates=config["use_coordinates"],
                    )
                    for item, y, pred, spec in zip(subset, target, prediction, spectral):
                        rows.append({
                            "seed": args.seed, "test_country": test_country,
                            "validation_country": validation_country, "area": item.area,
                            "training_protocol": args.training_protocol,
                            "training_mode": training_mode,
                            "model": model_name, "failure_mode": mode, "target": y,
                            "prediction": pred, "spectral": spec,
                            "failed_count": item.failed_count,
                            "relative_eigengap": item.relative_eigengap,
                            "base_density": item.base_density, "nodes": item.x.shape[0],
                            "train_seconds": train_seconds,
                        })
    pd.DataFrame(rows).to_csv(args.output / "predictions.csv", index=False)
    metadata = {"seed": args.seed, "epochs": args.epochs,
                "scenarios_per_area_mode": args.scenarios_per_area_mode,
                "protocol": "leave-one-country-out; next country in manifest order validates",
                "training_protocol": args.training_protocol,
                "modes": list(MODES), "folds": folds}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
