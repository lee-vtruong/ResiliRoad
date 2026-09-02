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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leave-one-area-out evaluation with geographically disjoint splits."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenarios-per-site-mode", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--osm-manifest", type=Path, default=Path("data/osm/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.osm_manifest.read_text(encoding="utf-8"))
    modes = ("independent", "spatial_cluster")
    by_area = {}
    for area_index, site in enumerate(manifest):
        graph = load_preprocessed_network(Path(site["path"]))
        parts = []
        for mode_index, mode in enumerate(modes):
            parts.extend(generate_scenarios_for_graph(
                graph, args.scenarios_per_site_mode,
                seed=args.seed + 10_000 + area_index * 100 + mode_index * 10,
                graph_id=area_index, failure_mode=mode, area=site["name"],
            ))
        by_area[site["name"]] = parts

    areas = list(by_area)
    rows, fold_metadata = [], []
    for held_index, held_area in enumerate(areas):
        validation_area = areas[(held_index + 1) % len(areas)]
        training_areas = [a for a in areas if a not in {held_area, validation_area}]
        train = [x for area in training_areas for x in by_area[area]]
        validation = list(by_area[validation_area])
        test = list(by_area[held_area])
        fold_metadata.append({"held_area": held_area, "validation_area": validation_area,
                              "training_areas": training_areas})
        for model_name, config in CONFIGS.items():
            started = perf_counter()
            model, history, device = train_model(
                list(train), list(validation), epochs=args.epochs,
                seed=args.seed + held_index, **config,
            )
            train_seconds = perf_counter() - started
            pd.DataFrame(history).to_csv(
                args.output / f"history_{held_area}_{model_name}.csv", index=False
            )
            for mode in modes:
                subset = [x for x in test if x.failure_mode == mode]
                _, target, prediction, spectral = evaluate(
                    model, train, subset, device,
                    use_fiedler=config["use_fiedler"],
                    use_coordinates=config["use_coordinates"],
                )
                for item, y, pred, spec in zip(subset, target, prediction, spectral):
                    rows.append({
                        "seed": args.seed, "held_area": held_area,
                        "validation_area": validation_area, "model": model_name,
                        "failure_mode": mode, "target": y, "prediction": pred,
                        "spectral": spec, "failed_count": item.failed_count,
                        "connected": item.damaged_connected,
                        "spectral_clipped": item.spectral_clipped,
                        "base_density": item.base_density,
                        "train_seconds": train_seconds,
                    })

    pd.DataFrame(rows).to_csv(args.output / "predictions.csv", index=False)
    metadata = {
        "seed": args.seed, "epochs": args.epochs,
        "scenarios_per_site_mode": args.scenarios_per_site_mode,
        "protocol": "leave-one-area-out with one geographically distinct validation area",
        "folds": fold_metadata,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
