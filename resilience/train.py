from __future__ import annotations

import copy
import random

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

from .data import Scenario
from .model import ScenarioGCN


def split_by_graph(data: list[Scenario], seed: int = 42):
    """Prevent leakage: all scenarios from one base graph stay in one split."""
    graph_ids = sorted({item.graph_id for item in data})
    random.Random(seed).shuffle(graph_ids)
    n = len(graph_ids)
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.15 * n))
    train_ids = set(graph_ids[:n_train])
    val_ids = set(graph_ids[n_train:n_train + n_val])
    test_ids = set(graph_ids[n_train + n_val:])
    if not test_ids:
        test_ids = {graph_ids[-1]}
        train_ids.discard(graph_ids[-1])
    return (
        [x for x in data if x.graph_id in train_ids],
        [x for x in data if x.graph_id in val_ids],
        [x for x in data if x.graph_id in test_ids],
    )


def _features(item: Scenario, use_fiedler: bool) -> torch.Tensor:
    if use_fiedler:
        return item.x
    # Feature 2 is the absolute normalized Fiedler coordinate.
    return item.x[:, [0, 1, 3, 4]]


def _predict(
    model: ScenarioGCN,
    data: list[Scenario],
    device: torch.device,
    use_fiedler: bool,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.array([
            model(
                _features(item, use_fiedler).to(device),
                item.adjacency.to(device),
                torch.tensor(item.spectral_prediction, dtype=torch.float32, device=device),
            ).cpu().item()
            for item in data
        ])


def train_model(
    train,
    validation,
    epochs=80,
    lr=2e-3,
    seed=42,
    use_fiedler=True,
    residual=False,
):
    # Dense 35--65 node graphs are faster with one BLAS thread; many threads
    # spend more time scheduling tiny matrix products than computing them.
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScenarioGCN(input_dim=5 if use_fiedler else 4, residual=residual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history = []
    patience = 15
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train)
        losses = []
        for item in train:
            optimizer.zero_grad()
            prediction = model(
                _features(item, use_fiedler).to(device),
                item.adjacency.to(device),
                torch.tensor(item.spectral_prediction, dtype=torch.float32, device=device),
            )
            target = torch.tensor(item.target, dtype=torch.float32, device=device)
            loss = loss_fn(prediction, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_predictions = _predict(model, validation, device, use_fiedler)
        val_targets = np.array([item.target for item in validation])
        val_loss = float(np.mean((val_predictions - val_targets) ** 2))
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_mse": val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, history, device


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    if np.std(targets) < 1e-12 or np.std(predictions) < 1e-12:
        correlation = 0.0
    else:
        correlation = spearmanr(targets, predictions).statistic
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "r2": float(r2_score(targets, predictions)),
        "spearman": float(correlation) if not np.isnan(correlation) else 0.0,
    }


def evaluate(model, train, test, device, use_fiedler=True):
    targets = np.array([item.target for item in test])
    gcn = _predict(model, test, device, use_fiedler)
    spectral = np.array([item.spectral_prediction for item in test])
    train_mean = float(np.mean([item.target for item in train]))
    constant = np.repeat(train_mean, len(targets))
    return {
        "gcn": regression_metrics(targets, gcn),
        "spectral_first_order": regression_metrics(targets, spectral),
        "train_mean_constant": regression_metrics(targets, constant),
    }, targets, gcn, spectral
