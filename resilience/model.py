from __future__ import annotations

import torch
from torch import nn


class ScenarioGCN(nn.Module):
    """Small dense GCN for graph-level regression on variable-size graphs."""

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 48,
        dropout: float = 0.15,
        residual: bool = False,
        reliability_aware: bool = False,
    ):
        super().__init__()
        self.residual = residual
        self.reliability_aware = reliability_aware
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1 + (4 if reliability_aware else 0), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        spectral_prediction: torch.Tensor | None = None,
        reliability_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = adjacency @ h
            h = self.dropout(torch.relu(layer(h)))
        mean_pool = h.mean(dim=0)
        max_pool = h.max(dim=0).values
        failure_rate = x[:, 1].mean().reshape(1)
        pooled = [mean_pool, max_pool, failure_rate]
        if self.reliability_aware:
            if reliability_context is None:
                raise ValueError("Reliability-aware mode requires context")
            pooled.append(reliability_context)
        raw = self.head(torch.cat(pooled)).squeeze()
        if self.residual:
            if spectral_prediction is None:
                raise ValueError("Residual mode requires a spectral prediction")
            # Learn a bounded correction to the analytical first-order estimate.
            return torch.clamp(spectral_prediction + torch.tanh(raw), 0.0, 1.0)
        return torch.sigmoid(raw)


class ScenarioGraphSAGE(nn.Module):
    """GraphSAGE-style mean aggregation for graph-level regression."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 48,
                 dropout: float = 0.15, residual: bool = False):
        super().__init__()
        self.residual = residual
        dimensions = [input_dim, hidden_dim, hidden_dim]
        self.layers = nn.ModuleList([
            nn.Linear(2 * dimensions[i], hidden_dim) for i in range(3)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, adjacency, spectral_prediction=None, reliability_context=None):
        h = x
        for layer in self.layers:
            neighbourhood = adjacency @ h
            h = self.dropout(torch.relu(layer(torch.cat([h, neighbourhood], dim=1))))
        pooled = torch.cat([h.mean(dim=0), h.max(dim=0).values,
                            x[:, 1].mean().reshape(1)])
        raw = self.head(pooled).squeeze()
        if self.residual:
            if spectral_prediction is None:
                raise ValueError("Residual mode requires a spectral prediction")
            return torch.clamp(spectral_prediction + torch.tanh(raw), 0.0, 1.0)
        return torch.sigmoid(raw)


class DeepSetsRegressor(nn.Module):
    """Permutation-invariant baseline over node features without message passing."""

    def __init__(self, input_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, adjacency, spectral_prediction=None, reliability_context=None):
        h = self.phi(x)
        pooled = torch.cat([h.mean(dim=0), h.max(dim=0).values])
        return torch.sigmoid(self.rho(pooled).squeeze())


class SummaryMLP(nn.Module):
    """Fixed-summary non-graph baseline."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 4 + 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, adjacency, spectral_prediction=None, reliability_context=None):
        summary = torch.cat([
            x.mean(dim=0), x.std(dim=0, unbiased=False),
            x.min(dim=0).values, x.max(dim=0).values,
            torch.tensor([x.shape[0] / 100.0], device=x.device),
            (adjacency > 0).float().mean().reshape(1),
        ])
        return torch.sigmoid(self.net(summary).squeeze())
