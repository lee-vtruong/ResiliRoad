from __future__ import annotations

import torch
from torch import nn


class ScenarioGCN(nn.Module):
    """Small dense GCN for graph-level regression on variable-size graphs."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 48, dropout: float = 0.15):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = adjacency @ h
            h = self.dropout(torch.relu(layer(h)))
        mean_pool = h.mean(dim=0)
        max_pool = h.max(dim=0).values
        failure_rate = x[:, 1].mean().reshape(1)
        return self.head(torch.cat([mean_pool, max_pool, failure_rate])).squeeze()

