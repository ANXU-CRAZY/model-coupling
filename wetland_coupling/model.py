"""A small residual MLP with two bounded Softmax gates; base models stay frozen."""
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, x):
        return x + self.net(x)


class DualGate(nn.Module):
    def __init__(self, n_features=6, width=32, dropout=0.1, min_weight=0.1,
                 active_heads=(True, True)):
        super().__init__()
        if not 0 < min_weight < 0.5:
            raise ValueError("min_weight must be between 0 and 0.5")
        self.min_weight = float(min_weight)
        self.active_heads = tuple(active_heads)
        self.encoder = nn.Sequential(nn.Linear(n_features, width), nn.GELU(),
                                     ResidualBlock(width, dropout), nn.LayerNorm(width))
        self.head = nn.Linear(width, 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def weights(self, x):
        p = self.head(self.encoder(x)).reshape(-1, 2, 2).softmax(dim=-1)
        w = self.min_weight + (1 - 2 * self.min_weight) * p
        active = torch.as_tensor(self.active_heads, dtype=torch.bool, device=x.device)[None, :, None]
        return torch.where(active, w, torch.full_like(w, 0.5))

    def forward(self, x, m, h, eligible, epsilon=1e-8):
        w = self.weights(x)
        factors = torch.stack([h, 1 - h], dim=1)
        scores = torch.exp(w[:, :, 0] * torch.log(m.clamp_min(epsilon))[:, None] +
                           w[:, :, 1] * torch.log(factors.clamp_min(epsilon)))
        scores = torch.where((m[:, None] == 0) | (factors == 0), torch.zeros_like(scores), scores)
        scores = torch.stack([scores[:, 0], torch.where(eligible, scores[:, 1],
                                                       torch.zeros_like(scores[:, 1]))], dim=1)
        return scores, w
