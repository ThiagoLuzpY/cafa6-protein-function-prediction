from __future__ import annotations
import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    """
    Cabeçote rápido: por padrão é uma REGRESSÃO LOGÍSTICA (apenas Linear).
    Se hidden>0, vira MLP: Linear -> ReLU -> Dropout -> Linear.
    """
    def __init__(self, input_dim: int, num_labels: int, hidden: int = 0, dropout: float = 0.3):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_labels),
            )
        else:
            self.net = nn.Linear(input_dim, num_labels)  # mais leve e rápido na CPU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits (sem sigmoid)
