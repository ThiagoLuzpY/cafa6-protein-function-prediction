import torch
import torch.nn as nn

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int=512, dropout: float=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)
