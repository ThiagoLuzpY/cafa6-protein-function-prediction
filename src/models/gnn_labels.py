# src/models/gnn_labels.py
from __future__ import annotations
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv
except Exception as e:  # fallback bem simples, caso torch_geometric não esteja disponível
    GCNConv = None

class LabelGCN(nn.Module):
    """
    GNN simples para refinar embeddings de rótulos (nós = termos GO).
    Usa 2 camadas GCNConv + ReLU.
    """
    def __init__(self, num_nodes: int, in_dim: int, hidden_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        if GCNConv is None:
            # fallback linear (sem grafo) só p/ não quebrar
            self.conv1 = nn.Linear(in_dim, hidden_dim)
            self.conv2 = nn.Linear(hidden_dim, in_dim)
            self.relu = nn.ReLU()
            self._use_gcn = False
        else:
            self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True, normalize=True)
            self.conv2 = GCNConv(hidden_dim, in_dim, add_self_loops=True, normalize=True)
            self.relu = nn.ReLU()
            self._use_gcn = True

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (M, in_dim)  |  edge_index: (2, E) long
        """
        if self._use_gcn:
            x = self.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
        else:
            # sem pyg: aplica MLP por nó (não usa edges)
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
        return x
