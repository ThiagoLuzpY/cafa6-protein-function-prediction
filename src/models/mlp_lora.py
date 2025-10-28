import torch
import torch.nn as nn
import math


class MLP_LoRA(nn.Module):
    # CORREÇÃO: Adicionamos 'dropout' (corrigido antes) e 'freeze_base' (resolvendo o novo TypeError).
    def __init__(self, input_dim: int, num_labels: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.2,
                 freeze_base: bool = False):
        super().__init__()

        self.alpha = alpha
        self.rank = rank

        # Camada Base (Full-Rank)
        self.W = nn.Linear(input_dim, num_labels, bias=False)

        # LoRA: decomposição de baixa dimensão
        self.A = nn.Linear(input_dim, rank, bias=False)
        self.B = nn.Linear(rank, num_labels, bias=False)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Lógica para congelar a base (Obriga o modelo a aprender apenas com LoRA)
        if freeze_base:
            for param in self.W.parameters():
                param.requires_grad = False

        # Inicialização do LoRA (Importante para estabilidade)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

        # NOTA: O self.sigmoid foi removido para usar BCEWithLogitsLoss no train_lora.py

    def forward(self, x: torch.Tensor):
        x_dropped = self.dropout_layer(x)

        # Forward pass da base (W)
        base = self.W(x_dropped)

        # Forward pass do LoRA (B * A) * (alpha / rank)
        delta = self.B(self.A(x_dropped)) * (self.alpha / self.rank)

        logits = base + delta

        # Retorna LOGITS crus (sem sigmoid)
        return logits