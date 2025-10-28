# src/train/train_lora.py
from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.models.mlp_lora import MLP_LoRA

# -------------------------- utils -------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_npz_X(path: str) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    if "X" not in z:
        raise ValueError(f"{path} não contém array 'X'")
    X = z["X"]
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    return X

def load_labels_npz(path: str):
    z = np.load(path, allow_pickle=True)
    if "Y" not in z:
        raise ValueError(f"{path} não contém array 'Y' (matriz de rótulos)")
    Y = z["Y"].astype(np.float32)
    go_list = z["go_list"].astype(str).tolist() if "go_list" in z else None
    return Y, go_list

def compute_pos_weight(Y: np.ndarray) -> torch.Tensor:
    # pos_weight = N_neg / N_pos (para BCEWithLogits; por classe)
    N = Y.shape[0]
    pos = Y.sum(0)                       # (M,)
    neg = N - pos
    pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1e-6), 1.0).astype(np.float32)
    return torch.from_numpy(pos_weight)

# -------------------------- train -------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_emb", required=True)
    ap.add_argument("--labels_npz", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_ckpt", default="models/mlp_lora.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- dados ----
    X = load_npz_X(args.train_emb)                  # (N, D)
    Y, go_list = load_labels_npz(args.labels_npz)   # (N, M)
    N, D = X.shape
    Ny, M = Y.shape
    assert N == Ny, f"Desalinhado: X tem {N} linhas e Y tem {Ny}"
    print(f"[LOAD] X: {X.shape} | Y: {Y.shape} | device={device}")

    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
    ds = TensorDataset(X_t, Y_t)

    val_size = max(1, int(args.val_frac * N))
    train_size = N - val_size
    train_ds, val_ds = random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    pin = torch.cuda.is_available()
    dl_tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False, pin_memory=pin)
    dl_va = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=pin)

    # ---- modelo ----
    model = MLP_LoRA(
        input_dim=D,
        num_labels=M,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        freeze_base=False,
    ).to(device)

    pos_weight = compute_pos_weight(Y).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        # treino
        model.train()
        tr_sum = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=pin)
            yb = yb.to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)              # (B, M)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_sum += loss.item() * xb.size(0)
        tr_loss = tr_sum / train_size

        # valida
        model.eval()
        va_sum = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device, non_blocking=pin)
                yb = yb.to(device, non_blocking=pin)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_sum += loss.item() * xb.size(0)
        va_loss = va_sum / val_size
        print(f"[Epoch {epoch:02d}] train_loss={tr_loss:.6f} | val_loss={va_loss:.6f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "D": D, "M": M, "go_list": go_list}, args.out_ckpt)
            print(f"  ↳ [SAVE] {args.out_ckpt} (best_val={best_val:.6f})")

    print("[DONE]")

if __name__ == "__main__":
    main()
