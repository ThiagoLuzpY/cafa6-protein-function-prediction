from __future__ import annotations
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from src.models.baseline_mlp import BaselineMLP

class NPZDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.astype(np.float32, copy=False)
        self.Y = Y.astype(np.float32, copy=False)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

def load_npz_X(path):
    z = np.load(path, allow_pickle=True)
    # nossos embeddings foram salvos como X=...
    X = z["X"]
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_emb", required=True)
    ap.add_argument("--labels_npz", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--out_ckpt", default="models/mlp_linear.pt")
    args = ap.parse_args()

    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # Carrega embeddings e rótulos
    X = load_npz_X(args.train_emb)          # (N, D)
    zlab = np.load(args.labels_npz, allow_pickle=True)
    Y = zlab["Y"]                           # (N, M)
    go_list = zlab["go_list"]               # (M,)

    N, D = X.shape
    M = Y.shape[1]
    print(f"[LOAD] X: {X.shape} | Y: {Y.shape} | D={D} | M={M}")

    # split treino/val
    n_val = int(N * args.val_frac)
    n_tr  = N - n_val
    ds = NPZDataset(X, Y)
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))

    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # modelo
    model = BaselineMLP(input_dim=D, num_labels=M, hidden=args.hidden).to(device)

    # BCE com pos_weight (desbalanceamento)
    pos = Y.sum(axis=0) + 1e-6
    neg = (N - pos) + 1e-6
    pos_weight = torch.from_numpy((neg / pos).astype(np.float32))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train(); loss_tr = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            loss_tr += float(loss) * xb.size(0)
        loss_tr /= n_tr

        model.eval(); loss_va = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss_va += float(loss) * xb.size(0)
        loss_va /= n_val
        print(f"[EP {ep}] train={loss_tr:.4f}  val={loss_va:.4f}")

        if loss_va < best:
            best = loss_va
            torch.save({"model": model.state_dict(), "D": D, "M": M, "go_list": go_list}, args.out_ckpt)
            print(f"[SAVE] {args.out_ckpt} (best so far: {best:.4f})")

if __name__ == "__main__":
    main()
