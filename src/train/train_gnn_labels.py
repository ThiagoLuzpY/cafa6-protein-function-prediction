# src/train/train_gnn_labels.py
from __future__ import annotations
import argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.gnn_labels import LabelGCN

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_go_list(labels_npz: str):
    z = np.load(labels_npz, allow_pickle=True)
    if "go_list" in z:
        return z["go_list"].astype(str).tolist()
    M = int(z["Y"].shape[1])
    return [f"GO:{str(i).zfill(7)}" for i in range(M)]

def _norm_go(s: str) -> str:
    s = s.strip()
    if s.startswith("GO:"): return s
    # se vier número puro, normaliza como GO:xxxxxxx
    if s.isdigit():
        return f"GO:{s.zfill(7)}"
    return s  # deixa como está

def load_edges_tsv(path: str):
    """Lê TSV src<TAB>dst (GO:* ou índices). Retorna listas de strings (src_list, dst_list)."""
    src_list, dst_list = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a, b = line.split("\t")[:2]
            src_list.append(_norm_go(a))
            dst_list.append(_norm_go(b))
    return src_list, dst_list

def map_edges_to_indices(src_list, dst_list, go_to_id):
    s_idx, d_idx, miss = [], [], 0
    for a, b in zip(src_list, dst_list):
        ia = go_to_id.get(a); ib = go_to_id.get(b)
        if ia is None or ib is None:
            miss += 1; continue
        s_idx.append(ia); d_idx.append(ib)
    if miss:
        print(f"[WARN] {miss} arestas ignoradas (rótulo fora de go_list).")
    if not s_idx:
        raise RuntimeError("Nenhuma aresta válida após o mapeamento; verifique o TSV.")
    edge_index = torch.tensor([s_idx, d_idx], dtype=torch.long)
    return edge_index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_npz", required=True, help="usa para obter go_list e M")
    ap.add_argument("--edges_tsv", required=True, help="TSV com arestas do DAG GO: src\\tdst")
    ap.add_argument("--in_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_ckpt", default="models/gnn_labels.pt")
    ap.add_argument("--out_npy",  default="models/label_embeds_gnn.npy")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- rótulos / arestas ---
    go_list = load_go_list(args.labels_npz)
    M = len(go_list)
    go_to_id = {g: i for i, g in enumerate(go_list)}
    src_list, dst_list = load_edges_tsv(args.edges_tsv)
    edge_index = map_edges_to_indices(src_list, dst_list, go_to_id).to(device)
    print(f"[LOAD] M={M} labels | edges={edge_index.size(1)} | device={device}")

    # --- embeddings treináveis + GCN ---
    emb_table = nn.Embedding(M, args.in_dim).to(device)
    nn.init.xavier_uniform_(emb_table.weight)

    gnn = LabelGCN(num_nodes=M, in_dim=args.in_dim, hidden_dim=args.hidden_dim).to(device)
    params = list(emb_table.parameters()) + list(gnn.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    def smoothness_loss(Z: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
        s, d = edge_idx[0], edge_idx[1]
        diff = Z[s] - Z[d]
        return (diff * diff).sum(dim=1).mean()

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        emb_table.train(); gnn.train()
        optim.zero_grad()
        X0 = emb_table.weight                     # (M, in_dim)
        Z  = gnn(X0, edge_index)                  # (M, in_dim)
        loss = smoothness_loss(Z, edge_index) + 1e-3 * (Z.pow(2).mean())
        loss.backward()
        optim.step()

        val = float(loss.item())
        print(f"[Epoch {ep:02d}] loss={val:.6f}")
        if val < best - 1e-6:
            best = val
            Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"gnn": gnn.state_dict(),
                 "emb_table": emb_table.state_dict(),
                 "in_dim": args.in_dim,
                 "hidden_dim": args.hidden_dim,
                 "go_list": go_list},
                args.out_ckpt,
            )
            np.save(args.out_npy, Z.detach().cpu().numpy().astype(np.float32))
            print(f"  ↳ [SAVE] {args.out_ckpt} | {args.out_npy}")

    print("[DONE]")

if __name__ == "__main__":
    main()
