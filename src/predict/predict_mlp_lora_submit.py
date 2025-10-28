# src/predict/predict_mlp_lora_submit.py
from __future__ import annotations
import argparse, inspect
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.models.mlp_lora import MLP_LoRA


# --------------------------- helpers ---------------------------

def read_ids(txt_path: str) -> list[str]:
    return [l.strip() for l in open(txt_path, encoding="utf-8") if l.strip()]

def fmt_go(x: str) -> str:
    s = str(x)
    return f"GO:{s.zfill(7)}" if not s.startswith("GO:") else s

def load_npz_X(path: str) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    if "X" not in z:
        raise ValueError(f"{path} não contém array 'X'")
    X = z["X"]
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    return X

def load_go_list(labels_npz: str, M_expected: int | None = None) -> list[str]:
    z = np.load(labels_npz, allow_pickle=True)
    if "go_list" in z:
        go = z["go_list"].astype(str).tolist()
    else:
        if "Y" in z:
            M = int(z["Y"].shape[1])
        elif M_expected is not None:
            M = int(M_expected)
        else:
            raise ValueError("labels_npz não tem go_list nem Y para inferir M.")
        go = [f"GO:{str(i).zfill(7)}" for i in range(M)]
    return [fmt_go(g) for g in go]


# ----------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt do MLP_LoRA")
    ap.add_argument("--test_emb", required=True, help="npz com X (N, D)")
    ap.add_argument("--test_ids", required=True, help="txt com IDs (N linhas)")
    ap.add_argument("--labels_npz", required=True, help="npz com go_list (ou Y)")
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--threshold", type=float, default=0.161)
    ap.add_argument("--topk", type=int, default=1500)
    ap.add_argument("--min_prob", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=64)
    # GNN opcional
    ap.add_argument("--label_embeds_npy", default=None,
                    help="npy (M, E) com embeddings de rótulos (do GNN)")
    ap.add_argument("--gnn_mix", type=float, default=0.30)
    ap.add_argument("--gnn_temp", type=float, default=12.0)
    args = ap.parse_args()

    device = torch.device("cpu")

    # ---- checkpoint / modelo ----
    ckpt = torch.load(args.ckpt, map_location="cpu")
    D = int(ckpt["D"])
    M = int(ckpt["M"])

    # instancia apenas com kwargs suportados
    sig = inspect.signature(MLP_LoRA.__init__).parameters
    kwargs = {"input_dim": D, "num_labels": M}
    if "rank" in sig:          kwargs["rank"] = ckpt.get("hparams", {}).get("rank", 8)
    if "alpha" in sig:         kwargs["alpha"] = ckpt.get("hparams", {}).get("alpha", 16)
    if "freeze_base" in sig:   kwargs["freeze_base"] = False
    # NÃO passar hidden/dropout se a classe não aceitar
    model = MLP_LoRA(**kwargs).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # ---- rótulos ----
    go_list = load_go_list(args.labels_npz, M_expected=M)

    # ---- dados de teste ----
    Xte = load_npz_X(args.test_emb)            # (N, D)
    test_ids = read_ids(args.test_ids)
    assert Xte.shape[0] == len(test_ids), "Xte e test_ids desalinharam"
    assert Xte.shape[1] == D, f"Dimensão de X ({Xte.shape[1]}) != D do modelo ({D})"
    N = Xte.shape[0]
    print(f"[LOAD] test X: {Xte.shape}  | ids: {len(test_ids)}")

    # ---- canal GNN opcional ----
    LnormT = None
    if args.label_embeds_npy:
        L = np.load(args.label_embeds_npy)
        if L.shape[0] != M:
            print(f"[WARN] label_embeds_npy M={L.shape[0]} != {M}. Desabilitando mix.")
        elif L.shape[1] != D:
            print(f"[WARN] label_embeds_npy dim={L.shape[1]} != D={D}. Desabilitando mix.")
        else:
            L = L.astype(np.float32)
            Lnorm = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
            LnormT = torch.from_numpy(Lnorm).to(device).t().contiguous()  # (D, M)
            print(f"[GNN] mix habilitado (λ={args.gnn_mix}, temp={args.gnn_temp})")

    # ---- saída ----
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w", encoding="utf-8")

    B = args.batch_size
    with torch.no_grad():
        for s in range(0, N, B):
            e = min(N, s + B)
            xb = torch.from_numpy(Xte[s:e]).to(device)  # (b, D)

            # forward padrão do modelo (respeita as camadas LoRA internas)
            logits = model(xb)                           # (b, M)
            probs = torch.sigmoid(logits)               # (b, M)

            # mistura com GNN (se disponível)
            if LnormT is not None:
                xbn = xb / (xb.norm(dim=1, keepdim=True) + 1e-8)
                sim = (xbn @ LnormT) / float(args.gnn_temp)   # (b, M)
                probs_gnn = torch.softmax(sim, dim=1)
                probs = (1.0 - args.gnn_mix) * probs + args.gnn_mix * probs_gnn

            rowset = probs.cpu().numpy()
            for i, pid in enumerate(test_ids[s:e]):
                row = rowset[i]
                keep = row >= args.threshold
                idx = np.nonzero(keep)[0]
                if idx.size == 0:
                    k = min(args.topk, row.size)
                    idx = np.argpartition(-row, k - 1)[:k]
                elif idx.size > args.topk:
                    top_local = np.argpartition(row[idx], -(args.topk))[-args.topk:]
                    idx = idx[top_local]
                idx = idx[np.argsort(-row[idx])]
                for j in idx:
                    p = float(max(args.min_prob, min(1.0, row[j])))
                    f.write(f"{pid}\t{fmt_go(go_list[j])}\t{p:.3f}\n")

            if ((s // B) % 50) == 0:
                print(f"  - {e}/{N}")

    f.close()
    print(f"[SAVE] {str(out_path)}")

if __name__ == "__main__":
    main()
