# src/postprocess/uncertainty_submit.py
from __future__ import annotations
import argparse, math, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.models.baseline_mlp import BaselineMLP
from src.models.mlp_lora import MLP_LoRA

def read_ids(p): return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
def fmt_go(x: str) -> str: s=str(x); return f"GO:{s.zfill(7)}" if not s.startswith("GO:") else s

def load_npz_X(path:str)->np.ndarray:
    z=np.load(path,allow_pickle=True); X=z["X"].astype(np.float32); return X

def load_labels(path:str):
    z=np.load(path,allow_pickle=True)
    Y = z["Y"].astype(np.float32) if "Y" in z else None
    if "go_list" in z: go_list = z["go_list"].astype(str).tolist()
    else:
        M = Y.shape[1]
        go_list = [f"GO:{str(i).zfill(7)}" for i in range(M)]
    return Y, [fmt_go(g) for g in go_list]

def ia_from_Y(Y: np.ndarray) -> np.ndarray:
    # IA ≈ -log(freq) normalizada em [0,1]
    freq = (Y.sum(0) / max(1, Y.shape[0])) + 1e-9
    ia = -np.log(freq)
    ia = (ia - ia.min()) / (ia.max() - ia.min() + 1e-8)
    return ia.astype(np.float32)

def load_edges_tsv(path: str):
    src, dst = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"): continue
            a,b = line.strip().split("\t")[:2]
            src.append(a); dst.append(b)
    return src, dst

def build_parents(go_list, src, dst):
    idx = {g:i for i,g in enumerate(go_list)}
    parents = [[] for _ in range(len(go_list))]
    miss = 0
    for a,b in zip(src,dst):
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None: miss += 1; continue
        parents[ib].append(ia)  # a -> b  (a é pai de b)
    if miss: print(f"[WARN] {miss} edges fora de go_list.")
    return parents

def propagate_and_clip(scores_row: np.ndarray, chosen_idx: np.ndarray, parents):
    # adiciona ancestrais e garante pai >= max(filhos)
    chosen = set(chosen_idx.tolist())
    stack = list(chosen_idx.tolist())
    while stack:
        j = stack.pop()
        for p in parents[j]:
            if p not in chosen:
                chosen.add(p); stack.append(p)
            # clipping: pai >= filho
            if scores_row[p] < scores_row[j]:
                scores_row[p] = scores_row[j]
    return np.array(sorted(chosen), dtype=np.int64), scores_row

def instantiate_model(model_type: str, D:int, M:int, ckpt):
    if model_type == "linear":
        model = BaselineMLP(input_dim=D, num_labels=M)
    elif model_type == "lora":
        hparams = ckpt.get("hparams", {})
        rank = hparams.get("rank", 8); alpha = hparams.get("alpha", 16)
        try:
            model = MLP_LoRA(input_dim=D, num_labels=M, rank=rank, alpha=alpha)
        except TypeError:
            model = MLP_LoRA(input_dim=D, num_labels=M)  # assinatura minimalista
    else:
        raise ValueError("--model_type deve ser 'linear' ou 'lora'")
    model.load_state_dict(ckpt["model"], strict=False)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["linear","lora"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_emb", required=True)
    ap.add_argument("--test_ids", required=True)
    ap.add_argument("--labels_npz", required=True)
    ap.add_argument("--edges_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--mc_passes", type=int, default=12)
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--conf_thr", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--topk_max", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=64)
    # opcional: GNN
    ap.add_argument("--label_embeds_npy", default=None)
    ap.add_argument("--gnn_mix", type=float, default=0.30)
    ap.add_argument("--gnn_temp", type=float, default=12.0)
    args = ap.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    D, M = int(ckpt["D"]), int(ckpt["M"])

    Xte = load_npz_X(args.test_emb)
    test_ids = read_ids(args.test_ids)
    assert Xte.shape[1] == D and len(test_ids) == Xte.shape[0]
    Ytr, go_list = load_labels(args.labels_npz)
    ia = ia_from_Y(Ytr) if Ytr is not None else np.ones(M, dtype=np.float32)

    # hierarquia
    src, dst = load_edges_tsv(args.edges_tsv)
    parents = build_parents(go_list, src, dst)

    # GNN opcional
    LnormT = None
    if args.label_embeds_npy:
        L = np.load(args.label_embeds_npy)
        if L.shape == (M, D):
            L = L.astype(np.float32)
            L = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-8)
            LnormT = torch.from_numpy(L).to(device).t().contiguous()
            print(f"[GNN] mix ativado λ={args.gnn_mix}, temp={args.gnn_temp}")
        else:
            print(f"[WARN] label_embeds_npy {L.shape} != (M={M}, D={D}); mix desativado.")

    model = instantiate_model(args.model_type, D, M, ckpt).to(device)
    model.eval()  # vamos alternar .train() só para ativar dropout nos passes

    out = Path(args.out_tsv); out.parent.mkdir(parents=True, exist_ok=True)
    f = open(out, "w", encoding="utf-8")

    B = args.batch_size
    N = Xte.shape[0]
    for s in range(0, N, B):
        e = min(N, s+B)
        xb_np = Xte[s:e]
        xb = torch.from_numpy(xb_np).to(device)

        # MC Dropout (sem guardar todas as passagens)
        sum_p = torch.zeros((e-s, M), dtype=torch.float32, device=device)
        sum_p2 = torch.zeros_like(sum_p)

        with torch.no_grad():
            for r in range(args.mc_passes):
                model.train()  # ativa dropout
                logits = model(xb)  # (b, M)
                probs = torch.sigmoid(logits)

                if LnormT is not None:
                    xbn = xb / (xb.norm(dim=1, keepdim=True) + 1e-8)
                    sim = (xbn @ LnormT) / float(args.gnn_temp)
                    probs_gnn = torch.softmax(sim, dim=1)
                    probs = (1.0 - args.gnn_mix) * probs + args.gnn_mix * probs_gnn

                sum_p += probs
                sum_p2 += probs * probs

        mu = (sum_p / args.mc_passes).cpu().numpy()             # (b, M)
        var = (sum_p2 / args.mc_passes - (sum_p/args.mc_passes)**2).cpu().numpy()
        # confiança normalizada: var máxima de Bernoulli ~0.25
        conf = 1.0 - np.clip(var / 0.25, 0.0, 1.0)              # (b, M)

        # IA em torch? mais simples em numpy
        ia_row = ia.reshape(1, -1)

        for i, pid in enumerate(test_ids[s:e]):
            m = mu[i]; c = conf[i]
            score = m * (np.power(c, args.beta)) * (np.power(ia_row, args.gamma)).ravel()

            # filtro por thr e conf_thr
            keep = (m >= args.thr) & (c >= args.conf_thr)
            idx = np.nonzero(keep)[0]
            if idx.size == 0:
                k = min(args.topk_max, score.size)
                idx = np.argpartition(-score, k-1)[:k]
            elif idx.size > args.topk_max:
                top = np.argpartition(score[idx], -(args.topk_max))[-args.topk_max:]
                idx = idx[top]

            # propaga e clipa
            idx, score_mod = propagate_and_clip(score.copy(), idx, parents)

            # ordena e escreve
            idx = idx[np.argsort(-score_mod[idx])]
            for j in idx:
                p = float(max(1e-6, min(1.0, score_mod[j])))
                f.write(f"{pid}\t{go_list[j]}\t{p:.3f}\n")

        if ((s//B) % 20) == 0:
            print(f"- {e}/{N}")

    f.close()
    print(f"[SAVE] {str(out)}")
    print("[DONE]")

if __name__ == "__main__":
    main()
