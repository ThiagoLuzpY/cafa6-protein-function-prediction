# src/eval/metrics.py
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------
# Métricas / utilidades
# ---------------------------

def f1_weighted_binary(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """F1 ponderado por weights (forma simples, binária). Mantido da sua versão."""
    eps = 1e-8
    tp = ((y_true == 1) & (y_pred == 1)).astype(float)
    fp = ((y_true == 0) & (y_pred == 1)).astype(float)
    fn = ((y_true == 1) & (y_pred == 0)).astype(float)
    w = weights.reshape(1, -1)
    p = (w * tp).sum() / ((w * (tp + fp)).sum() + eps)
    r = (w * tp).sum() / ((w * (tp + fn)).sum() + eps)
    return float(2 * p * r / (p + r + eps))

def fmax_weighted(Ytrue: np.ndarray, S: np.ndarray, w: np.ndarray, n_thresh: int = 256):
    """
    Fmax ponderado por IA via sweep em thresholds uniformes [0,1].
    Retorna (Fmax, best_threshold).
    """
    thr = np.linspace(0.0, 1.0, n_thresh, dtype=np.float32)
    bestF, bestT = 0.0, 0.0
    Yw = Ytrue * w[None, :]
    for t in thr:
        P = (S >= t).astype(np.float32)
        TP = (P * Yw).sum()
        FP = (P * (1 - Ytrue) * w[None, :]).sum()
        FN = ((1 - P) * Yw).sum()
        prec = TP / (TP + FP + 1e-9)
        rec  = TP / (TP + FN + 1e-9)
        F = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        if F > bestF:
            bestF, bestT = float(F), float(t)
    return bestF, bestT

# ---------------------------
# Suporte à validação holdout kNN (sem alterar estrutura)
# ---------------------------

def _load_npz_embeddings(npz_path: str) -> np.ndarray:
    X = np.load(npz_path)["X"].astype("float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms  # normaliza para cosseno

def _load_ids(txt_path: str) -> list[str]:
    return [l.strip() for l in open(txt_path, encoding="utf-8") if l.strip()]

def _build_go_ontology_masks(go_list: np.ndarray, terms_tsv: str):
    # mapeia GO -> ont usando o TSV propagado
    df = pd.read_csv(terms_tsv, sep="\t", header=None, names=["uniprot","go","ont"])
    go2ont = dict(df.drop_duplicates("go")[["go","ont"]].values)
    cols = np.arange(len(go_list))
    ont = np.array([go2ont.get(g, "NA") for g in go_list], dtype=object)
    mask_MF = ont == "MFO"
    mask_BP = ont == "BPO"
    mask_CC = ont == "CCO"
    return {"ALL": cols, "MF": cols[mask_MF], "BP": cols[mask_BP], "CC": cols[mask_CC]}

def _load_weights(go_list: np.ndarray, ia_tsv: str) -> np.ndarray:
    # IA.tsv: "GO:xxxxxx \t IA"
    ia = {}
    with open(ia_tsv, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            ia[parts[0]] = float(parts[1])
    return np.array([ia.get(g, 0.0) for g in go_list], dtype="float32")

def _knn_scores(Xtr: np.ndarray, Ytr: np.ndarray, Xval: np.ndarray, k: int) -> np.ndarray:
    S = Xval @ Xtr.T  # similaridade cosseno
    K = min(k, Xtr.shape[0])
    nn_idx = np.argpartition(-S, K-1, axis=1)[:, :K]
    row = np.arange(S.shape[0])[:, None]
    top_sim = S[row, nn_idx]
    order = np.argsort(-top_sim, axis=1)
    nn_idx = nn_idx[row, order]
    top_sim = top_sim[row, order]       # (T, K)
    T, M = Xval.shape[0], Ytr.shape[1]
    scores = np.zeros((T, M), dtype=np.float32)
    for t in range(T):
        Yk = Ytr[nn_idx[t]]             # (K, M)
        wk = top_sim[t][:, None]        # (K, 1)
        scores[t] = (Yk * wk).sum(axis=0)
    denom = top_sim.sum(axis=1, keepdims=True) + 1e-9
    return scores / denom

def _run_holdout(train_emb, train_ids, labels_npz, terms_tsv, ia_tsv, k, val_frac, seed):
    X = _load_npz_embeddings(train_emb)
    ids = np.array(_load_ids(train_ids))
    lab = np.load(labels_npz, allow_pickle=True)
    Y = lab["Y"].astype(np.float32)
    go_list = lab["go_list"].astype(object)
    assert len(ids) == X.shape[0] == Y.shape[0]

    # split reproduzível
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = max(1, int(N * val_frac))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    Xtr, Xval = X[tr_idx], X[val_idx]
    Ytr, Yval = Y[tr_idx], Y[val_idx]

    w_all = _load_weights(go_list, ia_tsv)
    masks = _build_go_ontology_masks(go_list, terms_tsv)

    Sval = _knn_scores(Xtr, Ytr, Xval, k=k)

    results = {}
    for name, cols in masks.items():
        if name == "ALL":
            w = w_all
            Yt = Yval
            Sv = Sval
        else:
            w = w_all[cols]
            Yt = Yval[:, cols]
            Sv = Sval[:, cols]
        F, T = fmax_weighted(Yt, Sv, w, n_thresh=256)
        results[name] = (F, T)

    print("=== Holdout kNN (subset) ===")
    print(f"N={N}  | train={len(tr_idx)}  val={len(val_idx)}  | k={k}")
    for kname, (F, T) in results.items():
        print(f"{kname}: Fmax={F:.4f} @ thr={T:.3f}")

# ---------------------------
# CLI (opcional) – roda a avaliação holdout usando ESTE arquivo
# ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fmax ponderado (holdout) usando baseline kNN e IA.tsv.")
    ap.add_argument("--train_emb", required=True)
    ap.add_argument("--train_ids", required=True)
    ap.add_argument("--labels_npz", required=True, help="npz com Y (N x M), go_list, train_ids")
    ap.add_argument("--terms_tsv", required=True, help="data/processed/train_terms_propagated.tsv")
    ap.add_argument("--ia_tsv", required=True, help="data/raw/IA.tsv")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _run_holdout(
        train_emb=args.train_emb,
        train_ids=args.train_ids,
        labels_npz=args.labels_npz,
        terms_tsv=args.terms_tsv,
        ia_tsv=args.ia_tsv,
        k=args.k,
        val_frac=args.val_frac,
        seed=args.seed,
    )
