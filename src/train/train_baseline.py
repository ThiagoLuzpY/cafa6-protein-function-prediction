# src/train/train_baseline.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def load_npz_embeddings(npz_path: str) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=True)
    # aceita tanto 'X' quanto 'x'
    key = "X" if "X" in z else "x"
    X = z[key].astype("float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

def load_ids(txt_path: str) -> list[str]:
    return [l.strip() for l in open(txt_path, encoding="utf-8") if l.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_emb", required=True)
    parser.add_argument("--test_emb", required=True)
    parser.add_argument("--train_ids", required=True)
    parser.add_argument("--test_ids", required=True)
    parser.add_argument("--labels_npz", required=True, help="npz com Y (N x M), go_list e train_ids")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--out_npz", default="data/interim/preds/test_scores.npz")
    args = parser.parse_args()

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)

    print("[LOAD] embeddings")
    Xtr = load_npz_embeddings(args.train_emb)  # (N, D)
    Xte = load_npz_embeddings(args.test_emb)   # (T, D)
    print("  train:", Xtr.shape, "| test:", Xte.shape)

    train_ids = np.array(load_ids(args.train_ids))
    test_ids  = np.array(load_ids(args.test_ids))

    lab = np.load(args.labels_npz, allow_pickle=True)
    Y = lab["Y"].astype("float32")   # (N, M) binária
    go_list = lab["go_list"].astype(object)    # (M,)
    lab_train_ids = lab["train_ids"]

    # sanity: garante mesma ordem dos ids
    assert len(lab_train_ids) == len(train_ids)
    if not np.all(lab_train_ids == train_ids):
        # reordena Y na ordem de train_ids atuais
        print("[WARN] Reordenando Y para casar com train_ids deste run.")
        ix = {pid: i for i, pid in enumerate(lab_train_ids)}
        order = np.array([ix[pid] for pid in train_ids], dtype=int)
        Y = Y[order]

    # similaridade cosseno via produto Xte @ Xtr^T (já estão normalizados)
    print("[SIM] calculando similaridades (cosseno)...")
    S = Xte @ Xtr.T  # (T, N)

    # pega top-k vizinhos e gera score por termo: soma ponderada por similaridade
    K = min(args.k, Xtr.shape[0])
    print(f"[AGG] agregando top-{K} vizinhos…")
    # índices dos vizinhos ordenados desc (maiores sim primeiro)
    nn_idx = np.argpartition(-S, K-1, axis=1)[:, :K]  # (T, K) não ordenado
    # ordena dentro do top-k
    row_arange = np.arange(S.shape[0])[:, None]
    top_sim = S[row_arange, nn_idx]
    order = np.argsort(-top_sim, axis=1)
    nn_idx = nn_idx[row_arange, order]
    top_sim = top_sim[row_arange, order]  # (T, K)

    # acumula scores: para cada teste t, scores[t, :] = sum_k( sim[t,k] * Y[nn_idx[t,k], :] )
    T, M = Xte.shape[0], Y.shape[1]
    scores = np.zeros((T, M), dtype=np.float32)
    for t in range(T):
        Yk = Y[nn_idx[t]]             # (K, M)
        wk = top_sim[t][:, None]      # (K, 1)
        scores[t] = (Yk * wk).sum(axis=0)

    # normaliza opcionalmente por soma dos pesos (para escalar em 0-1)
    denom = top_sim.sum(axis=1, keepdims=True) + 1e-9
    scores = scores / denom

    np.savez_compressed(args.out_npz,
                        scores=scores,
                        test_ids=test_ids,
                        go_list=go_list)
    print(f"[SAVE] {args.out_npz} | scores shape: {scores.shape}")

if __name__ == "__main__":
    main()
