# src/predict/predict_mlp_submit.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch

from src.models.baseline_mlp import BaselineMLP


# -------------------- Utils --------------------

def load_npz_X(npz_path: Path) -> np.ndarray:
    """
    Carrega embeddings de um .npz, aceitando chaves comuns.
    Retorna array float32 (N, D).
    """
    z = np.load(npz_path, allow_pickle=True)
    for k in ("X", "emb", "embs", "E", "arr_0"):
        if k in z:
            X = z[k]
            break
    else:
        raise KeyError(
            f"Nenhuma chave de embeddings encontrada em {npz_path} "
            "(procurado por: X, emb, embs, E, arr_0)."
        )
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Embeddings devem ser 2D (N, D); recebido shape={X.shape} em {npz_path}")
    return X


def read_ids(txt_path: Path) -> List[str]:
    with txt_path.open(encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def fmt_go(x: str) -> str:
    s = str(x)
    return f"GO:{s.zfill(7)}" if not s.startswith("GO:") else s


def infer_ids_path(test_emb_path: Path, explicit_ids: Optional[Path]) -> Path:
    """
    Resolve o caminho dos IDs do teste com as seguintes prioridades:
      1) --test_ids (se fornecido e existir)
      2) mesmo prefixo do .npz (com .txt)
      3) padrão 'test_ids_*.txt' no mesmo diretório (primeiro que existir)
      4) tentativa esperta substituindo prefixo do stem
    """
    # 1) explícito
    if explicit_ids:
        p = Path(explicit_ids)
        if p.exists():
            return p

    # 2) mesmo nome do .npz com .txt
    candidate2 = test_emb_path.with_suffix(".txt")
    if candidate2.exists():
        return candidate2

    # 3) glob por "test_ids_*.txt" no mesmo diretório
    for cand in test_emb_path.parent.glob("test_ids*.txt"):
        if cand.exists():
            return cand

    # 4) tentativa: trocar prefixos comuns no stem
    stem = test_emb_path.stem  # ex: test_esm2_t6_8M_subsetNone
    repls = [
        ("test_esm2_t6_8M_", "test_ids_"),
        ("test_esm2_", "test_ids_"),
        ("test_", "test_ids_"),
    ]
    for a, b in repls:
        if stem.startswith(a):
            candidate4 = test_emb_path.parent / f"{stem.replace(a, b, 1)}.txt"
            if candidate4.exists():
                return candidate4

    # Se nada deu certo, erra com mensagem amigável
    tried = [str(candidate2)]
    tried += [str(p) for p in test_emb_path.parent.glob("test_ids*.txt")]
    msg = (
        f"Arquivo de IDs do teste não encontrado.\n"
        f"Tentativas:\n  - {chr(10).join(tried)}\n"
        f"Sugestão: passe explicitamente com --test_ids PATH_DO_TXT."
    )
    raise FileNotFoundError(msg)


# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint (.pt) do MLP")
    ap.add_argument("--test_emb", required=True, help="NPZ com embeddings de teste (N, D)")
    ap.add_argument("--labels_npz", required=True, help="NPZ de labels (usa go_list consistente)")
    ap.add_argument("--out_tsv", required=True, help="Saída em formato Kaggle (sem header)")
    ap.add_argument("--threshold", type=float, default=0.161)
    ap.add_argument("--topk", type=int, default=1500)
    ap.add_argument("--min_prob", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--test_ids", type=str, default=None, help="(Opcional) TXT com IDs do teste")
    args = ap.parse_args()

    device = torch.device("cpu")

    # --- Carrega checkpoint ---
    ckpt = torch.load(args.ckpt, map_location="cpu")
    D, M = int(ckpt["D"]), int(ckpt["M"])

    # Compatível com nosso baseline (linear se hidden=0)
    # Ajuste aqui se seu BaselineMLP exigir assinatura diferente.
    model = BaselineMLP(input_dim=D, num_labels=M, hidden=0)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    # --- GO list coerente com labels_npz ---
    try:
        zlab = np.load(args.labels_npz, allow_pickle=True)
        if "go_list" in zlab:
            go_list = [fmt_go(x) for x in zlab["go_list"].astype(str).tolist()]
        elif "terms" in zlab:
            go_list = [fmt_go(x) for x in zlab["terms"].astype(str).tolist()]
        else:
            raise KeyError("go_list/terms não encontrado no labels_npz")
    except Exception:
        go_list = [fmt_go(x) for x in ckpt["go_list"]]

    if len(go_list) != M:
        raise ValueError(
            f"Incompatibilidade: M (ckpt)={M} vs len(go_list)={len(go_list)}"
        )

    # --- Embeddings e IDs ---
    test_emb_path = Path(args.test_emb)
    Xte = load_npz_X(test_emb_path)

    ids_path = infer_ids_path(test_emb_path, Path(args.test_ids) if args.test_ids else None)
    test_ids = read_ids(ids_path)

    if Xte.shape[0] != len(test_ids):
        raise ValueError(
            f"Desalinhado: embeddings N={Xte.shape[0]} vs test_ids={len(test_ids)}\n"
            f"test_emb: {test_emb_path}\nids_txt:  {ids_path}"
        )

    N = Xte.shape[0]
    print(f"[LOAD] test X: {Xte.shape}  | ids: {len(test_ids)}")
    print(f"[IDS]  usando: {ids_path}")

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Inferência em streaming ---
    B = max(1, int(args.batch_size))
    wrote = 0
    with out_path.open("w", encoding="utf-8", newline="") as f, torch.no_grad():
        for s in range(0, N, B):
            e = min(N, s + B)
            xb = torch.from_numpy(Xte[s:e]).to(device)  # (b, D)
            logits = model(xb)                          # (b, M)
            probs = torch.sigmoid(logits).cpu().numpy() # (b, M)

            for i, pid in enumerate(test_ids[s:e]):
                row = probs[i]
                # aplica threshold
                keep = row >= args.threshold
                idx = np.nonzero(keep)[0]

                if idx.size == 0:
                    # se nada passou, pega top-k bruto
                    k = min(args.topk, row.size)
                    idx = np.argpartition(-row, k - 1)[:k]
                elif idx.size > args.topk:
                    # se passou muita coisa no thr, limita ao top-k dentro do conjunto keep
                    top = np.argpartition(row[idx], -(args.topk))[-args.topk:]
                    idx = idx[top]

                # ordena para imprimir bonito
                idx = idx[np.argsort(-row[idx])]

                # escreve
                for j in idx:
                    p = float(max(args.min_prob, min(1.0, row[j])))
                    f.write(f"{pid}\t{go_list[j]}\t{p:.3f}\n")
                    wrote += 1

            if ((s // B) % 50) == 0:
                print(f"  - {e}/{N}")

    print(f"[SAVE] {out_path}  (linhas: {wrote}, formato Kaggle, sem header)")


if __name__ == "__main__":
    main()
