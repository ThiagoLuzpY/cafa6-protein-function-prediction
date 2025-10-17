# src/data/load_labels.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.paths import CFG

def load_ids(txt_path: str) -> list[str]:
    return [l.strip() for l in open(txt_path, encoding="utf-8") if l.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ids", required=True, help="arquivo com 1 UniProt por linha (mesmo usado nos embeddings)")
    parser.add_argument("--labels_tsv", default=str(Path(CFG["paths"]["data_processed"]) / "train_terms_propagated.tsv"),
                        help="TSV propagado: uniprot \\t go_id \\t ont")
    parser.add_argument("--out_npz", default="data/interim/labels/train_labels_subset.npz")
    args = parser.parse_args()

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)

    ids = load_ids(args.train_ids)
    id2row = {pid: i for i, pid in enumerate(ids)}
    print(f"[IDS] subset treinos: {len(ids)}")

    print(f"[LOAD] {args.labels_tsv}")
    df = pd.read_csv(args.labels_tsv, sep="\t", header=None, names=["uniprot", "go_id", "ont"])
    df = df[df["uniprot"].isin(id2row)]  # filtra só o subset
    print(f"[FILTER] linhas após filtrar pelo subset: {len(df)}")

    # mapeia GO -> col
    go_list = sorted(df["go_id"].unique().tolist())
    go2col = {g: j for j, g in enumerate(go_list)}
    print(f"[GO] termos únicos no subset: {len(go_list)}")

    # monta matriz binária (N x M) como uint8 (memória ok para subset)
    N, M = len(ids), len(go_list)
    Y = np.zeros((N, M), dtype=np.uint8)

    for pid, go in zip(df["uniprot"].values, df["go_id"].values):
        i = id2row.get(pid)
        j = go2col.get(go)
        if i is not None and j is not None:
            Y[i, j] = 1

    np.savez_compressed(args.out_npz, Y=Y, train_ids=np.array(ids), go_list=np.array(go_list, dtype=object))
    print(f"[SAVE] {args.out_npz} | Y shape: {Y.shape} | positivos: {int(Y.sum())}")

if __name__ == "__main__":
    main()
