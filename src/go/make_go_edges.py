# src/go/make_go_edges.py
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import obonet
import networkx as nx  # vem junto com obonet

def load_go_list(labels_npz: str):
    z = np.load(labels_npz, allow_pickle=True)
    if "go_list" in z:
        return [str(x) for x in z["go_list"].astype(str).tolist()]
    # fallback: deduz pelo nº de colunas de Y
    M = int(z["Y"].shape[1])
    return [f"GO:{str(i).zfill(7)}" for i in range(M)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo", required=True, help="caminho do go-basic.obo")
    ap.add_argument("--labels_npz", required=True, help="para filtrar termos válidos (go_list)")
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--undirected", action="store_true", help="escreve arestas nos dois sentidos")
    args = ap.parse_args()

    go_list = load_go_list(args.labels_npz)
    go_set  = set(go_list)

    print(f"[LOAD] lendo OBO: {args.obo}")
    g: nx.MultiDiGraph = obonet.read_obo(args.obo)

    keep_edges = set()
    n_all, n_kept = 0, 0
    for u, v, data in g.edges(data=True):
        n_all += 1
        # obonet coloca 'is_a' como aresta u(child)->v(parent)
        rel = data.get("relation") or data.get("rel") or data.get("type")
        is_is_a   = (rel is None)  # aresta 'is_a' não vem marcada
        is_partof = (rel == "part_of")

        if not (is_is_a or is_partof):
            continue
        if u in go_set and v in go_set:
            keep_edges.add((u, v))
            if args.undirected:
                keep_edges.add((v, u))
            n_kept += 1

    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tsv, "w", encoding="utf-8") as f:
        for a, b in sorted(keep_edges):
            f.write(f"{a}\t{b}\n")

    print(f"[OK] edges.tsv: {args.out_tsv}  | edges_in={n_all}  edges_out={len(keep_edges)}  (filtrados pelos {len(go_list)} termos)")
    print("[DONE]")

if __name__ == "__main__":
    main()
