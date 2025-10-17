# src/postprocess/make_submission.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def load_ids(path: str) -> list[str]:
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_npz", required=True, help="npz com scores (T x M), test_ids, go_list")
    parser.add_argument("--test_ids", required=True)
    parser.add_argument("--out_tsv", default="submissions/baseline_knn.tsv")
    parser.add_argument("--topk", type=int, default=1500, help="limite Kaggle por proteína")
    parser.add_argument("--min_prob", type=float, default=1e-5, help="evita 0.0")
    args = parser.parse_args()

    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)

    z = np.load(args.scores_npz, allow_pickle=True)
    scores = z["scores"].astype(np.float32)    # (T, M)
    go_list = z["go_list"].astype(object)      # (M,)
    test_ids_scores = z["test_ids"]

    test_ids = np.array(load_ids(args.test_ids))
    assert len(test_ids) == len(test_ids_scores), "test_ids diferentes dos que geraram os scores"

    with open(args.out_tsv, "w", encoding="utf-8") as f:
        for i, pid in enumerate(test_ids):
            row = scores[i]
            # pega índices ordenados por score desc
            idx = np.argsort(-row)
            # aplica min_prob e topk
            taken = 0
            for j in idx:
                p = float(max(args.min_prob, min(1.0, row[j])))
                term = go_list[j]
                f.write(f"{pid}\t{term}\t{p:.3f}\n")
                taken += 1
                if taken >= args.topk:
                    break

    print(f"[SAVE] {args.out_tsv}  (formato Kaggle, sem header)")

if __name__ == "__main__":
    main()
