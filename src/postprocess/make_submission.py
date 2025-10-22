# src/postprocess/make_submission.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores_npz",
        required=True,
        help="NPZ com scores (T x M), test_ids e go_list",
    )
    parser.add_argument(
        "--out_tsv",
        default="submissions/baseline_knn.tsv",
        help="Caminho do TSV de saída (formato Kaggle, sem header)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1500,
        help="Limite Kaggle por proteína (máx 1500 linhas/proteína)",
    )
    parser.add_argument(
        "--min_prob",
        type=float,
        default=1e-5,
        help="Probabilidade mínima (evita 0.0 no arquivo final)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Filtra scores abaixo deste valor (use o threshold do Fmax dev)",
    )
    args = parser.parse_args()

    # Garante diretório de saída
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)

    # Carrega dados
    try:
        z = np.load(args.scores_npz, allow_pickle=True)
        scores = z["scores"].astype(np.float32)  # (T, M)
        go_list = z["go_list"].astype(str)       # (M,)
        test_ids = z["test_ids"]                 # (T,)
        assert len(test_ids) == scores.shape[0], (
            f"Inconsistência: {len(test_ids)} IDs vs {scores.shape[0]} linhas de score"
        )
        assert scores.shape[1] == len(go_list), (
            f"Inconsistência: {scores.shape[1]} colunas de score vs {len(go_list)} termos GO"
        )
        # Normaliza formato dos termos GO
        go_list = [g if str(g).startswith("GO:") else f"GO:{str(g).zfill(7)}" for g in go_list]
    except Exception as e:
        logger.error(f"Erro ao carregar {args.scores_npz}: {e}")
        raise

    total_lines = 0
    with open(args.out_tsv, "w", encoding="utf-8") as f:
        for pid, row in zip(test_ids, scores):
            # 1) aplica threshold
            valid_idx = np.where(row >= args.threshold)[0]
            if valid_idx.size == 0:
                continue
            # 2) ordena desc apenas entre válidos
            ordered = valid_idx[np.argsort(-row[valid_idx])]
            # 3) aplica top-k
            ordered = ordered[: args.topk]

            for j in ordered:
                p = float(max(args.min_prob, min(1.0, row[j])))
                if not (0.0 <= p <= 1.0) or np.isnan(p):
                    continue
                term = go_list[j]
                # Linha no formato exigido pelo Kaggle
                f.write(f"{pid}\t{term}\t{p:.3f}\n")
                total_lines += 1

    logger.info(f"[SAVE] {args.out_tsv} (formato Kaggle, sem header) | linhas: {total_lines}")


if __name__ == "__main__":
    main()
