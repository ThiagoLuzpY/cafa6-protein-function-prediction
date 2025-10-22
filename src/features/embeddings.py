# src/features/embeddings.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from src.utils.paths import CFG, ensure_dirs

# -------- FASTA util (simples e robusto) --------
def read_fasta(fp: str, max_items: int | None = None) -> List[Tuple[str, str]]:
    """Retorna [(uniprot_id, sequence), ...] preservando ordem."""
    items = []
    with open(fp, "r", encoding="utf-8") as f:
        pid = None
        seq_parts = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if pid is not None:
                    items.append((pid, "".join(seq_parts)))
                    if max_items and len(items) >= max_items:
                        break
                # header como: sp|P9WHI7|RECN_MYCT ...
                hdr = line[1:].split()[0]
                # tenta extrair o accession (entre barras)
                if "|" in hdr:
                    # ex: sp|P9WHI7|RECN_MYCT -> P9WHI7
                    parts = hdr.split("|")
                    pid = parts[1] if len(parts) > 1 else parts[-1]
                else:
                    pid = hdr
                seq_parts = []
            else:
                seq_parts.append(line)
        # último
        if pid is not None and (not max_items or len(items) < max_items):
            items.append((pid, "".join(seq_parts)))
    return items

# -------- ESM embedding --------
@torch.no_grad()
def embed_batch(
    model: EsmModel,
    tokenizer,
    seqs: List[str],
    device: torch.device,
    pooling: str = "mean",
    max_len: int = 1022,  # limite prático do ESM2
) -> np.ndarray:
    toks = tokenizer(
        seqs,
        return_tensors="pt",
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    out = model(**toks)
    hidden = out.last_hidden_state  # [B, T, H]

    # removendo tokens especiais <cls>=0 e <eos>=T-1 para pooling
    # máscara: 1 nos tokens válidos (aminoácidos), 0 nos especiais/padding
    attn = toks["attention_mask"].clone()
    if hidden.size(1) >= 2:
        attn[:, 0] = 0
        attn[:, -1] = 0

    if pooling == "mean":
        attn = attn.unsqueeze(-1)  # [B, T, 1]
        summed = (hidden * attn).sum(dim=1)
        lens = attn.sum(dim=1).clamp(min=1)
        emb = summed / lens
    elif pooling == "cls":
        emb = hidden[:, 0, :]
    elif pooling == "max":
        attn = attn.unsqueeze(-1)
        masked = hidden.masked_fill(attn == 0, float("-inf"))
        emb = masked.max(dim=1).values
    else:
        raise ValueError(f"Pooling desconhecido: {pooling}")

    return emb.cpu().float().numpy()

def main():
    ensure_dirs()
    cfg = CFG
    p = cfg["paths"]
    model_name = cfg.get("model", {}).get("backbone", "facebook/esm2_t6_8M_UR50D")
    pooling = cfg.get("model", {}).get("pooling", "mean")

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--max_proteins", type=int, default=None, help="subset para rodar em CPU rapidamente")
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    fasta = p["train_fasta"] if args.split == "train" else p["test_fasta"]
    out_dir = Path(p["data_interim"]) / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"{args.split}_esm2_t6_8M_subset{args.max_proteins}.npz"
    out_ids = out_dir / f"{args.split}_ids_subset{args.max_proteins}.txt"

    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Lendo FASTA: {fasta}")
    items = read_fasta(fasta, max_items=args.max_proteins)
    ids = [pid for pid, _ in items]
    seqs = [s for _, s in items]
    print(f"[INFO] Proteínas lidas: {len(ids)}")

    device = torch.device("cpu")  # sem GPU
    torch.set_num_threads(max(1, torch.get_num_threads()))  # mantém padrão do sistema

    print(f"[ESM] Carregando tokenizer/modelo: {model_name} (CPU)")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = EsmModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    # gera embeddings em lotes e concatena
    all_embs = []
    bs = max(1, args.batch_size)
    for i in range(0, len(seqs), bs):
        batch = seqs[i:i+bs]
        embs = embed_batch(model, tokenizer, batch, device, pooling=pooling)
        all_embs.append(embs)
        if (i // bs) % 50 == 0:
            print(f"  - lote {i//bs:05d}  ({i+len(batch)}/{len(seqs)})")

    embs = np.concatenate(all_embs, axis=0).astype("float32")
    print(f"[OK] Embeddings shape: {embs.shape}  (dtype={embs.dtype})")

    # salva
    np.savez_compressed(out_npz, X=embs)
    with open(out_ids, "w", encoding="utf-8") as f:
        for pid in ids:
            f.write(pid + "\n")

    print(f"[SAVE] {out_npz}")
    print(f"[SAVE] {out_ids}")
    print("[DONE]")

if __name__ == "__main__":
    main()