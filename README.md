# CAFA 6 – Protein Function Prediction

## Objetivo
Prever termos GO (MF, BP, CC) para proteínas a partir da sequência.

## Dados
- `train_sequences.fasta`, `train_terms.tsv`, `train_taxonomy.tsv`
- `go-basic.obo`, `IA.tsv`
- `testsuperset.fasta`

## Pipeline (resumo)
1) Parse FASTA/GO/IA → 2) Embeddings (ESM2/ProtBERT) → 
3) Classificador multilabel → 4) Propagação GO → 
5) Avaliação local (F1 ponderado) → 6) Geração de `submission.tsv`.

## Como rodar
Ver `configs/config.yaml` e os comandos em `src/train/train_baseline.py`.
