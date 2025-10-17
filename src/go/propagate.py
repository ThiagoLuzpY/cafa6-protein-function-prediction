# src/go/propagate.py
from pathlib import Path
import sqlite3
import pandas as pd
from tqdm import tqdm
import obonet
import networkx as nx
from src.utils.paths import CFG, ensure_dirs

BATCH_SIZE = 2000  # lote de inserts no SQLite (ajuste se quiser)

def load_go_graph(obo_path: str):
    print(f"[GO] Lendo OBO: {obo_path}")
    G = obonet.read_obo(obo_path)
    # Normaliza como grafo direcionado (em obonet, child -> parent costuma ser edge)
    G = nx.DiGraph(G)
    return G

def build_parent_map(G: nx.DiGraph):
    """Mapeia termo -> lista de pais imediatos (is_a / part_of)."""
    parent_map = {}
    for node in G.nodes:
        # successors(node) retorna nós alcançados por edges node->successor
        # Em GO (obonet), costuma ser child->parent
        parent_map[node] = list(G.successors(node))
    return parent_map

def all_ancestors(term: str, parent_map: dict[str, list[str]]):
    """Sobe na hierarquia acumulando ancestrais sem recursão profunda."""
    anc = set()
    stack = list(parent_map.get(term, []))
    while stack:
        u = stack.pop()
        if u in anc:
            continue
        anc.add(u)
        stack.extend(parent_map.get(u, []))
    return anc

def open_sqlite(db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=MEMORY;")
    cur.execute("PRAGMA synchronous=OFF;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS terms (
            uniprot TEXT NOT NULL,
            go_id   TEXT NOT NULL,
            ont     TEXT NOT NULL,
            PRIMARY KEY (uniprot, go_id, ont)
        );
    """)
    conn.commit()
    return conn

def export_sqlite_to_tsv(conn: sqlite3.Connection, out_tsv: Path):
    print(f"[SQL] Exportando para TSV: {out_tsv}")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8") as f:
        for row in conn.execute("SELECT uniprot, go_id, ont FROM terms;"):
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
    print("[SQL] Export completo.")

def main():
    ensure_dirs()
    p = CFG["paths"]
    obo = p["go_obo"]
    in_terms = p["train_terms"]
    out_terms = Path(p["data_processed"]) / "train_terms_propagated.tsv"
    db_path = Path(CFG["paths"]["data_interim"]) / "propagate.db"

    # 1) GO
    G = load_go_graph(obo)
    parent_map = build_parent_map(G)
    del G  # libera memória do grafo completo

    # 2) SQLite
    print(f"[SQL] Banco: {db_path}")
    conn = open_sqlite(db_path)
    cur = conn.cursor()

    # 3) Ler anotações em chunks para economizar RAM
    print(f"[DATA] Lendo: {in_terms}")
    total_rows = 0
    inserted = 0

    # vamos ler tudo de uma vez (arquivo não é gigante) — se quiser, pode usar chunksize
    df = pd.read_csv(in_terms, sep="\t", header=None, names=["uniprot", "go_id", "ont"])
    total_rows = len(df)
    print(f"[DATA] Linhas originais: {total_rows}")

    batch = []
    with tqdm(total=total_rows, desc="Propagando", unit="row") as pbar:
        for uniprot, go_id, ont in df.itertuples(index=False):
            # termo original
            batch.append((uniprot, go_id, ont))
            # ancestrais
            for anc in all_ancestors(go_id, parent_map):
                batch.append((uniprot, anc, ont))

            if len(batch) >= BATCH_SIZE:
                cur.executemany("INSERT OR IGNORE INTO terms (uniprot, go_id, ont) VALUES (?, ?, ?);", batch)
                conn.commit()
                inserted += len(batch)
                batch.clear()

            pbar.update(1)

    # flush final
    if batch:
        cur.executemany("INSERT OR IGNORE INTO terms (uniprot, go_id, ont) VALUES (?, ?, ?);", batch)
        conn.commit()
        inserted += len(batch)
        batch.clear()

    print(f"[SQL] Linhas (com duplicatas já ignoradas) inseridas/processadas ~ {inserted:,}")

    # 4) Export para TSV final
    export_sqlite_to_tsv(conn, out_terms)
    # fechar db
    conn.close()

    # 5) Estatística rápida
    n_out = sum(1 for _ in open(out_terms, "r", encoding="utf-8"))
    print(f"[OK] Salvo: {out_terms} | linhas: {n_out:,}")

if __name__ == "__main__":
    main()