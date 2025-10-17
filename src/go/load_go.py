import obonet
import networkx as nx
import pandas as pd
from typing import Tuple

def load_go_and_ia(obo_path: str, ia_path: str) -> Tuple[nx.DiGraph, dict]:
    """
    Lê o GO como grafo direcionado acíclico e IA.tsv como dict {go_id: ia}.
    """
    G: nx.DiGraph = obonet.read_obo(obo_path)
    ia_df = pd.read_csv(ia_path, sep="\t", header=None, names=["go_id", "ia"])
    ia = dict(zip(ia_df["go_id"], ia_df["ia"]))
    return G, ia
