from Bio import SeqIO
from typing import Dict

def read_fasta(path: str) -> Dict[str, str]:
    """Retorna {uniprot_id: sequence}"""
    seqs = {}
    for record in SeqIO.parse(path, "fasta"):
        # header ex: sp|P9WHI7|RECN_MYCT ...
        header = record.id
        uid = header.split("|")[1] if "|" in header else header
        seqs[uid] = str(record.seq)
    return seqs
