from pathlib import Path
import yaml

# defaults só para chaves que podem faltar no YAML
DEFAULTS = {
    "paths": {
        "data_raw": "data/raw",
        "data_processed": "data/processed",
        "submissions": "submissions",
        # data_interim não está no seu YAML — criamos por default:
        "data_interim": "data/interim",
        # arquivos (mantemos defaults, mas o seu YAML já define todos)
        "go_obo": "data/raw/go-basic.obo",
        "ia_tsv": "data/raw/IA.tsv",
        "train_fasta": "data/raw/train_sequences.fasta",
        "train_terms": "data/raw/train_terms.tsv",
        "train_taxonomy": "data/raw/train_taxonomy.tsv",
        "test_fasta": "data/raw/testsuperset.fasta",
    }
}

def load_config(path: str = "configs/config.yaml"):
    """Carrega o YAML do usuário e faz merge raso com DEFAULTS (sem exigir chaves extras)."""
    cfg = DEFAULTS.copy()
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        if isinstance(user, dict):
            # merge nível 1
            for k, v in user.items():
                if k not in cfg:
                    cfg[k] = v
                else:
                    if isinstance(cfg[k], dict) and isinstance(v, dict):
                        cfg[k].update(v)
                    else:
                        cfg[k] = v
    return cfg

CFG = load_config()

def ensure_dirs():
    """Cria as pastas necessárias caso não existam (inclui data_interim mesmo se não estiver no YAML)."""
    p = CFG["paths"]
    for key in ["data_raw", "data_processed", "data_interim", "submissions"]:
        Path(p[key]).mkdir(parents=True, exist_ok=True)
    # reports
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/logs").mkdir(parents=True, exist_ok=True)

def path_exists_report():
    """Retorna um dict booleano checando arquivos críticos em data/raw/."""
    p = CFG["paths"]
    required = {
        "go_obo": p["go_obo"],
        "ia_tsv": p["ia_tsv"],
        "train_fasta": p["train_fasta"],
        "train_terms": p["train_terms"],
        "train_taxonomy": p["train_taxonomy"],
        "test_fasta": p["test_fasta"],
    }
    return {k: Path(v).exists() for k, v in required.items()}

if __name__ == "__main__":
    ensure_dirs()
    print("Pastas criadas/verificadas.")
    print("Arquivos essenciais:", path_exists_report())
