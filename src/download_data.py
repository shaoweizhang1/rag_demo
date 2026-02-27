import json
from pathlib import Path

from datasets import load_dataset


# ===== config =====
OUT_DIR = Path("data")


# ===== helpers =====

def save_jsonl(ds, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for x in ds:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


# ===== main =====

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    corpus  = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")
    queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")
    qrels   = load_dataset("BeIR/nfcorpus-qrels", split="test")

    save_jsonl(corpus,  OUT_DIR / "corpus.jsonl")
    save_jsonl(queries, OUT_DIR / "queries.jsonl")
    save_jsonl(qrels,   OUT_DIR / "qrels_test.jsonl")

    print("saved to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
