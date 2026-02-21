import json
from datasets import load_dataset
from pathlib import Path

out = Path("data")
out.mkdir(exist_ok=True)

corpus = load_dataset("BeIR/nfcorpus","corpus",split="corpus")
queries = load_dataset("BeIR/nfcorpus","queries",split="queries")
test = load_dataset("BeIR/nfcorpus-qrels",split="test")

def save_jsonl(ds, path):
    with open(path, "w", encoding="utf-8") as f:
        for x in ds:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

save_jsonl(corpus, out / "corpus.jsonl")
save_jsonl(queries, out / "queries.jsonl")
save_jsonl(test, out / "qrels_test.jsonl")

print("done")