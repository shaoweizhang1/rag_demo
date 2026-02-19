from datasets import load_dataset
from pathlib import Path

out = Path("data")
out.mkdir(exist_ok=True)

load_dataset("BeIR/nfcorpus","corpus",split="corpus").save_to_disk(out/"beir_nfcorpus_corpus")
load_dataset("BeIR/nfcorpus","queries",split="queries").save_to_disk(out/"beir_nfcorpus_queries")
load_dataset("BeIR/nfcorpus-qrels",split="test").save_to_disk(out/"beir_nfcorpus_qrels_test")

print("saved to", out.resolve())

