import json
from pathlib import Path

import torch
import faiss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

CORPUS_PATH = Path("data/corpus.jsonl")

MODEL_DIR = "models/bge_m3"
OUT_DIR = Path("vector_base")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHUNK_SIZE = 300
OVERLAP = 50
BATCH_SIZE = 32
MAX_EMBED_LEN = 512


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def mean_pooling(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.inference_mode()
def embed_texts(model, tokenizer, texts):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_EMBED_LEN,
        return_tensors="pt",
    ).to(DEVICE)
    out = model(**enc)
    emb = mean_pooling(out.last_hidden_state, enc["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy()


def chunk_by_tokens(tokenizer, text: str, chunk_size: int, overlap: int):
    # No special tokens
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(ids), step):
        end = start + chunk_size
        piece = ids[start:end]
        if not piece:
            continue
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True).strip())
        if end >= len(ids):
            break
    return chunks


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).to(DEVICE)
    model.eval()

    # 1) corpus -> chunk -> docstore
    docstore = []
    texts_for_index = []

    for x in tqdm(list(read_jsonl(CORPUS_PATH)), desc="read+chunk"):
        parent_id = x.get("_id") or x.get("id") or x.get("doc_id")
        title = (x.get("title") or "").strip()
        text = (x.get("text") or x.get("contents") or "").strip()
        full = (title + "\n" + text).strip() if title else text
        if not full:
            continue

        chunks = chunk_by_tokens(tokenizer, full, CHUNK_SIZE, OVERLAP)
        for i, ch in enumerate(chunks):
            if not ch:
                continue
            chunk_id = f"{parent_id}__{i}"
            docstore.append(
                {
                    "_id": chunk_id,
                    "parent_id": parent_id,
                    "chunk_no": i,
                    "text": ch,
                }
            )
            texts_for_index.append(ch)

    print("chunks:", len(texts_for_index))

    # 2) FAISS
    first = embed_texts(model, tokenizer, texts_for_index[: min(BATCH_SIZE, len(texts_for_index))])
    dim = first.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(first)

    for i in tqdm(range(BATCH_SIZE, len(texts_for_index), BATCH_SIZE), desc="embed+add"):
        vecs = embed_texts(model, tokenizer, texts_for_index[i : i + BATCH_SIZE])
        index.add(vecs)

    faiss.write_index(index, str(OUT_DIR / "index.faiss"))

    # 3) docstore
    with (OUT_DIR / "docstore.jsonl").open("w", encoding="utf-8") as f:
        for row in docstore:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "corpus_path": str(CORPUS_PATH),
        "model_dir": MODEL_DIR,
        "chunk_size_tokens": CHUNK_SIZE,
        "overlap_tokens": OVERLAP,
        "total_chunks": len(docstore),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()