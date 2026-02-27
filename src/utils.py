import json
from pathlib import Path

import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===== I/O =====

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_docstore(path: Path) -> list:
    return list(read_jsonl(path))


def avg_latency_from_jsonl(path: Path) -> float:
    values = [row["latency_ms"] for row in read_jsonl(path) if "latency_ms" in row]
    return sum(values) / len(values) if values else 0.0


# ===== embedding =====

@torch.inference_mode()
def embed_query(model, tokenizer, query: str):
    enc = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    out = model(**enc)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    emb  = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    emb  = F.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy()


# ===== reranker =====

@torch.inference_mode()
def rerank_scores(model, tokenizer, query: str, candidates: list, batch_size: int) -> list:
    scores = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        enc   = tokenizer(
            [query] * len(batch),
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits.squeeze(-1)
        scores.extend(logits.detach().cpu().tolist())
    return scores


# ===== context builder =====

def build_context(docstore: list, hit_indices: list, ctx_k: int = 4, max_chars: int = 8000):
    ctx_parts = []
    ctx_ids   = []
    for idx in hit_indices[:ctx_k]:
        if idx < 0:
            continue
        row  = docstore[idx]
        cid  = row.get("_id", str(idx))
        text = (row.get("text") or "").strip()
        if not text:
            continue
        ctx_ids.append(cid)
        ctx_parts.append(f"[{cid}]\n{text}")

    ctx = "\n\n---\n\n".join(ctx_parts)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars]
    return ctx_ids, ctx
