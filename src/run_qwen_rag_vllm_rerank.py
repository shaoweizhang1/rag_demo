import json
from pathlib import Path
import math

import torch
import faiss
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification


# ===== paths =====
QWEN_DIR = "models/Qwen2.5_3B_Instruct"
EMB_DIR = "models/bge_m3"
RERANK_DIR = "models/bge_reranker_v2_m3"

QUERIES_PATH = Path("data/queries.jsonl")
INDEX_PATH = Path("vector_base/index.faiss")
DOCSTORE_PATH = Path("vector_base/docstore.jsonl")

OUT_PATH = Path("result/qwen2.5_rag.jsonl")


# ===== rag params =====
TOP_K = 20
CTX_K = 4
MAX_CTX_CHARS = 8000  
RERANK_BS = 16


# ===== gen params =====
MAX_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9
BATCH = 32


device = "cuda" if torch.cuda.is_available() else "cpu"


def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_docstore(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean_pooling(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.inference_mode()
def embed_query(emb_model, emb_tok, query: str):
    enc = emb_tok(
        [query],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    out = emb_model(**enc)
    emb = mean_pooling(out.last_hidden_state, enc["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy()


@torch.inference_mode()
def rerank(rerank_model, rerank_tok, query: str, candidates: list[str], batch_size: int):
    scores = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        enc = rerank_tok(
            [query] * len(batch),
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        logits = rerank_model(**enc).logits.squeeze(-1)  # (bs,)
        scores.extend(logits.detach().cpu().tolist())

    order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)
    return order, scores


def build_context(docstore, hit_indices, ctx_k=4):
    ctx_parts = []
    ctx_ids = []
    for idx in hit_indices[:ctx_k]:
        if idx < 0:
            continue
        row = docstore[idx]
        cid = row.get("_id", str(idx))
        text = (row.get("text") or "").strip()
        if not text:
            continue
        ctx_ids.append(cid)
        ctx_parts.append(f"[{cid}]\n{text}")

    ctx = "\n\n---\n\n".join(ctx_parts)
    if len(ctx) > MAX_CTX_CHARS:
        ctx = ctx[:MAX_CTX_CHARS]
    return ctx_ids, ctx


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # load faiss + docstore
    index = faiss.read_index(str(INDEX_PATH))
    docstore = load_docstore(DOCSTORE_PATH)

    # embedding model
    emb_tok = AutoTokenizer.from_pretrained(EMB_DIR, trust_remote_code=True)
    emb_model = AutoModel.from_pretrained(EMB_DIR, trust_remote_code=True).to(device).eval()

    # reranker
    rerank_tok = AutoTokenizer.from_pretrained(RERANK_DIR, trust_remote_code=True)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_DIR, trust_remote_code=True
    ).to(device).eval()

    # qwen via vllm
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)

    llm = LLM(
        model=QWEN_DIR, 
        trust_remote_code=True, 
        model_impl="transformers",
        runner='generate', 
        gpu_memory_utilization=0.80
    )
    
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)

    rows = list(read_jsonl(QUERIES_PATH))
    total_batches = math.ceil(len(rows) / BATCH)

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for bi in tqdm(range(total_batches), desc="rag+rerank vllm"):
            batch_rows = rows[bi * BATCH : (bi + 1) * BATCH]

            prompts = []
            metas = []

            for x in batch_rows:
                qid = x.get("_id") or x.get("id") or x.get("query_id")
                query = (x.get("text") or x.get("query") or "").strip()
                if not query:
                    continue

                # 1) retrieve
                qvec = embed_query(emb_model, emb_tok, query)
                _, idxs = index.search(qvec, TOP_K)
                hit_indices = idxs[0].tolist()

                # 2) rerank
                cand_texts = [(docstore[i].get("text") or "").strip() if i >= 0 else "" for i in hit_indices]
                order, _ = rerank(rerank_model, rerank_tok, query, cand_texts, RERANK_BS)
                reranked = [hit_indices[j] for j in order if hit_indices[j] >= 0][:CTX_K]

                # 3) build ctx
                ctx_ids, ctx = build_context(docstore, reranked, ctx_k=CTX_K)

                # 4) build prompt
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer using the given context. If the context is insufficient, say you don't know.",
                    },
                    {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion:\n{query}"},
                ]
                prompt = qwen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                prompts.append(prompt)
                metas.append({"qid": qid, "query": query, "ctx_ids": ctx_ids, "ctx": ctx})

            if not prompts:
                continue

            outs = llm.generate(prompts, sp, use_tqdm=False)

            for meta, out in zip(metas, outs):
                answer = out.outputs[0].text.strip()
                out_f.write(
                    json.dumps(
                        {
                            "qid": meta["qid"],
                            "query": meta["query"],
                            "answer": answer,
                            "ctx_ids": meta["ctx_ids"],
                            "ctx": meta["ctx"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print("saved to", OUT_PATH)


if __name__ == "__main__":
    main()