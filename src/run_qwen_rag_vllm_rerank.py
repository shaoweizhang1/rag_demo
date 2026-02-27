import json
import math
import time
from pathlib import Path

import faiss
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from vllm import LLM, SamplingParams

from utils import build_context, device, embed_query, load_docstore, read_jsonl, rerank_scores


# ===== config =====
QWEN_DIR    = "models/Qwen2.5_3B_Instruct"
EMB_DIR     = "models/bge_m3"
RERANK_DIR  = "models/bge_reranker_v2_m3"

QUERIES_PATH  = Path("data/queries.jsonl")
INDEX_PATH    = Path("vector_base/index.faiss")
DOCSTORE_PATH = Path("vector_base/docstore.jsonl")
OUT_PATH      = Path("result/qwen2.5_rag_rerank.jsonl")

TOP_K         = 20
CTX_K         = 4
MAX_CTX_CHARS = 8000
RERANK_BS     = 16

MAX_TOKENS  = 256
TEMPERATURE = 0.2
TOP_P       = 0.9
BATCH       = 32


# ===== main =====

def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    index    = faiss.read_index(str(INDEX_PATH))
    docstore = load_docstore(DOCSTORE_PATH)

    emb_tok   = AutoTokenizer.from_pretrained(EMB_DIR, trust_remote_code=True)
    emb_model = AutoModel.from_pretrained(EMB_DIR, trust_remote_code=True).to(device).eval()

    rerank_tok   = AutoTokenizer.from_pretrained(RERANK_DIR, trust_remote_code=True)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_DIR, trust_remote_code=True
    ).to(device).eval()

    qwen_tok = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
    llm      = LLM(
        model=QWEN_DIR,
        trust_remote_code=True,
        model_impl="transformers",
        runner="generate",
        gpu_memory_utilization=0.80,
    )
    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)

    rows          = list(read_jsonl(QUERIES_PATH))
    total_batches = math.ceil(len(rows) / BATCH)

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for bi in tqdm(range(total_batches), desc="rag+rerank"):
            batch_rows = rows[bi * BATCH : (bi + 1) * BATCH]

            prompts = []
            metas   = []

            for x in batch_rows:
                qid   = x.get("_id") or x.get("id") or x.get("query_id")
                query = (x.get("text") or x.get("query") or "").strip()
                if not query:
                    continue

                # ---- retrieve ----
                t0 = time.perf_counter()
                qvec = embed_query(emb_model, emb_tok, query)
                _, idxs = index.search(qvec, TOP_K)
                hit_indices = idxs[0].tolist()

                # ---- rerank ----
                cand_texts = [
                    (docstore[i].get("text") or "").strip() if i >= 0 else ""
                    for i in hit_indices
                ]
                scores  = rerank_scores(rerank_model, rerank_tok, query, cand_texts, RERANK_BS)
                order   = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
                reranked = [hit_indices[j] for j in order if hit_indices[j] >= 0][:CTX_K]
                retr_rerank_ms = (time.perf_counter() - t0) * 1000

                # ---- build context + prompt ----
                ctx_ids, ctx = build_context(docstore, reranked, ctx_k=CTX_K, max_chars=MAX_CTX_CHARS)
                messages = [
                    {
                        "role":    "system",
                        "content": "You are a helpful assistant. Answer using the given context. If the context is insufficient, say you don't know.",
                    },
                    {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion:\n{query}"},
                ]
                prompts.append(
                    qwen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                )
                metas.append({"qid": qid, "query": query, "ctx_ids": ctx_ids, "ctx": ctx, "retr_rerank_ms": retr_rerank_ms})

            if not prompts:
                continue

            t1 = time.perf_counter()
            outputs = llm.generate(prompts, sp, use_tqdm=False)
            gen_ms_per = (time.perf_counter() - t1) * 1000 / len(prompts)

            for meta, out in zip(metas, outputs):
                out_f.write(
                    json.dumps(
                        {
                            "qid":           meta["qid"],
                            "query":         meta["query"],
                            "answer":        out.outputs[0].text.strip(),
                            "ctx_ids":       meta["ctx_ids"],
                            "ctx":           meta["ctx"],
                            "output_tokens": len(out.outputs[0].token_ids),
                            "latency_ms":    round(meta["retr_rerank_ms"] + gen_ms_per, 2),
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )

    print("saved to", OUT_PATH)


if __name__ == "__main__":
    main()
