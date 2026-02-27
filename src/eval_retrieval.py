import json
import math
import time
from collections import defaultdict
from pathlib import Path

import faiss
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


# ===== config =====
EMB_DIR = "models/bge_m3"
RERANK_DIR = "models/bge_reranker_v2_m3"

QUERIES_PATH = Path("data/queries.jsonl")
QRELS_PATH = Path("data/qrels_test.jsonl")
INDEX_PATH = Path("vector_base/index.faiss")
DOCSTORE_PATH = Path("vector_base/docstore.jsonl")
OUT_PATH = Path("result/eval_retrieval.json")

TOP_K = 20       # number of candidates retrieved from FAISS
RERANK_BS = 16   # reranker batch size
K_VALUES = [1, 5, 10]

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===== I/O helpers =====

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_qrels(path: Path) -> dict:
    """
    Returns {qid: {doc_id: relevance_score}}.
    Handles BeIR field names (query-id / corpus-id) and alternatives.
    Only retains entries with score > 0.
    """
    qrels: dict = defaultdict(dict)
    for row in read_jsonl(path):
        qid = str(
            row.get("query-id") or row.get("qid") or row.get("query_id") or ""
        )
        did = str(
            row.get("corpus-id") or row.get("doc_id") or row.get("corpus_id") or ""
        )
        score = int(row.get("score", 1))
        if qid and did and score > 0:
            qrels[qid][did] = score
    return dict(qrels)


def load_queries(path: Path) -> list:
    """Returns list of dicts with keys qid and text."""
    rows = []
    for x in read_jsonl(path):
        qid = str(x.get("_id") or x.get("id") or x.get("query_id") or "")
        text = (x.get("text") or x.get("query") or "").strip()
        if qid and text:
            rows.append({"qid": qid, "text": text})
    return rows


def load_docstore(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.detach().cpu().numpy()


# ===== reranker =====

@torch.inference_mode()
def rerank_scores(model, tokenizer, query: str, candidates: list, batch_size: int) -> list:
    scores = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        enc = tokenizer(
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


# ===== chunk -> doc mapping =====

def chunks_to_doc_ranking(hit_indices: list, docstore: list) -> list:
    """
    Convert a ranked list of chunk indices into a deduplicated ranked list of
    parent document IDs.  The rank of a document is determined by the rank of
    its first (highest-scoring) chunk.
    """
    seen: set = set()
    doc_ids = []
    for idx in hit_indices:
        if idx < 0:
            continue
        row = docstore[idx]
        parent_id = row.get("parent_id") or row.get("_id", "").rsplit("__", 1)[0]
        if parent_id and parent_id not in seen:
            seen.add(parent_id)
            doc_ids.append(parent_id)
    return doc_ids


# ===== IR metrics =====

def dcg(relevances: list, k: int) -> float:
    return sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(doc_ranking: list, qrel: dict, k: int) -> float:
    if not qrel:
        return 0.0
    rels = [qrel.get(d, 0) for d in doc_ranking[:k]]
    ideal = sorted(qrel.values(), reverse=True)
    idcg = dcg(ideal, k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0


def recall_at_k(doc_ranking: list, qrel: dict, k: int) -> float:
    relevant = {d for d, s in qrel.items() if s > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(doc_ranking[:k])) / len(relevant)


def mean_reciprocal_rank(doc_ranking: list, qrel: dict) -> float:
    relevant = {d for d, s in qrel.items() if s > 0}
    for rank, doc_id in enumerate(doc_ranking, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


# ===== main =====

def main():
    print(f"device: {device}")

    print("loading index and data...")
    index = faiss.read_index(str(INDEX_PATH))
    docstore = load_docstore(DOCSTORE_PATH)
    qrels_all = load_qrels(QRELS_PATH)
    all_queries = load_queries(QUERIES_PATH)

    queries = [q for q in all_queries if q["qid"] in qrels_all]
    print(f"total queries: {len(all_queries)}  |  queries with qrels: {len(queries)}")

    print("loading embedding model...")
    emb_tok = AutoTokenizer.from_pretrained(EMB_DIR, trust_remote_code=True)
    emb_model = AutoModel.from_pretrained(EMB_DIR, trust_remote_code=True).to(device).eval()

    print("loading reranker model...")
    rerank_tok = AutoTokenizer.from_pretrained(RERANK_DIR, trust_remote_code=True)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_DIR, trust_remote_code=True
    ).to(device).eval()

    faiss_ndcg   = {k: [] for k in K_VALUES}
    faiss_recall = {k: [] for k in K_VALUES}
    faiss_mrr    = []
    faiss_lat    = []

    rerank_ndcg   = {k: [] for k in K_VALUES}
    rerank_recall = {k: [] for k in K_VALUES}
    rerank_mrr    = []
    rerank_lat    = []

    for q in tqdm(queries, desc="evaluating"):
        qid   = q["qid"]
        query = q["text"]
        qrel  = qrels_all[qid]

        # ---- FAISS retrieval ----
        t0 = time.perf_counter()
        qvec = embed_query(emb_model, emb_tok, query)
        _, idxs = index.search(qvec, TOP_K)
        faiss_ms = (time.perf_counter() - t0) * 1000
        faiss_lat.append(faiss_ms)

        hit_indices = idxs[0].tolist()
        faiss_ranking = chunks_to_doc_ranking(hit_indices, docstore)

        for k in K_VALUES:
            faiss_ndcg[k].append(ndcg_at_k(faiss_ranking, qrel, k))
            faiss_recall[k].append(recall_at_k(faiss_ranking, qrel, k))
        faiss_mrr.append(mean_reciprocal_rank(faiss_ranking, qrel))

        # ---- reranker ----
        t1 = time.perf_counter()
        cand_texts = [
            (docstore[i].get("text") or "").strip() if i >= 0 else ""
            for i in hit_indices
        ]
        scores = rerank_scores(rerank_model, rerank_tok, query, cand_texts, RERANK_BS)
        order = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)
        reranked_indices = [hit_indices[j] for j in order if hit_indices[j] >= 0]
        rerank_ms = (time.perf_counter() - t1) * 1000
        rerank_lat.append(faiss_ms + rerank_ms)

        reranked_ranking = chunks_to_doc_ranking(reranked_indices, docstore)

        for k in K_VALUES:
            rerank_ndcg[k].append(ndcg_at_k(reranked_ranking, qrel, k))
            rerank_recall[k].append(recall_at_k(reranked_ranking, qrel, k))
        rerank_mrr.append(mean_reciprocal_rank(reranked_ranking, qrel))

    # ===== print table =====
    print("\n" + "=" * 56)
    print(f"  {'Metric':<22} {'FAISS only':>12} {'FAISS+Rerank':>14}")
    print("=" * 56)
    for k in K_VALUES:
        fn = mean(faiss_ndcg[k])
        rn = mean(rerank_ndcg[k])
        print(f"  NDCG@{k:<17} {fn:>12.4f} {rn:>14.4f}")
    print("-" * 56)
    for k in K_VALUES:
        fr = mean(faiss_recall[k])
        rr = mean(rerank_recall[k])
        print(f"  Recall@{k:<15} {fr:>12.4f} {rr:>14.4f}")
    print("-" * 56)
    fm = mean(faiss_mrr)
    rm = mean(rerank_mrr)
    print(f"  {'MRR':<22} {fm:>12.4f} {rm:>14.4f}")
    print("=" * 56)
    print(f"\n  Latency (avg per query)")
    print(f"    FAISS only  :  {mean(faiss_lat):6.1f} ms")
    print(f"    FAISS+Rerank:  {mean(rerank_lat):6.1f} ms")

    # ===== save =====
    result = {
        "num_queries": len(queries),
        "top_k_retrieval": TOP_K,
        "faiss": {
            **{f"ndcg@{k}": mean(faiss_ndcg[k]) for k in K_VALUES},
            **{f"recall@{k}": mean(faiss_recall[k]) for k in K_VALUES},
            "mrr": fm,
            "avg_latency_ms": mean(faiss_lat),
        },
        "faiss_rerank": {
            **{f"ndcg@{k}": mean(rerank_ndcg[k]) for k in K_VALUES},
            **{f"recall@{k}": mean(rerank_recall[k]) for k in K_VALUES},
            "mrr": rm,
            "avg_latency_ms": mean(rerank_lat),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
