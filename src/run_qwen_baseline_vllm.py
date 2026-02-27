import json
import time
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import read_jsonl


# ===== config =====
MODEL_DIR    = "models/Qwen2.5_3B_Instruct"
QUERIES_PATH = Path("data/queries.jsonl")
OUT_PATH     = Path("result/qwen2.5.jsonl")

MAX_TOKENS  = 256
TEMPERATURE = 0.2
TOP_P       = 0.9
BATCH       = 64


# ===== main =====

def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    llm = LLM(
        model=MODEL_DIR,
        trust_remote_code=True,
        model_impl="transformers",
        runner="generate",
    )

    sp   = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
    rows = list(read_jsonl(QUERIES_PATH))

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(rows), BATCH), desc="qwen baseline"):
            batch = rows[i : i + BATCH]

            qids, queries, prompts = [], [], []
            for x in batch:
                qid   = x.get("_id") or x.get("id") or x.get("query_id")
                query = (x.get("text") or x.get("query") or "").strip()
                if not query:
                    continue
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user",   "content": query},
                ]
                qids.append(qid)
                queries.append(query)
                prompts.append(
                    tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                )

            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sp, use_tqdm=False)
            gen_ms_per = (time.perf_counter() - t0) * 1000 / max(len(prompts), 1)

            for qid, query, o in zip(qids, queries, outputs):
                out_f.write(
                    json.dumps(
                        {
                            "qid":        qid,
                            "query":      query,
                            "answer":     o.outputs[0].text.strip(),
                            "latency_ms": round(gen_ms_per, 2),
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )

    print("saved to", OUT_PATH)


if __name__ == "__main__":
    main()
