import json
from pathlib import Path
from tqdm import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL_DIR = "models/Qwen2.5_3B_Instruct"
QUERIES_PATH = Path("data/queries.jsonl")
OUT_PATH = Path("result/qwen2.5.jsonl")

MAX_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    llm = LLM(
        model=MODEL_DIR,
        trust_remote_code=True,
    )

    sp = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for x in tqdm(list(read_jsonl(QUERIES_PATH)), desc="vllm baseline"):
            qid = x.get("_id") or x.get("id") or x.get("query_id")
            query = (x.get("text") or x.get("query") or "").strip()
            if not query:
                continue

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ]

            prompt = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            out = llm.generate([prompt], sp)[0]
            answer = out.outputs[0].text.strip()

            out_f.write(json.dumps({"qid": qid, "query": query, "answer": answer}, ensure_ascii=False) + "\n")
            out_f.flush()

    print("saved to", OUT_PATH)

if __name__ == "__main__":
    main()