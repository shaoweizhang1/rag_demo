from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


# ===== config =====
# (model_name, save_dir, model_class)
MODELS = [
    ("BAAI/bge-m3",                "models/bge_m3",                AutoModel),
    ("BAAI/bge-reranker-v2-m3",   "models/bge_reranker_v2_m3",   AutoModelForSequenceClassification),
    ("Qwen/Qwen2.5-3B-Instruct",  "models/Qwen2.5_3B_Instruct",  AutoModel),
]


# ===== main =====

def main() -> None:
    for model_name, save_dir, model_cls in MODELS:
        print(f"downloading {model_name} -> {save_dir}")
        model     = model_cls.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"saved {model_name}")


if __name__ == "__main__":
    main()
