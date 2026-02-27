from transformers import AutoModel, AutoTokenizer


# ===== config =====
MODEL_NAMES = [
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
    "Qwen/Qwen2.5-3B-Instruct",
]

SAVE_DIRS = [
    "models/bge_m3",
    "models/bge_reranker_v2_m3",
    "models/Qwen2.5_3B_Instruct",
]


# ===== main =====

def main() -> None:
    for model_name, save_dir in zip(MODEL_NAMES, SAVE_DIRS):
        print(f"downloading {model_name} -> {save_dir}")
        model     = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"saved {model_name}")


if __name__ == "__main__":
    main()
