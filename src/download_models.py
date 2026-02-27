from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer


# ===== config =====
# Models downloaded via save_pretrained (no custom modeling code needed)
STANDARD_MODELS = [
    ("BAAI/bge-m3",               "models/bge_m3",               AutoModel),
    ("Qwen/Qwen2.5-3B-Instruct",  "models/Qwen2.5_3B_Instruct",  AutoModel),
]

# Models downloaded via snapshot_download to preserve custom modeling files
SNAPSHOT_MODELS = [
    ("BAAI/bge-reranker-v2-m3",  "models/bge_reranker_v2_m3"),
]


# ===== main =====

def main() -> None:
    for model_name, save_dir, model_cls in STANDARD_MODELS:
        print(f"downloading {model_name} -> {save_dir}")
        model     = model_cls.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"saved {model_name}")

    for model_name, save_dir in SNAPSHOT_MODELS:
        print(f"downloading {model_name} -> {save_dir}  (snapshot)")
        snapshot_download(repo_id=model_name, local_dir=save_dir)
        print(f"saved {model_name}")


if __name__ == "__main__":
    main()
