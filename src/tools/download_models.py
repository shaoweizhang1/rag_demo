from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


model_names = ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3", "Qwen/Qwen2.5-7B"]
save_dirs = ["models/bge_m3", "models/bge_reranker_v2_m3", "models/Qwen2.5_7B"]


for model_name, save_dir in zip(model_names, save_dirs):

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(model)
    print(tokenizer)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

