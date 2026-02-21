from transformers import AutoTokenizer, AutoModel 
from tqdm import tqdm 

MODEL_NAMES = ["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3", "Qwen/Qwen2.5-3B-Instruct"] 
SAVE_DIRS = ["models/bge_m3", "models/bge_reranker_v2_m3", "models/Qwen2.5_3B_Instruct"] 

for model_name, save_dir in zip(MODEL_NAMES, SAVE_DIRS): 
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True) 
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 
    
    print(model) 
    print(tokenizer) 

    model.save_pretrained(save_dir) 
    tokenizer.save_pretrained(save_dir)