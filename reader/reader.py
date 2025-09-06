from huggingface_hub import snapshot_download
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch

def construct_prompt(query, retrieved_chunks):
    prompt = "Use the following chunks to answer the question: " + query + "\n"
    for i in range(len(retrieved_chunks)):
        chunk = " ".join(retrieved_chunks[i])
        prompt += chunk + "\n"
    return prompt

def load_reader_model():
    model_name = "tiiuae/falcon-7b-instruct"

    local_dir = snapshot_download(
        repo_id="tiiuae/falcon-7b-instruct",
        cache_dir="hf_cache",
        local_files_only=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder="hf_offload"
    )
    tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)

    llm = pipeline('text-generation', model=model, tokenizer=tokenizer)
    return llm