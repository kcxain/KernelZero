from utils.io import load_json
from dotenv import load_dotenv
import os
import json

load_dotenv()

cudaLLM_sft_path = os.getenv("CUDALLM_SFT_DATA_PATH")
cudaLLM_rl_path = os.getenv("CUDALLM_RL_DATA_PATH")


cudaLLM_SFT = load_json(cudaLLM_sft_path)
cudaLLM_rl = load_json(cudaLLM_rl_path)

ops_all = set()

def get_ops(data):
    for item in data:
        ops = json.loads(item["ops"])
        ops_all.add(frozenset(ops))
    # print(len(data))
    print(len(ops_all))



if __name__ == "__main__":
    # get_ops(cudaLLM_SFT)
    get_ops(cudaLLM_rl)
    # print(ops_all)