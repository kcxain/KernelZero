from utils.io import load_json
from dotenv import load_dotenv
import os
import json

load_dotenv()

cudaLLM_sft_path = os.getenv("CUDALLM_SFT_DATA_PATH")
cudaLLM_rl_path = os.getenv("CUDALLM_RL_DATA_PATH")


cudaLLM_SFT = load_json(cudaLLM_sft_path)
cudaLLM_rl = load_json(cudaLLM_rl_path)

torch_ops = set()
torch_nn_ops = set()
torch_nn_functional_ops = set()

def get_ops(data):
    for item in data:
        ops = json.loads(item["ops"])
        for op in ops:
            if op.startswith("torch."):
                torch_ops.add(op)
            elif op.startswith("nn."):
                torch_nn_ops.add("torch." + op)
            elif op.startswith("F."):
                torch_nn_functional_ops.add("torch.nn.functional." + op[2:])
            else:
                print(op)
    print(len(torch_ops))
    print(len(torch_nn_ops))
    print(len(torch_nn_functional_ops))
    print(list(torch_ops))
    print(list(torch_nn_ops))
    print(list(torch_nn_functional_ops))

if __name__ == "__main__":
    get_ops(cudaLLM_rl)