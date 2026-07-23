import argparse
import multiprocessing
import os
import json
from transformers import AutoTokenizer

STORAGE_PATH = os.getenv("STORAGE_PATH")

def main(args):
    # 设置 vLLM 环境变量，禁用 V1 引擎避免 fork 问题
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['VLLM_USE_V1'] = '0'
    print("[DEBUG] Set VLLM_USE_V1=0 to disable V1 engine")
    
    # 延迟导入 vLLM，确保在设置 multiprocessing 之后导入
    import vllm
    import evaluation.datasets_loader as datasets_loader

    
    print("STORAGE_PATH")
    print(STORAGE_PATH)
    print(args.model, args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=0.85,
        enforce_eager=True,  # 禁用 CUDA graph，避免某些多进程问题
        disable_custom_all_reduce=True,  # 禁用自定义 all-reduce，避免多进程问题
    )
    sample_params = vllm.SamplingParams(
        max_tokens=4096,
        temperature=0.0,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    handler = datasets_loader.get_dataset_handler(args.dataset,args.name)
    questions, answers = handler.load_data()
    chats=[[{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},{"role": "user", "content": question}] for question in questions]
    if tokenizer.chat_template:
        prompts = [tokenizer.apply_chat_template(chat, tokenize=False,add_generation_prompt=True, add_special_tokens=True) for chat in chats]
    else:
        prompts = ["system: " + chat[0]["content"] + '\n' + "user: " + chat[1]["content"] + '\nPlease reason step by step, and put your final answer within \\boxed{}.' for chat in chats]
    responses = model.generate(prompts, sampling_params=sample_params,use_tqdm=True)
    responses = [response.outputs[0].text for response in responses]
    scores,average_score = handler.get_score(responses, answers)
    results = [{"question": question, "answer": answer, "response": response, "score": score} for question, answer, response, score in zip(questions, answers, responses, scores)]
    print(f"Average score: {average_score}")
    results.append({"average_score": average_score})
    os.makedirs(f"{STORAGE_PATH}/evaluation/{args.model.replace('/', '_')}", exist_ok=True)
    with open(f"{STORAGE_PATH}/evaluation/{args.model.replace('/', '_')}/results_{args.dataset}.json", "w") as f:
        json.dump(results, f, indent=4)

    
if __name__ == "__main__":
    # 设置 multiprocessing start method 为 spawn，避免 CUDA fork 问题
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("[DEBUG] Set multiprocessing start method to 'spawn'")
    except RuntimeError as e:
        print(f"[DEBUG] Multiprocessing start method already set: {e}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", type=str, default="math")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    main(args)
