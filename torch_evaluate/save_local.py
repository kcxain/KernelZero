"""
将生成的训练数据保存到本地 parquet 格式，而不是上传到 HuggingFace
适用于训练节点不能联网的场景
"""
import json
import os
import argparse
from datasets import Dataset, DatasetDict

STORAGE_PATH = os.getenv("STORAGE_PATH")
print(f"STORAGE_PATH: {STORAGE_PATH}")

parser = argparse.ArgumentParser()
parser.add_argument("--repo_name", type=str, default="")
parser.add_argument("--max_score", type=float, default=0.7)
parser.add_argument("--min_score", type=float, default=0.3)
parser.add_argument("--experiment_name", type=str, default="Qwen_Qwen3-4B-Base_all")
args = parser.parse_args()

# 读取所有生成的问题结果
datas = []
for i in range(8):
    try:
        with open(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json', 'r') as f:
            data = json.load(f)
            datas.extend(data)
    except:
        print(f"File {args.experiment_name}_{i}_results.json not found")
        continue

# 删除临时文件（可选）
for i in range(8):
    try:
        os.remove(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json')
    except:
        print(f"File {args.experiment_name}_{i}_results.json not found")
        continue

# 过滤数据
if not args.repo_name == "":
    filtered_datas = [
        {'problem': data['question'], 'answer': data['answer'], 'score': data['score']} 
        for data in datas 
        if data['score'] >= args.min_score 
        and data['score'] <= args.max_score 
        and data['answer'] != '' 
        and data['answer'] != 'None'
    ]
    print(f"Filtered {len(filtered_datas)} samples (from {len(datas)} total)")
    
    # 创建 Dataset
    train_dataset = Dataset.from_list(filtered_datas)
    dataset_dict = {"train": train_dataset}
    dataset = DatasetDict(dataset_dict)
    
    # 保存到本地 parquet 格式
    local_dataset_path = f"{STORAGE_PATH}/generated_question/{args.experiment_name}_train"
    os.makedirs(local_dataset_path, exist_ok=True)
    
    # 保存为 parquet 格式（保存到目录，让 load_dataset 可以从目录加载）
    # 使用 to_parquet 保存单个文件，或者使用 save_to_disk 保存完整数据集
    # 为了兼容 load_dataset("parquet", data_dir=...)，我们保存为 parquet 文件
    parquet_file = f"{local_dataset_path}/train-00000-of-00001.parquet"
    dataset["train"].to_parquet(parquet_file)
    
    print(f"Dataset saved to: {local_dataset_path}")
    print(f"Parquet file: {parquet_file}")
    print(f"Use this path in solver_train.sh: {local_dataset_path}")

