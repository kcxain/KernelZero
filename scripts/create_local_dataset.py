"""
创建一个最小的本地数据集用于离线训练

对于 questioner 训练：
- questioner 使用固定的 prompt，不依赖数据集内容
- 只需要提供足够的样本数量即可

对于 solver 训练：
- 需要真实的问题和答案对
- 如果还没有生成的数据，可以先用占位符数据

依赖要求：
- 需要安装 datasets 库: pip install datasets pyarrow
"""
import os
import sys
import argparse

try:
    from datasets import Dataset, DatasetDict
except ImportError:
    print("ERROR: 'datasets' module is not installed.")
    print("Please install it using: pip install datasets pyarrow")
    sys.exit(1)

STORAGE_PATH = os.getenv("STORAGE_PATH", "./data")
LOCAL_DATA_PATH = f"../local_datasets"

def create_minimal_dataset_for_questioner(num_samples=1000, output_dir=None, create_val=True):
    """
    为 questioner 创建最小的本地数据集
    
    Questioner 使用固定的 prompt（"Generate one new, challenging reasoning question now."），
    所以数据集中的 problem 和 answer 字段实际上不会被使用。
    只需要提供足够的样本数量即可。
    
    数据集大小计算：
    - questioner_train_penalty.sh 中 max_steps=6, rollout_batch_size=512
    - 最小样本数 = max_steps × rollout_batch_size = 6 × 512 = 3,072
    - 建议至少 5,000-10,000 个样本（留有调整空间）
    
    注意：验证数据集是必需的（代码中有 assert），所以即使不想验证也需要提供
    这里会创建一个最小的验证数据集（1个样本）
    """
    if output_dir is None:
        output_dir = f"{LOCAL_DATA_PATH}/questioner_dummy"
    
    # 创建占位符数据
    # 注意：对于 questioner_format，这些值虽然会被读取，但实际上不会被使用
    # 因为 _build_messages 函数会直接返回固定的 prompt，不依赖数据集内容
    train_data = []
    for i in range(num_samples):
        train_data.append({
            "problem": f"Dummy problem {i}",  # 不会被使用（questioner 使用固定 prompt）
            "answer": f"Dummy answer {i}",    # 不会被使用（questioner 不使用 ground_truth）
        })
    
    # 创建训练 Dataset
    train_dataset = Dataset.from_list(train_data)
    
    # 保存训练数据为 parquet 格式
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    train_file = f"{output_dir}/train/train-00000-of-00001.parquet"
    train_dataset.to_parquet(train_file)
    
    # 验证文件是否成功创建
    if not os.path.exists(train_file):
        raise RuntimeError(f"Failed to create train parquet file: {train_file}")
    
    # 创建最小的验证数据集（即使不想验证，代码中也需要 val_dataloader，所以至少需要1个样本）
    if create_val:
        val_data = [{
            "problem": "Dummy validation problem",  # 不会被使用
            "answer": "Dummy validation answer",    # 不会被使用
        }]
        val_dataset = Dataset.from_list(val_data)
        os.makedirs(f"{output_dir}/test", exist_ok=True)
        val_file = f"{output_dir}/test/test-00000-of-00001.parquet"
        val_dataset.to_parquet(val_file)
        
        if not os.path.exists(val_file):
            raise RuntimeError(f"Failed to create val parquet file: {val_file}")
    
    print(f"Created minimal dataset for questioner at: {output_dir}")
    print(f"Train file: {train_file}")
    if create_val:
        print(f"Validation file: {val_file}")
    print(f"Number of train samples: {num_samples}")
    print(f"Number of validation samples: {1 if create_val else 0}")
    print(f"\nUse these paths in config.yaml or training script:")
    print(f"  data.train_files={output_dir}/train")
    if create_val:
        print(f"  data.val_files={output_dir}/test")
    
    return output_dir


def create_minimal_dataset_for_solver(num_samples=100, output_dir=None):
    """
    为 solver 创建最小的本地数据集
    
    注意：Solver 需要真实的问题和答案对。
    这个函数创建的是占位符数据，仅用于测试。
    实际训练时应该使用 questioner 生成的数据。
    """
    if output_dir is None:
        output_dir = f"{LOCAL_DATA_PATH}/solver_dummy"
    
    # 创建占位符数据（仅用于测试）
    data = []
    for i in range(num_samples):
        data.append({
            "problem": f"What is 2 + {i}?",  # 占位符问题
            "answer": str(2 + i),            # 占位符答案
        })
    
    # 创建 Dataset
    train_dataset = Dataset.from_list(data)
    test_dataset = Dataset.from_list(data)  # 使用相同数据作为测试集
    
    # 保存为 parquet 格式
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)
    
    train_file = f"{output_dir}/train/train-00000-of-00001.parquet"
    test_file = f"{output_dir}/test/test-00000-of-00001.parquet"
    
    train_dataset.to_parquet(train_file)
    test_dataset.to_parquet(test_file)
    
    # 验证文件是否成功创建
    if not os.path.exists(train_file):
        raise RuntimeError(f"Failed to create train parquet file: {train_file}")
    if not os.path.exists(test_file):
        raise RuntimeError(f"Failed to create test parquet file: {test_file}")
    
    print(f"Created minimal dataset for solver at: {output_dir}")
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}")
    print(f"Number of samples: {num_samples}")
    print(f"\nWARNING: These are placeholder data. For real training, use generated data from questioner.")
    print(f"\nUse these paths in config.yaml or training script:")
    print(f"  data.train_files={output_dir}/train")
    print(f"  data.val_files={output_dir}/test")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Create minimal local datasets for offline training")
    parser.add_argument("--type", type=str, choices=["questioner", "solver", "both"], 
                       default="both", help="Type of dataset to create")
    parser.add_argument("--num_samples_questioner", type=int, default=10000,
                       help="Number of samples for questioner dataset (min: ~3500, recommended: 5000-10000)")
    parser.add_argument("--num_samples_solver", type=int, default=100,
                       help="Number of samples for solver dataset")
    parser.add_argument("--output_dir_questioner", type=str, default=None,
                       help="Output directory for questioner dataset")
    parser.add_argument("--output_dir_solver", type=str, default=None,
                       help="Output directory for solver dataset")
    
    args = parser.parse_args()
    
    if args.type in ["questioner", "both"]:
        create_minimal_dataset_for_questioner(
            num_samples=args.num_samples_questioner,
            output_dir=args.output_dir_questioner,
            create_val=True  # 验证数据集是必需的（代码中有 assert），即使不验证也需要至少1个样本
        )
    
    if args.type in ["solver", "both"]:
        create_minimal_dataset_for_solver(
            num_samples=args.num_samples_solver,
            output_dir=args.output_dir_solver
        )


if __name__ == "__main__":
    main()
