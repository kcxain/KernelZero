#!/bin/bash
# 创建本地数据集的便捷脚本

STORAGE_PATH=${STORAGE_PATH:-./data}
export STORAGE_PATH

# 检查是否安装了必要的依赖
echo "Checking dependencies..."
python3 -c "import datasets" 2>/dev/null || {
    echo "ERROR: 'datasets' module is not installed."
    echo "Please install it using: pip install datasets pyarrow"
    exit 1
}

# 创建本地数据集目录
mkdir -p ../local_datasets

# 创建 questioner 和 solver 的本地数据集
echo "Creating local datasets..."
python3 create_local_dataset.py \
    --type both \
    --num_samples_questioner 10000 \
    --num_samples_solver 100

if [ $? -eq 0 ]; then
    echo "Dataset creation completed successfully!"
else
    echo "ERROR: Failed to create datasets. Please check the error messages above."
    exit 1
fi

echo ""
echo "=========================================="
echo "本地数据集创建完成！"
echo ""
echo "生成的路径："
echo "  Questioner 训练数据: ../local_datasets/questioner_dummy/train"
echo "  Questioner 验证数据: ../local_datasets/questioner_dummy/test (最小验证集，1个样本)"
echo "  Solver 训练数据: ../local_datasets/solver_dummy/train"
echo "  Solver 验证数据: ../local_datasets/solver_dummy/test"
echo ""
echo "使用方法："
echo "1. 在训练脚本中覆盖 train_files 和 val_files 参数："
echo "   data.train_files=../local_datasets/questioner_dummy/train"
echo "   data.val_files=../local_datasets/questioner_dummy/test"
echo ""
echo "注意：验证数据集是必需的（代码中有 assert），即使设置 val_freq=-1 不进行验证"
echo "      也需要提供验证数据集。脚本会自动创建最小的验证数据集（1个样本）。"
echo ""
echo "2. 或者修改 config.yaml 中的配置"
echo "=========================================="
