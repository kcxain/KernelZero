#!/bin/bash
# PyTorch 算子爬虫运行脚本

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TORCH_POOL_DIR="$PROJECT_ROOT/torch_pool"

# 检查依赖
echo "检查依赖..."
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "缺少依赖，请安装:"
    echo "  pip install torch"
    exit 1
fi

# 运行爬虫
echo "开始运行爬虫..."
cd "$TORCH_POOL_DIR"
python3 scraper.py

echo "完成！"

