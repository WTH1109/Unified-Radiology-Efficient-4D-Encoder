#!/bin/bash

# UME训练启动脚本
# 设置虚拟环境和启动训练

echo "=== UME Training Startup Script ==="

# 设置项目根目录
PROJECT_ROOT="/mnt/cfs/a8bga2/huawei/code/Unified-Radiology-Efficient-4D-Encoder"
cd $PROJECT_ROOT

# 检查虚拟环境
echo "Checking conda environment..."
if conda env list | grep -q "ume"; then
    echo "✓ UME environment found"
else
    echo "✗ UME environment not found. Please create it first."
    exit 1
fi

# 检查配置文件
CONFIG_FILE="./configs/brats_ume_config.json"
if [ -f "$CONFIG_FILE" ]; then
    echo "✓ Configuration file found: $CONFIG_FILE"
else
    echo "✗ Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# 创建必要的目录
echo "Creating directories..."
mkdir -p logs checkpoints jsons

# 安装依赖（如果需要）
echo "Installing dependencies..."
conda run -n ume pip install -q torch torchvision numpy monai nibabel timm transformers matplotlib scikit-learn tqdm tensorboard scipy

# 检查GPU
echo "Checking GPU availability..."
conda run -n ume python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 启动训练
echo "Starting UME training..."
echo "Configuration: $CONFIG_FILE"
echo "Project root: $PROJECT_ROOT"
echo ""

# 运行训练脚本
conda run -n ume python ume/training/train_ume.py \
    --config $CONFIG_FILE \
    --distributed false

echo "Training script finished."