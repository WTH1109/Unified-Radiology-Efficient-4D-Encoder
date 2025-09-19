#!/bin/bash

# UME 快速开始脚本
# 用于验证环境配置和快速启动训练

echo "=== UME 快速开始脚本 ==="
echo "检查环境配置和启动训练"
echo

# 检查Python环境
echo "1. 检查Python环境..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Python未安装或环境未激活"
    echo "请运行: conda activate ume"
    exit 1
fi
echo "✅ Python环境正常"

# 检查PyTorch
echo
echo "2. 检查PyTorch安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
if [ $? -ne 0 ]; then
    echo "❌ PyTorch未安装"
    echo "请运行: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"
    exit 1
fi
echo "✅ PyTorch安装正常"

# 检查CUDA
echo
echo "3. 检查CUDA可用性..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')" 2>/dev/null
echo "✅ CUDA检查完成"

# 检查MONAI
echo
echo "4. 检查MONAI安装..."
python -c "import monai; print(f'MONAI版本: {monai.__version__}')"
if [ $? -ne 0 ]; then
    echo "❌ MONAI未安装"
    echo "请运行: pip install monai[all]"
    exit 1
fi
echo "✅ MONAI安装正常"

# 检查项目结构
echo
echo "5. 检查项目结构..."
if [ ! -f "ume/models/ume_model.py" ]; then
    echo "❌ 项目结构不完整，请确保在正确的项目根目录"
    exit 1
fi
echo "✅ 项目结构正常"

# 检查配置文件
echo
echo "6. 检查配置文件..."
if [ ! -f "configs/brats_ume_config.json" ]; then
    echo "❌ 配置文件缺失"
    exit 1
fi
echo "✅ 配置文件存在"

# 提供训练选项
echo
echo "=== 环境检查完成 ==="
echo
echo "请选择训练模式："
echo "1. VoCo自监督训练 (推荐，主要训练方式)"
echo "2. 传统分割训练 (对比)"
echo "3. 快速测试训练 (小模型，适合验证)"
echo "4. 仅检查环境 (不启动训练)"
echo
read -p "请输入选择 (1/2/3/4): " choice

case $choice in
    1)
        echo "启动VoCo自监督训练..."
        python ume/training/train_ume.py --config configs/brats_ume_voco.json
        ;;
    2)
        echo "启动传统分割训练..."
        python ume/training/train_ume.py --config configs/brats_ume_config.json
        ;;
    3)
        echo "启动快速测试训练..."
        python ume/training/train_ume.py --config configs/brats_ume_fast.json
        ;;
    4)
        echo "环境检查完成，未启动训练"
        ;;
    *)
        echo "无效选择，退出"
        exit 1
        ;;
esac

echo
echo "=== 脚本执行完成 ==="