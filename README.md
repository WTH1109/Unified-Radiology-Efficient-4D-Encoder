# UME: Unified Medical imaging Efficient 4D-Encoder

## 项目简介

**UME (Unified Medical imaging Efficient 4D-Encoder)** 是一个完整的医学影像AI系统，专为BraTS脑肿瘤分割任务设计。该项目实现了自适应多模态动态采样编码器和4D医学影像数据的分层融合。

**核心创新**: 自适应多模态动态采样编码器，能够智能地选择关键帧并执行分层模态融合，支持任意数量模态的高效处理。

### 核心特性

- **动态采样**: 自适应关键帧选择，智能选择最相关的帧
- **多模态融合**: 分层融合任意数量模态，支持Intra/Inter/Global三级融合
- **Transformer架构**: 基于Vision Transformer (ViT) 骨干网络，专为医学影像优化
- **智能关键帧选择**: 多策略动态采样 (均匀 + 内容感知 + 注意力引导)
- **配置驱动**: 支持多种训练模式和模型大小的JSON配置
- **完整数据流**: 一次读入完整(4, 256, 256, 128)数据，交给网络智能处理
- **双推理模式**: 支持关键帧推理和逐帧推理
- **监督学习**: 传统监督训练，使用分割标签

## 项目结构

```
Unified-Radiology-Efficient-4D-Encoder/
├── ume/                          # 核心代码
│   ├── __init__.py
│   ├── core/                     # 核心模块
│   │   ├── __init__.py
│   │   ├── keyframe_selector.py  # 智能关键帧选择 (246 lines)
│   │   └── modal_fusion.py       # 多层次模态融合 (333 lines)
│   ├── models/                   # 模型定义
│   │   ├── __init__.py
│   │   └── ume_model.py          # UME主模型 (118M参数, 451MB)
│   ├── data/                     # 数据加载
│   │   ├── __init__.py
│   │   └── brats_loader.py       # BraTS数据加载器
│   ├── training/                 # 训练脚本
│   │   ├── __init__.py
│   │   └── train_ume.py          # 主训练脚本 (436 lines)
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       ├── losses.py             # 损失函数
│       ├── metrics.py            # 评估指标
├── configs/                      # 配置文件
│   ├── brats_ume_config.json     # 传统分割训练配置
│   ├── brats_ume_main.json       # 主要训练配置
│   ├── brats_ume_fast.json       # 快速测试配置
│   └── brats_ume_test.json       # 测试配置
├── jsons/                        # 数据集配置
│   └── brats21_folds.json        # 自动生成的数据配置
├── checkpoints/                  # 模型检查点
├── logs/                         # 训练日志
├── paper/                        # 论文相关资料
├── modification_requirements/     # 修改需求文档
├── UME_PRD/                      # 产品需求文档
├── quick_start.sh                # 快速启动脚本
├── start_training.sh             # 训练启动脚本
├── test_medclip.py              # MedCLIP测试脚本
├── requirements.txt              # 依赖列表
├── CLAUDE.md                     # 项目指令文档
├── PROJECT_INFO.md               # 项目信息
├── PROJECT_COMPLETION_REPORT.md  # 项目完成报告
├── CODE_STATISTICS.md            # 代码统计
├── MED_CLIP_USAGE.md             # MedCLIP使用说明
└── README.md                     # 本文档
```

## 环境配置与安装

### 系统要求

- **Python**: 3.9+
- **CUDA**: 12.1+ (GPU训练必需)
- **内存**: 建议16GB以上系统内存
- **GPU内存**: 建议8GB以上 (可通过配置调整)

### 核心依赖

- **PyTorch ≥2.0.0**: 深度学习框架
- **MONAI ≥1.3.0**: 医学影像AI工具包，提供变换、数据集和指标
- **timm**: 预训练视觉模型和Transformer实现
- **transformers**: Hugging Face Transformer架构
- **nibabel**: 医学影像格式处理 (.nii.gz文件)
- **tensorboard**: 训练可视化和日志记录

### 快速安装

#### 步骤1: 创建环境
```bash
# 创建并激活conda环境
conda create -n ume python=3.9
conda activate ume
```

#### 步骤2: 安装依赖
```bash
# 安装PyTorch (GPU版本)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装所有依赖
pip install -r requirements.txt
```

#### 步骤3: 验证安装
```bash
# 运行综合环境检查和交互式训练启动
./quick_start.sh
```

**快速启动脚本功能**:
- ✅ Python环境检查
- ✅ PyTorch和CUDA可用性验证
- ✅ MONAI安装确认
- ✅ 项目结构完整性检查
- ✅ 配置文件存在性验证
- 🚀 交互式训练模式选择

### 开发环境验证

```bash
# 快速环境验证命令
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import monai; print(f'MONAI版本: {monai.__version__}')"

# 测试核心组件
python test_medclip.py
```

## 数据准备

### BraTS数据集结构

确保BraTS数据集按以下结构组织：

```
/path/to/brats21/
├── BraTS21_Training_001/
│   ├── BraTS21_Training_001_t1.nii.gz
│   ├── BraTS21_Training_001_t2.nii.gz
│   ├── BraTS21_Training_001_t1ce.nii.gz
│   ├── BraTS21_Training_001_flair.nii.gz
│   └── BraTS21_Training_001_seg.nii.gz  # 可选
├── BraTS21_Training_002/
│   └── ...
```

### 配置数据路径

修改配置文件中的数据路径。以主配置为例：

```json
// configs/brats_ume_main.json
{
  "data": {
    "data_dir": "/path/to/your/brats21",
    "target_size": [256, 256, 128],  // 始终完整128帧
    ...
  }
}
```

**注意**: 所有配置文件都需要更新 `data_dir` 字段为您的BraTS数据集路径。

## 开发工作流程

### 快速启动

```bash
# 激活环境
conda activate ume

# 方法1: 交互式环境验证和训练启动 (推荐)
./quick_start.sh

# 方法2: 自动化训练启动脚本
./start_training.sh
```

### 开发命令

```bash
# 快速环境验证和交互式训练
./quick_start.sh

# 自动化训练启动 (包含环境检查)
./start_training.sh

# 测试Med-CLIP集成和关键帧选择器
python test_medclip.py

# 开发迭代 (使用快速配置)
python ume/training/train_ume.py --config configs/brats_ume_fast.json
```

### 主要训练命令

```bash
# 主要监督训练 (主要/推荐训练方式)
python ume/training/train_ume.py --config configs/brats_ume_main.json

# 传统监督训练 (替代配置)
python ume/training/train_ume.py --config configs/brats_ume_config.json

# 快速测试配置 (小模型)
python ume/training/train_ume.py --config configs/brats_ume_fast.json

# 分布式训练
python ume/training/train_ume.py --config configs/brats_ume_main.json --distributed

# 从检查点恢复训练
python ume/training/train_ume.py --config configs/brats_ume_main.json --resume /path/to/checkpoint.pth
```

### 测试基础设施

```bash
# 测试Med-CLIP集成和关键帧选择器功能
python test_medclip.py

# 快速环境验证 (在quick_start.sh中自动化)
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import monai; print(f'MONAI版本: {monai.__version__}')"
```

### 开发测试策略

- **集成测试**: `test_medclip.py` 验证核心组件
- **快速迭代**: 使用 `brats_ume_fast.json` 进行快速验证循环
- **环境验证**: `quick_start.sh` 提供全面的环境检查
- **无正式单元测试**: 项目依赖集成测试和训练验证

### 配置说明

#### 模型配置

```json
{
  "model": {
    "input_channels": 4,        // BraTS四个模态
    "embed_dim": 768,           // 嵌入维度
    "num_heads": 12,            // 注意力头数
    "num_layers": 12,           // Transformer层数
    "patch_size": 32,           // 图像块大小
    "image_size": [256, 256],   // 输入图像尺寸
    "num_classes": 4,           // 分割类别数
    "max_keyframes": 10,        // 最大关键帧数
    "compression_ratio": 4      // 特征压缩比例
  }
}
```

#### 数据配置

```json
{
  "data": {
    "data_dir": "/path/to/brats21",     // 数据目录
    "target_size": [256, 256, 128],     // 目标尺寸（完整128帧）
    "batch_size": 2,                    // 批次大小（建议较小）
    "num_workers": 4,                   // 数据加载进程数
    "cache_rate": 0.5,                  // 缓存比例
    "use_cache": true                   // 是否使用缓存
  }
}
```


#### 训练配置

```json
{
  "training": {
    "epochs": 100,                      // 训练轮数
    "learning_rate": 1e-4,              // 学习率
    "weight_decay": 1e-5,               // 权重衰减
    "warmup_epochs": 10,                // 预热轮数
    "loss_weights": {
      "diversity": 0.1,                 // 多样性损失权重
      "segmentation": 1.0,              // 分割损失权重
      "consistency": 0.05,              // 一致性损失权重
      "temporal": 0.03                  // 时序损失权重
    }
  }
}
```

## 核心技术

### 1. 智能关键帧选择

- **均匀采样**: 在时间维度上均匀分布
- **内容感知**: 基于帧间差异选择关键帧
- **注意力引导**: 使用注意力机制识别重要帧
- **策略融合**: 自适应权重融合多种策略

### 2. 多层次模态融合

- **Intra-Modal**: 同模态内不同切片融合
- **Inter-Modal**: 跨模态信息交互
- **Global**: 全局特征增强

### 3. 多模态动态采样

自适应多模态动态采样是本项目的核心创新：

- **完整数据读取**: 一次读入完整(4, 256, 256, 128)数据，交给网络智能处理
- **关键帧选择**: 网络动态选择K≤10个关键帧进行特征提取
- **Patch组织**: 256×256 → 64个32×32 patches → (4, 64, 32, 32, K)
- **多模态融合**: 分层融合支持任意数量的模态
- **监督训练**: 使用分割标签进行标准监督学习
- **自适应采样**: 智能选择最相关的帧进行处理

### 4. 训练策略

- **监督训练**: 主要训练方式，基于分割标签
- **关键帧网络**: 智能选择关键帧进行特征提取
- **多样性约束**: 确保选择的帧具有多样性

## 性能监控

监督训练过程中会显示以下指标：

```
Training Epoch 0: 100%|██████████| 250/250 [02:05<00:00, 1.99it/s, loss=1.8245, seg=1.2150, div=0.1895, cons=0.0567]
```

- **loss**: 总损失
- **seg**: 分割损失 (segmentation loss)
- **div**: 关键帧多样性损失 (diversity loss)
- **cons**: 一致性损失 (consistency loss)

传统分割训练（对比模式）显示指标：

```
Training Epoch 0: 100%|██████████| 250/250 [02:05<00:00, 1.99it/s, loss=2.2136, div=0.7695, seg=2.1366]
```

- **seg**: 分割损失 (segmentation loss)

## 调试和性能优化

### 内存管理

- **标准配置**: 需要约8GB GPU内存
- **内存问题**: 将 `batch_size` 从2减少到1，使用 `brats_ume_fast.json` 配置
- **缓存管理**: 如果内存受限，在配置中设置 `"use_cache": false`
- **分布式训练**: 多GPU设置使用 `--distributed` 标志

### 常见开发问题解决方案

```bash
# 首次运行数据加载过慢
# 解决方案: CacheDataset会预处理数据，后续运行会更快

# CUDA内存不足
# 解决方案: 减少配置中的batch_size，使用快速配置，或禁用缓存

# 维度不匹配错误
# 解决方案: 确保crop尺寸匹配ViT patch维度 (32×32)

# 缺少BraTS数据
# 解决方案: 更新配置文件中的data_dir，确保.nii.gz文件存在
```

### 性能优化策略

- **快速开发**: 使用 `configs/brats_ume_fast.json` (384 embed_dim, 6层)
- **生产环境**: 使用 `configs/brats_ume_main.json` (768 embed_dim, 12层)
- **缓存优化**: 初始数据处理后启用以加快训练
- **批次大小**: 从2开始，如有内存问题则减少到1

### 内存不足优化设置

1. **减少批次大小**:
```json
"batch_size": 1  // 从2减少到1
```

2. **使用快速配置模型**:
```json
"embed_dim": 384,    // 从768减少到384
"num_heads": 6,      // 从12减少到6
"num_layers": 6      // 从12减少到6
```

3. **禁用缓存**:
```json
"use_cache": false,
"cache_rate": 0.0
```

4. **减少关键帧数**:
```json
"max_keyframes": 6,     // 从10减少到6
"training_frames": 6    // 从8减少到6
```

## 常见问题

### Q: 数据加载很慢怎么办？
A: 这是正常现象。首次运行时CacheDataset会预处理所有数据到内存，后续训练会很快。可以：
- 使用 `"use_cache": false` 进行即时加载
- 增加 `num_workers` 数量
- 使用SSD存储加速I/O

### Q: GPU内存不足怎么办？
A: 参考"内存优化"部分，逐步减少模型大小和批次大小。

### Q: 如何修改数据路径？
A: 修改配置文件中的 `data_dir` 字段为您的BraTS数据集路径。

### Q: 如何监控训练进度？
A: 训练过程会显示实时损失，同时会保存检查点到工作目录。

## 项目文档

项目包含详细的技术文档和报告：

- **CLAUDE.md**: Claude Code使用指南和项目详细说明
- **PROJECT_INFO.md**: 项目概述和关键信息
- **PROJECT_COMPLETION_REPORT.md**: 项目完成报告和技术总结
- **CODE_STATISTICS.md**: 代码统计和架构分析
- **MED_CLIP_USAGE.md**: MedCLIP集成使用说明
- **UME_PRD/**: 详细的产品需求文档和技术设计

## 架构技术栈

### 技术支持

- **模型架构**: Vision Transformer (ViT) + Multi-Level Modal Fusion
- **数据处理**: MONAI医学影像框架 + CacheDataset内存优化
- **训练优化**: Mixed Precision + Gradient Scaling
- **内存管理**: 智能关键帧选择 + 动态采样策略
- **监督学习**: 传统分割监督训练
- **配置管理**: JSON驱动的多模式训练配置

### 架构模式

- **模块化设计**: 数据(`ume/data/`)、模型(`ume/models/`)、训练(`ume/training/`)和工具(`ume/utils/`)清晰分离
- **监督学习**: 分割标签作为主要训练范式
- **多模态融合**: 4个BraTS模态的分层融合，支持Intra/Inter/Global三级融合
- **Transformer驱动**: 基于ViT骨干网络，专为医学影像优化
- **配置驱动**: 支持不同训练模式和模型大小的JSON配置

## 开发状态

项目当前为**生产就绪**状态，具备以下特性：

### ✅ 完整功能
- 完整的监督训练流程
- 多配置支持（主要/传统/快速测试）
- 完整的数据处理管道
- 智能关键帧选择和模态融合
- 分布式训练支持
- 详细的文档和使用指南

### 🔧 开发工具
- 交互式环境验证 (`quick_start.sh`)
- 自动化训练启动 (`start_training.sh`)
- 核心组件集成测试 (`test_medclip.py`)
- 调试和性能优化指南

### 📚 文档体系
- **CLAUDE.md**: 开发者技术指南
- **README.md**: 用户使用手册
- **UME_PRD/**: 详细产品需求文档
- **技术报告**: 项目完成报告和代码统计

## 致谢

本项目整合了多种先进的医学影像处理技术，特别感谢MONAI社区和相关研究工作的贡献。**自适应多模态动态采样编码器是本项目的核心创新**，为医学影像的智能关键帧选择和多模态融合提供了新的解决方案，能够高效处理任意数量的模态并智能选择最相关的帧。