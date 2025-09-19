# UME: Unified Medical imaging Efficient 4D-Encoder

## 项目简介

UME (Unified Medical imaging Efficient 4D-Encoder) 是一个专为医学影像设计的高效4D编码器，特别优化了BraTS脑肿瘤分割任务。该项目整合了智能关键帧选择、多层次模态融合和时序建模等先进技术。

### 核心特性

- **智能关键帧选择**: 多策略抽帧 (均匀采样 + 内容感知 + 注意力引导)
- **多层次模态融合**: Intra-Modal + Inter-Modal + Global融合
- **VoCo自监督训练**: 基于随机crop和相交面积比的对比学习
- **完整数据处理**: 一次读入完整256×256×128数据，交给网络智能选择
- **逐帧推理**: 测试时支持frame-by-frame预测
- **BraTS优化**: 专门针对256×256×128×4的脑肿瘤数据

## 项目结构

```
Unified-Radiology-Efficient-4D-Encoder/
├── ume/                          # 核心代码
│   ├── __init__.py
│   ├── core/                     # 核心模块
│   │   ├── __init__.py
│   │   ├── keyframe_selector.py  # 智能关键帧选择
│   │   └── modal_fusion.py       # 多层次模态融合
│   ├── models/                   # 模型定义
│   │   ├── __init__.py
│   │   └── ume_model.py          # UME主模型
│   ├── data/                     # 数据加载
│   │   ├── __init__.py
│   │   └── brats_loader.py       # BraTS数据加载器
│   ├── training/                 # 训练脚本
│   │   ├── __init__.py
│   │   └── train_ume.py          # 主训练脚本
│   └── utils/                    # 工具函数
│       ├── __init__.py
│       ├── losses.py             # 损失函数
│       └── metrics.py            # 评估指标
├── configs/                      # 配置文件
│   ├── brats_ume_config.json     # 标准训练配置
│   └── brats_ume_fast.json       # 快速测试配置
├── jsons/                        # 数据集配置
│   └── brats21_folds.json        # 自动生成的数据配置
├── CLAUDE.md                     # 项目指令文档
└── README.md                     # 本文档
```

## 安装说明

### 1. 环境要求

- Python 3.9+
- CUDA 12.1+ (GPU训练)
- conda/miniconda

### 2. 创建环境

```bash
# 创建conda环境
conda create -n ume python=3.9
conda activate ume
```

### 3. 安装依赖

```bash
# 安装PyTorch (GPU版本)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装MONAI及相关依赖
pip install monai[all]

# 安装其他依赖
pip install timm transformers tqdm
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

修改 `configs/brats_ume_config.json` 中的数据路径：

```json
{
  "data": {
    "data_dir": "/path/to/your/brats21",
    ...
  }
}
```

## 使用方法

### 训练模型

```bash
# 激活环境
conda activate ume

# VoCo自监督训练 (推荐，主要训练方式)
python ume/training/train_ume.py --config configs/brats_ume_voco.json

# 传统分割训练 (仅对比)
python ume/training/train_ume.py --config configs/brats_ume_config.json

# 快速测试训练 (较小配置)
python ume/training/train_ume.py --config configs/brats_ume_fast.json

# 分布式训练 (多GPU)
python ume/training/train_ume.py --config configs/brats_ume_voco.json --distributed

# 从检查点恢复训练
python ume/training/train_ume.py --config configs/brats_ume_voco.json --resume /path/to/checkpoint.pth
```

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
    "batch_size": 2,                    // 批次大小（VoCo训练建议较小）
    "num_workers": 4,                   // 数据加载进程数
    "cache_rate": 0.5,                  // 缓存比例
    "use_cache": true                   // 是否使用缓存
  }
}
```

#### VoCo配置

```json
{
  "training": {
    "loss_weights": {
      "voco": 1.0,                      // VoCo总损失权重
      "diversity": 0.1,                 // 关键帧多样性损失权重
      "contrastive": 0.5,               // 对比学习损失权重
      "area": 0.3                       // 面积预测损失权重
    }
  },
  "voco": {
    "crop_size": [64, 64, 32],          // 随机crop尺寸
    "overlap_ratio": 0.25,              // 重叠比例
    "temperature": 0.07,                // 对比学习温度参数
    "lambda_area": 1.0                  // 面积损失权重
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

### 3. VoCo自监督训练

- **完整数据读取**: 一次读入256×256×128完整数据
- **智能crop选择**: 随机选择主crop + 4个相交neighbor crops
- **面积比监督**: 计算crop相交面积比作为监督信号
- **对比学习**: 基于相交程度的特征对比学习
- **无需分割标签**: 纯自监督训练，不依赖人工标注

### 4. 训练策略

- **VoCo训练**: 主要训练方式，基于crop对比学习
- **关键帧网络**: 智能选择关键帧进行特征提取
- **多样性约束**: 确保选择的帧具有多样性

## 性能监控

VoCo训练过程中会显示以下指标：

```
Training Epoch 0: 100%|██████████| 250/250 [02:05<00:00, 1.99it/s, loss=1.8245, voco=1.2150, div=0.1895, area=0.4200]
```

- **loss**: 总损失
- **voco**: VoCo对比学习损失
- **div**: 关键帧多样性损失 (diversity loss)
- **area**: 面积预测损失 (area prediction loss)

传统分割训练（对比模式）显示指标：

```
Training Epoch 0: 100%|██████████| 250/250 [02:05<00:00, 1.99it/s, loss=2.2136, div=0.7695, seg=2.1366]
```

- **seg**: 分割损失 (segmentation loss)

## 内存优化

### 对于内存不足的情况：

1. **减少批次大小**:
```json
"batch_size": 2  // 从4减少到2
```

2. **使用较小模型**:
```json
"embed_dim": 384,    // 从768减少到384
"num_heads": 6,      // 从12减少到6
"num_layers": 6      // 从12减少到6
```

3. **关闭缓存**:
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

## 技术支持

- 模型架构: 基于ViT + Multi-Level Modal Fusion
- 数据处理: MONAI框架
- 训练优化: Mixed Precision + Gradient Scaling
- 内存管理: CacheDataset + 智能采样

## 致谢

本项目整合了多种先进的医学影像处理技术，特别感谢MONAI社区和相关研究工作的贡献。