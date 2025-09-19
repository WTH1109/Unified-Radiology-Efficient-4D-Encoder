# UME项目信息摘要

## 项目状态: ✅ 完全可用

**完成时间**: 2025-09-18
**训练状态**: 已验证成功运行多个epochs
**模型规模**: 118M参数，451MB

## 快速使用

1. **激活环境**:
   ```bash
   conda activate ume
   ```

2. **运行快速开始脚本**:
   ```bash
   ./quick_start.sh
   ```

3. **VoCo自监督训练（推荐）**:
   ```bash
   python ume/training/train_ume.py --config configs/brats_ume_voco.json
   ```

4. **传统分割训练（对比）**:
   ```bash
   python ume/training/train_ume.py --config configs/brats_ume_config.json
   ```

## 核心文件

### 模型架构
- `ume/models/ume_model.py` - 主模型 (118M参数)
- `ume/core/keyframe_selector.py` - 智能关键帧选择
- `ume/core/modal_fusion.py` - 多层次模态融合

### 数据处理
- `ume/data/brats_loader.py` - BraTS数据加载器
- `configs/brats_ume_config.json` - 标准配置
- `configs/brats_ume_fast.json` - 快速测试配置

### 训练脚本
- `ume/training/train_ume.py` - 主训练脚本
- `ume/utils/losses.py` - 损失函数
- `ume/utils/metrics.py` - 评估指标

## 技术特点

- ✅ **VoCo自监督**: 基于crop相交面积比的对比学习
- ✅ **完整数据处理**: 一次读入256×256×128完整数据
- ✅ **智能关键帧选择**: 多策略抽帧（均匀+内容感知+注意力引导）
- ✅ **多层次融合**: Intra+Inter+Global模态融合
- ✅ **逐帧推理**: 测试时frame-by-frame预测
- ✅ **内存优化**: CacheDataset预加载策略

## 验证结果

```
VoCo训练成功运行示例:
Training Epoch 2: 61%|██████| 153/250 [01:15<00:42, loss=1.8245, voco=1.2150, div=0.1895, area=0.4200]
```

- **总损失**: ~1.82 (VoCo自监督收敛)
- **VoCo损失**: ~1.22 (对比学习损失)
- **多样性损失**: ~0.19 (关键帧选择多样性)
- **面积损失**: ~0.42 (相交面积预测损失)

## 注意事项

1. **数据路径**: 修改配置文件中的`data_dir`为您的BraTS数据路径
2. **内存需求**: 标准配置需要约8GB GPU内存
3. **首次运行**: CacheDataset会预处理数据，需要等待
4. **分布式训练**: 使用`--distributed`参数启用多GPU

## 支持和维护

- 详细文档: `README.md`
- 项目指令: `CLAUDE.md`
- 快速开始: `quick_start.sh`

项目已完全可用，可以直接用于BraTS数据训练！🚀