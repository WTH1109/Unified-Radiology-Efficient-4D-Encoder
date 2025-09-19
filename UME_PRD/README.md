# UME (统一影像学4D编码器) 产品需求文档包

## 📋 文档概览

本文档包包含了统一影像学4D编码器(UME)项目的完整产品需求文档，包括技术架构、实现方案和设计规范。

## 📂 文档结构

### 核心文档

1. **`UME_Enhanced_PRD.md`** - 🎯 **主要文档**
   - 增强版产品需求文档
   - 整合了MedTransformer模态融合策略
   - 包含长视频推理模型启发的抽帧策略
   - 完整的技术架构和实现计划

2. **`UME_Original_PRD.md`** - 📝 原始版本
   - 基础产品需求文档
   - 初始设计思路和功能需求
   - 保留作为设计演进记录

3. **`UME_Architecture_Design.md`** - 🏗️ 架构设计
   - 详细的系统架构设计
   - 核心模块实现规范
   - 数学公式和算法细节

4. **`UME_DataLoader_Design.md`** - 📊 数据处理
   - 数据加载器设计方案
   - BraTS数据集适配策略
   - 配置文件和使用示例

## 🔍 关键技术特性

### 智能抽帧策略
- **多策略融合**: 均匀采样 + 内容感知 + 注意力引导
- **自适应选择**: 根据序列复杂度动态调整
- **投票机制**: 加权投票融合不同策略结果

### 多层次模态融合 (参考MedTransformer)
- **Intra-Modal融合**: 同模态内不同切片融合
- **Inter-Modal融合**: 跨模态信息交互
- **Global融合**: 全局特征增强

### 长视频启发的时序建模
- **多尺度注意力**: 短期、中期、长期时序关系
- **位置编码**: 时序位置信息编码
- **关系融合**: 多尺度时序特征融合

## 📈 核心创新点

### 1. 智能抽帧策略设计
```python
# 多策略融合示例
strategies = {
    'uniform': uniform_sampling(D, K),
    'content_aware': content_aware_sampling(features, K),
    'attention_guided': attention_guided_sampling(scores, K)
}
final_indices = fuse_strategies(strategies, target_frames=K)
```

### 2. MedTransformer启发的融合机制
```python
# Intra-dimension融合
S^(t·s)_l = IntraCAE_t(S^(t·s)_(l-1))
# 融合注意力
QK^T = [Q^class_1 K^class_1        (Q_1+Q_2)K^class_1    ]
       [Q^class_1(K_1+K_2)    (Q_1+Q_2)(K_1+K_2)]
```

### 3. 增强的损失函数
```python
L_total = λ1×L_diversity + λ2×L_segmentation +
          λ3×L_consistency + λ4×L_temporal
```

## 🎯 技术目标

| 指标类别 | 目标值 | 基线对比 |
|---------|-------|---------|
| 分割精度 (Dice) | > 0.87 | +3% |
| 推理速度 | < 800ms | +20% |
| 内存效率 | batch_size ≥ 4 | +100% |
| 策略一致性 | > 0.8 | 新指标 |

## 🛠️ 实现路径

### Phase 1: 基础架构 (Week 1-3)
- [x] 环境搭建和PRD文档
- [ ] 智能抽帧模块实现
- [ ] 基础ViT编码器集成

### Phase 2: 核心功能 (Week 4-6)
- [ ] 多层次模态融合实现
- [ ] 时序关系建模集成
- [ ] 增强损失函数设计

### Phase 3: 优化验证 (Week 7-9)
- [ ] 性能优化和调优
- [ ] 消融实验设计执行
- [ ] 基线方法对比

### Phase 4: 扩展部署 (Week 10-12)
- [ ] 插件化架构实现
- [ ] 多数据集验证
- [ ] 最终性能评估

## 📊 数据支持

### 主要数据集
- **BraTS21**: 脑肿瘤分割数据集
- **多模态**: T1, T2, T1ce, FLAIR
- **格式**: NIfTI医学影像格式

### 预处理流程
```
原始数据 → 加载 → 归一化 → 重采样 → 维度标准化 → 数据增强
```

## 🔧 开发环境

### 新环境: UME
- **框架**: PyTorch + MONAI
- **依赖**: timm, transformers, nibabel
- **硬件**: CUDA支持的GPU

## 📖 使用指南

1. **阅读顺序建议**:
   - 先阅读 `UME_Enhanced_PRD.md` 了解整体架构
   - 参考 `UME_Architecture_Design.md` 了解实现细节
   - 查看 `UME_DataLoader_Design.md` 了解数据处理

2. **开发参考**:
   - 核心算法实现参考架构设计文档
   - 数据加载使用DataLoader设计方案
   - 配置文件使用提供的模板

## 📞 技术支持

如需技术支持或有疑问，请参考各文档中的详细设计说明和代码示例。

---

**文档版本**: v2.0 Enhanced
**最后更新**: 2025-09-18
**创建者**: Claude Code AI Assistant