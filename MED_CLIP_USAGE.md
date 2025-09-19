# Med-CLIP集成使用说明

## 功能说明
Med-CLIP（BiomedCLIP）已成功集成到多样性损失计算中，用于更准确地评估关键帧之间的语义相似度。

## 实现特点

### 1. 多模态独立编码
- BraTS的4个MRI模态（T1, T2, T1ce, FLAIR）分别通过Med-CLIP编码
- 每个模态作为灰度图扩展为3通道RGB输入
- 4个模态的特征通过平均池化合并

### 2. 正确的处理流程
```python
# 对每个选中的帧
for k in range(K):
    frame = selected_frames[:, :, :, :, k]  # [B, 4, H, W]

    # 对4个模态分别编码
    for modality_idx in range(4):  # T1, T2, T1ce, FLAIR
        modality = frame[:, modality_idx:modality_idx+1, :, :]  # [B, 1, H, W]
        modality_rgb = modality.repeat(1, 3, 1, 1)  # 转为RGB格式
        modality_resized = F.interpolate(modality_rgb, size=(224, 224))  # 调整尺寸

        # Med-CLIP编码
        features = medclip_encoder(modality_resized)

    # 合并4个模态的特征
    combined_features = mean(modality_features)
```

## 使用方法

### 启用Med-CLIP（默认禁用）

1. **在代码中启用**：
```python
# 修改 ume/core/keyframe_selector.py
self.use_medclip = True  # 第89行
```

2. **或在初始化时设置**：
```python
selector = EnhancedKeyFrameSelector(
    input_channels=4,
    embed_dim=768,
    max_frames=10
)
selector.use_medclip = True
```

### 性能考虑
- Med-CLIP编码会显著增加计算时间（约4倍）
- 建议在需要高质量多样性评估时使用
- 常规训练可使用默认的池化特征方法

## 测试验证

运行测试脚本验证Med-CLIP集成：
```bash
export CUDA_VISIBLE_DEVICES=2
python test_medclip.py
```

预期输出：
```
✓ Med-CLIP集成测试通过!
- BraTS的4个模态分别通过Med-CLIP编码
- 每个模态作为灰度图扩展为3通道RGB
- 4个模态的特征通过平均池化合并
```

## 注意事项
1. 需要安装`open_clip_torch`库
2. 首次运行会自动下载BiomedCLIP模型（约400MB）
3. 需要足够的GPU内存（建议8GB以上）