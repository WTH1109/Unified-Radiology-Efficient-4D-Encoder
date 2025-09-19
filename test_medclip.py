#!/usr/bin/env python
"""测试Med-CLIP集成是否正常工作"""

import torch
import time
from ume.core.keyframe_selector import EnhancedKeyFrameSelector

# 设置GPU (使用CUDA_VISIBLE_DEVICES指定的设备)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("测试Med-CLIP集成...")

# 创建keyframe selector
print("1. 创建EnhancedKeyFrameSelector...")
selector = EnhancedKeyFrameSelector(input_channels=4, embed_dim=768, max_frames=10)
selector = selector.to(device)
selector.eval()

# 创建测试数据 (小批量)
print("2. 创建测试数据...")
x = torch.randn(1, 4, 256, 256, 32).to(device)  # 减少深度维度以加快测试
print(f"   输入形状: {x.shape}")

# 测试前向传播
print("3. 测试前向传播...")
start_time = time.time()
with torch.no_grad():
    selected_frames, indices, losses = selector(x)

end_time = time.time()

print(f"   选中帧形状: {selected_frames.shape}")
print(f"   选中索引: {indices}")
print(f"   多样性损失: {losses['diversity_loss'].item():.4f}")
print(f"   前向传播时间: {end_time - start_time:.2f}秒")

print("\n✓ Med-CLIP集成测试通过!")
print("\n注意：")
print("- BraTS的4个模态分别通过Med-CLIP编码")
print("- 每个模态作为灰度图扩展为3通道RGB")
print("- 4个模态的特征通过平均池化合并")