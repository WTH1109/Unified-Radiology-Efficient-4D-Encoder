# UME架构设计与技术规范

## 1. 系统架构概览

### 1.1 整体架构图
```
输入数据 → 维度标准化模块 → 关键帧选择模块 → 帧位置编码模块 → ViT2D编码模块 → 模态融合模块 → 特征输出
   ↓              ↓                ↓              ↓              ↓              ↓            ↓
  多维度         统一4D格式        压缩4D        位置增强4D      Token嵌入      融合特征      最终特征
(2D/3D/4D)    (C×H×W×D)       (C×H×W×K)    (C×H×W×K)     (CK×P×Emb)   (融合+压缩)   (输出维度)
                                   ↓              ↓                              ↓
                                帧索引信息      位置编码信息                   分割验证网络
                               [d1,d2...dK]   [pos_enc_K×E]
```

### 1.2 数据流详细描述

#### 阶段1：维度标准化
- **输入**：多种维度格式
- **处理**：统一到4D格式
- **输出**：C×H×W×D标准格式

#### 阶段2：关键帧选择
- **输入**：C×H×W×D
- **处理**：注意力权重计算 + Top-K选择
- **输出**：C×H×W×K (K≤10) + 帧索引信息 [d1,d2,...,dK]

#### 阶段3：帧位置编码 (新增)
- **输入**：C×H×W×K + 帧索引 [d1,d2,...,dK] + 选择权重
- **处理**：多尺度位置编码 + 自适应权重融合
- **输出**：位置编码增强的4D特征 + 位置编码矩阵 [K×E]

#### 阶段4：ViT编码
- **输入**：位置增强的C×H×W×K + 位置编码信息
- **处理**：Patch分割 + 层次化位置融合 + Transformer编码
- **输出**：CK×P×Emb (包含位置信息)

#### 阶段5：模态融合
- **输入**：CK×P×Emb
- **处理**：跨模态注意力融合
- **输出**：全局特征 + 压缩特征

#### 阶段6：特征输出
- **输入**：融合特征
- **处理**：特征拼接与重塑
- **输出**：PKD(1+C//α)×Emb

## 2. 核心模块详细设计

### 2.1 维度标准化模块 (DimensionNormalizer)

#### 设计目标
将不同维度的医学影像数据统一为4D格式，便于后续处理。

#### 实现逻辑
```python
class DimensionNormalizer(nn.Module):
    def forward(self, x):
        if x.dim() == 2:  # H×W
            return x.unsqueeze(0).unsqueeze(-1)  # 1×H×W×1
        elif x.dim() == 3:  # H×W×D
            return x.unsqueeze(0)  # 1×H×W×D
        elif x.dim() == 4:  # C×H×W×D
            return x  # 保持原格式
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
```

### 2.2 增强关键帧选择模块 (Enhanced KeyFrameSelector)

#### 设计目标
设计一个复杂的多策略融合关键帧选择器，能够：
- 自适应确定最优帧数K
- 多策略融合选择（内容感知、时序感知、重要性感知）
- 上下文感知的重要性评估
- 多尺度特征提取和分析
- 动态权重调整和策略优化

#### 整体架构

```
输入X[C×H×W×D] → 多尺度特征提取 → 多策略评估 → 动态K选择 → 策略融合 → 最终选择
                     ↓              ↓           ↓         ↓         ↓
                   特征金字塔      [内容/时序/     自适应K     加权投票    关键帧输出
                   [多个尺度]       重要性评估]   [4-16]      融合       [C×H×W×K]
```

#### 核心算法设计

##### 1. 多尺度特征提取器
```python
class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取，捕获不同层次的信息"""

    def __init__(self, input_channels, embed_dim=256):
        super().__init__()

        # 多尺度卷积特征提取器
        self.scale_extractors = nn.ModuleList([
            # 局部特征 (小感受野)
            nn.Sequential(
                nn.Conv3d(input_channels, embed_dim//4, kernel_size=(1,3,3), padding=(0,1,1)),
                nn.ReLU(),
                nn.Conv3d(embed_dim//4, embed_dim//4, kernel_size=(3,1,1), padding=(1,0,0)),
                nn.ReLU()
            ),
            # 中等特征 (中感受野)
            nn.Sequential(
                nn.Conv3d(input_channels, embed_dim//4, kernel_size=(1,5,5), padding=(0,2,2)),
                nn.ReLU(),
                nn.Conv3d(embed_dim//4, embed_dim//4, kernel_size=(5,1,1), padding=(2,0,0)),
                nn.ReLU()
            ),
            # 全局特征 (大感受野)
            nn.Sequential(
                nn.Conv3d(input_channels, embed_dim//4, kernel_size=(1,7,7), padding=(0,3,3)),
                nn.ReLU(),
                nn.Conv3d(embed_dim//4, embed_dim//4, kernel_size=(7,1,1), padding=(3,0,0)),
                nn.ReLU()
            ),
            # 时序全局特征
            nn.Sequential(
                nn.AdaptiveAvgPool3d((None, 1, 1)),  # 全局空间池化，保持时序
                nn.Conv3d(input_channels, embed_dim//4, kernel_size=(3,1,1), padding=(1,0,0)),
                nn.ReLU()
            )
        ])

        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Dropout3d(0.1),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        )

        # 深度维度特征聚合
        self.depth_aggregator = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # [B, embed_dim, D, 1, 1]
            nn.Flatten(start_dim=3),             # [B, embed_dim, D]
            nn.Transpose(-1, -2)                 # [B, D, embed_dim]
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W, D]
        Returns:
            multi_scale_features: [B, D, embed_dim] - 每帧的多尺度融合特征
            scale_specific_features: List[[B, D, embed_dim//4]] - 各尺度特征
        """
        # 提取多尺度特征
        scale_features = []
        for extractor in self.scale_extractors:
            scale_feat = extractor(x)  # [B, embed_dim//4, H, W, D]
            scale_feat = self.depth_aggregator(scale_feat)  # [B, D, embed_dim//4]
            scale_features.append(scale_feat)

        # 融合多尺度特征
        concatenated = torch.cat(scale_features, dim=-1)  # [B, D, embed_dim]

        # 还原到3D进行卷积融合
        B, D, embed_dim = concatenated.shape
        reshaped = concatenated.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # [B, embed_dim, D, 1, 1]
        fused_features = self.feature_fusion(reshaped)  # [B, embed_dim, D, 1, 1]
        fused_features = fused_features.squeeze(-1).squeeze(-1).transpose(1, 2)  # [B, D, embed_dim]

        return fused_features, scale_features
```

##### 2. 多策略重要性评估器
```python
class MultiStrategyImportanceEvaluator(nn.Module):
    """多策略重要性评估：内容、时序、任务相关性"""

    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim

        # 策略1: 内容变化感知网络
        self.content_evaluator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, 1),
            nn.Sigmoid()
        )

        # 策略2: 时序连续性评估网络
        self.temporal_evaluator = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim//2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.temporal_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, 1),
            nn.Sigmoid()
        )

        # 策略3: 任务相关性评估网络（可学习任务嵌入）
        self.task_embedding = nn.Parameter(torch.randn(1, embed_dim))
        self.task_evaluator = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        self.task_scorer = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # 策略4: 空间复杂度评估
        self.spatial_evaluator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """
        Args:
            features: [B, D, embed_dim] - 多尺度融合特征
        Returns:
            strategy_scores: Dict - 各策略的重要性分数
            combined_scores: [B, D] - 融合后的综合分数
        """
        B, D, embed_dim = features.shape

        # 策略1: 内容变化重要性
        content_scores = self.content_evaluator(features).squeeze(-1)  # [B, D]

        # 计算内容变化程度
        if D > 1:
            content_diff = torch.norm(features[:, 1:] - features[:, :-1], p=2, dim=-1)  # [B, D-1]
            content_diff = F.pad(content_diff, (1, 0), value=0)  # [B, D] 第一帧设为0
            content_scores = content_scores * (1 + content_diff)  # 变化大的帧权重更高

        # 策略2: 时序连续性重要性
        temporal_features, _ = self.temporal_evaluator(features)  # [B, D, embed_dim]
        temporal_scores = self.temporal_scorer(temporal_features).squeeze(-1)  # [B, D]

        # 策略3: 任务相关性重要性
        task_embed = self.task_embedding.expand(B, 1, embed_dim)  # [B, 1, embed_dim]
        task_attended, _ = self.task_evaluator(task_embed, features, features)  # [B, 1, embed_dim]

        # 计算每帧与任务的相关性
        task_relevance = torch.sum(features * task_attended, dim=-1)  # [B, D]
        task_scores = torch.sigmoid(task_relevance)

        # 策略4: 空间复杂度重要性
        spatial_scores = self.spatial_evaluator(features).squeeze(-1)  # [B, D]

        strategy_scores = {
            'content': content_scores,
            'temporal': temporal_scores,
            'task': task_scores,
            'spatial': spatial_scores
        }

        return strategy_scores

class AdaptiveStrategyFusion(nn.Module):
    """自适应策略融合网络"""

    def __init__(self, num_strategies=4):
        super().__init__()
        self.num_strategies = num_strategies

        # 策略权重学习网络
        self.strategy_weight_net = nn.Sequential(
            nn.Linear(num_strategies, num_strategies * 2),
            nn.ReLU(),
            nn.Linear(num_strategies * 2, num_strategies),
            nn.Softmax(dim=-1)
        )

        # 上下文感知权重调整
        self.context_net = nn.Sequential(
            nn.Linear(1, num_strategies),  # 基于序列长度的上下文
            nn.ReLU(),
            nn.Linear(num_strategies, num_strategies),
            nn.Sigmoid()
        )

    def forward(self, strategy_scores, sequence_length):
        """
        Args:
            strategy_scores: Dict[str, Tensor[B, D]] - 各策略分数
            sequence_length: int - 序列长度
        Returns:
            fused_scores: [B, D] - 融合后的分数
            strategy_weights: [B, num_strategies] - 学习到的策略权重
        """
        B, D = next(iter(strategy_scores.values())).shape

        # 堆叠所有策略分数
        stacked_scores = torch.stack(list(strategy_scores.values()), dim=-1)  # [B, D, num_strategies]

        # 计算全局策略统计
        strategy_stats = torch.mean(stacked_scores, dim=1)  # [B, num_strategies]

        # 学习策略权重
        strategy_weights = self.strategy_weight_net(strategy_stats)  # [B, num_strategies]

        # 上下文调整
        context_input = torch.tensor([sequence_length], dtype=torch.float32, device=stacked_scores.device)
        context_input = context_input.unsqueeze(0).expand(B, 1)  # [B, 1]
        context_weights = self.context_net(context_input)  # [B, num_strategies]

        # 最终权重
        final_weights = strategy_weights * context_weights  # [B, num_strategies]

        # 加权融合
        fused_scores = torch.sum(stacked_scores * final_weights.unsqueeze(1), dim=-1)  # [B, D]

        return fused_scores, final_weights
```

##### 3. 动态K值选择器
```python
class DynamicKSelector(nn.Module):
    """动态确定最优关键帧数量"""

    def __init__(self, embed_dim=256, min_k=4, max_k=16):
        super().__init__()
        self.min_k = min_k
        self.max_k = max_k

        # K值预测网络
        self.k_predictor = nn.Sequential(
            nn.Linear(embed_dim + 3, embed_dim//2),  # +3 for [seq_len, content_var, temporal_smooth]
            nn.ReLU(),
            nn.Linear(embed_dim//2, embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, 1),
            nn.Sigmoid()
        )

        # 复杂度评估器
        self.complexity_evaluator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//4),
            nn.ReLU(),
            nn.Linear(embed_dim//4, 1)
        )

    def forward(self, features, importance_scores):
        """
        Args:
            features: [B, D, embed_dim]
            importance_scores: [B, D]
        Returns:
            optimal_k: [B] - 每个样本的最优K值
            complexity_score: [B] - 序列复杂度分数
        """
        B, D, embed_dim = features.shape

        # 计算序列统计特征
        seq_len_norm = torch.tensor([D / 100.0], device=features.device).expand(B, 1)

        # 内容变化方差
        if D > 1:
            content_var = torch.var(importance_scores, dim=1, keepdim=True)  # [B, 1]
        else:
            content_var = torch.zeros(B, 1, device=features.device)

        # 时序平滑度
        if D > 2:
            temporal_diff = torch.abs(importance_scores[:, 2:] - 2*importance_scores[:, 1:-1] + importance_scores[:, :-2])
            temporal_smooth = torch.mean(temporal_diff, dim=1, keepdim=True)  # [B, 1]
        else:
            temporal_smooth = torch.zeros(B, 1, device=features.device)

        # 全局特征
        global_features = torch.mean(features, dim=1)  # [B, embed_dim]

        # 拼接所有特征
        k_input = torch.cat([global_features, seq_len_norm, content_var, temporal_smooth], dim=1)

        # 预测归一化的K值
        k_normalized = self.k_predictor(k_input).squeeze(-1)  # [B]

        # 转换为实际K值
        optimal_k = self.min_k + (self.max_k - self.min_k) * k_normalized
        optimal_k = torch.round(optimal_k).long()

        # 确保K值在合理范围内
        optimal_k = torch.clamp(optimal_k, self.min_k, min(D, self.max_k))

        # 计算复杂度分数
        complexity_score = self.complexity_evaluator(global_features).squeeze(-1)

        return optimal_k, complexity_score
```

##### 4. 智能选择与优化
```python
class IntelligentFrameSelector(nn.Module):
    """智能帧选择与优化"""

    def __init__(self):
        super().__init__()

    def diverse_top_k_selection(self, scores, features, k, diversity_weight=0.3):
        """多样性感知的Top-K选择"""
        B, D = scores.shape
        selected_indices = []

        for b in range(B):
            batch_scores = scores[b]  # [D]
            batch_features = features[b]  # [D, embed_dim]
            current_k = k[b].item()

            indices = []
            remaining_indices = list(range(D))

            # 选择第一个帧（最高分数）
            first_idx = torch.argmax(batch_scores).item()
            indices.append(first_idx)
            remaining_indices.remove(first_idx)

            # 迭代选择剩余帧
            for _ in range(current_k - 1):
                if not remaining_indices:
                    break

                best_idx = None
                best_score = float('-inf')

                for idx in remaining_indices:
                    # 原始重要性分数
                    importance = batch_scores[idx].item()

                    # 计算与已选择帧的最小距离（多样性）
                    selected_features = batch_features[indices]  # [len(indices), embed_dim]
                    current_feature = batch_features[idx:idx+1]   # [1, embed_dim]

                    similarities = F.cosine_similarity(current_feature, selected_features, dim=1)
                    max_similarity = torch.max(similarities).item()
                    diversity = 1 - max_similarity

                    # 综合分数
                    combined_score = importance + diversity_weight * diversity

                    if combined_score > best_score:
                        best_score = combined_score
                        best_idx = idx

                if best_idx is not None:
                    indices.append(best_idx)
                    remaining_indices.remove(best_idx)

            # 填充到统一长度（如果需要）
            while len(indices) < current_k:
                if remaining_indices:
                    indices.append(remaining_indices.pop(0))
                else:
                    indices.append(indices[-1])  # 重复最后一个

            selected_indices.append(torch.tensor(indices, device=scores.device))

        return selected_indices

    def temporal_constraint_selection(self, selected_indices, min_interval=2):
        """时序约束选择，确保选择的帧之间有最小间隔"""
        constrained_indices = []

        for indices in selected_indices:
            sorted_indices = torch.sort(indices)[0]
            filtered = [sorted_indices[0].item()]

            for i in range(1, len(sorted_indices)):
                if sorted_indices[i].item() - filtered[-1] >= min_interval:
                    filtered.append(sorted_indices[i].item())

            # 如果过滤后数量不够，均匀填充
            while len(filtered) < len(indices):
                # 找到最大间隔并插入
                max_gap = 0
                insert_pos = 0
                for i in range(len(filtered) - 1):
                    gap = filtered[i+1] - filtered[i]
                    if gap > max_gap:
                        max_gap = gap
                        insert_pos = i

                if max_gap > min_interval:
                    new_idx = (filtered[insert_pos] + filtered[insert_pos + 1]) // 2
                    filtered.insert(insert_pos + 1, new_idx)
                else:
                    break

            constrained_indices.append(torch.tensor(filtered[:len(indices)], device=indices.device))

        return constrained_indices
```

##### 5. 完整的增强关键帧选择器
```python
class EnhancedKeyFrameSelector(nn.Module):
    """增强的关键帧选择器 - 集成所有组件"""

    def __init__(self, input_channels, embed_dim=256, min_k=4, max_k=16):
        super().__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.min_k = min_k
        self.max_k = max_k

        # 核心组件
        self.feature_extractor = MultiScaleFeatureExtractor(input_channels, embed_dim)
        self.importance_evaluator = MultiStrategyImportanceEvaluator(embed_dim)
        self.strategy_fusion = AdaptiveStrategyFusion(num_strategies=4)
        self.k_selector = DynamicKSelector(embed_dim, min_k, max_k)
        self.frame_selector = IntelligentFrameSelector()

        # 学习参数
        self.diversity_weight = nn.Parameter(torch.tensor(0.3))
        self.temporal_constraint = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W, D] - 输入4D数据
        Returns:
            selected_frames: [B, C, H, W, K] - 选择的关键帧
            frame_indices: List[[K]] - 每个样本选择的帧索引
            selection_weights: [B, K] - 选择权重
            selection_metadata: Dict - 选择过程的元数据
        """
        B, C, H, W, D = x.shape

        # 1. 多尺度特征提取
        multi_scale_features, scale_features = self.feature_extractor(x)  # [B, D, embed_dim]

        # 2. 多策略重要性评估
        strategy_scores = self.importance_evaluator(multi_scale_features)

        # 3. 自适应策略融合
        fused_scores, strategy_weights = self.strategy_fusion(strategy_scores, D)

        # 4. 动态K值选择
        optimal_k, complexity_scores = self.k_selector(multi_scale_features, fused_scores)

        # 5. 智能帧选择
        selected_indices = self.frame_selector.diverse_top_k_selection(
            fused_scores, multi_scale_features, optimal_k, self.diversity_weight
        )

        # 6. 时序约束优化
        constrained_indices = self.frame_selector.temporal_constraint_selection(
            selected_indices, min_interval=max(1, int(self.temporal_constraint))
        )

        # 7. 构建输出
        selected_frames = []
        selection_weights = []
        max_k = max([len(indices) for indices in constrained_indices])

        for b, indices in enumerate(constrained_indices):
            # 选择帧
            batch_frames = x[b, :, :, :, indices]  # [C, H, W, K_b]

            # 获取权重
            batch_weights = fused_scores[b, indices]  # [K_b]

            # 填充到统一大小
            if len(indices) < max_k:
                # 重复最后一帧
                last_frame = batch_frames[:, :, :, -1:].expand(-1, -1, -1, max_k - len(indices))
                batch_frames = torch.cat([batch_frames, last_frame], dim=-1)

                # 填充权重
                last_weight = batch_weights[-1:].expand(max_k - len(indices))
                batch_weights = torch.cat([batch_weights, last_weight], dim=0)

            selected_frames.append(batch_frames)
            selection_weights.append(batch_weights)

        selected_frames = torch.stack(selected_frames, dim=0)  # [B, C, H, W, max_K]
        selection_weights = torch.stack(selection_weights, dim=0)  # [B, max_K]

        # 元数据
        selection_metadata = {
            'strategy_scores': strategy_scores,
            'strategy_weights': strategy_weights,
            'optimal_k': optimal_k,
            'complexity_scores': complexity_scores,
            'diversity_weight': self.diversity_weight,
            'scale_features': scale_features
        }

        return selected_frames, constrained_indices, selection_weights, selection_metadata

    def compute_selection_loss(self, selection_metadata, selected_frames):
        """计算选择相关的损失"""
        losses = {}

        # 1. 多样性损失
        diversity_loss = self.compute_diversity_loss(selected_frames)
        losses['diversity_loss'] = diversity_loss

        # 2. 策略平衡损失
        strategy_weights = selection_metadata['strategy_weights']  # [B, num_strategies]
        strategy_balance_loss = torch.var(strategy_weights.mean(dim=0))  # 策略权重方差
        losses['strategy_balance_loss'] = strategy_balance_loss

        # 3. 复杂度一致性损失
        complexity_scores = selection_metadata['complexity_scores']
        optimal_k = selection_metadata['optimal_k'].float()
        complexity_consistency = F.mse_loss(
            complexity_scores,
            (optimal_k - self.min_k) / (self.max_k - self.min_k)
        )
        losses['complexity_consistency_loss'] = complexity_consistency

        return losses

    def compute_diversity_loss(self, selected_frames):
        """计算多样性损失"""
        B, C, H, W, K = selected_frames.shape

        # 展平空间维度
        frames_flat = selected_frames.view(B, C*H*W, K)  # [B, CHW, K]

        # 计算帧间相似度
        similarities = []
        for k1 in range(K):
            for k2 in range(k1+1, K):
                sim = F.cosine_similarity(frames_flat[:, :, k1], frames_flat[:, :, k2], dim=1)
                similarities.append(sim)

        if similarities:
            avg_similarity = torch.stack(similarities, dim=1).mean()
            return avg_similarity  # 最小化相似度
        else:
            return torch.tensor(0.0, device=selected_frames.device)
```

### 2.3 帧位置编码模块 (FramePositionEncoder)

#### 设计目标
在关键帧选择后，将选中帧的原始位置信息（d/D）编码到特征表示中，保留重要的时空序列信息，增强模型对帧位置的感知能力。

#### 多维度位置信息
```python
position_info = {
    'absolute_pos': [d1, d2, ..., dK],           # 绝对帧索引
    'relative_pos': [d1/D, d2/D, ..., dK/D],     # 相对位置 [0,1]
    'selection_weight': [w1, w2, ..., wK],       # 选择权重
    'sequence_length': D                          # 原始序列长度
}
```

#### 多尺度帧位置编码公式
```python
# 1. 绝对位置编码
abs_pos_enc = sinusoidal_encoding(frame_indices, embed_dim // 4)

# 2. 相对位置编码
relative_positions = frame_indices / sequence_length  # [0, 1]
rel_pos_enc = sinusoidal_encoding(relative_positions * 1000, embed_dim // 4)

# 3. 序列感知编码
seq_aware_enc = learned_encoding(sequence_length, embed_dim // 4)

# 4. 帧间距离编码
frame_distances = calculate_frame_distances(frame_indices)
distance_enc = distance_encoding(frame_distances, embed_dim // 4)

# 融合所有位置信息
frame_pos_encoding = concat([abs_pos_enc, rel_pos_enc, seq_aware_enc, distance_enc])
```

#### 自适应权重融合
```python
class AdaptiveFramePositionEncoding(nn.Module):
    def __init__(self, embed_dim, max_sequence_length=512):
        super().__init__()
        # 可学习的位置嵌入表
        self.absolute_pos_embedding = nn.Parameter(
            torch.randn(max_sequence_length, embed_dim // 4)
        )

        # 相对位置编码网络
        self.relative_pos_net = nn.Sequential(
            nn.Linear(1, embed_dim // 8),
            nn.ReLU(),
            nn.Linear(embed_dim // 8, embed_dim // 4)
        )

        # 权重融合网络
        self.weight_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, frame_indices, selection_weights, sequence_length):
        # 多尺度位置编码计算
        all_encodings = self.compute_multi_scale_encoding(
            frame_indices, selection_weights, sequence_length
        )

        # 自适应权重融合
        weighted_encoding = self.weight_fusion(all_encodings)

        # 基于选择权重的最终调制
        selection_weights_expanded = selection_weights.unsqueeze(-1).expand(-1, self.embed_dim)
        final_encoding = weighted_encoding * selection_weights_expanded

        return final_encoding  # [K, embed_dim]
```

#### 层次化位置融合
```python
class HierarchicalPositionFusion(nn.Module):
    """将帧位置编码与ViT的2D空间位置编码进行层次化融合"""

    def __init__(self, embed_dim, patch_size, image_size):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        # 2D空间位置编码 (ViT标准)
        self.spatial_pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )

        # 跨维度融合网络
        self.cross_dimension_fusion = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )

        # 位置编码融合权重
        self.fusion_weights = nn.Parameter(torch.tensor([0.6, 0.4]))  # [spatial, temporal]

    def forward(self, patch_embeddings, frame_pos_encodings, frame_indices):
        """
        融合空间位置编码和时序位置编码

        Args:
            patch_embeddings: [B, CK, P, embed_dim] - ViT patch embeddings
            frame_pos_encodings: [K, embed_dim] - 帧位置编码
            frame_indices: [K] - 帧索引
        """
        # 空间 + 时序位置编码的层次化融合
        enhanced_embeddings = self.hierarchical_fusion(
            patch_embeddings, frame_pos_encodings
        )

        return enhanced_embeddings
```

#### 增强的位置编码质量评估（集成CLIP）
```python
import clip
from PIL import Image
import torchvision.transforms as transforms

class CLIPBasedPositionEvaluator:
    """基于CLIP的位置编码质量评估器"""

    def __init__(self, clip_model_name="ViT-B/32", device="cuda"):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device)
        self.clip_model.eval()

        # 医学图像预处理
        self.medical_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # 转换为3通道
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_clip_features(self, frames):
        """
        提取帧的CLIP特征

        Args:
            frames: [K, H, W] or [K, C, H, W] - 关键帧
        Returns:
            clip_features: [K, clip_dim] - CLIP特征
        """
        if frames.dim() == 3:
            frames = frames.unsqueeze(1)  # [K, 1, H, W]
        elif frames.dim() == 4 and frames.shape[1] > 1:
            # 多通道转单通道（取平均）
            frames = frames.mean(dim=1, keepdim=True)  # [K, 1, H, W]

        K, C, H, W = frames.shape
        clip_features = []

        for k in range(K):
            # 预处理单帧
            frame = frames[k, 0]  # [H, W]

            # 归一化到[0, 1]
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)

            # 转换为PIL图像并预处理
            frame_pil = transforms.ToPILImage()(frame)
            frame_processed = self.clip_preprocess(frame_pil).unsqueeze(0).to(self.device)

            # 提取CLIP特征
            clip_feat = self.clip_model.encode_image(frame_processed)
            clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)  # L2归一化
            clip_features.append(clip_feat)

        return torch.cat(clip_features, dim=0)  # [K, clip_dim]

def evaluate_position_encoding_quality(
    frame_pos_encodings,
    frame_indices,
    original_frames=None,
    use_clip=True,
    clip_weight=0.4
):
    """
    增强的位置编码质量评估（集成CLIP语义距离）

    Args:
        frame_pos_encodings: [K, embed_dim] - 帧位置编码
        frame_indices: [K] - 帧索引
        original_frames: [K, H, W] or [K, C, H, W] - 原始帧（用于CLIP特征提取）
        use_clip: bool - 是否使用CLIP评估
        clip_weight: float - CLIP距离的权重

    Returns:
        metrics: Dict - 评估指标
    """
    metrics = {}
    K = len(frame_indices)

    # 1. 基础几何距离评估
    # 1.1 位置单调性
    position_similarity = []
    encoding_distances_geometric = []

    for i in range(K - 1):
        # 余弦相似度
        similarity = F.cosine_similarity(
            frame_pos_encodings[i:i+1], frame_pos_encodings[i+1:i+2], dim=1
        )
        position_similarity.append(similarity.item())

        # 欧氏距离
        geo_dist = torch.norm(frame_pos_encodings[i+1] - frame_pos_encodings[i], p=2).item()
        encoding_distances_geometric.append(geo_dist)

    metrics['position_monotonicity'] = np.mean(position_similarity)

    # 1.2 距离保持性（几何）
    actual_distances = [frame_indices[i+1] - frame_indices[i] for i in range(K - 1)]

    if encoding_distances_geometric:
        correlation_geo = np.corrcoef(actual_distances, encoding_distances_geometric)[0, 1]
        metrics['geometric_distance_correlation'] = correlation_geo
    else:
        metrics['geometric_distance_correlation'] = 0.0

    # 2. CLIP语义距离评估
    if use_clip and original_frames is not None:
        try:
            clip_evaluator = CLIPBasedPositionEvaluator()
            clip_features = clip_evaluator.extract_clip_features(original_frames)  # [K, clip_dim]

            # 2.1 CLIP语义距离
            clip_distances = []
            clip_similarities = []

            for i in range(K - 1):
                # 语义距离（1 - 余弦相似度）
                clip_sim = F.cosine_similarity(
                    clip_features[i:i+1], clip_features[i+1:i+2], dim=1
                ).item()
                clip_similarities.append(clip_sim)

                # 语义距离
                clip_dist = 1.0 - clip_sim
                clip_distances.append(clip_dist)

            # 2.2 语义距离与位置编码的相关性
            if clip_distances:
                correlation_clip = np.corrcoef(actual_distances, clip_distances)[0, 1]
                metrics['clip_semantic_correlation'] = correlation_clip
                metrics['clip_semantic_distances'] = clip_distances
                metrics['clip_similarities'] = clip_similarities

            # 2.3 位置编码与CLIP特征的对齐度
            # 计算位置编码距离和CLIP语义距离的一致性
            pos_enc_clip_consistency = []
            for i in range(K - 1):
                pos_enc_dist = encoding_distances_geometric[i]
                clip_dist = clip_distances[i]

                # 归一化距离
                pos_enc_dist_norm = pos_enc_dist / (max(encoding_distances_geometric) + 1e-8)
                clip_dist_norm = clip_dist / (max(clip_distances) + 1e-8)

                # 计算一致性（距离越接近一致性越高）
                consistency = 1.0 - abs(pos_enc_dist_norm - clip_dist_norm)
                pos_enc_clip_consistency.append(consistency)

            metrics['position_clip_consistency'] = np.mean(pos_enc_clip_consistency)

            # 2.4 语义渐变性评估
            # 相邻帧的语义相似度应该较高，远距离帧相似度较低
            semantic_gradient_scores = []
            for i in range(K):
                for j in range(i+2, K):  # 跳过相邻帧
                    actual_gap = frame_indices[j] - frame_indices[i]
                    clip_sim = F.cosine_similarity(
                        clip_features[i:i+1], clip_features[j:j+1], dim=1
                    ).item()

                    # 期望：距离越远，相似度越低
                    expected_sim = max(0, 1.0 - actual_gap / max(actual_distances))
                    gradient_score = 1.0 - abs(clip_sim - expected_sim)
                    semantic_gradient_scores.append(gradient_score)

            if semantic_gradient_scores:
                metrics['semantic_gradient_quality'] = np.mean(semantic_gradient_scores)

        except Exception as e:
            print(f"CLIP评估失败: {e}")
            # 如果CLIP评估失败，设置默认值
            metrics['clip_semantic_correlation'] = 0.0
            metrics['position_clip_consistency'] = 0.0
            metrics['semantic_gradient_quality'] = 0.0

    # 3. 综合质量评估
    # 3.1 编码区分度
    pairwise_distances = []
    for i in range(K):
        for j in range(i+1, K):
            dist = torch.norm(frame_pos_encodings[i] - frame_pos_encodings[j], p=2).item()
            pairwise_distances.append(dist)

    metrics['encoding_diversity'] = np.std(pairwise_distances) if pairwise_distances else 0.0

    # 3.2 位置编码平滑度
    if K > 2:
        smoothness_scores = []
        for i in range(1, K-1):
            # 计算二阶差分（平滑度指标）
            prev_diff = frame_pos_encodings[i] - frame_pos_encodings[i-1]
            next_diff = frame_pos_encodings[i+1] - frame_pos_encodings[i]

            second_diff = torch.norm(next_diff - prev_diff, p=2).item()
            smoothness_scores.append(second_diff)

        metrics['position_smoothness'] = np.mean(smoothness_scores)
    else:
        metrics['position_smoothness'] = 0.0

    # 3.3 综合质量分数
    quality_components = []

    # 几何质量
    geo_quality = max(0, metrics['geometric_distance_correlation'])
    quality_components.append(geo_quality * (1 - clip_weight))

    # 语义质量（如果使用CLIP）
    if use_clip and 'clip_semantic_correlation' in metrics:
        semantic_quality = (
            max(0, metrics['clip_semantic_correlation']) * 0.4 +
            metrics['position_clip_consistency'] * 0.3 +
            metrics['semantic_gradient_quality'] * 0.3
        )
        quality_components.append(semantic_quality * clip_weight)

    # 多样性和平滑度
    diversity_quality = min(1.0, metrics['encoding_diversity'] / 2.0)  # 归一化
    smoothness_quality = max(0, 1.0 - metrics['position_smoothness'])  # 平滑度越低越好

    quality_components.extend([diversity_quality * 0.1, smoothness_quality * 0.1])

    metrics['overall_quality_score'] = sum(quality_components)

    # 4. 质量等级评估
    overall_score = metrics['overall_quality_score']
    if overall_score >= 0.8:
        metrics['quality_grade'] = 'Excellent'
    elif overall_score >= 0.6:
        metrics['quality_grade'] = 'Good'
    elif overall_score >= 0.4:
        metrics['quality_grade'] = 'Fair'
    else:
        metrics['quality_grade'] = 'Poor'

    return metrics

# 使用示例和可视化工具
def visualize_position_encoding_quality(metrics, save_path=None):
    """可视化位置编码质量评估结果"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 距离相关性对比
    ax1 = axes[0, 0]
    correlations = []
    labels = []

    if 'geometric_distance_correlation' in metrics:
        correlations.append(metrics['geometric_distance_correlation'])
        labels.append('Geometric')

    if 'clip_semantic_correlation' in metrics:
        correlations.append(metrics['clip_semantic_correlation'])
        labels.append('CLIP Semantic')

    if correlations:
        bars = ax1.bar(labels, correlations, color=['blue', 'orange'])
        ax1.set_ylabel('Correlation')
        ax1.set_title('Distance Correlation Analysis')
        ax1.set_ylim([-1, 1])

        # 添加数值标签
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05 if height > 0 else height - 0.1,
                    f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top')

    # 2. 质量组成分析
    ax2 = axes[0, 1]
    quality_metrics = {
        'Position-CLIP\nConsistency': metrics.get('position_clip_consistency', 0),
        'Semantic\nGradient': metrics.get('semantic_gradient_quality', 0),
        'Encoding\nDiversity': min(1.0, metrics['encoding_diversity'] / 2.0),
        'Position\nSmoothness': max(0, 1.0 - metrics['position_smoothness'])
    }

    bars = ax2.bar(quality_metrics.keys(), quality_metrics.values(),
                   color=['green', 'purple', 'red', 'brown'])
    ax2.set_ylabel('Quality Score')
    ax2.set_title('Quality Components')
    ax2.set_ylim([0, 1])
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    # 3. 整体质量评估
    ax3 = axes[1, 0]
    overall_score = metrics['overall_quality_score']
    grade = metrics['quality_grade']

    colors = {'Excellent': 'darkgreen', 'Good': 'green', 'Fair': 'orange', 'Poor': 'red'}

    ax3.bar(['Overall Quality'], [overall_score], color=colors.get(grade, 'gray'))
    ax3.set_ylabel('Quality Score')
    ax3.set_title(f'Overall Quality: {grade} ({overall_score:.3f})')
    ax3.set_ylim([0, 1])

    # 4. CLIP语义距离可视化（如果可用）
    ax4 = axes[1, 1]
    if 'clip_semantic_distances' in metrics:
        clip_distances = metrics['clip_semantic_distances']
        indices_pairs = [f'{i}-{i+1}' for i in range(len(clip_distances))]

        ax4.plot(indices_pairs, clip_distances, 'o-', color='orange', label='CLIP Semantic Distance')
        ax4.set_xlabel('Frame Pairs')
        ax4.set_ylabel('Semantic Distance')
        ax4.set_title('CLIP Semantic Distance Between Adjacent Frames')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'CLIP Analysis\nNot Available',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('CLIP Semantic Analysis')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"质量评估可视化已保存到: {save_path}")

    plt.show()

    return fig
```

### 2.4 ViT2D编码模块 (ViT2DEncoder)

#### 设计目标
将选择的关键帧通过ViT架构转换为高维特征表示。

#### 架构参数
- **Patch大小**：32×32
- **嵌入维度**：768 (可配置)
- **注意力头数**：12
- **层数**：12
- **MLP隐藏层维度**：3072

#### 输入处理
```
输入：C×H×W×K
重塑：CK×H×W (通道维度展开)
Patch分割：CK×P×(patch_size²×3) 其中P = (H//patch_size) × (W//patch_size)
线性投影：CK×P×Emb
```

#### 位置编码
```python
# 2D位置编码
pos_embed = nn.Parameter(torch.randn(1, P, Emb))
# 模态位置编码 (区分不同模态)
modal_embed = nn.Parameter(torch.randn(CK, 1, Emb))
```

#### 共享注意力优化
```python
# 在同一模态内共享Key和Query计算
class SharedAttention(nn.Module):
    def __init__(self, dim, num_heads):
        self.shared_kq = nn.Linear(dim, dim * 2)  # 共享K, Q
        self.separate_v = nn.Linear(dim, dim)     # 独立V
```

### 2.5 模态融合模块 (ModalFusion)

#### 设计目标
实现跨模态信息交互，生成全局融合特征和各模态的压缩表示。

#### 融合策略

##### 跨模态注意力
```
给定不同模态特征 {X_1, X_2, ..., X_C} ∈ R^(P×Emb×K)

1. 模态间交叉注意力：
   Q_i = X_i * W_q^i
   K_j = X_j * W_k^j
   V_j = X_j * W_v^j

   Attention(Q_i, K_j, V_j) = softmax(Q_i K_j^T / √d_k) V_j

2. 全局融合：
   X_global = Σ(i=1 to C) w_i * Attention(Q_i, K_global, V_global)
   其中 w_i 为模态权重

3. 压缩表示：
   X_compressed_i = compress(X_i)  # 压缩到 (P//α)×Emb×K
```

##### 注意力机制实现
```python
class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_modals):
        super().__init__()
        self.num_modals = num_modals
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.modal_weights = nn.Parameter(torch.ones(num_modals) / num_modals)

    def forward(self, modal_features):
        # modal_features: List[Tensor(P, Emb, K)]
        global_features = []

        for i, query_modal in enumerate(modal_features):
            attended_features = []
            for j, key_modal in enumerate(modal_features):
                if i != j:  # 跨模态注意力
                    attn_out, _ = self.multihead_attn(
                        query_modal, key_modal, key_modal
                    )
                    attended_features.append(attn_out)

            # 加权融合
            weighted_features = torch.stack(attended_features).mean(0)
            global_features.append(weighted_features * self.modal_weights[i])

        return torch.stack(global_features).sum(0)
```

### 2.6 特征压缩模块 (FeatureCompressor)

#### 设计目标
减少特征维度，提高计算效率，同时保持重要信息。

#### 压缩策略
```python
class FeatureCompressor(nn.Module):
    def __init__(self, input_dim, compression_ratio=4):
        super().__init__()
        self.alpha = compression_ratio
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, input_dim // self.alpha),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // self.alpha, input_dim // self.alpha)
        )

    def forward(self, x):
        # x: P×Emb×K
        P, Emb, K = x.shape
        compressed_P = P // self.alpha

        # 空间压缩
        x_reshaped = x.view(compressed_P, self.alpha * Emb, K)
        x_compressed = self.compressor(x_reshaped)

        return x_compressed  # (P//α)×Emb×K
```

### 2.7 分割验证模块 (SegmentationHead)

#### 设计目标
通过分割任务验证特征表示的质量，提供额外的监督信号。

#### 网络架构
```python
class SegmentationHead(nn.Module):
    def __init__(self, input_features, num_classes, image_size):
        super().__init__()
        self.feature_proj = nn.Linear(input_features, 256)
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 上采样
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, 4, 2, 1)  # 最终分割图
        )

    def forward(self, compressed_features):
        # compressed_features: C×(P//α)×Emb×K
        batch_size, spatial_dim, emb_dim, k_frames = compressed_features.shape

        # 投影到分割特征空间
        features = self.feature_proj(compressed_features)

        # 重塑为2D特征图
        h = w = int(sqrt(spatial_dim))
        features_2d = features.view(batch_size * k_frames, -1, h, w)

        # 上采样到原图尺寸
        segmentation_logits = self.upsampler(features_2d)

        return segmentation_logits
```

## 3. 损失函数设计

### 3.1 多样性损失 (Diversity Loss)

#### 目标
确保选择的关键帧在语义上尽可能不同，提高特征的代表性。

#### 数学表示
```
L_diversity = (1/(K(K-1))) * Σ(i=1 to K) Σ(j=1 to K, i≠j) similarity(f_i, f_j)

其中：
- f_i, f_j 为第i和第j个选择帧的特征表示
- similarity 可以是余弦相似度或点积相似度
- 目标是最小化此损失
```

#### 实现
```python
def diversity_loss(selected_features):
    """
    selected_features: C×H×W×K
    """
    C, H, W, K = selected_features.shape
    features_flat = selected_features.view(C, -1, K)  # C×(H*W)×K
    features_normalized = F.normalize(features_flat, dim=1)

    # 计算相似度矩阵
    similarity_matrix = torch.bmm(
        features_normalized.transpose(1, 2),  # C×K×(H*W)
        features_normalized                    # C×(H*W)×K
    )  # C×K×K

    # 移除对角线元素（自相似度）
    mask = ~torch.eye(K, dtype=torch.bool, device=selected_features.device)
    similarity_values = similarity_matrix[:, mask].view(C, K * (K - 1))

    # 平均相似度作为损失
    loss = similarity_values.mean()
    return loss
```

### 3.2 分割损失 (Segmentation Loss)

#### 组合损失
```python
class SegmentationLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, pred, target):
        smooth = 1e-5
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
```

### 3.3 位置编码一致性损失 (Position Encoding Consistency Loss)

#### 目标
确保帧位置编码能够准确反映真实的时序关系，保持位置信息的一致性和单调性。

#### 数学表示
```python
L_position = L_distance + L_monotonicity + L_smoothness

其中：
- L_distance: 距离保持性损失
- L_monotonicity: 位置单调性损失
- L_smoothness: 编码平滑性损失
```

#### 实现
```python
class PositionEncodingConsistencyLoss(nn.Module):
    def __init__(self, distance_weight=0.4, monotonic_weight=0.3, smooth_weight=0.3):
        super().__init__()
        self.distance_weight = distance_weight
        self.monotonic_weight = monotonic_weight
        self.smooth_weight = smooth_weight

    def forward(self, frame_pos_encodings, frame_indices, sequence_length):
        """
        Args:
            frame_pos_encodings: [K, embed_dim] - 帧位置编码
            frame_indices: [K] - 帧索引
            sequence_length: int - 原始序列长度
        """
        # 1. 距离保持性损失
        distance_loss = self.compute_distance_preservation_loss(
            frame_pos_encodings, frame_indices, sequence_length
        )

        # 2. 位置单调性损失
        monotonic_loss = self.compute_monotonicity_loss(
            frame_pos_encodings, frame_indices
        )

        # 3. 编码平滑性损失
        smoothness_loss = self.compute_smoothness_loss(frame_pos_encodings)

        total_loss = (
            self.distance_weight * distance_loss +
            self.monotonic_weight * monotonic_loss +
            self.smooth_weight * smoothness_loss
        )

        return {
            'position_loss': total_loss,
            'distance_loss': distance_loss,
            'monotonic_loss': monotonic_loss,
            'smoothness_loss': smoothness_loss
        }

    def compute_distance_preservation_loss(self, encodings, indices, seq_len):
        """确保编码距离与实际帧距离成正比"""
        if len(indices) < 2:
            return torch.tensor(0.0, device=encodings.device)

        # 计算实际帧间距离（归一化）
        actual_distances = []
        encoding_distances = []

        for i in range(len(indices) - 1):
            actual_dist = (indices[i+1] - indices[i]).float() / seq_len
            actual_distances.append(actual_dist)

            encoding_dist = torch.norm(encodings[i+1] - encodings[i], p=2)
            encoding_distances.append(encoding_dist)

        actual_distances = torch.stack(actual_distances)
        encoding_distances = torch.stack(encoding_distances)

        # 归一化编码距离
        encoding_distances = encoding_distances / (encoding_distances.max() + 1e-8)

        # MSE损失确保距离的一致性
        distance_loss = F.mse_loss(encoding_distances, actual_distances)

        return distance_loss

    def compute_monotonicity_loss(self, encodings, indices):
        """确保位置编码保持单调性"""
        if len(indices) < 2:
            return torch.tensor(0.0, device=encodings.device)

        monotonic_violations = 0
        total_pairs = 0

        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                # 实际位置关系
                actual_order = indices[i] < indices[j]

                # 编码相似度关系（位置越近相似度越高）
                similarity = F.cosine_similarity(
                    encodings[i:i+1], encodings[j:j+1], dim=1
                ).item()

                # 距离关系
                distance = torch.norm(encodings[j] - encodings[i], p=2).item()

                # 期望：位置距离越大，编码距离也应该越大
                expected_order = distance > 0

                if actual_order != expected_order:
                    monotonic_violations += 1

                total_pairs += 1

        # 单调性违反比例作为损失
        monotonic_loss = torch.tensor(
            monotonic_violations / max(total_pairs, 1),
            device=encodings.device
        )

        return monotonic_loss

    def compute_smoothness_loss(self, encodings):
        """确保相邻位置编码的平滑变化"""
        if len(encodings) < 2:
            return torch.tensor(0.0, device=encodings.device)

        # 计算相邻编码的差异
        differences = []
        for i in range(len(encodings) - 1):
            diff = torch.norm(encodings[i+1] - encodings[i], p=2)
            differences.append(diff)

        differences = torch.stack(differences)

        # 平滑性损失：相邻差异的方差应该较小
        smoothness_loss = torch.var(differences)

        return smoothness_loss
```

### 3.4 总损失函数

```python
class UMELoss(nn.Module):
    def __init__(
        self,
        diversity_weight=0.1,
        segmentation_weight=1.0,
        position_weight=0.05
    ):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.segmentation_weight = segmentation_weight
        self.position_weight = position_weight

        self.segmentation_loss = SegmentationLoss()
        self.position_loss = PositionEncodingConsistencyLoss()

    def forward(
        self,
        selected_features,
        segmentation_pred,
        segmentation_target,
        frame_pos_encodings=None,
        frame_indices=None,
        sequence_length=None
    ):
        # 基础损失
        diversity_loss = self.diversity_loss(selected_features)
        segmentation_loss = self.segmentation_loss(segmentation_pred, segmentation_target)

        total_loss = (
            self.diversity_weight * diversity_loss +
            self.segmentation_weight * segmentation_loss
        )

        loss_dict = {
            'total_loss': total_loss,
            'diversity_loss': diversity_loss,
            'segmentation_loss': segmentation_loss
        }

        # 位置编码损失（如果提供相关信息）
        if (frame_pos_encodings is not None and
            frame_indices is not None and
            sequence_length is not None):

            position_losses = self.position_loss(
                frame_pos_encodings, frame_indices, sequence_length
            )

            position_loss = position_losses['position_loss']
            total_loss += self.position_weight * position_loss

            # 更新损失字典
            loss_dict['total_loss'] = total_loss
            loss_dict.update(position_losses)

        return loss_dict
```

## 4. 性能优化策略

### 4.1 内存优化

#### 梯度检查点
```python
from torch.utils.checkpoint import checkpoint

class UMEEncoder(nn.Module):
    def forward(self, x):
        # 使用检查点减少内存占用
        x = checkpoint(self.dimension_normalizer, x)
        x = checkpoint(self.keyframe_selector, x)
        x = checkpoint(self.vit_encoder, x)
        x = checkpoint(self.modal_fusion, x)
        return x
```

#### 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

@autocast()
def forward_pass(model, x):
    return model(x)
```

### 4.2 计算优化

#### 批量处理优化
```python
class BatchOptimizedUME(nn.Module):
    def forward(self, batch):
        # 批量处理不同尺寸的输入
        normalized_batch = []
        for item in batch:
            normalized_batch.append(self.dimension_normalizer(item))

        # 填充到统一尺寸
        max_dims = self.get_max_dimensions(normalized_batch)
        padded_batch = self.pad_to_size(normalized_batch, max_dims)

        return self.process_batch(padded_batch)
```

#### 动态图优化
```python
# 根据输入动态调整网络结构
class DynamicUME(nn.Module):
    def forward(self, x):
        if x.shape[-1] <= 5:  # 少量切片时使用简化网络
            return self.simplified_forward(x)
        else:
            return self.full_forward(x)
```

## 5. 实验配置

### 5.1 超参数设置

#### 模型参数
```yaml
model:
  embed_dim: 768
  num_heads: 12
  num_layers: 12
  patch_size: 32
  compression_ratio: 4
  max_keyframes: 10

training:
  batch_size: 2
  learning_rate: 1e-4
  weight_decay: 1e-5
  num_epochs: 200
  warmup_epochs: 10

loss:
  diversity_weight: 0.1
  segmentation_weight: 1.0
  dice_weight: 0.5
  ce_weight: 0.5
```

#### 数据配置
```yaml
data:
  image_size: [256, 256]
  roi_size: [256, 256, 64]
  spacing: [1.0, 1.0, 1.0]
  intensity_range: [0, 1]
  cache_rate: 0.8
```

### 5.2 评估指标

#### 分割性能
- Dice系数
- Hausdorff距离
- 平均表面距离

#### 计算效率
- 推理时间
- 内存占用
- FLOPs计算

#### 特征质量
- 特征可视化
- 降维分析 (t-SNE, UMAP)
- 聚类分析

这个架构设计文档提供了UME系统的详细技术规范，可以作为实现的蓝图。