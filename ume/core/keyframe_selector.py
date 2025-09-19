"""
UME核心模块：智能抽帧策略
Enhanced Key Frame Selection Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open_clip
from typing import List, Tuple, Optional, Dict


class EnhancedKeyFrameSelector(nn.Module):
    """
    增强的关键帧选择器
    融合多种抽帧策略：均匀采样、内容感知、注意力引导
    """

    def __init__(self, input_channels: int = 4, embed_dim: int = 768, max_frames: int = 10,
                 freeze_area_predictor: bool = False):
        super().__init__()
        self.max_frames = max_frames
        self.embed_dim = embed_dim
        self.freeze_area_predictor = freeze_area_predictor

        # 内容感知网络 - 用于分割面积预测
        self.content_analyzer = nn.Sequential(
            nn.Conv3d(input_channels, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # 保持深度维度 -> [B, 128, D, 1, 1]
        )

        # 分割面积预测网络 - 预测通用mask的面积（单一值）
        self.area_predictor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 预测单一mask的面积（归一化值）
        )

        # 如果需要冻结面积预测网络
        if self.freeze_area_predictor:
            for param in self.area_predictor.parameters():
                param.requires_grad = False

        # 注意力特征提取网络
        self.attention_feature_extractor = nn.Sequential(
            nn.Conv3d(input_channels, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 4, 4))  # [B, 128, D, 4, 4]
        )

        # 注意力评分网络
        self.attention_scorer = nn.Sequential(
            nn.Linear(128 * 16, 256),  # 128 * 4 * 4
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # 策略融合网络 - 可学习的融合机制
        self.strategy_fusion = nn.Sequential(
            nn.Linear(max_frames * 3, 128),  # 3个策略，每个max_frames个候选
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, max_frames)  # 输出最终选择的帧索引权重
        )

        # Med-CLIP特征提取器（用于多样性损失）
        # 注意：BiomedCLIP需要使用open_clip
        # 默认禁用以加快训练速度，需要时可以设置use_medclip=True
        self.use_medclip = True

        if self.use_medclip:
            try:
                model, preprocess = open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.medclip_encoder = model.visual
                for param in self.medclip_encoder.parameters():
                    param.requires_grad = False
                print("Med-CLIP loaded successfully for diversity loss calculation")
            except Exception as e:
                print(f"Failed to load Med-CLIP: {e}")
                self.use_medclip = False


    def uniform_sampling(self, sequence_length: int, target_frames: int) -> List[int]:
        """均匀采样策略"""
        if sequence_length <= target_frames:
            return list(range(sequence_length))

        step = sequence_length / target_frames
        indices = [int(i * step) for i in range(target_frames)]
        return indices

    def content_aware_sampling(self, content_scores: torch.Tensor, target_frames: int,
                              nms_window: int = 5) -> List[int]:
        """
        内容感知采样策略 - 基于分割面积的非极大值抑制
        Args:
            content_scores: [B, D] 每帧的内容得分（反映肿瘤面积大小）
            target_frames: 目标帧数
            nms_window: NMS窗口大小
        """
        B, D = content_scores.shape
        indices_list = []

        for b in range(B):
            scores = content_scores[b]  # [D]

            # 非极大值抑制 (NMS)
            selected_indices = []
            scores_clone = scores.clone()

            for _ in range(min(target_frames, D)):
                # 找到当前最大值的索引
                max_idx = torch.argmax(scores_clone).item()
                selected_indices.append(max_idx)

                # 抑制窗口内的其他值
                start_idx = max(0, max_idx - nms_window // 2)
                end_idx = min(D, max_idx + nms_window // 2 + 1)
                scores_clone[start_idx:end_idx] = -float('inf')

                # 如果已经选够了帧，退出
                if len(selected_indices) >= target_frames:
                    break

            # 确保按时间顺序排序
            indices = sorted(selected_indices)[:target_frames]
            indices_list.append(indices)

        return indices_list[0] if len(indices_list) == 1 else indices_list

    def attention_guided_sampling(self, attention_scores: torch.Tensor, target_frames: int) -> List[int]:
        """注意力引导采样策略"""
        B, D = attention_scores.shape
        indices_list = []

        for b in range(B):
            scores = attention_scores[b]  # [D]
            _, top_indices = torch.topk(scores, min(target_frames, D))
            indices = sorted(top_indices.tolist())
            indices_list.append(indices)

        return indices_list[0] if len(indices_list) == 1 else indices_list

    def fuse_strategies(self, uniform_idx: List[int], content_idx: List[int],
                       attention_idx: List[int], target_frames: int) -> List[int]:
        """
        融合多种策略的结果 - 使用可学习的网络进行融合
        """
        device = next(self.parameters()).device
        D = max(max(uniform_idx), max(content_idx), max(attention_idx)) + 1

        # 创建策略编码向量
        strategy_encoding = torch.zeros(3 * self.max_frames, device=device)

        # 编码每个策略的选择
        for i, idx in enumerate(uniform_idx[:self.max_frames]):
            strategy_encoding[i] = idx / D  # 归一化到[0,1]

        for i, idx in enumerate(content_idx[:self.max_frames]):
            strategy_encoding[self.max_frames + i] = idx / D

        for i, idx in enumerate(attention_idx[:self.max_frames]):
            strategy_encoding[2 * self.max_frames + i] = idx / D

        # 通过融合网络计算权重
        fusion_weights = self.strategy_fusion(strategy_encoding)  # [max_frames]
        fusion_weights = torch.softmax(fusion_weights, dim=0)

        # 收集所有候选帧
        all_candidates = list(set(uniform_idx + content_idx + attention_idx))

        # 计算每个候选帧的最终得分
        candidate_scores = {}
        for idx in all_candidates:
            score = 0
            if idx in uniform_idx:
                pos = uniform_idx.index(idx)
                if pos < self.max_frames:
                    score += fusion_weights[pos].item()
            if idx in content_idx:
                pos = content_idx.index(idx)
                if pos < self.max_frames:
                    score += fusion_weights[pos].item()
            if idx in attention_idx:
                pos = attention_idx.index(idx)
                if pos < self.max_frames:
                    score += fusion_weights[pos].item()
            candidate_scores[idx] = score

        # 选择得分最高的帧
        sorted_indices = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        final_indices = [idx for idx, _ in sorted_indices[:target_frames]]

        return sorted(final_indices)

    def calculate_diversity_loss(self, selected_frames: torch.Tensor) -> torch.Tensor:
        """
        计算多样性损失，使用Med-CLIP编码特征或改进的特征相似度
        """
        B, C, H, W, K = selected_frames.shape

        if self.use_medclip and hasattr(self, 'medclip_encoder'):
            # 使用Med-CLIP编码特征 - 每个模态分别编码
            features_list = []
            for k in range(K):
                frame = selected_frames[:, :, :, :, k]  # [B, C, H, W]  C=4 (T1, T2, T1ce, FLAIR)
                frame_features = []

                # 对4个模态分别进行Med-CLIP编码
                for modality_idx in range(frame.shape[1]):  # 遍历4个模态
                    # 提取单个模态，扩展为3通道（灰度图复制为RGB）
                    modality = frame[:, modality_idx:modality_idx+1, :, :]  # [B, 1, H, W]
                    modality_rgb = modality.repeat(1, 3, 1, 1)  # [B, 3, H, W] 复制为3通道

                    # BiomedCLIP需要224x224输入，调整尺寸
                    modality_resized = F.interpolate(modality_rgb, size=(224, 224), mode='bilinear', align_corners=False)

                    # 归一化到[0,1]范围
                    if modality_resized.min() < 0:
                        modality_resized = (modality_resized - modality_resized.min()) / (modality_resized.max() - modality_resized.min() + 1e-8)

                    with torch.no_grad():
                        # BiomedCLIP编码单个模态
                        medclip_feat = self.medclip_encoder(modality_resized)
                        if isinstance(medclip_feat, tuple):
                            medclip_feat = medclip_feat[0]
                        # 如果是3D tensor，取平均池化
                        if medclip_feat.dim() > 2:
                            medclip_feat = medclip_feat.mean(dim=list(range(1, medclip_feat.dim()-1)))
                        frame_features.append(medclip_feat)

                # 合并4个模态的特征（可以用平均或拼接）
                # 方案1：平均池化
                frame_combined_features = torch.stack(frame_features, dim=1).mean(dim=1)  # [B, feature_dim]
                # 方案2：拼接（会增加维度）
                # frame_combined_features = torch.cat(frame_features, dim=-1)  # [B, feature_dim*4]

                features_list.append(frame_combined_features)

            features = torch.stack(features_list, dim=1)  # [B, K, feature_dim]
        else:
            # 使用改进的特征提取方法，而不是简单的展平
            # 通过卷积网络提取更有意义的特征
            features_list = []
            for k in range(K):
                frame = selected_frames[:, :, :, :, k]  # [B, C, H, W]
                # 使用全局平均池化和最大池化组合
                avg_pool = F.adaptive_avg_pool2d(frame, (8, 8))  # [B, C, 8, 8]
                max_pool = F.adaptive_max_pool2d(frame, (8, 8))  # [B, C, 8, 8]
                # 组合特征
                combined = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2*C, 8, 8]
                features_list.append(combined.flatten(1))  # [B, 2*C*64]

            features = torch.stack(features_list, dim=1)  # [B, K, feature_dim]

        # 计算特征相似度矩阵
        features_norm = F.normalize(features, dim=2)
        similarity_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))  # [B, K, K]

        # 排除对角线（自相似度）
        mask = ~torch.eye(K, dtype=torch.bool, device=selected_frames.device).unsqueeze(0).repeat(B, 1, 1)
        similarity_values = similarity_matrix[mask].view(B, K * (K - 1))

        # 最小化相似度（鼓励多样性）
        return similarity_values.mean()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int], Dict[str, torch.Tensor]]:
        """
        前向传播
        Args:
            x: Input tensor [B, C, H, W, D]
        Returns:
            selected_frames: [B, C, H, W, K]
            final_indices: 选择的帧索引
            losses: 包含多样性损失和面积预测损失的字典
        """
        B, C, H, W, D = x.shape
        target_frames = min(D, self.max_frames)
        losses = {}

        # 策略1: 均匀采样
        uniform_indices = self.uniform_sampling(D, target_frames)

        # 策略2: 内容感知采样（基于分割面积）
        # 将输入从[B, C, H, W, D]转换为Conv3d期望的[B, C, D, H, W]格式
        x_permuted = x.permute(0, 1, 4, 2, 3)  # [B, C, D, H, W]
        content_features = self.content_analyzer(x_permuted)  # [B, 128, D, 1, 1]
        content_features = content_features.squeeze(-1).squeeze(-1)  # [B, 128, D]
        content_features = content_features.permute(0, 2, 1)  # [B, D, 128]

        # 预测每帧的分割面积（单一mask的面积）
        area_predictions = self.area_predictor(content_features)  # [B, D, 1]
        # 使用面积预测作为内容得分
        content_scores = area_predictions.squeeze(-1)  # [B, D]

        content_indices_batch = self.content_aware_sampling(content_scores, target_frames)
        # 确保content_indices是批次列表格式
        if not isinstance(content_indices_batch[0], list):
            content_indices_batch = [content_indices_batch]  # 单样本情况

        # 策略3: 注意力引导采样（使用特征提取网络）
        attention_features = self.attention_feature_extractor(x_permuted)  # [B, 128, D, 4, 4]
        # 调整维度并展平空间特征
        attention_features = attention_features.permute(0, 2, 1, 3, 4)  # [B, D, 128, 4, 4]
        attention_features_flat = attention_features.reshape(B, D, -1)  # [B, D, 128*16]

        # 计算注意力分数
        attention_scores = self.attention_scorer(attention_features_flat).squeeze(-1)  # [B, D]
        attention_indices_batch = self.attention_guided_sampling(attention_scores, target_frames)
        # 确保attention_indices是批次列表格式
        if not isinstance(attention_indices_batch[0], list):
            attention_indices_batch = [attention_indices_batch]  # 单样本情况

        # 对batch中每个样本分别进行策略融合
        final_indices_batch = []
        for b in range(B):
            # 对当前样本进行策略融合
            sample_final_indices = self.fuse_strategies(
                uniform_indices,
                content_indices_batch[b],
                attention_indices_batch[b],
                target_frames
            )
            final_indices_batch.append(sample_final_indices)

        # 正确处理每个样本的不同帧索引
        selected_frames_list = []
        selected_area_predictions_list = []

        for b in range(B):
            # 每个样本使用自己的帧索引
            batch_final_indices = final_indices_batch[b]
            sample_frames = x[b:b+1, :, :, :, batch_final_indices]  # [1, C, H, W, K]
            sample_area_preds = area_predictions[b:b+1, batch_final_indices, :]  # [1, K, 1]

            selected_frames_list.append(sample_frames)
            selected_area_predictions_list.append(sample_area_preds)

        # 合并所有样本的结果
        selected_frames = torch.cat(selected_frames_list, dim=0)  # [B, C, H, W, K]
        selected_area_predictions = torch.cat(selected_area_predictions_list, dim=0)  # [B, K, 1]

        # 返回所有样本的索引列表，供损失函数使用
        final_indices = final_indices_batch

        losses['area_predictions'] = selected_area_predictions

        # 计算多样性损失
        diversity_loss = self.calculate_diversity_loss(selected_frames)
        losses['diversity_loss'] = diversity_loss

        return selected_frames, final_indices, losses


class FrameWiseProcessor:
    """
    帧级处理器：用于训练时的帧采样和测试时的逐帧预测
    """

    def __init__(self, frame_selector: EnhancedKeyFrameSelector):
        self.frame_selector = frame_selector

    def training_frame_sampling(self, x: torch.Tensor, num_samples: int = 8) -> torch.Tensor:
        """
        训练时的帧采样：从D维度中采样指定数量的帧
        Args:
            x: Input tensor [B, C, H, W, D] - BraTS shape (B, 4, 256, 256, 128)
            num_samples: 采样的帧数
        Returns:
            sampled_frames: [B, C, H, W, num_samples]
        """
        B, C, H, W, D = x.shape

        # 使用关键帧选择器采样
        with torch.no_grad():
            self.frame_selector.max_frames = num_samples
            selected_frames, indices, _ = self.frame_selector(x)

        return selected_frames

    def frame_by_frame_prediction(self, x: torch.Tensor, model: nn.Module,
                                 overlap: int = 4) -> torch.Tensor:
        """
        测试时的逐帧预测：一帧一帧进行预测然后合并
        Args:
            x: Input tensor [B, C, H, W, D] - BraTS shape (B, 4, 256, 256, 128)
            model: 预测模型
            overlap: 重叠帧数，用于平滑预测
        Returns:
            predictions: [B, num_classes, H, W, D]
        """
        B, C, H, W, D = x.shape
        device = x.device

        # 初始化预测结果容器
        predictions = []

        # 逐帧预测
        for i in range(D):
            # 获取当前帧及其邻近帧
            start_idx = max(0, i - overlap // 2)
            end_idx = min(D, i + overlap // 2 + 1)

            # 提取帧序列
            frame_sequence = x[:, :, :, :, start_idx:end_idx]  # [B, C, H, W, seq_len]

            # 如果序列长度不足，进行填充
            seq_len = frame_sequence.shape[-1]
            if seq_len < overlap + 1:
                # 重复边界帧进行填充
                if start_idx == 0:
                    # 向后填充
                    pad_frame = frame_sequence[:, :, :, :, -1:].repeat(1, 1, 1, 1, overlap + 1 - seq_len)
                    frame_sequence = torch.cat([frame_sequence, pad_frame], dim=-1)
                else:
                    # 向前填充
                    pad_frame = frame_sequence[:, :, :, :, :1].repeat(1, 1, 1, 1, overlap + 1 - seq_len)
                    frame_sequence = torch.cat([pad_frame, frame_sequence], dim=-1)

            # 模型预测
            with torch.no_grad():
                frame_pred = model(frame_sequence)  # [B, num_classes, H, W]

            predictions.append(frame_pred.unsqueeze(-1))  # [B, num_classes, H, W, 1]

        # 合并所有帧的预测
        predictions = torch.cat(predictions, dim=-1)  # [B, num_classes, H, W, D]

        return predictions