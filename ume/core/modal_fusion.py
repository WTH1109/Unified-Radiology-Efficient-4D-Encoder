"""
UME核心模块：多层次模态融合
Multi-Level Modal Fusion Module (参考MedTransformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class IntraModalFusion(nn.Module):
    """
    同模态内不同切片的融合
    参考MedTransformer的Intra-dimension Cross-Attention
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.fusion_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, modal_sequences: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            modal_sequences: List of sequences from same modality
                           Each sequence: [B, seq_len, embed_dim]
        Returns:
            outputs: List of fused sequences
        """
        if len(modal_sequences) == 1:
            return modal_sequences

        # 构建融合的Key和Query矩阵
        fused_keys = []
        fused_queries = []

        for seq in modal_sequences:
            # 分离class token和patch embeddings
            class_token = seq[:, 0:1, :]  # [B, 1, embed_dim]
            patch_embeddings = seq[:, 1:, :]  # [B, seq_len-1, embed_dim]

            # 融合patch embeddings（求和）
            if len(fused_keys) == 0:
                fused_patch = patch_embeddings
            else:
                # 从前一个融合结果中提取patch部分（移除class token）
                prev_fused_patch = fused_keys[-1][:, 1:, :]  # [B, seq_len-1, embed_dim]
                fused_patch = prev_fused_patch + patch_embeddings

            # 构建融合的K和Q
            fused_k = torch.cat([class_token, fused_patch], dim=1)
            fused_q = torch.cat([class_token, fused_patch], dim=1)

            fused_keys.append(fused_k)
            fused_queries.append(fused_q)

        # 应用融合注意力
        outputs = []
        for i, (q, k) in enumerate(zip(fused_queries, fused_keys)):
            v = modal_sequences[i]  # 使用原始序列作为Value

            attn_output, _ = self.fusion_attention(q, k, v)
            outputs.append(attn_output)

        return outputs


class InterModalFusion(nn.Module):
    """
    跨模态融合
    """

    def __init__(self, embed_dim: int, num_heads: int, num_modalities: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        # 跨模态注意力
        self.cross_modal_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # 模态权重学习
        self.modal_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)

        # 融合投影层
        self.fusion_proj = nn.Linear(embed_dim * num_modalities, embed_dim)

    def forward(self, modal_features: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            modal_features: List of features from different modalities
                          Each: [B, seq_len, embed_dim]
        Returns:
            global_features: [B, seq_len, embed_dim]
            cross_attended: List of cross-attended features
        """
        # 跨模态注意力计算
        cross_attended = []
        for i, query_modal in enumerate(modal_features):
            attended_features = []

            for j, key_modal in enumerate(modal_features):
                if i != j:  # 跨模态注意力
                    attn_out, _ = self.cross_modal_attention(
                        query_modal, key_modal, key_modal
                    )
                    attended_features.append(attn_out * self.modal_weights[j])

            # 加权融合其他模态的信息
            if attended_features:
                cross_attended.append(torch.stack(attended_features).sum(0))
            else:
                cross_attended.append(query_modal)

        # 全局融合
        global_features = torch.cat(cross_attended, dim=-1)
        fused_features = self.fusion_proj(global_features)

        return fused_features, cross_attended


class MultiLevelModalFusion(nn.Module):
    """
    多层次模态融合架构
    """

    def __init__(self, embed_dim: int, num_heads: int, num_modalities: int):
        super().__init__()

        # Level 1: Intra-modal fusion
        self.intra_fusion = IntraModalFusion(embed_dim, num_heads)

        # Level 2: Inter-modal fusion
        self.inter_fusion = InterModalFusion(embed_dim, num_heads, num_modalities)

        # Level 3: Global fusion
        self.global_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, multi_modal_data: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            multi_modal_data: Dict with modality_name -> List[sequences]
                             For BraTS: {'t1': [seq], 't2': [seq], 't1ce': [seq], 'flair': [seq]}
        Returns:
            Dict containing fused features
        """
        # Level 1: 同模态内融合
        intra_fused = {}
        for modality, sequences in multi_modal_data.items():
            if isinstance(sequences, list) and len(sequences) > 1:
                fused_sequences = self.intra_fusion(sequences)
                intra_fused[modality] = fused_sequences[0]  # 取第一个输出
            else:
                intra_fused[modality] = sequences[0] if isinstance(sequences, list) else sequences

        # Level 2: 跨模态融合
        modal_features = list(intra_fused.values())
        global_features, modal_specific = self.inter_fusion(modal_features)

        # Level 3: 全局特征增强
        enhanced_global = self.global_fusion(global_features)

        return {
            'global_features': enhanced_global,
            'modal_specific': modal_specific,
            'intra_fused': intra_fused
        }


class TemporalRelationshipModeling(nn.Module):
    """
    基于长视频理解的时序关系建模
    """

    def __init__(self, embed_dim: int, num_heads: int, max_sequence_length: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_sequence_length = max_sequence_length

        # 时序位置编码
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(max_sequence_length, embed_dim)
        )

        # 多尺度时序注意力
        self.multi_scale_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(3)  # 短期、中期、长期
        ])

        # 时序融合层
        self.temporal_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence_features: [B, seq_len, embed_dim]
        Returns:
            fused_features: [B, seq_len, embed_dim]
        """
        B, seq_len, embed_dim = sequence_features.shape

        # 添加时序位置编码
        pos_encoded = sequence_features + self.temporal_pos_encoding[:seq_len].unsqueeze(0)

        # 多尺度时序注意力
        # 短期关系 (局部窗口)
        short_term, _ = self.multi_scale_attention[0](pos_encoded, pos_encoded, pos_encoded)

        # 中期关系 (跨步采样)
        if seq_len > 4:
            step = seq_len // 4
            medium_indices = list(range(0, seq_len, step))
            medium_features = pos_encoded[:, medium_indices, :]
            medium_term, _ = self.multi_scale_attention[1](
                medium_features, medium_features, medium_features
            )
            # 插值回原长度
            medium_term = F.interpolate(
                medium_term.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            medium_term = short_term

        # 长期关系 (全局注意力)
        long_term, _ = self.multi_scale_attention[2](pos_encoded, pos_encoded, pos_encoded)

        # 融合多尺度特征
        multi_scale_features = torch.cat([short_term, medium_term, long_term], dim=-1)
        fused_features = self.temporal_fusion(multi_scale_features)

        return fused_features


class BraTSModalityFusion(nn.Module):
    """
    专门为BraTS数据集设计的模态融合模块
    处理T1, T2, T1ce, FLAIR四种模态
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_names = ['t1', 't2', 't1ce', 'flair']

        # 多层次模态融合
        self.modal_fusion = MultiLevelModalFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_modalities=4
        )

        # 时序关系建模
        self.temporal_modeling = TemporalRelationshipModeling(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_sequence_length=128  # BraTS D维度最大值
        )

        # 模态特征投影
        self.modality_projectors = nn.ModuleDict({
            modality: nn.Linear(embed_dim, embed_dim)
            for modality in self.modality_names
        })

    def prepare_brats_input(self, brats_tensor: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        将BraTS张量 [B, 4, H, W, D] 转换为模态字典
        Args:
            brats_tensor: [B, 4, 256, 256, 128] - BraTS多模态数据
        Returns:
            modality_dict: Dict with modality_name -> List[features]
        """
        B, C, H, W, D = brats_tensor.shape
        assert C == 4, f"Expected 4 modalities, got {C}"

        modality_dict = {}
        for i, modality in enumerate(self.modality_names):
            # 提取单个模态: [B, 1, H, W, D]
            modal_data = brats_tensor[:, i:i+1, :, :, :]

            # 转换为patch序列 (简化版本，实际需要ViT patch embedding)
            # 这里先用全局平均池化作为特征提取
            modal_features = modal_data.mean(dim=[2, 3])  # [B, 1, D]
            modal_features = modal_features.squeeze(1).transpose(1, 0)  # [D, B]

            # 投影到embed_dim
            modal_features = self.modality_projectors[modality](
                modal_features.float()
            )  # [D, embed_dim]

            # 重新整形为 [B, seq_len, embed_dim] 格式
            modal_features = modal_features.unsqueeze(0).repeat(B, 1, 1)  # [B, D, embed_dim]

            modality_dict[modality] = [modal_features]

        return modality_dict

    def forward(self, brats_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        BraTS多模态融合前向传播
        Args:
            brats_tensor: [B, 4, 256, 256, 128] - BraTS数据
        Returns:
            融合后的特征字典
        """
        # 准备多模态输入
        modality_dict = self.prepare_brats_input(brats_tensor)

        # 多层次模态融合
        fusion_results = self.modal_fusion(modality_dict)

        # 时序关系建模
        global_features = fusion_results['global_features']
        temporal_features = self.temporal_modeling(global_features)

        # 更新结果
        fusion_results['temporal_features'] = temporal_features
        fusion_results['final_features'] = temporal_features

        return fusion_results