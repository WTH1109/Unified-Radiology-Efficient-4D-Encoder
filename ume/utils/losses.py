"""
UME损失函数模块
包含多种损失函数的组合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class DiceLoss(nn.Module):
    """Dice损失函数"""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, C, H, W] or [B, C, H, W, D]
            targets: [B, C, H, W] or [B, C, H, W, D]
        """
        # 应用softmax
        predictions = F.softmax(predictions, dim=1)

        # 计算每个类别的Dice
        dice_scores = []
        num_classes = predictions.shape[1]

        for c in range(num_classes):
            pred_c = predictions[:, c]
            target_c = targets[:, c] if targets.shape[1] > 1 else (targets == c).float()

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        return 1.0 - torch.stack(dice_scores).mean()


class FocalLoss(nn.Module):
    """Focal损失函数"""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiversityLoss(nn.Module):
    """多样性损失函数，确保选择的帧具有多样性"""

    def __init__(self):
        super().__init__()

    def forward(self, selected_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            selected_features: List of frame features or tensor [B, K, feature_dim]
        """
        if isinstance(selected_features, list):
            # 如果是列表，先转换为tensor
            features = torch.stack([f.mean(dim=1) for f in selected_features], dim=1)  # [B, K, embed_dim]
        else:
            features = selected_features

        if features.dim() == 3:
            B, K, feature_dim = features.shape
        else:
            # 如果是其他维度，尝试重新整形
            features = features.view(features.shape[0], -1, features.shape[-1])
            B, K, feature_dim = features.shape

        if K <= 1:
            return torch.tensor(0.0, device=features.device)

        # 计算特征相似度矩阵
        features_norm = F.normalize(features, dim=-1)
        similarity_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))  # [B, K, K]

        # 排除对角线（自相似度）
        mask = ~torch.eye(K, dtype=torch.bool, device=features.device).unsqueeze(0)
        similarity_values = similarity_matrix[mask].view(B, K * (K - 1))

        # 最小化相似度
        return similarity_values.mean()


class ConsistencyLoss(nn.Module):
    """一致性损失函数，确保不同策略的特征一致性"""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, strategy_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            strategy_features: Dict containing features from different strategies
        """
        if len(strategy_features) < 2:
            return torch.tensor(0.0, device=next(iter(strategy_features.values())).device)

        features_list = list(strategy_features.values())
        total_loss = 0.0
        num_pairs = 0

        # 计算所有特征对之间的一致性损失
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                feat1 = F.normalize(features_list[i], dim=-1)
                feat2 = F.normalize(features_list[j], dim=-1)

                # 计算余弦相似度
                similarity = torch.sum(feat1 * feat2, dim=-1) / self.temperature

                # 使用对比学习损失
                pos_loss = -torch.log(torch.exp(similarity).mean())
                total_loss += pos_loss
                num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)


class TemporalConsistencyLoss(nn.Module):
    """时序一致性损失函数"""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: [B, seq_len, embed_dim]
        """
        if temporal_features.shape[1] <= 1:
            return torch.tensor(0.0, device=temporal_features.device)

        # 计算相邻帧特征差异
        diff_features = temporal_features[:, 1:] - temporal_features[:, :-1]

        # L2范数
        diff_norms = torch.norm(diff_features, p=2, dim=-1)

        # 平滑性损失（相邻帧差异不应过大）
        smoothness_loss = torch.relu(diff_norms - self.margin).mean()

        return smoothness_loss


class AreaPredictionLoss(nn.Module):
    """
    面积预测损失函数
    用于预测mask的面积，支持BraTS数据集的多mask并集
    """

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def compute_intersection_area(self, masks: torch.Tensor) -> torch.Tensor:
        """
        计算BraTS 3种mask的交集面积
        Args:
            masks: [B, H, W, D] 或 [H, W, D] 的mask tensor，包含BraTS标签
                   1: 坏死(necrotic), 2: 水肿(edema), 3: 增强肿瘤(enhancing tumor)
        Returns:
            areas: [B, D] 或 [D] 每帧的归一化交集面积（0-1之间）
        """
        if masks.dim() == 3:  # [H, W, D] - 单个样本
            # 提取3种类别的mask
            necrotic_mask = (masks == 1).float()     # 坏死
            edema_mask = (masks == 2).float()        # 水肿
            enhancing_mask = (masks == 3).float()    # 增强肿瘤

            # 计算交集：所有3种mask都为1的地方
            intersection_mask = necrotic_mask * edema_mask * enhancing_mask  # [H, W, D]

            # 计算每帧的交集面积
            H, W, D = intersection_mask.shape
            areas = intersection_mask.sum(dim=[0, 1]) / (H * W)  # [D] 归一化到0-1

        elif masks.dim() == 4:  # [B, H, W, D]
            # 提取3种类别的mask
            necrotic_mask = (masks == 1).float()     # 坏死
            edema_mask = (masks == 2).float()        # 水肿
            enhancing_mask = (masks == 3).float()    # 增强肿瘤

            # 计算交集：所有3种mask都为1的地方
            intersection_mask = necrotic_mask * edema_mask * enhancing_mask  # [B, H, W, D]

            # 计算每帧的交集面积
            B, H, W, D = intersection_mask.shape
            areas = intersection_mask.sum(dim=[1, 2]) / (H * W)  # [B, D] 归一化到0-1

        elif masks.dim() == 5:  # [B, num_classes, H, W, D] - 如果是one-hot编码
            if masks.shape[1] >= 4:  # 至少有4个类别（包括背景）
                necrotic_mask = masks[:, 1:2, :, :, :]      # [B, 1, H, W, D] 坏死
                edema_mask = masks[:, 2:3, :, :, :]         # [B, 1, H, W, D] 水肿
                enhancing_mask = masks[:, 3:4, :, :, :]     # [B, 1, H, W, D] 增强肿瘤

                # 计算交集
                intersection_mask = (necrotic_mask * edema_mask * enhancing_mask).squeeze(1)  # [B, H, W, D]

                # 计算每帧的交集面积
                B, H, W, D = intersection_mask.shape
                areas = intersection_mask.sum(dim=[1, 2]) / (H * W)  # [B, D] 归一化到0-1
            else:
                raise ValueError(f"Expected at least 4 classes for BraTS, got {masks.shape[1]}")
        else:
            raise ValueError(f"Unsupported mask dimension: {masks.dim()}")

        return areas

    def forward(
        self,
        predictions: torch.Tensor,  # [B, K, 1] 或 [B, D, 1] 预测的面积
        masks: Optional[torch.Tensor] = None,  # Ground truth masks
        selected_indices: Optional[List[int]] = None  # 如果是选中的K帧，需要indices
    ) -> torch.Tensor:
        """
        计算面积预测损失
        """
        if masks is None:
            # 如果没有ground truth，返回0损失
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # 计算ground truth的交集面积
        gt_areas = self.compute_intersection_area(masks)  # [B, D] 或 [D]

        # 处理不同维度的gt_areas
        if gt_areas.dim() == 1:  # [D] - 来自3维mask
            # 需要添加batch维度以匹配predictions
            gt_areas = gt_areas.unsqueeze(0)  # [1, D]

        # 如果有selected_indices，处理帧选择
        if selected_indices is not None:
            if isinstance(selected_indices, list) and len(selected_indices) > 0 and isinstance(selected_indices[0], list):
                # selected_indices是每个样本的索引列表 [[sample0_indices], [sample1_indices], ...]
                B, K, _ = predictions.shape
                selected_gt_areas = []

                for b in range(B):
                    sample_indices = selected_indices[b]
                    sample_gt = gt_areas[b:b+1, sample_indices]  # [1, K]
                    selected_gt_areas.append(sample_gt)

                gt_areas = torch.cat(selected_gt_areas, dim=0)  # [B, K]
                gt_areas = gt_areas.unsqueeze(-1)  # [B, K, 1]
            else:
                # selected_indices是单一索引列表（所有样本相同）
                gt_areas = gt_areas[:, selected_indices]  # [B, K]
                gt_areas = gt_areas.unsqueeze(-1)  # [B, K, 1]
        else:
            # selected_indices为None表示predictions已经是选择后的结果
            gt_areas = gt_areas.unsqueeze(-1)  # [B, D, 1] 或适当调整

        # 确保predictions和gt_areas维度一致
        if predictions.shape != gt_areas.shape:
            # 可能需要调整维度
            if predictions.dim() == 2:  # [B, K] or [B, D]
                predictions = predictions.unsqueeze(-1)  # [B, K, 1] or [B, D, 1]

        # 计算MSE损失
        loss = self.mse_loss(predictions, gt_areas)

        return loss


class UMELoss(nn.Module):
    """
    UME组合损失函数
    包含分割损失、多样性损失、一致性损失等
    """

    def __init__(
        self,
        loss_weights: Dict[str, float] = None,
        use_focal: bool = False,
        dice_smooth: float = 1e-6,
        use_voco: bool = False
    ):
        super().__init__()

        # 默认权重
        if use_voco:
            default_weights = {
                'voco': 1.0,
                'diversity': 0.1,
                'contrastive': 0.5,
                'area': 0.3
            }
        else:
            default_weights = {
                'segmentation': 1.0,
                'diversity': 0.1,
                'consistency': 0.05,
                'temporal': 0.03
            }
        self.loss_weights = loss_weights or default_weights
        self.use_voco = use_voco

        # 分割损失
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss() if use_focal else None

        # 其他损失
        self.diversity_loss = DiversityLoss()
        self.consistency_loss = ConsistencyLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        self.area_prediction_loss = AreaPredictionLoss()  # 添加面积预测损失

        # VoCo损失
        if use_voco:
            # 延迟导入避免循环依赖
            from ume.utils.voco_supervision import VoCoLoss
            self.voco_loss = VoCoLoss(temperature=0.07, lambda_area=1.0)

    def forward(
        self,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        diversity_loss: Optional[torch.Tensor] = None,
        selected_frames: Optional[torch.Tensor] = None,
        strategy_features: Optional[Dict[str, torch.Tensor]] = None,
        temporal_features: Optional[torch.Tensor] = None,
        # VoCo相关参数
        main_features: Optional[torch.Tensor] = None,
        neighbor_features: Optional[torch.Tensor] = None,
        overlap_areas: Optional[torch.Tensor] = None,
        # 面积预测相关参数
        area_predictions: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        selected_indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合损失
        """
        losses = {}
        total_loss = 0.0

        # VoCo训练模式
        if self.use_voco and main_features is not None and neighbor_features is not None and overlap_areas is not None:
            voco_results = self.voco_loss(main_features, neighbor_features, overlap_areas)

            losses['voco_total_loss'] = voco_results['total_loss']
            losses['contrastive_loss'] = voco_results['contrastive_loss']
            losses['area_loss'] = voco_results['area_loss']

            total_loss += self.loss_weights['voco'] * voco_results['total_loss']
            total_loss += self.loss_weights.get('contrastive', 0.0) * voco_results['contrastive_loss']
            total_loss += self.loss_weights.get('area', 0.0) * voco_results['area_loss']

        # 分割损失 (用于传统训练模式)
        elif predictions is not None and targets is not None:
            if targets.dim() == predictions.dim() - 1:
                # targets: [B, H, W], predictions: [B, C, H, W]
                seg_loss_ce = self.ce_loss(predictions, targets.long())

                # 为Dice损失转换targets格式
                num_classes = predictions.shape[1]
                targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 3, 1, 2).float()
                seg_loss_dice = self.dice_loss(predictions, targets_one_hot)
            else:
                # targets已经是one-hot格式
                seg_loss_dice = self.dice_loss(predictions, targets)
                seg_loss_ce = self.ce_loss(predictions, targets.argmax(dim=1).long())

            # 如果使用Focal损失
            if self.focal_loss is not None:
                seg_loss_focal = self.focal_loss(predictions, targets.argmax(dim=1).long())
                seg_loss = seg_loss_dice + seg_loss_ce + seg_loss_focal
            else:
                seg_loss = seg_loss_dice + seg_loss_ce

            losses['segmentation_loss'] = seg_loss
            total_loss += self.loss_weights['segmentation'] * seg_loss

        # 多样性损失
        if diversity_loss is not None:
            losses['diversity_loss'] = diversity_loss
            total_loss += self.loss_weights['diversity'] * diversity_loss
        elif selected_frames is not None:
            div_loss = self.diversity_loss(selected_frames)
            losses['diversity_loss'] = div_loss
            total_loss += self.loss_weights['diversity'] * div_loss

        # 一致性损失
        if strategy_features is not None:
            cons_loss = self.consistency_loss(strategy_features)
            losses['consistency_loss'] = cons_loss
            total_loss += self.loss_weights['consistency'] * cons_loss

        # 时序一致性损失
        if temporal_features is not None:
            temp_loss = self.temporal_loss(temporal_features)
            losses['temporal_loss'] = temp_loss
            total_loss += self.loss_weights['temporal'] * temp_loss

        # 面积预测损失
        if area_predictions is not None:
            area_pred_loss = self.area_prediction_loss(
                predictions=area_predictions,
                masks=masks,
                selected_indices=selected_indices
            )
            losses['area_prediction_loss'] = area_pred_loss
            # 使用'area'权重或默认权重
            area_weight = self.loss_weights.get('area', 0.3)
            total_loss += area_weight * area_pred_loss

        losses['total_loss'] = total_loss
        return losses