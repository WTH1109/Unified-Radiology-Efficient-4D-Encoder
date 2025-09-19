"""
UME评估指标模块
包含分割任务的常用评估指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import directed_hausdorff


class UMEMetrics:
    """UME评估指标计算器"""

    def __init__(self, num_classes: int = 4, ignore_background: bool = True):
        self.num_classes = num_classes
        self.ignore_background = ignore_background

    def compute_dice(self, predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> Dict[str, float]:
        """
        计算Dice系数
        Args:
            predictions: [B, C, H, W] or [B, C, H, W, D]
            targets: [B, C, H, W] or [B, C, H, W, D] or [B, H, W] or [B, H, W, D]
        """
        # 应用softmax
        predictions = F.softmax(predictions, dim=1)

        # 处理targets格式
        if targets.dim() == predictions.dim() - 1:
            # targets: [B, H, W], 需要转换为one-hot
            targets = F.one_hot(targets.long(), self.num_classes).permute(0, -1, *range(1, targets.dim())).float()

        dice_scores = {}
        dice_list = []

        start_class = 1 if self.ignore_background else 0
        for c in range(start_class, self.num_classes):
            pred_c = predictions[:, c]
            target_c = targets[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores[f'dice_class_{c}'] = dice.item()
            dice_list.append(dice.item())

        dice_scores['dice_mean'] = np.mean(dice_list)
        return dice_scores

    def compute_iou(self, predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> Dict[str, float]:
        """
        计算IoU (Intersection over Union)
        """
        # 应用softmax
        predictions = F.softmax(predictions, dim=1)

        # 处理targets格式
        if targets.dim() == predictions.dim() - 1:
            targets = F.one_hot(targets.long(), self.num_classes).permute(0, -1, *range(1, targets.dim())).float()

        iou_scores = {}
        iou_list = []

        start_class = 1 if self.ignore_background else 0
        for c in range(start_class, self.num_classes):
            pred_c = predictions[:, c]
            target_c = targets[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection

            iou = (intersection + smooth) / (union + smooth)
            iou_scores[f'iou_class_{c}'] = iou.item()
            iou_list.append(iou.item())

        iou_scores['iou_mean'] = np.mean(iou_list)
        return iou_scores

    def compute_hausdorff_distance(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        计算Hausdorff距离
        注意：这个计算较慢，通常用于最终评估
        """
        # 转换为numpy并获取预测类别
        pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()

        if targets.dim() == predictions.dim() - 1:
            target_classes = targets.cpu().numpy()
        else:
            target_classes = torch.argmax(targets, dim=1).cpu().numpy()

        hausdorff_scores = {}
        hausdorff_list = []

        start_class = 1 if self.ignore_background else 0

        for c in range(start_class, self.num_classes):
            class_hausdorff = []

            for b in range(pred_classes.shape[0]):
                pred_mask = (pred_classes[b] == c).astype(np.uint8)
                target_mask = (target_classes[b] == c).astype(np.uint8)

                # 获取边界点
                pred_points = np.argwhere(pred_mask)
                target_points = np.argwhere(target_mask)

                if len(pred_points) == 0 or len(target_points) == 0:
                    # 如果其中一个mask为空，设置一个很大的距离
                    class_hausdorff.append(100.0)
                else:
                    hd = max(
                        directed_hausdorff(pred_points, target_points)[0],
                        directed_hausdorff(target_points, pred_points)[0]
                    )
                    class_hausdorff.append(hd)

            mean_hd = np.mean(class_hausdorff)
            hausdorff_scores[f'hausdorff_class_{c}'] = mean_hd
            hausdorff_list.append(mean_hd)

        hausdorff_scores['hausdorff_mean'] = np.mean(hausdorff_list)
        return hausdorff_scores

    def compute_surface_distance(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        计算平均表面距离 (Average Surface Distance, ASD)
        """
        # 简化版本的表面距离计算
        pred_classes = torch.argmax(predictions, dim=1)

        if targets.dim() == predictions.dim() - 1:
            target_classes = targets
        else:
            target_classes = torch.argmax(targets, dim=1)

        asd_scores = {}
        asd_list = []

        start_class = 1 if self.ignore_background else 0

        for c in range(start_class, self.num_classes):
            pred_mask = (pred_classes == c).float()
            target_mask = (target_classes == c).float()

            # 简化的表面距离计算（基于边界）
            pred_boundary = self._get_boundary(pred_mask)
            target_boundary = self._get_boundary(target_mask)

            if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
                asd = 100.0  # 大的距离值
            else:
                # 简化计算（实际应该计算每个边界点到最近边界点的距离）
                asd = F.mse_loss(pred_boundary, target_boundary).item()

            asd_scores[f'asd_class_{c}'] = asd
            asd_list.append(asd)

        asd_scores['asd_mean'] = np.mean(asd_list)
        return asd_scores

    def _get_boundary(self, mask: torch.Tensor) -> torch.Tensor:
        """获取mask的边界"""
        # 简化的边界检测
        if mask.dim() == 4:  # [B, H, W, D]
            kernel = torch.ones(1, 1, 3, 3, 3, device=mask.device)
            eroded = F.conv3d(mask.unsqueeze(1), kernel, padding=1) < 27
        else:  # [B, H, W]
            kernel = torch.ones(1, 1, 3, 3, device=mask.device)
            eroded = F.conv2d(mask.unsqueeze(1), kernel, padding=1) < 9

        boundary = mask.unsqueeze(1) - eroded.float()
        return boundary.squeeze(1)

    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, List[float]]:
        """
        计算所有主要指标
        """
        dice_scores = self.compute_dice(predictions, targets)
        iou_scores = self.compute_iou(predictions, targets)

        # 为了保持与训练循环的兼容性，返回列表格式
        batch_size = predictions.shape[0]

        return {
            'dice': [dice_scores['dice_mean']] * batch_size,
            'iou': [iou_scores['iou_mean']] * batch_size,
            'detailed_dice': dice_scores,
            'detailed_iou': iou_scores
        }

    def compute_comprehensive_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        计算全面的评估指标（包括较慢的指标）
        """
        metrics = {}

        # 基础指标
        dice_scores = self.compute_dice(predictions, targets)
        iou_scores = self.compute_iou(predictions, targets)

        metrics.update(dice_scores)
        metrics.update(iou_scores)

        # 表面距离指标
        try:
            asd_scores = self.compute_surface_distance(predictions, targets)
            metrics.update(asd_scores)
        except Exception as e:
            print(f"Warning: Could not compute surface distance: {e}")

        # Hausdorff距离（较慢）
        try:
            hausdorff_scores = self.compute_hausdorff_distance(predictions, targets)
            metrics.update(hausdorff_scores)
        except Exception as e:
            print(f"Warning: Could not compute Hausdorff distance: {e}")

        return metrics


def evaluate_frame_consistency(frame_predictions: List[torch.Tensor]) -> Dict[str, float]:
    """
    评估逐帧预测的一致性
    Args:
        frame_predictions: List of predictions for each frame
    Returns:
        consistency metrics
    """
    if len(frame_predictions) < 2:
        return {'frame_consistency': 1.0}

    consistency_scores = []

    for i in range(len(frame_predictions) - 1):
        pred1 = F.softmax(frame_predictions[i], dim=1)
        pred2 = F.softmax(frame_predictions[i + 1], dim=1)

        # 计算相邻帧预测的一致性
        consistency = F.cosine_similarity(
            pred1.view(pred1.shape[0], -1),
            pred2.view(pred2.shape[0], -1),
            dim=1
        ).mean().item()

        consistency_scores.append(consistency)

    return {
        'frame_consistency': np.mean(consistency_scores),
        'frame_consistency_std': np.std(consistency_scores)
    }