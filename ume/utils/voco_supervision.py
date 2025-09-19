"""
VoCo (Volume-based Context) 监督实现
基于随机crop和相交面积比的自监督方法 - 2D版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
import random


class VoCoSupervision:
    """
    VoCo监督信号生成器 - 2D版本
    对每一帧进行2D crop和相交面积计算
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (256, 256),  # H, W
        crop_size: int = 32,  # 固定大小的正方形crop
        overlap_ratio: float = 0.25
    ):
        self.input_size = input_size
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio

        # 计算crop的网格数量
        self.grid_size = (
            input_size[0] // crop_size,
            input_size[1] // crop_size
        )
        self.num_crops = self.grid_size[0] * self.grid_size[1]

    def generate_crops_for_frame(self, batch_size: int, num_frames: int, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """
        为每个帧生成2D crops
        对于(B, C, H, W, K)，每一帧(B, C, H, W, 1)切分成crop^2个patches，
        然后随机选择一个主patch和相邻patches计算面积

        Args:
            batch_size: 批次大小
            num_frames: 帧数K
            device: 设备

        Returns:
            Dict包含：
            - main_crop_indices: 主要crop的索引 [B, K]
            - neighbor_crops_indices: 相邻crop的索引 [B, K, 4]
            - overlap_areas: 相交面积比 [B, K, 4]
        """
        main_crop_indices_list = []
        neighbor_crops_indices_list = []
        overlap_areas_list = []

        for b in range(batch_size):
            frame_main_indices = []
            frame_neighbor_indices = []
            frame_overlap_areas = []

            for k in range(num_frames):
                # 1. 随机选择一个主patch索引（0到num_crops-1）
                main_idx = random.randint(0, self.num_crops - 1)

                # 2. 生成相邻patches的索引
                neighbor_indices = self._generate_neighbor_indices(main_idx)

                # 3. 计算相交面积比
                overlap_areas = self._calculate_2d_overlap_ratios(main_idx, neighbor_indices)

                frame_main_indices.append(main_idx)
                frame_neighbor_indices.append(neighbor_indices)
                frame_overlap_areas.append(overlap_areas)

            main_crop_indices_list.append(frame_main_indices)
            neighbor_crops_indices_list.append(frame_neighbor_indices)
            overlap_areas_list.append(frame_overlap_areas)

        # 如果指定了设备，直接创建在该设备上
        if device is not None:
            return {
                'main_crop_indices': torch.tensor(main_crop_indices_list, dtype=torch.long, device=device),
                'neighbor_crops_indices': torch.tensor(neighbor_crops_indices_list, dtype=torch.long, device=device),
                'overlap_areas': torch.tensor(overlap_areas_list, dtype=torch.float32, device=device)
            }
        else:
            return {
                'main_crop_indices': torch.tensor(main_crop_indices_list, dtype=torch.long),
                'neighbor_crops_indices': torch.tensor(neighbor_crops_indices_list, dtype=torch.long),
                'overlap_areas': torch.tensor(overlap_areas_list, dtype=torch.float32)
            }

    def _generate_neighbor_indices(self, main_idx: int) -> List[int]:
        """
        生成相邻patch的索引
        main_idx是线性索引，需要转换为2D坐标
        """
        # 将线性索引转换为2D网格坐标
        grid_h, grid_w = self.grid_size
        row = main_idx // grid_w
        col = main_idx % grid_w

        neighbors = []

        # 4个邻近方向：右、下、右下、左下
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            new_row = row + dr
            new_col = col + dc

            # 检查边界
            if 0 <= new_row < grid_h and 0 <= new_col < grid_w:
                neighbor_idx = new_row * grid_w + new_col
            else:
                # 如果超出边界，使用随机有效索引
                neighbor_idx = random.randint(0, self.num_crops - 1)

            neighbors.append(neighbor_idx)

        return neighbors

    def _calculate_2d_overlap_ratios(self, main_idx: int, neighbor_indices: List[int]) -> List[float]:
        """
        计算2D patches之间的重叠面积比
        由于patches是不重叠的网格，相邻的patches重叠为0，需要特殊处理
        """
        grid_h, grid_w = self.grid_size
        main_row = main_idx // grid_w
        main_col = main_idx % grid_w

        overlap_ratios = []

        for neighbor_idx in neighbor_indices:
            neighbor_row = neighbor_idx // grid_w
            neighbor_col = neighbor_idx % grid_w

            # 计算曼哈顿距离
            distance = abs(main_row - neighbor_row) + abs(main_col - neighbor_col)

            # 根据距离给定重叠比例（模拟相邻关系）
            if distance == 0:  # 同一个patch
                overlap_ratio = 1.0
            elif distance == 1:  # 直接相邻
                overlap_ratio = 0.5
            elif distance == 2:  # 对角相邻
                overlap_ratio = 0.25
            else:  # 较远
                overlap_ratio = 0.1

            overlap_ratios.append(overlap_ratio)

        return overlap_ratios

    def extract_patches_from_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        从帧中提取所有的2D patches

        Args:
            frames: [B, C, H, W, K] 选中的K帧

        Returns:
            patches: [B, K, num_crops, C, crop_size, crop_size] 所有的patches
        """
        B, C, H, W, K = frames.shape
        crop_size = self.crop_size
        grid_h, grid_w = self.grid_size

        patches = torch.zeros(B, K, self.num_crops, C, crop_size, crop_size,
                             device=frames.device)

        for b in range(B):
            for k in range(K):
                frame = frames[b, :, :, :, k]  # [C, H, W]

                # 将帧切分成patches
                patch_idx = 0
                for i in range(grid_h):
                    for j in range(grid_w):
                        h_start = i * crop_size
                        w_start = j * crop_size
                        h_end = h_start + crop_size
                        w_end = w_start + crop_size

                        patches[b, k, patch_idx] = frame[:, h_start:h_end, w_start:w_end]
                        patch_idx += 1

        return patches

    def extract_selected_patches(self, patches: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        根据索引提取选中的patches

        Args:
            patches: [B, K, num_crops, C, crop_size, crop_size] 所有patches
            indices: [B, K] or [B, K, N] patch索引

        Returns:
            selected_patches: 选中的patches
        """
        B, K = indices.shape[:2]

        if len(indices.shape) == 2:  # [B, K]
            # 提取主patches
            selected = torch.zeros(B, K, patches.shape[3], patches.shape[4], patches.shape[5],
                                  device=patches.device)
            for b in range(B):
                for k in range(K):
                    selected[b, k] = patches[b, k, indices[b, k]]
        else:  # [B, K, N]
            N = indices.shape[2]
            selected = torch.zeros(B, K, N, patches.shape[3], patches.shape[4], patches.shape[5],
                                  device=patches.device)
            for b in range(B):
                for k in range(K):
                    for n in range(N):
                        selected[b, k, n] = patches[b, k, indices[b, k, n]]

        return selected


class VoCoLoss(nn.Module):
    """
    VoCo损失函数 - 2D版本
    基于2D patches的对比学习
    """

    def __init__(self, temperature: float = 0.07, lambda_area: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_area = lambda_area
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        main_features: torch.Tensor,      # [B, K, embed_dim] 每帧的主patch特征
        neighbor_features: torch.Tensor,  # [B, K, 4, embed_dim] 每帧的邻居patch特征
        overlap_areas: torch.Tensor       # [B, K, 4] 相交面积比
    ) -> Dict[str, torch.Tensor]:
        """
        计算VoCo损失 - 对K帧分别计算并平均

        Args:
            main_features: 主patch的特征
            neighbor_features: 邻居patches的特征
            overlap_areas: 相交面积比

        Returns:
            包含各项损失的字典
        """
        B, K, embed_dim = main_features.shape
        B, K, num_neighbors, embed_dim = neighbor_features.shape

        total_contrastive_loss = 0
        total_area_loss = 0

        # 对每帧分别计算损失
        for k in range(K):
            main_feat = main_features[:, k, :]  # [B, embed_dim]
            neighbor_feat = neighbor_features[:, k, :, :]  # [B, 4, embed_dim]
            overlap = overlap_areas[:, k, :]  # [B, 4]

            # 1. 归一化特征
            main_feat_norm = F.normalize(main_feat, dim=1)  # [B, embed_dim]
            neighbor_feat_norm = F.normalize(neighbor_feat, dim=2)  # [B, 4, embed_dim]

            # 2. 计算相似度
            similarities = torch.bmm(
                neighbor_feat_norm,  # [B, 4, embed_dim]
                main_feat_norm.unsqueeze(2)  # [B, embed_dim, 1]
            ).squeeze(2) / self.temperature  # [B, 4]

            # 3. 基于面积的相似度预测
            area_similarities = torch.sigmoid(similarities)

            # 4. 面积预测损失
            frame_area_loss = self.mse_loss(area_similarities, overlap)

            # 5. 对比学习损失
            positive_logits = similarities * overlap
            negative_logits = similarities * (1 - overlap)

            frame_contrastive_loss = -torch.log(
                torch.exp(positive_logits).sum(dim=1) /
                (torch.exp(positive_logits).sum(dim=1) + torch.exp(negative_logits).sum(dim=1) + 1e-8)
            ).mean()

            total_contrastive_loss += frame_contrastive_loss
            total_area_loss += frame_area_loss

        # 平均损失
        contrastive_loss = total_contrastive_loss / K
        area_loss = total_area_loss / K

        # 总损失
        total_loss = contrastive_loss + self.lambda_area * area_loss

        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'area_loss': area_loss
        }


class VoCoAugmentation:
    """
    VoCo数据增强
    生成多个sub-volumes用于对比学习
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (256, 256, 128),
        num_crops: int = 8,
        crop_size: Tuple[int, int, int] = (64, 64, 32)
    ):
        self.voco_supervision = VoCoSupervision(input_size, crop_size)
        self.num_crops = num_crops

    def __call__(self, volume: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        对单个volume进行VoCo增强

        Args:
            volume: [C, H, W, D] 单个volume

        Returns:
            增强后的数据字典
        """
        # 添加batch维度
        volume = volume.unsqueeze(0)  # [1, C, H, W, D]

        # 生成VoCo crops
        voco_data = self.voco_supervision.generate_crops(batch_size=1)

        # 提取主crop
        main_crop = self.voco_supervision.extract_crops_from_volume(
            volume, voco_data['main_crop_coords'].unsqueeze(1)
        ).squeeze(1)  # [1, C, h, w, d]

        # 提取neighbor crops
        neighbor_crops = self.voco_supervision.extract_crops_from_volume(
            volume, voco_data['neighbor_crops_coords']
        )  # [1, 4, C, h, w, d]

        return {
            'main_crop': main_crop.squeeze(0),  # [C, h, w, d]
            'neighbor_crops': neighbor_crops.squeeze(0),  # [4, C, h, w, d]
            'overlap_areas': voco_data['overlap_areas'].squeeze(0),  # [4]
            'main_crop_coords': voco_data['main_crop_coords'].squeeze(0),  # [6]
            'neighbor_crops_coords': voco_data['neighbor_crops_coords'].squeeze(0)  # [4, 6]
        }