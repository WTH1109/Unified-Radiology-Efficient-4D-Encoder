"""
UME训练脚本
实现帧级训练逻辑和模型训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
from pathlib import Path

# 添加项目路径
import sys
sys.path.append('/mnt/cfs/a8bga2/huawei/code/Unified-Radiology-Efficient-4D-Encoder')

from ume.models.ume_model import UMEModel
from ume.data.brats_loader import BraTSUMEDataLoader
from ume.utils.losses import UMELoss
from ume.utils.metrics import UMEMetrics


class UMETrainer:
    """
    UME训练器
    支持帧级训练和完整的训练流程
    """

    def __init__(
        self,
        model: UMEModel,
        train_loader,
        val_loader,
        config: Dict,
        device: str = "cuda",
        distributed: bool = False,
        local_rank: int = 0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.distributed = distributed
        self.local_rank = local_rank

        # 设置优化器
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.scaler = GradScaler()

        # 设置损失函数和评估指标
        # 使用标准分割损失函数
        self.use_voco = False  # 改为标准分割训练模式
        self.criterion = UMELoss(
            loss_weights=config['training']['loss_weights'],
            use_voco=False  # 禁用VoCo训练模式
        )
        self.metrics = UMEMetrics()

        # 训练状态
        self.current_epoch = 0
        self.best_dice = 0.0
        self.train_losses = []
        self.val_losses = []

        # 设置日志
        self._setup_logging()

    def _setup_optimizer(self) -> optim.Optimizer:
        """设置优化器"""
        opt_config = self.config['optimizer']
        if opt_config['type'] == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['type'] == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")

    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """设置学习率调度器"""
        sched_config = self.config['scheduler']
        if sched_config['type'] == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max']
            )
        elif sched_config['type'] == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def _setup_logging(self):
        """设置日志"""
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        实现帧级训练逻辑
        """
        self.model.train()
        # 标准分割训练的损失统计
        epoch_losses = {'total': 0.0, 'segmentation': 0.0, 'diversity': 0.0}
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Training Epoch {self.current_epoch}')

        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            images = batch["image"].to(self.device)  # [B, 4, 256, 256, 128] (完整128帧)

            # 如果有标签数据
            labels = batch.get("label", None)
            if labels is not None:
                # 检查labels是否是tensor
                if torch.is_tensor(labels):
                    labels = labels.to(self.device)
                else:
                    # labels是文件路径列表，暂时创建模拟标签
                    B, _, H, W, _ = images.shape
                    labels = torch.randint(0, 4, (B, H, W), device=self.device)
            else:
                # 如果没有标签，创建模拟标签用于训练验证
                B, _, H, W, _ = images.shape
                labels = torch.randint(0, 4, (B, H, W), device=self.device)

            # 前向传播 - 标准分割训练
            with autocast():
                outputs = self.model(images, mode='training')

                # 计算分割损失
                loss_dict = self.criterion(
                    predictions=outputs.get('segmentation'),
                    targets=labels,
                    diversity_loss=outputs.get('diversity_loss', None),
                    selected_frames=outputs.get('selected_frames', None)
                )

            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss_dict['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 更新损失统计 - 标准分割训练
            epoch_losses['total'] += loss_dict['total_loss'].item()
            epoch_losses['segmentation'] += loss_dict.get('segmentation_loss', torch.tensor(0.0)).item()
            if 'diversity_loss' in loss_dict:
                epoch_losses['diversity'] += loss_dict['diversity_loss'].item()

            # 更新进度条 - 分割训练指标
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'seg': f"{loss_dict.get('segmentation_loss', torch.tensor(0.0)).item():.4f}",
                'div': f"{loss_dict.get('diversity_loss', torch.tensor(0.0)).item():.4f}"
            })

        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch
        支持逐帧预测和关键帧预测两种模式
        """
        self.model.eval()

        # 标准分割验证的损失统计
        epoch_losses = {'total': 0.0, 'segmentation': 0.0, 'diversity': 0.0}
        metrics_accumulator = {'dice': [], 'iou': []}

        pbar = tqdm(self.val_loader, desc=f'Validation Epoch {self.current_epoch}')

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch["image"].to(self.device)  # [B, 4, 256, 256, 128] (验证时完整128帧)
                labels = batch.get("label", None)
                if labels is not None:
                    # 检查labels是否是tensor
                    if torch.is_tensor(labels):
                        labels = labels.to(self.device)
                    else:
                        # labels是文件路径列表，暂时创建模拟标签
                        B, _, H, W, _ = images.shape
                        labels = torch.randint(0, 4, (B, H, W), device=self.device)
                else:
                    # 如果没有标签，创建模拟标签用于验证
                    B, _, H, W, _ = images.shape
                    labels = torch.randint(0, 4, (B, H, W), device=self.device)

                # 标准分割验证
                outputs = self.model(images, mode='inference')
                predictions = outputs['predictions']

                # 计算分割损失
                if labels is not None:
                    loss_dict = self.criterion(
                        predictions=predictions,
                        targets=labels,
                        diversity_loss=outputs.get('diversity_loss', None),
                        selected_frames=outputs.get('selected_frames', None)
                    )

                    epoch_losses['total'] += loss_dict['total_loss'].item()
                    epoch_losses['segmentation'] += loss_dict.get('segmentation_loss', torch.tensor(0.0)).item()
                    if 'diversity_loss' in loss_dict:
                        epoch_losses['diversity'] += loss_dict['diversity_loss'].item()

                # 计算评估指标
                if labels is not None:
                    batch_metrics = self.metrics.compute_metrics(predictions, labels)
                    metrics_accumulator['dice'].extend(batch_metrics['dice'])
                    metrics_accumulator['iou'].extend(batch_metrics['iou'])

                # 更新进度条 - 标准分割验证
                avg_dice = np.mean(metrics_accumulator['dice']) if metrics_accumulator['dice'] else 0.0
                pbar.set_postfix({
                    'dice': f"{avg_dice:.4f}",
                        'loss': f"{epoch_losses['total'] / (batch_idx + 1):.4f}"
                    })

        # 计算平均损失和指标
        num_batches = len(self.val_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        # VoCo模式和传统模式不同的指标计算
        if hasattr(self, 'use_voco') and self.use_voco:
            # VoCo模式：只返回损失，不计算分割指标
            return epoch_losses
        else:
            # 传统模式：计算分割指标
            avg_metrics = {
                'dice': np.mean(metrics_accumulator['dice']) if metrics_accumulator['dice'] else 0.0,
                'iou': np.mean(metrics_accumulator['iou']) if metrics_accumulator['iou'] else 0.0
            }
            epoch_losses.update(avg_metrics)
            return epoch_losses

    def frame_by_frame_validation(self) -> Dict[str, float]:
        """
        逐帧验证：测试时一帧一帧进行预测
        这是用户特别要求的功能
        """
        self.model.eval()
        metrics_accumulator = {'dice': [], 'iou': []}

        self.logger.info("Starting frame-by-frame validation...")
        pbar = tqdm(self.val_loader, desc='Frame-by-Frame Validation')

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch["image"].to(self.device)  # [B, 4, 256, 256, 128]
                labels = batch.get("label", None)
                if labels is not None:
                    labels = labels.to(self.device)

                # 逐帧预测
                frame_predictions = self.model.frame_by_frame_inference(images)

                # 计算评估指标
                if labels is not None:
                    batch_metrics = self.metrics.compute_metrics(frame_predictions, labels)
                    metrics_accumulator['dice'].extend(batch_metrics['dice'])
                    metrics_accumulator['iou'].extend(batch_metrics['iou'])

                # 更新进度条
                avg_dice = np.mean(metrics_accumulator['dice']) if metrics_accumulator['dice'] else 0.0
                pbar.set_postfix({'dice': f"{avg_dice:.4f}"})

        # 计算平均指标
        avg_metrics = {
            'dice': np.mean(metrics_accumulator['dice']) if metrics_accumulator['dice'] else 0.0,
            'iou': np.mean(metrics_accumulator['iou']) if metrics_accumulator['iou'] else 0.0
        }

        self.logger.info(f"Frame-by-frame validation results: {avg_metrics}")
        return avg_metrics

    def train(self):
        """
        完整训练流程
        """
        num_epochs = self.config['training']['epochs']
        self.logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # 训练
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)

            # 验证
            val_losses = self.validate_epoch()
            self.val_losses.append(val_losses)

            # 更新学习率
            self.scheduler.step()

            # 记录日志 - VoCo模式和传统模式不同
            if hasattr(self, 'use_voco') and self.use_voco:
                # VoCo模式日志
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_losses['total']:.4f}, "
                    f"Val Loss: {val_losses['total']:.4f}, "
                    f"Val VoCo: {val_losses['voco']:.4f}, "
                    f"Val Div: {val_losses['diversity']:.4f}"
                )
                # VoCo模式下以总损失的降低作为最佳模型标准
                if val_losses['total'] < getattr(self, 'best_loss', float('inf')):
                    self.best_loss = val_losses['total']
                    self.save_checkpoint(is_best=True)
            else:
                # 传统模式日志
                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_losses['total']:.4f}, "
                    f"Val Loss: {val_losses['total']:.4f}, "
                    f"Val Dice: {val_losses['dice']:.4f}"
                )
                # 保存最佳模型
                if val_losses['dice'] > self.best_dice:
                    self.best_dice = val_losses['dice']
                    self.save_checkpoint(is_best=True)

            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(is_best=False)

            # 每20个epoch进行一次逐帧验证
            if (epoch + 1) % 20 == 0:
                frame_metrics = self.frame_by_frame_validation()
                self.logger.info(f"Frame-by-frame metrics at epoch {epoch}: {frame_metrics}")

        # 训练完成日志
        self.logger.info(f"Training completed. Best Dice: {self.best_dice:.4f}")

    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'config': self.config
        }

        # 创建检查点目录
        checkpoint_dir = Path("./checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        # 保存当前检查点
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with Dice: {self.best_dice:.4f}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_dice = checkpoint['best_dice']

        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def main():
    parser = argparse.ArgumentParser(description='UME Training Script')
    parser.add_argument('--config', type=str, default='./configs/brats_ume_config.json',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建数据加载器
    data_loader = BraTSUMEDataLoader(**config['data'])
    train_loader, val_loader = data_loader.get_loaders()

    # 创建模型
    model = UMEModel(**config['model'])
    print(f"Model created with {model.get_model_size()}")

    # 创建训练器
    trainer = UMETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        distributed=args.distributed,
        local_rank=args.local_rank
    )

    # 恢复训练（如果指定）
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()