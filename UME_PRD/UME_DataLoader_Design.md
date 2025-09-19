# UME数据处理流程与DataLoader设计

## 1. 数据处理流程总览

### 1.1 数据流水线架构

```
原始数据 → 预处理 → 维度标准化 → 数据增强 → 批量组装 → 模型输入
    ↓        ↓         ↓          ↓         ↓        ↓
  多格式    MONAI      统一4D      VoCo      批处理    训练/推理
  医学影像   变换       格式        增强      优化
```

### 1.2 支持的数据格式

#### 输入数据类型
| 数据类型 | 格式 | 来源 | 预处理策略 |
|---------|------|------|----------|
| T1脑部MRI | H×W×D | BraTS21 | 3D→4D扩展 |
| T2脑部MRI | H×W×D | BraTS21 | 3D→4D扩展 |
| T1ce脑部MRI | H×W×D | BraTS21 | 3D→4D扩展 |
| FLAIR脑部MRI | H×W×D | BraTS21 | 3D→4D扩展 |
| 多模态组合 | 4×H×W×D | BraTS21 | 保持4D格式 |
| 单层切片 | H×W | 提取切片 | 2D→4D扩展 |

## 2. UME DataLoader设计

### 2.1 核心设计原则

#### 2.1.1 兼容性原则
- 支持现有BraTS数据集格式
- 兼容MONAI transforms
- 可扩展到其他医学影像数据集

#### 2.1.2 效率原则
- 支持多进程数据加载
- 内存高效的缓存机制
- 动态批处理优化

#### 2.1.3 灵活性原则
- 可配置的预处理流水线
- 支持不同维度输入混合训练
- 运行时数据增强策略

### 2.2 UME Dataset类设计

```python
from monai.data import Dataset, CacheDataset
from monai.transforms import Compose
import torch
import numpy as np

class UMEDataset(Dataset):
    def __init__(
        self,
        data_list,
        transforms=None,
        dimension_mode='mixed',  # 'mixed', '2d', '3d', '4d'
        max_cache_size=1000,
        cache_rate=0.8
    ):
        """
        UME统一数据集类

        Args:
            data_list: 数据列表，包含路径和标签信息
            transforms: MONAI变换流水线
            dimension_mode: 维度处理模式
            max_cache_size: 最大缓存大小
            cache_rate: 缓存比例
        """
        super().__init__(data_list, transforms)
        self.dimension_mode = dimension_mode
        self.max_cache_size = max_cache_size
        self.cache_rate = cache_rate

        # 初始化缓存
        if cache_rate > 0:
            self._init_cache()

    def _init_cache(self):
        """初始化缓存系统"""
        cache_size = min(
            int(len(self.data) * self.cache_rate),
            self.max_cache_size
        )
        self.cached_data = {}
        self.cache_indices = set(range(cache_size))

    def __getitem__(self, index):
        """获取单个数据样本"""
        # 检查缓存
        if hasattr(self, 'cached_data') and index in self.cached_data:
            return self.cached_data[index]

        # 加载原始数据
        data_item = self.data[index]

        # 应用变换
        if self.transforms:
            data_item = self.transforms(data_item)

        # 维度标准化
        data_item = self._normalize_dimensions(data_item)

        # 缓存数据
        if (hasattr(self, 'cached_data') and
            index in self.cache_indices and
            len(self.cached_data) < self.max_cache_size):
            self.cached_data[index] = data_item

        return data_item

    def _normalize_dimensions(self, data_item):
        """维度标准化处理"""
        image = data_item['image']

        if self.dimension_mode == 'mixed':
            # 混合模式：根据输入自动判断
            if image.dim() == 3:  # H×W×D
                data_item['image'] = image.unsqueeze(0)  # 1×H×W×D
                data_item['input_type'] = '3d'
            elif image.dim() == 2:  # H×W
                data_item['image'] = image.unsqueeze(0).unsqueeze(-1)  # 1×H×W×1
                data_item['input_type'] = '2d'
            elif image.dim() == 4:  # C×H×W×D
                data_item['input_type'] = '4d'
            else:
                raise ValueError(f"Unsupported image dimension: {image.dim()}")

        elif self.dimension_mode == '4d':
            # 强制4D模式：所有输入转换为4D
            if image.dim() == 3:
                data_item['image'] = image.unsqueeze(0)
            elif image.dim() == 2:
                data_item['image'] = image.unsqueeze(0).unsqueeze(-1)
            data_item['input_type'] = '4d'

        return data_item
```

### 2.3 BraTS数据集适配器

```python
class BraTSUMEDataset(UMEDataset):
    def __init__(self, json_path, cache_dir=None, **kwargs):
        """
        BraTS数据集适配UME格式

        Args:
            json_path: BraTS数据集JSON配置文件路径
            cache_dir: 缓存目录
        """
        self.json_path = json_path
        self.cache_dir = cache_dir

        # 加载BraTS数据列表
        data_list = self._load_brats_data()

        # 初始化变换
        transforms = self._get_brats_transforms()

        super().__init__(data_list, transforms, **kwargs)

    def _load_brats_data(self):
        """加载BraTS数据列表"""
        import json

        with open(self.json_path, 'r') as f:
            brats_data = json.load(f)

        data_list = []
        for item in brats_data:
            # BraTS数据项格式转换
            data_item = {
                'image_t1': item.get('t1', ''),
                'image_t2': item.get('t2', ''),
                'image_t1ce': item.get('t1ce', ''),
                'image_flair': item.get('flair', ''),
                'label': item.get('seg', ''),
                'patient_id': item.get('name', ''),
            }
            data_list.append(data_item)

        return data_list

    def _get_brats_transforms(self):
        """获取BraTS数据变换流水线"""
        from monai.transforms import (
            LoadImaged, EnsureChannelFirstd, Orientationd,
            ScaleIntensityd, CropForegroundd, Resized,
            RandRotated, RandFlipd, RandGaussianNoised,
            Compose
        )

        # 加载多模态图像的keys
        image_keys = ['image_t1', 'image_t2', 'image_t1ce', 'image_flair']

        transforms = Compose([
            LoadImaged(keys=image_keys + ['label']),
            EnsureChannelFirstd(keys=image_keys + ['label']),
            Orientationd(keys=image_keys + ['label'], axcodes="RAS"),
            ScaleIntensityd(keys=image_keys, a_min=0, a_max=1, b_min=0.0, b_max=1.0),
            CropForegroundd(keys=image_keys + ['label'], source_key='image_t1'),
            Resized(
                keys=image_keys + ['label'],
                spatial_size=[128, 128, 128],  # 标准尺寸
                mode=['bilinear'] * len(image_keys) + ['nearest']
            ),

            # 数据增强（训练时）
            RandRotated(
                keys=image_keys,
                range_x=0.1, range_y=0.1, range_z=0.1,
                prob=0.2
            ),
            RandFlipd(keys=image_keys + ['label'], prob=0.5, spatial_axis=0),
            RandGaussianNoised(keys=image_keys, prob=0.1, std=0.01),

            # 多模态合并
            self._merge_modalities,
        ])

        return transforms

    def _merge_modalities(self, data_item):
        """合并多模态图像为4D张量"""
        modalities = ['image_t1', 'image_t2', 'image_t1ce', 'image_flair']

        # 提取有效模态
        available_modalities = []
        for modality in modalities:
            if modality in data_item and data_item[modality] is not None:
                available_modalities.append(data_item[modality])

        if available_modalities:
            # 合并为4D图像: C×H×W×D
            merged_image = torch.stack(available_modalities, dim=0)
            data_item['image'] = merged_image
            data_item['num_modalities'] = len(available_modalities)
        else:
            raise ValueError("No valid modalities found")

        # 清理原始模态数据以节省内存
        for modality in modalities:
            if modality in data_item:
                del data_item[modality]

        return data_item
```

### 2.4 动态批处理器

```python
class UMECollator:
    def __init__(self, pad_value=0, max_batch_dimensions=None):
        """
        UME自定义批处理器

        Args:
            pad_value: 填充值
            max_batch_dimensions: 最大批处理维度限制
        """
        self.pad_value = pad_value
        self.max_batch_dimensions = max_batch_dimensions

    def __call__(self, batch):
        """
        处理批量数据

        Args:
            batch: 数据批次列表

        Returns:
            Dict: 批处理后的数据
        """
        # 分析批次中的数据维度
        batch_info = self._analyze_batch(batch)

        # 统一数据维度
        unified_batch = self._unify_dimensions(batch, batch_info)

        # 构建批处理张量
        batch_tensors = self._build_batch_tensors(unified_batch)

        return batch_tensors

    def _analyze_batch(self, batch):
        """分析批次数据特征"""
        batch_info = {
            'max_channels': 0,
            'max_height': 0,
            'max_width': 0,
            'max_depth': 0,
            'input_types': [],
            'num_modalities': []
        }

        for item in batch:
            image = item['image']
            C, H, W, D = image.shape

            batch_info['max_channels'] = max(batch_info['max_channels'], C)
            batch_info['max_height'] = max(batch_info['max_height'], H)
            batch_info['max_width'] = max(batch_info['max_width'], W)
            batch_info['max_depth'] = max(batch_info['max_depth'], D)
            batch_info['input_types'].append(item.get('input_type', 'unknown'))
            batch_info['num_modalities'].append(item.get('num_modalities', C))

        return batch_info

    def _unify_dimensions(self, batch, batch_info):
        """统一批次中的数据维度"""
        target_shape = (
            batch_info['max_channels'],
            batch_info['max_height'],
            batch_info['max_width'],
            batch_info['max_depth']
        )

        unified_batch = []
        for item in batch:
            image = item['image']

            # 填充到目标尺寸
            padded_image = self._pad_to_shape(image, target_shape)

            unified_item = item.copy()
            unified_item['image'] = padded_image
            unified_batch.append(unified_item)

        return unified_batch

    def _pad_to_shape(self, tensor, target_shape):
        """将张量填充到目标形状"""
        current_shape = tensor.shape

        pad_dims = []
        for i in range(len(target_shape)):
            pad_size = target_shape[i] - current_shape[i]
            if pad_size > 0:
                pad_dims.extend([0, pad_size])
            else:
                pad_dims.extend([0, 0])

        # PyTorch padding格式是反向的
        pad_dims = pad_dims[::-1]

        return torch.nn.functional.pad(tensor, pad_dims, value=self.pad_value)

    def _build_batch_tensors(self, unified_batch):
        """构建最终的批处理张量"""
        batch_size = len(unified_batch)

        # 获取统一的图像尺寸
        sample_image = unified_batch[0]['image']
        C, H, W, D = sample_image.shape

        # 初始化批处理张量
        images = torch.zeros(batch_size, C, H, W, D)
        labels = []
        metadata = []

        for i, item in enumerate(unified_batch):
            images[i] = item['image']

            if 'label' in item:
                labels.append(item['label'])

            # 收集元数据
            meta = {
                'input_type': item.get('input_type', 'unknown'),
                'num_modalities': item.get('num_modalities', C),
                'patient_id': item.get('patient_id', f'unknown_{i}')
            }
            metadata.append(meta)

        result = {
            'images': images,
            'metadata': metadata
        }

        if labels:
            # 处理标签批次
            if all(isinstance(label, torch.Tensor) for label in labels):
                # 统一标签尺寸
                max_label_shape = self._get_max_shape([l.shape for l in labels])
                padded_labels = []
                for label in labels:
                    padded_label = self._pad_to_shape(label, max_label_shape)
                    padded_labels.append(padded_label)
                result['labels'] = torch.stack(padded_labels)
            else:
                result['labels'] = labels

        return result

    def _get_max_shape(self, shapes):
        """获取最大形状"""
        max_shape = list(shapes[0])
        for shape in shapes[1:]:
            for i, dim in enumerate(shape):
                max_shape[i] = max(max_shape[i], dim)
        return tuple(max_shape)
```

## 3. 数据加载器工厂

```python
class UMEDataLoaderFactory:
    """UME数据加载器工厂类"""

    @staticmethod
    def create_brats_loader(
        json_path,
        batch_size=2,
        num_workers=4,
        cache_rate=0.8,
        train_mode=True,
        **kwargs
    ):
        """创建BraTS数据加载器"""

        # 创建数据集
        dataset = BraTSUMEDataset(
            json_path=json_path,
            cache_rate=cache_rate if train_mode else 0,
            dimension_mode='mixed',
            **kwargs
        )

        # 创建批处理器
        collator = UMECollator()

        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=train_mode,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=num_workers > 0
        )

        return dataloader

    @staticmethod
    def create_mixed_dimension_loader(
        data_configs,
        batch_size=2,
        num_workers=4,
        **kwargs
    ):
        """创建混合维度数据加载器"""

        # 合并多个数据源
        all_data = []
        for config in data_configs:
            if config['type'] == 'brats':
                dataset = BraTSUMEDataset(**config['params'])
                all_data.extend(dataset.data)
            # 可以添加其他数据集类型

        # 创建统一数据集
        unified_dataset = UMEDataset(
            data_list=all_data,
            dimension_mode='mixed',
            **kwargs
        )

        # 创建数据加载器
        collator = UMECollator()
        dataloader = torch.utils.data.DataLoader(
            dataset=unified_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )

        return dataloader
```

## 4. 配置文件模板

### 4.1 UME训练配置

```yaml
# UME训练配置文件 - ume_config.yaml

# 数据配置
data:
  train:
    type: "brats"
    json_path: "./jsons/brats21_train.json"
    cache_rate: 0.8
    cache_dir: "/tmp/ume_cache/train"

  validation:
    type: "brats"
    json_path: "./jsons/brats21_val.json"
    cache_rate: 0.5
    cache_dir: "/tmp/ume_cache/val"

# 数据加载器配置
dataloader:
  train:
    batch_size: 2
    num_workers: 4
    shuffle: True
    pin_memory: True

  validation:
    batch_size: 1
    num_workers: 2
    shuffle: False
    pin_memory: True

# 数据预处理配置
preprocessing:
  target_size: [128, 128, 128]
  intensity_range: [0, 1]
  normalize: True

  # 数据增强配置
  augmentation:
    rotation_range: 0.1
    flip_probability: 0.5
    noise_probability: 0.1
    noise_std: 0.01

# 模型配置
model:
  embed_dim: 768
  num_heads: 12
  num_layers: 12
  patch_size: 32
  compression_ratio: 4
  max_keyframes: 10

  # 损失函数权重
  loss_weights:
    diversity: 0.1
    segmentation: 1.0
    dice: 0.5
    cross_entropy: 0.5

# 训练配置
training:
  epochs: 200
  learning_rate: 1e-4
  weight_decay: 1e-5
  warmup_epochs: 10

  # 优化器配置
  optimizer: "adamw"
  scheduler: "cosine"

  # 检查点配置
  save_interval: 10
  checkpoint_dir: "./checkpoints/ume"

  # 验证配置
  val_interval: 5
  early_stopping_patience: 20

# 环境配置
environment:
  device: "cuda"
  mixed_precision: True
  gradient_checkpointing: True
  distributed: False
```

### 4.2 数据集配置生成器

```python
class UMEConfigGenerator:
    """UME配置文件生成器"""

    @staticmethod
    def generate_brats_config(
        data_root="/mnt/cfs/a8bga2/huawei/code/Unified-Radiology-Efficient-4D-Encoder/data",
        output_path="./configs/ume_brats_config.yaml"
    ):
        """生成BraTS数据集配置"""

        config = {
            'data': {
                'train': {
                    'type': 'brats',
                    'json_path': f'{data_root}/jsons/brats21_train.json',
                    'cache_rate': 0.8,
                    'cache_dir': f'{data_root}/cache/ume_train'
                },
                'validation': {
                    'type': 'brats',
                    'json_path': f'{data_root}/jsons/brats21_val.json',
                    'cache_rate': 0.5,
                    'cache_dir': f'{data_root}/cache/ume_val'
                }
            },

            'dataloader': {
                'train': {
                    'batch_size': 2,
                    'num_workers': 4,
                    'shuffle': True,
                    'pin_memory': True
                },
                'validation': {
                    'batch_size': 1,
                    'num_workers': 2,
                    'shuffle': False,
                    'pin_memory': True
                }
            },

            'preprocessing': {
                'target_size': [128, 128, 128],
                'intensity_range': [0, 1],
                'normalize': True,
                'augmentation': {
                    'rotation_range': 0.1,
                    'flip_probability': 0.5,
                    'noise_probability': 0.1,
                    'noise_std': 0.01
                }
            },

            'model': {
                'embed_dim': 768,
                'num_heads': 12,
                'num_layers': 12,
                'patch_size': 32,
                'compression_ratio': 4,
                'max_keyframes': 10,
                'loss_weights': {
                    'diversity': 0.1,
                    'segmentation': 1.0,
                    'dice': 0.5,
                    'cross_entropy': 0.5
                }
            },

            'training': {
                'epochs': 200,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'warmup_epochs': 10,
                'optimizer': 'adamw',
                'scheduler': 'cosine',
                'save_interval': 10,
                'checkpoint_dir': './checkpoints/ume',
                'val_interval': 5,
                'early_stopping_patience': 20
            },

            'environment': {
                'device': 'cuda',
                'mixed_precision': True,
                'gradient_checkpointing': True,
                'distributed': False
            }
        }

        # 保存配置文件
        import yaml
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        return config
```

## 5. 使用示例

### 5.1 基本数据加载示例

```python
# 创建BraTS数据加载器
train_loader = UMEDataLoaderFactory.create_brats_loader(
    json_path="./jsons/brats21_train.json",
    batch_size=2,
    num_workers=4,
    cache_rate=0.8,
    train_mode=True
)

# 使用数据加载器
for batch_idx, batch_data in enumerate(train_loader):
    images = batch_data['images']  # shape: [batch_size, C, H, W, D]
    labels = batch_data['labels']  # shape: [batch_size, 1, H, W, D]
    metadata = batch_data['metadata']  # List of metadata dicts

    print(f"Batch {batch_idx}:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Modalities: {[m['num_modalities'] for m in metadata]}")

    # 模型训练代码
    # ...

    if batch_idx >= 5:  # 仅显示前5个批次
        break
```

### 5.2 配置文件驱动的训练

```python
import yaml

# 加载配置文件
with open('./configs/ume_brats_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建数据加载器
train_loader = UMEDataLoaderFactory.create_brats_loader(
    **config['data']['train'],
    **config['dataloader']['train']
)

val_loader = UMEDataLoaderFactory.create_brats_loader(
    **config['data']['validation'],
    **config['dataloader']['validation'],
    train_mode=False
)

# 创建模型（基于配置）
from ume_model import UMEEncoder

model = UMEEncoder(**config['model'])

# 训练循环
for epoch in range(config['training']['epochs']):
    # 训练阶段
    model.train()
    for batch_data in train_loader:
        # 训练逻辑
        pass

    # 验证阶段
    if epoch % config['training']['val_interval'] == 0:
        model.eval()
        for batch_data in val_loader:
            # 验证逻辑
            pass
```

这个数据处理流程设计确保了UME系统能够高效地处理多维度医学影像数据，同时保持与现有BraTS数据集的兼容性和可扩展性。