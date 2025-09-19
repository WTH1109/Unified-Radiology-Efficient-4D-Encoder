# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **UME (Unified Medical imaging Efficient 4D-Encoder)** project - a complete medical imaging AI system designed for BraTS brain tumor segmentation. The project implements an adaptive multi-modal dynamic sampling encoder for 4D medical imaging data.

**Key Innovation**: Adaptive multi-modal dynamic sampling encoder that intelligently selects keyframes and performs hierarchical modal fusion, enabling efficient processing of any number of modalities with dynamic frame selection.

## Core Architecture

### Architectural Patterns
- **Modular Design**: Clear separation between data (`ume/data/`), models (`ume/models/`), training (`ume/training/`), and utilities (`ume/utils/`)
- **Dynamic Sampling**: Adaptive keyframe selection that intelligently chooses the most relevant frames
- **Multi-Modal Fusion**: Hierarchical fusion of arbitrary number of modalities with Intra/Inter/Global fusion levels
- **Transformer-Based**: Vision Transformer (ViT) backbone with medical imaging adaptations
- **Configuration-Driven**: JSON-based configs for different training modes and model sizes
- **Supervised Learning**: Traditional supervised training with segmentation supervision

### Key Dependencies
- **PyTorch ≥2.0.0**: Core deep learning framework
- **MONAI ≥1.3.0**: Medical imaging AI toolkit providing transforms, datasets, and metrics
- **timm**: Pre-trained vision models and transformer implementations
- **transformers**: Hugging Face transformer architectures
- **nibabel**: Medical image format handling (.nii.gz files)
- **tensorboard**: Training visualization and logging

### Data Flow Design
The system follows this precise pipeline:
1. **Input**: BraTS data (4, 256, 256, 128) - 4 modalities, complete 128 frames
2. **Keyframe Selection**: Dynamic sampling → (4, 256, 256, K) where K≤10 frames
3. **Patch Organization**: 256×256 → 64 patches of 32×32 → (4, 64, 32, 32, K)
4. **Multi-Modal Fusion**: Hierarchical fusion across modalities and frames
5. **Segmentation**: Standard supervised training with segmentation targets

### Key Components

- **`ume/models/ume_model.py`**: Main UME model (118M parameters, 451MB)
  - Integrates keyframe selection, modal fusion, and supervised training
  - Supports both keyframe and frame-by-frame inference modes

- **`ume/core/keyframe_selector.py`**: Intelligent frame selection (246 lines)
  - Multi-strategy: uniform + content-aware + attention-guided sampling
  - Adaptive weighting and diversity constraints
  - Med-CLIP integration for medical image understanding

- **`ume/core/modal_fusion.py`**: Multi-level modal fusion (333 lines)
  - Intra-Modal: within-modality feature fusion
  - Inter-Modal: cross-modality information exchange
  - Global: overall feature enhancement

- **`ume/training/train_ume.py`**: Complete training pipeline (436 lines)
  - Supervised training with segmentation targets
  - Frame-by-frame validation for testing
  - Multi-modal loss function integration

## Development Workflow

### Environment Setup
```bash
# Create and activate environment
conda create -n ume python=3.9
conda activate ume

# Install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# Verify setup and get training options
./quick_start.sh
```

### Development Commands
```bash
# Quick environment validation and interactive training
./quick_start.sh

# Automated training startup with environment checks
./start_training.sh

# Test Med-CLIP integration and keyframe selector
python test_medclip.py

# Development iteration with fast config
python ume/training/train_ume.py --config configs/brats_ume_fast.json
```

### Main Training Commands
```bash
# Main Supervised Training (Primary/Recommended)
python ume/training/train_ume.py --config configs/brats_ume_main.json

# Alternative Configuration (Same as main)
python ume/training/train_ume.py --config configs/brats_ume_config.json

# Fast Testing Configuration (Smaller model)
python ume/training/train_ume.py --config configs/brats_ume_fast.json

# Distributed Training
python ume/training/train_ume.py --config configs/brats_ume_main.json --distributed

# Resume from Checkpoint
python ume/training/train_ume.py --config configs/brats_ume_main.json --resume /path/to/checkpoint.pth
```

## Configuration Files

### Main Training Config (`configs/brats_ume_main.json`)
- **Primary training mode** - supervised training with segmentation targets
- Loss weights: `segmentation: 1.0, diversity: 0.1, consistency: 0.05, temporal: 0.03`
- Full model: 768 embed_dim, 12 layers, 10 max keyframes
- Requires ground truth segmentation labels

### Alternative Config (`configs/brats_ume_config.json`)
- Same as main config - alternative training configuration
- Uses segmentation supervision with standard parameters

### Fast Config (`configs/brats_ume_fast.json`)
- Smaller model for testing: reduced embed_dim, layers, and frames
- Faster iterations for development/debugging

## Training Process Specifics

### Training Pipeline
1. **Complete Data Loading**: Always read full (4, 256, 256, 128) volumes
2. **Keyframe Selection**: Network intelligently selects K≤10 key frames
3. **Patch Extraction**: 256×256 → 64 patches of 32×32 each
4. **Multi-Modal Fusion**: Hierarchical fusion across modalities and frames
5. **Segmentation Training**: Standard supervised training with ground truth masks

### Training Metrics Display
```
Training Epoch 0: 62%|██████▏| 312/500 [03:00<01:30, loss=1.2345, seg=1.0123, div=0.1111, cons=0.0567]
```
- **loss**: Total loss
- **seg**: Segmentation loss
- **div**: Keyframe diversity loss
- **cons**: Consistency loss

### Validation Metrics
```
Validation Epoch 1: 89%|████████▉| 112/126 [01:01<00:13, loss=1.1234, dice=0.8567, iou=0.7890]
```
Validation computes segmentation metrics including Dice score and IoU.

## Data Requirements

### BraTS Dataset Structure
```
/path/to/brats21/
├── BraTS21_Training_001/
│   ├── BraTS21_Training_001_t1.nii.gz
│   ├── BraTS21_Training_001_t2.nii.gz
│   ├── BraTS21_Training_001_t1ce.nii.gz
│   ├── BraTS21_Training_001_flair.nii.gz
│   └── BraTS21_Training_001_seg.nii.gz  # Required for training
```

### Configuration Update
Update `data_dir` in config files to point to your BraTS dataset:
```json
{
  "data": {
    "data_dir": "/path/to/your/brats21",
    "target_size": [256, 256, 128]  # Always complete 128 frames
  }
}
```

## Technical Implementation Details

### Frame Selection Strategy
- **NO frame sampling during data loading** - always load complete 128 frames
- Intelligent keyframe selector network performs sampling internally
- Multiple strategies combined: uniform, content-aware, attention-guided

### Training Mode
- **Supervised Training**: Uses ground truth segmentation masks
  - Standard medical imaging training approach with segmentation supervision
  - Multi-loss training including diversity, consistency, and temporal losses

### Memory Optimization
- **Standard Config**: ~8GB GPU memory required
- **CacheDataset**: Pre-loads data to memory for faster training
- **Batch Size**: Recommend 2-4 for optimal training performance

## Testing and Validation

### Testing Infrastructure
```bash
# Test Med-CLIP integration and keyframe selector functionality
python test_medclip.py

# Quick environment validation (automated in quick_start.sh)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import monai; print(f'MONAI version: {monai.__version__}')"
```

### Development Testing Strategy
- **Integration Tests**: `test_medclip.py` validates core components
- **Fast Iteration**: Use `brats_ume_fast.json` for quick validation cycles
- **Environment Validation**: `quick_start.sh` provides comprehensive environment checks
- **No Formal Unit Tests**: Project relies on integration testing and training validation

### Frame-by-Frame Inference
The model supports two inference modes:
1. **Keyframe Inference**: Fast, uses selected key frames
2. **Frame-by-Frame**: Slower but more accurate, processes each frame individually

### Model Checkpoints
- Automatically saves best model based on dice score improvement
- Regular checkpoints every 10 epochs
- Resume training with `--resume` flag

### Debugging and Performance

#### Memory Management
- **Standard Config**: ~8GB GPU memory required
- **Memory Issues**: Reduce `batch_size` from 2 to 1, use `brats_ume_fast.json` config
- **Cache Management**: Set `"use_cache": false` in config if memory constrained
- **Distributed Training**: Use `--distributed` flag for multi-GPU setups

#### Common Development Issues
```bash
# Data loading too slow on first run
# Solution: CacheDataset pre-processes data, subsequent runs are faster

# CUDA out of memory
# Solution: Reduce batch_size in config, use fast config, or disable caching

# Dimension mismatch errors
# Solution: Ensure crop sizes match ViT patch dimensions (32×32)

# Missing BraTS data
# Solution: Update data_dir in config files, ensure .nii.gz files present
```

#### Performance Optimization
- **Fast Development**: Use `configs/brats_ume_fast.json` (384 embed_dim, 6 layers)
- **Production**: Use `configs/brats_ume_main.json` (768 embed_dim, 12 layers)
- **Caching**: Enable for faster training after initial data processing
- **Batch Size**: Start with 2, reduce to 1 if memory issues occur

## Development Notes

### Key Design Principles
1. **Complete Data Input**: Never sample frames during data loading
2. **Network-Based Sampling**: Let keyframe selector handle intelligent sampling
3. **Supervised Training**: Primary training paradigm with segmentation supervision
4. **Multi-Modal Fusion**: Leverage all 4 BraTS modalities effectively

### Common Issues
- **Slow Initial Loading**: CacheDataset pre-processes all data on first run
- **Memory Requirements**: Adjust batch_size and model parameters if GPU memory limited
- **Dimension Mismatches**: Ensure crop sizes match ViT patch dimensions (32×32)

### Code Organization
- **`ume/`**: Core implementation modules
  - `core/`: Keyframe selection and modal fusion
  - `data/`: BraTS data loading and preprocessing
  - `models/`: UME model architecture
  - `training/`: Training pipeline and scripts
  - `utils/`: Loss functions, metrics, and supervision utilities
- **`configs/`**: Training configurations for different modes
- **`jsons/`**: Dataset fold definitions (auto-generated)
- **`checkpoints/`**: Model checkpoints and saved states
- **`logs/`**: Training logs and tensorboard outputs
- **Scripts**: `quick_start.sh`, `start_training.sh`, `test_medclip.py`

### Configuration Management
Each config file serves a specific purpose:
- **`brats_ume_main.json`**: Production supervised training
- **`brats_ume_config.json`**: Traditional supervised training for comparison
- **`brats_ume_fast.json`**: Development/testing with smaller model
- **`brats_ume_test.json`**: Additional test configuration

Critical config parameters:
- `data_dir`: Must point to your BraTS dataset location
- `batch_size`: Adjust based on GPU memory (start with 2, reduce to 1 if needed)
- `embed_dim`/`num_layers`: Model size parameters (768/12 for production, 384/6 for development)
- `max_keyframes`: Maximum frames to select (10 for production, 6 for fast testing)

The project is production-ready with verified supervised training pipeline and complete documentation.