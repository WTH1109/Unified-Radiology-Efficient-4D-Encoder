# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **UME (Unified Medical imaging Efficient 4D-Encoder)** project - a complete medical imaging AI system designed for BraTS brain tumor segmentation. The project implements VoCo (Volume-based Context) self-supervised training and multi-modal fusion for 4D medical imaging data.

**Key Innovation**: VoCo training uses crop-based contrastive learning instead of traditional segmentation supervision, requiring no manual labels during pre-training.

## Core Architecture

### Data Flow Design
The system follows this precise pipeline:
1. **Input**: BraTS data (4, 256, 256, 128) - 4 modalities, complete 128 frames
2. **Keyframe Selection**: Dynamic sampling → (4, 256, 256, K) where K≤10 frames
3. **Patch Organization**: 256×256 → 64 patches of 32×32 → (4, 64, 32, 32, K)
4. **VoCo Training**: Random crops (4, sw_nu, 32, 32, K) for contrastive learning

### Key Components

- **`ume/models/ume_model.py`**: Main UME model (118M parameters, 451MB)
  - Integrates keyframe selection, modal fusion, and VoCo training
  - Two forward modes: `'voco_training'` and `'inference'`

- **`ume/core/keyframe_selector.py`**: Intelligent frame selection (246 lines)
  - Multi-strategy: uniform + content-aware + attention-guided sampling
  - Adaptive weighting and diversity constraints

- **`ume/core/modal_fusion.py`**: Multi-level modal fusion (333 lines)
  - Intra-Modal: within-modality feature fusion
  - Inter-Modal: cross-modality information exchange
  - Global: overall feature enhancement

- **`ume/utils/voco_supervision.py`**: VoCo self-supervised training (292 lines)
  - Generates random crops and calculates overlap area ratios
  - Implements contrastive learning based on spatial relationships

- **`ume/training/train_ume.py`**: Complete training pipeline (436 lines)
  - Supports both VoCo and traditional training modes
  - Frame-by-frame validation for testing

## Training Commands

### Environment Setup
```bash
# Activate environment
conda activate ume

# Verify setup
./quick_start.sh
```

### Main Training Commands
```bash
# VoCo Self-Supervised Training (Primary/Recommended)
python ume/training/train_ume.py --config configs/brats_ume_voco.json

# Traditional Supervised Training (Comparison only)
python ume/training/train_ume.py --config configs/brats_ume_config.json

# Fast Testing Configuration (Smaller model)
python ume/training/train_ume.py --config configs/brats_ume_fast.json

# Distributed Training
python ume/training/train_ume.py --config configs/brats_ume_voco.json --distributed

# Resume from Checkpoint
python ume/training/train_ume.py --config configs/brats_ume_voco.json --resume /path/to/checkpoint.pth
```

## Configuration Files

### VoCo Training Config (`configs/brats_ume_voco.json`)
- **Primary training mode** - uses crop-based contrastive learning
- Loss weights: `voco: 1.0, diversity: 0.1, contrastive: 0.5, area: 0.3`
- Crop size: `[32, 32, 32]` to match ViT patch size
- No segmentation labels required

### Traditional Config (`configs/brats_ume_config.json`)
- Comparison/baseline mode - uses segmentation supervision
- Requires ground truth labels

### Fast Config (`configs/brats_ume_fast.json`)
- Smaller model for testing: reduced embed_dim, layers, and frames
- Faster iterations for development/debugging

## VoCo Training Specifics

### Training Process
1. **Complete Data Loading**: Always read full (4, 256, 256, 128) volumes
2. **Keyframe Selection**: Network intelligently selects K≤10 key frames
3. **Patch Extraction**: 256×256 → 64 patches of 32×32 each
4. **Random Crop Selection**: Choose main crop + 4 neighboring crops
5. **Contrastive Learning**: Calculate overlap area ratios as supervision signal

### Training Metrics Display
```
Training Epoch 0: 62%|██████▏| 312/500 [03:00<01:30, loss=11.0134, voco=7.3598, div=0.4443, area=0.3533]
```
- **loss**: Total loss
- **voco**: VoCo contrastive learning loss
- **div**: Keyframe diversity loss
- **area**: Area overlap prediction loss

### Validation Metrics
```
Validation Epoch 1: 89%|████████▉| 112/126 [01:01<00:13, loss=9.7183, voco=6.4930, div=0.5401]
```
VoCo validation computes the same losses on validation crops (no segmentation metrics).

## Data Requirements

### BraTS Dataset Structure
```
/path/to/brats21/
├── BraTS21_Training_001/
│   ├── BraTS21_Training_001_t1.nii.gz
│   ├── BraTS21_Training_001_t2.nii.gz
│   ├── BraTS21_Training_001_t1ce.nii.gz
│   ├── BraTS21_Training_001_flair.nii.gz
│   └── BraTS21_Training_001_seg.nii.gz  # Optional for VoCo
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

### VoCo vs Traditional Mode
- **VoCo Mode**: Self-supervised with crop contrastive learning
  - No manual labels needed during training
  - Learns spatial relationships through crop overlap ratios

- **Traditional Mode**: Supervised with segmentation labels
  - Uses ground truth segmentation masks
  - Standard medical imaging training approach

### Memory Optimization
- **Standard Config**: ~8GB GPU memory required
- **CacheDataset**: Pre-loads data to memory for faster training
- **Batch Size**: Recommend 2 for VoCo training due to memory requirements

## Testing and Validation

### Frame-by-Frame Inference
The model supports two inference modes:
1. **Keyframe Inference**: Fast, uses selected key frames
2. **Frame-by-Frame**: Slower but more accurate, processes each frame individually

### Model Checkpoints
- Automatically saves best model based on loss reduction (VoCo) or dice score (traditional)
- Regular checkpoints every 10 epochs
- Resume training with `--resume` flag

## Development Notes

### Key Design Principles
1. **Complete Data Input**: Never sample frames during data loading
2. **Network-Based Sampling**: Let keyframe selector handle intelligent sampling
3. **VoCo Self-Supervision**: Primary training paradigm, no labels needed
4. **Multi-Modal Fusion**: Leverage all 4 BraTS modalities effectively

### Common Issues
- **Slow Initial Loading**: CacheDataset pre-processes all data on first run
- **Memory Requirements**: Adjust batch_size and model parameters if GPU memory limited
- **Dimension Mismatches**: Ensure crop sizes match ViT patch dimensions (32×32)

### Code Organization
- **`ume/`**: Core implementation modules
- **`configs/`**: Training configurations
- **`jsons/`**: Dataset fold definitions (auto-generated)
- **Logs/Checkpoints**: Saved to working directory during training

The project is production-ready with verified VoCo training pipeline and complete documentation.