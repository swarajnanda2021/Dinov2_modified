# TME-DINOv2: Improving Cancer Mutation Predictions using Vision Encoders by Enforcing Tissue Microenvironment Priors

This repository contains the official implementation of TME-DINOv2, a self-supervised learning framework that incorporates tissue microenvironment (TME) priors through semantic masking augmentations for histopathology image analysis.

## Overview

TME-DINOv2 extends the DINOv2 self-supervised learning framework by introducing semantically-guided augmentations that leverage tissue microenvironment structure. Instead of relying solely on standard geometric and photometric augmentations, we generate additional views using learned or structured masks that capture biologically meaningful regions (e.g., nuclei, stroma, background).

### Key Features

- **Semantic Mask Augmentations**: Three complementary masking strategies for generating tissue-aware views
  - **Adversarial Masks (ADIOS-style)**: Learned UNet-based masks that identify semantically coherent regions
  - **CellViT Masks**: Nuclei/background separation using pre-trained segmentation
  - **Random Rectangular Masks**: Baseline random masking for comparison
  
- **Sequence Packing**: Efficient multi-crop processing using xformers block-diagonal attention, enabling variable-size crops in a single forward pass

- **Flexible Augmentation Pipeline**: Configurable number of global views, local crops, and masked views per augmentation strategy

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Image                              │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Standard DINO │     │  Mask Model     │     │    CellViT      │
│ Augmentations │     │  (Adversarial)  │     │  (Nuclei/BG)    │
└───────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Global Views  │     │  Masked Global  │     │ Nuclei/BG Views │
│ Local Crops   │     │  Masked Local   │     │ Channel Crops   │
└───────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                                ▼
                ┌───────────────────────────┐
                │    Sequence Packing       │
                │  (Block-Diagonal Attn)    │
                └───────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────┐
                │   Vision Transformer      │
                │   (Student / Teacher)     │
                └───────────────────────────┘
                                │
                                ▼
                ┌───────────────────────────┐
                │   DINO Loss (CLS Token)   │
                │   + iBOT Loss (Patches)   │
                │   + KoLeo Regularization  │
                └───────────────────────────┘
```

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/tme-dinov2.git
cd tme-dinov2

# Create conda environment
conda create -n tme-dinov2 python=3.10
conda activate tme-dinov2

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install xformers for efficient attention
pip install xformers

# Install remaining dependencies
pip install timm openslide-python pillow matplotlib scipy scikit-learn submitit
```

## Dataset Preparation

TME-DINOv2 expects datasets organized as zip files containing image patches, with an index file for efficient loading.

### Directory Structure

```
/path/to/dataset/
├── dataset_index.pkl           # Index mapping zip files to images
├── dataset_index_metadata.pkl  # Lightweight metadata
├── patches_001.zip
├── patches_002.zip
└── ...
```

### Creating Dataset Index

```python
import pickle
import zipfile
from pathlib import Path

def create_dataset_index(base_dir, output_path):
    """Create index file for dataset."""
    index = []
    for zip_path in sorted(Path(base_dir).glob("*.zip")):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            images = [f for f in zf.namelist() if f.endswith(('.png', '.webp', '.jpg'))]
            if images:
                index.append((str(zip_path), sorted(images)))
    
    with open(output_path, 'wb') as f:
        pickle.dump(index, f)
```

## Training

### Single Node Training

```bash
python main_train.py \
    --dataset_sources "DATASET:/path/to/data:dataset_index.pkl" \
    --output_dir ./outputs \
    --batch_size_per_gpu 32 \
    --total_iterations 150000 \
    --use_adversarial_mask_augmentation True \
    --mask_checkpoint /path/to/mask_model.pth \
    --num_masks 3 \
    --crops_per_mask 1
```

### Multi-Node Training with SLURM

```bash
python run_with_submitit.py \
    --nodes 2 \
    --ngpus 4 \
    --partition gpu \
    --timeout 4320
```

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--global_views` | 2 | Number of global (224×224) views |
| `--n_standard_local_crops` | 6 | Number of standard local (96×96) crops |
| `--use_adversarial_mask_augmentation` | False | Enable ADIOS-style semantic masks |
| `--num_masks` | 3 | Number of adversarial mask channels |
| `--crops_per_mask` | 1 | Local crops per mask channel |
| `--use_cellvit_augmentation` | False | Enable CellViT nuclei/background masks |
| `--cellvit_crops_per_channel` | 1 | Local crops per CellViT channel |
| `--use_random_mask_augmentation` | False | Enable random rectangular masks |
| `--batch_size_per_gpu` | 32 | Batch size per GPU |
| `--total_iterations` | 300000 | Total training iterations |
| `--lr` | 5e-4 | Base learning rate |
| `--grad_checkpointing` | False | Enable gradient checkpointing |

### Augmentation Configurations

**Standard DINOv2 (baseline):**
```bash
--global_views 2 --n_standard_local_crops 8
```

**With Adversarial Masks:**
```bash
--global_views 2 --n_standard_local_crops 6 \
--use_adversarial_mask_augmentation True \
--mask_checkpoint /path/to/adios_model.pth \
--num_masks 3 --crops_per_mask 1
```

**With CellViT Masks:**
```bash
--global_views 2 --n_standard_local_crops 6 \
--use_cellvit_augmentation True \
--cellvit_checkpoint /path/to/cellvit_model.pth \
--cellvit_crops_per_channel 1
```

**Combined (Adversarial + CellViT):**
```bash
--global_views 2 --n_standard_local_crops 4 \
--use_adversarial_mask_augmentation True \
--mask_checkpoint /path/to/adios_model.pth \
--num_masks 3 --crops_per_mask 0 \
--use_cellvit_augmentation True \
--cellvit_checkpoint /path/to/cellvit_model.pth \
--cellvit_crops_per_channel 1
```

## Model Architecture

### Vision Transformer Configurations

| Model | Embedding Dim | Depth | Heads | Parameters |
|-------|--------------|-------|-------|------------|
| ViT-S/16 | 384 | 12 | 6 | 22M |
| ViT-B/16 | 768 | 12 | 12 | 86M |
| ViT-L/16 | 1024 | 24 | 16 | 307M |

Configure via:
```bash
--patch_size 16 --embeddingdim 768 --vitdepth 12 --vitheads 12
```

## Checkpoints and Inference

### Loading a Trained Model

```python
import torch
from models.vision_transformer import VisionTransformer

# Load checkpoint
checkpoint = torch.load('checkpoint.pth', map_location='cpu')
args = checkpoint['args']

# Reconstruct model
model = VisionTransformer(
    img_size=224,
    patch_size=args.patch_size,
    embed_dim=args.embeddingdim,
    depth=args.vitdepth,
    num_heads=args.vitheads,
    mlp_ratio=4.0,
    qkv_bias=True,
    num_register_tokens=4,
)

# Load teacher weights (recommended for inference)
teacher_state = checkpoint['teacher']
backbone_state = {k.replace('module.backbone.', ''): v 
                  for k, v in teacher_state.items() 
                  if k.startswith('module.backbone.')}
model.load_state_dict(backbone_state, strict=False)
model.eval()
```

### Feature Extraction

```python
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

def extract_features(model, image_path, device='cuda'):
    """Extract CLS token features from an image."""
    # Load and preprocess
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    
    # Normalize (histopathology statistics)
    mean = np.array([0.6816, 0.5640, 0.7232])
    std = np.array([0.1617, 0.1714, 0.1389])
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - mean) / std
    
    # To tensor
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(tensor)
        cls_token = output['clstoken']  # [1, embed_dim]
        patch_tokens = output['patchtokens']  # [1, 196, embed_dim]
    
    return cls_token, patch_tokens
```

## Repository Structure

```
tme-dinov2/
├── configs/
│   ├── __init__.py
│   └── config.py              # Argument parser and training config
├── data/
│   ├── __init__.py
│   ├── datasets.py            # Dataset classes with sharding
│   └── transforms.py          # DINO augmentation transforms
├── losses/
│   ├── __init__.py
│   ├── dino_loss.py           # DINO CLS token loss
│   ├── ibot_loss.py           # iBOT patch token loss
│   └── koleo_loss.py          # KoLeo uniformity regularization
├── models/
│   ├── __init__.py
│   ├── dinov2_model.py        # Combined student/teacher model
│   └── vision_transformer/
│       ├── __init__.py
│       ├── modern_vit.py      # ViT with xformers and sequence packing
│       └── auxiliary_models.py # DINO head, mask models, CellViT
├── training/
│   ├── __init__.py
│   ├── trainer.py             # Main training loop
│   └── helpers.py             # Mask loading, visualization utilities
├── visualizations/
│   ├── plot_and_save_loss.py  # Training loss plots
│   └── plot_pca.py            # PCA visualization of features
├── main_train.py              # Entry point for training
├── run_with_submitit.py       # SLURM job submission
├── utils.py                   # Distributed training utilities
└── README.md
```

## Visualizations

### Training Loss Curves

```bash
python visualizations/plot_and_save_loss.py \
    --base_path "./logs/*_log.out" \
    --output_path training_losses.png
```

### PCA Feature Visualization

```bash
python visualizations/plot_pca.py \
    --checkpoint ./logs/checkpoint.pth \
    --output_dir ./pca_visualizations \
    --n_regions 3
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{tme-dinov2,
  title={Tissue Microenvironment as Prior for Self-Supervised Learning in Histopathology},
  author={},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## Acknowledgements

This work builds upon:

- [DINOv2](https://github.com/facebookresearch/dinov2) - Self-supervised vision transformers
- [DINO](https://github.com/facebookresearch/dino) - Self-distillation with no labels
- [iBOT](https://github.com/bytedance/ibot) - Image BERT pre-training
- [ADIOS](https://github.com/YugeTen/adios) - Adversarial masking for self-supervised learning
- [CellViT](https://github.com/TIO-IKIM/CellViT) - Cell segmentation in histopathology
- [xformers](https://github.com/facebookresearch/xformers) - Memory-efficient attention

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2024-2025

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```