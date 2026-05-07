# S2P — Semantic Segmentation Pipeline

> **ResNet50 → ASPP → ResNet50-Style Decoder** trained on two datasets:
> Pascal VOC 2012 and Syn-Mediverse (healthcare synthetic scenes).
> Built as part of the **Neuro Nexus** autonomous hospital navigation system.

---

## Architecture

Both models share the **exact same architecture**. Only the dataset, loss, and training config differ.

```
Input Image
    │
    ▼
ResNet50 Encoder (ImageNet pretrained, dilated)
    ├── layer0  →  /4
    ├── layer1  →  /4,  256ch   (low-level features)
    ├── layer2  →  /8,  512ch   (mid-level features)
    ├── layer3  →  /16, 1024ch  (dilation=2, no stride)
    └── layer4  →  /16, 2048ch  (dilation=4, no stride)
                        │
                        ▼
              ASPP Module (Atrous Spatial Pyramid Pooling)
              dilations: [1, 6, 12, 18] + Global Avg Pool
              → 256 output channels
                        │
                        ▼
         ResNet50-Style Decoder (skip connections)
              low_proj  : 256ch → 48ch
              mid_proj  : 512ch → 96ch
              Concat + 3× Residual Blocks (400→256→256→128)
                        │
                        ▼
              1×1 Conv → Bilinear Upsample to input size
                        │
                        ▼
              Segmentation Map (H × W × num_classes)
```

**Key design choices:**
- Layers 3 & 4 use **dilation instead of stride** — maintains spatial resolution without losing receptive field
- **Skip connections** from layer1 (low) and layer2 (mid) are fused into the decoder
- Backbone and head trained at **different learning rates** (10× lower for backbone)

---

## Dataset Details

### Pascal VOC 2012
| Property | Value |
|---|---|
| Classes | 21 (20 objects + background) |
| Train set | 1,464 (VOC) + 8,498 (SBD augmented) = **10,582** |
| Val set | 1,449 |
| Image size | 512 × 512 |
| Auto-download | Yes (torchvision) |

### Syn-Mediverse
| Property | Value |
|---|---|
| Classes | 21 (floor, wall, bed, IV stand, wheelchair, surgical robot, patient, staff, etc.) |
| Source | [Uni Freiburg RA-L 2024](https://syn-mediverse.cs.uni-freiburg.de/) |
| Format | COCO-style segmentation masks (0–20, 255=void) |
| Image size | 640 × 640 |
| Download | Manual (see setup below) |

---

## Training Parameters

| Parameter | Pascal VOC | Syn-Mediverse |
|---|---|---|
| Image size | 512 | 640 |
| Batch size | 8 | 6 (×2 grad accum = eff. 12) |
| Epochs | 100 | 150 |
| Warmup epochs | 5 | 8 |
| LR (head) | 1e-4 | 1e-4 |
| LR (backbone) | 1e-5 | 1e-5 |
| Weight decay | 1e-4 | 1e-4 |
| LR schedule | Poly decay + warmup | Cosine annealing + warmup |
| Optimizer | AdamW | AdamW |
| Mixed precision (AMP) | No | Yes |

### Loss Functions

**Pascal VOC:** `0.7 × OHEM-CrossEntropy + 0.3 × Dice`

**Syn-Mediverse:** `0.7 × Class-Weighted OHEM-CE (label_smoothing=0.05) + 0.3 × Dice`
- Class-frequency weights computed from 30% of training masks at startup
- Critical for handling background-dominated medical scenes

### Syn-Mediverse Extras
- **AMP** (`torch.cuda.amp`) — 1.5–2× faster training
- **Multi-scale TTA** at validation — scales [0.75, 1.0, 1.25]
- **Progressive backbone unfreezing:**
  - Epoch 0: unfreeze layer4
  - Epoch 20: unfreeze layer3
  - Epoch 40: unfreeze layer2
  - Epoch 60: unfreeze layer1 + layer0 (full backbone)

### Augmentations
| Augmentation | Pascal VOC | Syn-Mediverse |
|---|---|---|
| Random horizontal flip | ✓ | ✓ |
| Random vertical flip | ✗ | ✓ (p=0.3) |
| Random scale [0.5, 2.0] | ✓ | ✓ |
| Pad + random crop | ✓ | ✓ |
| Color jitter | ✓ | ✓ |
| Random rotation | ±10° | ±15° |
| Gaussian blur | ✓ (p=0.5) | ✓ (p=0.5) |
| Random grayscale | ✗ | ✓ (p=0.15, simulates low-light OR) |

---

## Setup

### 1. Clone & create environment

```bash
git clone https://github.com/SiDhU31/s2p-semantic-segmentation.git
cd s2p-semantic-segmentation

conda create -n s2p python=3.10 -y
conda activate s2p
```

### 2. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python pillow matplotlib seaborn tqdm
```

### 3. Dataset setup

**Pascal VOC** — auto-downloaded by torchvision on first run. No extra steps needed.

**Syn-Mediverse** — manual download:
```bash
# Option A: shell script
bash download_mediverse.sh /data/synmed --img --label

# Option B: download syn-mediverse_coco_format.zip from
# https://syn-mediverse.cs.uni-freiburg.de/
# Extract so layout is:
#
# /your/path/syn-mediverse/
#   train/                  ← RGB images
#   val/
#   stuffthingmaps/
#     train/                ← grayscale masks (0-20, 255=void)
#     val/
```

Then update `data_dir` in `s2p_synmediverse.py`:
```python
CFG = {
    "data_dir": "/your/path/syn-mediverse",   # ← change this line
    ...
}
```

---

## How to Run

### Pascal VOC 2012

```bash
conda activate s2p
cd /path/to/s2p-semantic-segmentation

python s2p_pascalvoc.py
```

What happens step by step:
1. Pascal VOC 2012 auto-downloads to `output1/VOCdata/`
2. SBD dataset auto-downloads to `output1/SBDdata/`
3. Combined train set: 10,582 masks
4. Trains 100 epochs, poly LR with 5-epoch warmup
5. Best checkpoint → `output1/experiment/checkpoints/best_model.pth`
6. Training graphs saved every 5 epochs to `output1/experiment/graphs/`
7. Full logs at `output1/experiment/logs/training.log`

---

### Syn-Mediverse

```bash
conda activate s2p
cd /path/to/s2p-semantic-segmentation

python s2p_synmediverse.py
```

What happens step by step:
1. Loads dataset from `CFG["data_dir"]` (must be set manually — see setup)
2. Scans 30% of training masks to compute class-frequency weights
3. Backbone is frozen initially; progressive unfreezing starts at epoch 0
4. Trains 150 epochs with cosine annealing LR + AMP + gradient accumulation
5. TTA (3 scales: 0.75, 1.0, 1.25) runs at every validation pass
6. Best checkpoint → `output_synmed/experiment/checkpoints/best_model.pth`
7. Graphs → `output_synmed/experiment/graphs/`

---

## Output Structure

```
output1/experiment/          ← Pascal VOC outputs
output_synmed/experiment/    ← Syn-Mediverse outputs
├── checkpoints/
│   ├── best_model.pth       ← best val mIoU checkpoint
│   └── latest_model.pth     ← most recent epoch
├── graphs/
│   ├── loss_miou_ep5.png
│   ├── metrics_dashboard_ep5.png
│   ├── per_class_iou_ep5.png
│   ├── confusion_matrix_ep5.png
│   ├── lr_schedule_ep5.png
│   └── research_summary_FINAL.png   ← 4-panel publication figure
└── logs/
    ├── training.log
    ├── training_history.json
    └── val_stats.json
```

---

## Loading a Checkpoint

```python
import torch

ckpt = torch.load("best_model.pth")
# keys: epoch, model_state, optimizer_state, miou, stats, config

model.load_state_dict(ckpt["model_state"])
print(f"Best mIoU: {ckpt['miou']:.2f}% at epoch {ckpt['epoch']}")
```

---

## Hardware Used

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4060 |
| Framework | PyTorch |
| OS | Ubuntu 22.04 |
| Conda env | `s2p` (Python 3.10) |

---

## Project Context

S2P is the perception module for **Neuro Nexus** — an autonomous hospital navigation system (TurtleBot4 + Intel RealSense D435i + ROS2 Humble). The trained SynMediverse model is deployed as a ROS2 node, feeding semantic costmaps into Nav2.

---

## Team

**Amrita School of Engineering, Coimbatore — B.Tech Automation & Robotics, 2026**

| Name | Role |
|---|---|
| Sidharth (CB.EN.U4ARE22046) | Architecture, Training, ROS2 Integration |
| Parvathy Vinod K V | Dataset Curation, Annotation |
| Sanjit R K | YOLO+SAM Mask Pipeline, Evaluation |

Internal Guide: Dr. G. Sivasankar | External Guide: Dr. Ankit Ravankar (Tohoku University)
