# S2P — Semantic Segmentation For Perception

> **ResNet50 → ASPP → ResNet50-Style Decoder** trained on two datasets:
> Pascal VOC 2012 and Syn-Mediverse (healthcare synthetic scenes).
> Built as part of autonomous hospital navigation system.

---

## Architecture

Both models share the **exact same architecture**. Only the dataset, loss, and training config differ.




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


---

## Setup

### 1. Clone & create environment

```bash
git clone https://github.com/SiDhU31/s2p-semantic-segmentation.git
cd s2p-semantic-segmentation

conda create -n s2p python=3.10 -y
conda activate s2p ( change according to your workbench )
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

---

## Hardware Used

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX A4000 |
| Framework | PyTorch |
| OS | Ubuntu 22.04 |
| Conda env | `s2p` (Python 3.10) |

---

## Project Context

S2P is the perception module for an autonomous hospital navigation system (TurtleBot4 + Intel RealSense D435i + ROS2 Humble). T

---

## Team

**Amrita School of Engineering, Coimbatore — B.Tech Automation & Robotics, 2026**


