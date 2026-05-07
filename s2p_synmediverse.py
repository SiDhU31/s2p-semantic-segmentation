"""
Semantic Segmentation with ResNet50 Encoder-ASPP-ResNet50 Decoder
Dataset  : Syn-Mediverse (Uni Freiburg, RA-L 2024) — COCO format download
           https://syn-mediverse.cs.uni-freiburg.de/
           bash download_mediverse.sh /data/synmed --img --label
           OR download syn-mediverse_coco_format.zip directly

Architecture : ResNet50 Encoder → ASPP Bottleneck → ResNet50-style Decoder  [UNCHANGED]
Target mIoU  : >90%  (synthetic data → achievable with class-weighted loss + AMP)

Key changes vs. VOC version
─────────────────────────────────────────────────────────────────────────────
1. SynMediverseDataset  — replaces VOCSegDataset / SBDSegDataset
2. Class names / colormap updated to 21 Syn-Mediverse COCO training classes
   (update SYN_CLASSES from label_helper.py if needed)
3. Class-frequency weighted CE loss  — critical for bg-dominated medical scenes
4. CosineAnnealing LR with warmup   — better convergence than poly for synthetic
5. AMP (torch.cuda.amp)             — 1.5-2x faster => more effective epochs
6. Multi-scale TTA at validation    — squeezes 1-2% extra mIoU for free
7. Progressive backbone unfreezing  — avoids early over-fitting backbone
8. Additional augmentations         — vflip, grayscale for OR scenes
9. Resolution bumped to 640         — more detail, helps small medical objects
10. 150 epochs, gradient accumulation=2
─────────────────────────────────────────────────────────────────────────────
ARCHITECTURE IS NOT MODIFIED — ResNet50Encoder, ASPPModule,
ResNet50StyleDecoder, SegmentationModel are identical to original.
"""

import os, sys, time, json, logging, random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # ── Dataset ──────────────────────────────────────────────────────────────
    # After running:
    #   bash download_mediverse.sh /data/synmed --img --label
    # the COCO-format zip extracts to a structure like:
    #   data_dir/
    #     train/           <- RGB images
    #     val/
    #     stuffthingmaps/
    #       train/         <- segmentation masks (0-20, 255=void)
    #       val/
    #
    # Update data_dir to your actual extract path.
    "data_dir": "/home/roboticslab/datasets/syn-mediverse",   # <- CHANGE THIS

    # ── Model ─────────────────────────────────────────────────────────────────
    "num_classes":   21,            # Syn-Mediverse COCO format: 21 training classes
    "ignore_index":  255,
    "img_size":      640,           # up from 512 — helps small medical objects

    # ── Training ──────────────────────────────────────────────────────────────
    "batch_size":    6,             # effective=12 with grad_accum=2
    "grad_accum":    2,
    "num_epochs":    150,
    "warmup_epochs": 8,
    "lr":            1e-4,
    "weight_decay":  1e-4,
    "use_amp":       True,

    # ── ASPP (unchanged from original) ────────────────────────────────────────
    "aspp_dilations":     [1, 6, 12, 18],
    "aspp_out_channels":  256,
    "decoder_channels":   256,
    "dropout":            0.1,

    # ── Loss ──────────────────────────────────────────────────────────────────
    "ohem_thresh":     0.7,
    "ohem_min_kept":   100_000,
    "dice_alpha":      0.3,         # weight for Dice term in combined loss
    "label_smoothing": 0.05,

    # ── Progressive backbone unfreezing ───────────────────────────────────────
    # Backbone is frozen at epoch 0, gradually opened to avoid overfitting
    "unfreeze_schedule": {
        0:  ["layer4"],             # epoch 0:  unfreeze only layer4
        20: ["layer3"],             # epoch 20: also unfreeze layer3
        40: ["layer2"],             # epoch 40: also unfreeze layer2
        60: ["layer1", "layer0"],   # epoch 60: full backbone
    },

    # ── TTA ───────────────────────────────────────────────────────────────────
    "use_tta":    True,
    "tta_scales": [0.75, 1.0, 1.25],

    # ── Misc ──────────────────────────────────────────────────────────────────
    "seed":         42,
    "save_dir":     "output_synmed/experiment",
    "log_interval": 20,
    "val_interval": 1,
}

# ─────────────────────────────────────────────────────────────────────────────
# SYN-MEDIVERSE CLASS DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
# 21 COCO-format training classes (indices 0-20).
# Update names from label_helper.py once downloaded.
SYN_CLASSES = [
    'background',        # 0
    'floor',             # 1
    'wall',              # 2
    'ceiling',           # 3
    'door',              # 4
    'window',            # 5
    'bed',               # 6
    'chair',             # 7
    'table',             # 8
    'cabinet',           # 9
    'monitor_screen',    # 10
    'iv_stand',          # 11
    'wheelchair',        # 12
    'medical_cart',      # 13
    'surgical_robot',    # 14
    'surgical_tool',     # 15
    'patient',           # 16
    'staff',             # 17
    'oxygen_tank',       # 18
    'medical_equipment', # 19
    'misc_object',       # 20
]

SYN_COLORMAP = [
    [0,   0,   0  ],  # background
    [128, 64,  128],  # floor
    [244, 35,  232],  # wall
    [70,  70,  70 ],  # ceiling
    [102, 102, 156],  # door
    [190, 153, 153],  # window
    [153, 153, 153],  # bed
    [250, 170, 30 ],  # chair
    [220, 220, 0  ],  # table
    [107, 142, 35 ],  # cabinet
    [152, 251, 152],  # monitor_screen
    [70,  130, 180],  # iv_stand
    [220, 20,  60 ],  # wheelchair
    [255, 0,   0  ],  # medical_cart
    [0,   0,   142],  # surgical_robot
    [0,   0,   70 ],  # surgical_tool
    [0,   60,  100],  # patient
    [0,   80,  100],  # staff
    [0,   0,   230],  # oxygen_tank
    [119, 11,  32 ],  # medical_equipment
    [64,  128, 128],  # misc_object
]

for sub in ["checkpoints", "graphs", "logs", "visualizations"]:
    os.makedirs(os.path.join(CFG["save_dir"], sub), exist_ok=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(CFG["seed"])

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
log_path = os.path.join(CFG["save_dir"], "logs", "training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─────────────────────────────────────────────────────────────────────────────
# SYN-MEDIVERSE DATASET
# ─────────────────────────────────────────────────────────────────────────────
class Augmentations:
    """Joint augmentation — same API as original, extended for OR scenes."""

    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train

    def __call__(self, img, mask):
        return self._train_transform(img, mask) if self.is_train \
               else self._val_transform(img, mask)

    def _train_transform(self, img, mask):
        # 1. Random horizontal flip
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        # 2. Random vertical flip (valid for OR/top-down views)
        if random.random() > 0.7:
            img  = TF.vflip(img)
            mask = TF.vflip(mask)

        # 3. Random scale [0.5, 2.0]
        scale = random.uniform(0.5, 2.0)
        w, h  = img.size
        nw, nh = int(w * scale), int(h * scale)
        img  = TF.resize(img,  (nh, nw), interpolation=Image.BILINEAR)
        mask = TF.resize(mask, (nh, nw), interpolation=Image.NEAREST)

        # 4. Pad + random crop
        img, mask = self._random_crop_or_pad(img, mask)

        # 5. Color jitter
        img = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(img)

        # 6. Random rotation +-15 deg
        angle = random.uniform(-15, 15)
        img  = TF.rotate(img,  angle, interpolation=Image.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=Image.NEAREST, fill=255)

        # 7. Gaussian blur
        if random.random() > 0.5:
            img = img.filter(
                __import__('PIL').ImageFilter.GaussianBlur(
                    radius=random.uniform(0.1, 2.0)))

        # 8. Random grayscale — simulates low-light OR cameras
        if random.random() > 0.85:
            img = TF.to_grayscale(img, num_output_channels=3)

        return img, mask

    def _val_transform(self, img, mask):
        s = self.img_size
        img  = TF.resize(img,  (s, s), interpolation=Image.BILINEAR)
        mask = TF.resize(mask, (s, s), interpolation=Image.NEAREST)
        return img, mask

    def _random_crop_or_pad(self, img, mask):
        s = self.img_size
        w, h = img.size
        ph, pw = max(s - h, 0), max(s - w, 0)
        if ph > 0 or pw > 0:
            img  = TF.pad(img,  (0, 0, pw, ph), fill=0)
            mask = TF.pad(mask, (0, 0, pw, ph), fill=255)
        w, h = img.size
        x = random.randint(0, w - s)
        y = random.randint(0, h - s)
        return TF.crop(img, y, x, s, s), TF.crop(mask, y, x, s, s)


def to_tensor_normalize(img, mask):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    img  = TF.to_tensor(img)
    img  = TF.normalize(img, mean, std)
    mask = torch.from_numpy(np.array(mask)).long()
    return img, mask


class SynMediverseDataset(Dataset):
    """
    Syn-Mediverse COCO-format dataset loader.

    Expected directory layout (after extracting syn-mediverse_coco_format.zip):
        {root}/
          train/                <- RGB .jpg/.png images
          val/
          stuffthingmaps/
            train/              <- grayscale .png masks (0-20 classes, 255=void)
            val/

    If your layout differs (e.g. from the shell-script download), point
    img_subdir and mask_subdir to the correct relative paths.
    """

    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.PNG', '.JPG')

    def __init__(self, root, split='train', img_size=640,
                 img_subdir=None, mask_subdir=None):
        assert split in ('train', 'val', 'test')
        self.aug  = Augmentations(img_size, is_train=(split == 'train'))
        root      = Path(root)
        img_dir   = root / (img_subdir  or split)
        mask_dir  = root / (mask_subdir or f'stuffthingmaps/{split}')

        img_paths = sorted([p for p in img_dir.iterdir()
                            if p.suffix in self.IMG_EXTS])
        if not img_paths:
            raise FileNotFoundError(
                f"No images in {img_dir}. Check data_dir.")

        self.pairs = []
        for ip in img_paths:
            mp = mask_dir / (ip.stem + '.png')
            if mp.exists():
                self.pairs.append((ip, mp))
            else:
                logger.warning(f"Mask missing for {ip.name}, skipping.")

        logger.info(f"  Syn-Mediverse [{split}]: {len(self.pairs)} pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        img, mask = self.aug(img, mask)
        return to_tensor_normalize(img, mask)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS FREQUENCY WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(dataset, num_classes, ignore_index=255,
                          sample_frac=0.3, dev='cpu'):
    """
    Scan ~sample_frac of training masks and compute inverse-frequency class
    weights. Handles the extreme background-dominance in medical scene datasets.
    """
    logger.info("  Computing class frequency weights ...")
    counts  = np.zeros(num_classes, dtype=np.float64)
    n       = max(1, int(len(dataset) * sample_frac))
    indices = random.sample(range(len(dataset)), n)
    for i in indices:
        _, mask = dataset[i]
        m = mask.numpy()
        valid = m[m != ignore_index]
        for c in range(num_classes):
            counts[c] += (valid == c).sum()
    freq    = counts / (counts.sum() + 1e-8)
    weights = 1.0 / (freq + 1e-4)
    weights = weights / weights.mean()
    weights = torch.tensor(weights, dtype=torch.float32).to(dev)
    logger.info(f"  Weights — min:{weights.min():.2f}  max:{weights.max():.2f}  mean:1.00")
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  <-- UNCHANGED FROM ORIGINAL
# ─────────────────────────────────────────────────────────────────────────────
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, stride,
                      padding=dilation if dilation > 1 else padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)


class ASPPModule(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256,
                 dilations=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        self.branches.append(
            ConvBNReLU(in_channels, out_channels, k=1, padding=0, dilation=1))
        for d in dilations[1:]:
            self.branches.append(
                ConvBNReLU(in_channels, out_channels, k=3, dilation=d))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels, out_channels, k=1, padding=0)
        )
        self.project = nn.Sequential(
            ConvBNReLU((len(dilations) + 1) * out_channels, out_channels,
                       k=1, padding=0),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        h, w  = x.shape[2:]
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.gap(x), (h, w),
                                   mode='bilinear', align_corners=True))
        return self.project(torch.cat(feats, dim=1))


class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        bb = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.layer0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self._make_dilated(self.layer3, stride=1, dilation=2)
        self._make_dilated(self.layer4, stride=1, dilation=4)

    def _make_dilated(self, layer, stride, dilation):
        for m in layer.modules():
            if isinstance(m, nn.Conv2d):
                if m.stride == (2, 2):
                    m.stride = (stride, stride)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilation, dilation)
                    m.padding  = (dilation, dilation)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class ResNet50StyleDecoder(nn.Module):
    def __init__(self, aspp_channels=256, num_classes=21):
        super().__init__()
        self.low_proj = ConvBNReLU(256, 48,  k=1, padding=0)
        self.mid_proj = ConvBNReLU(512, 96,  k=1, padding=0)
        fuse_in = aspp_channels + 48 + 96
        self.fuse1 = self._make_resblock(fuse_in, 256)
        self.fuse2 = self._make_resblock(256, 256)
        self.fuse3 = self._make_resblock(256, 128)
        self.dropout    = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def _make_resblock(self, in_c, out_c):
        layers   = nn.Sequential(
            ConvBNReLU(in_c,  out_c, k=1, padding=0),
            ConvBNReLU(out_c, out_c, k=3, padding=1))
        shortcut = (nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False),
                                   nn.BatchNorm2d(out_c))
                    if in_c != out_c else nn.Identity())
        class ResBlock(nn.Module):
            def forward(self_, x):
                return F.relu(layers(x) + shortcut(x), inplace=True)
        b = ResBlock(); b.layers = layers; b.shortcut = shortcut
        return b

    def forward(self, low_feat, mid_feat, aspp_feat, target_size):
        aspp_up = F.interpolate(aspp_feat, size=low_feat.shape[2:],
                                mode='bilinear', align_corners=True)
        mid_up  = F.interpolate(self.mid_proj(mid_feat),
                                size=low_feat.shape[2:],
                                mode='bilinear', align_corners=True)
        low_p   = self.low_proj(low_feat)
        x = torch.cat([aspp_up, low_p, mid_up], dim=1)
        x = self.fuse1(x); x = self.fuse2(x); x = self.fuse3(x)
        x = self.dropout(x); x = self.classifier(x)
        return F.interpolate(x, size=target_size,
                             mode='bilinear', align_corners=True)


class SegmentationModel(nn.Module):
    """ResNet50 -> ASPP -> ResNet50Decoder  [UNCHANGED]"""
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained=pretrained)
        self.aspp    = ASPPModule(in_channels=2048, out_channels=256,
                                   dilations=CFG["aspp_dilations"])
        self.decoder = ResNet50StyleDecoder(aspp_channels=256,
                                             num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        low, mid, _, high = self.encoder(x)
        return self.decoder(low, mid, self.aspp(high), target_size=(h, w))


# ─────────────────────────────────────────────────────────────────────────────
# LOSS — class-weighted OHEM-CE + Dice (structure identical, weights added)
# ─────────────────────────────────────────────────────────────────────────────
class OHEMCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100_000,
                 weight=None, label_smoothing=0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh   = thresh
        self.min_kept = min_kept
        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction='none',
            weight=weight, label_smoothing=label_smoothing)

    def forward(self, pred, target):
        losses = self.ce(pred, target)
        mask   = target != self.ignore_index
        losses_flat = losses[mask]
        if losses_flat.numel() == 0:
            return losses.mean()
        n_kept    = max(self.min_kept,
                        int(losses_flat.numel() * (1 - self.thresh)))
        n_kept    = min(n_kept, losses_flat.numel())
        thresh_v  = losses_flat.sort(descending=True).values[n_kept - 1].item()
        hard_mask = (losses >= thresh_v) & mask
        return losses[hard_mask].mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        prob  = F.softmax(pred, dim=1)
        valid = target != self.ignore_index
        total, count = 0.0, 0
        for c in range(self.num_classes):
            tgt_c = ((target == c) & valid).float()
            if tgt_c.sum() == 0:
                continue
            prd_c = prob[:, c][valid]; tgt_c = tgt_c[valid]
            inter = (prd_c * tgt_c).sum()
            union = prd_c.sum() + tgt_c.sum()
            total += 1.0 - (2 * inter + self.smooth) / (union + self.smooth)
            count += 1
        return total / max(count, 1)


class CombinedLoss(nn.Module):
    """Weighted OHEM-CE (0.7) + Dice (0.3) with class-frequency weights."""
    def __init__(self, num_classes=21, ignore_index=255,
                 class_weights=None, label_smoothing=0.05):
        super().__init__()
        self.ohem = OHEMCrossEntropyLoss(
            ignore_index=ignore_index,
            thresh=CFG["ohem_thresh"],
            min_kept=CFG["ohem_min_kept"],
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.w = 1.0 - CFG["dice_alpha"]   # 0.7

    def forward(self, pred, target):
        return self.w * self.ohem(pred, target) + \
               CFG["dice_alpha"] * self.dice(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes),
                                   dtype=np.int64)

    def update(self, pred, target):
        pred   = pred.cpu().numpy()
        target = target.cpu().numpy()
        mask   = target != self.ignore_index
        pred   = pred[mask]; target = target[mask]
        idx    = target * self.num_classes + pred
        self.confusion += np.bincount(idx, minlength=self.num_classes**2)\
                            .reshape(self.num_classes, self.num_classes)

    def compute(self):
        cm    = self.confusion
        tp    = np.diag(cm)
        fp    = cm.sum(0) - tp
        fn    = cm.sum(1) - tp
        denom = tp + fp + fn
        iou   = np.where(denom > 0, tp / denom, np.nan)
        miou  = np.nanmean(iou)
        prec  = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
        rec   = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
        f1    = np.where((prec + rec) > 0, 2*prec*rec/(prec+rec), np.nan)
        total = cm.sum()
        px_acc = tp.sum() / total if total > 0 else 0.0
        mn_acc = np.nanmean(np.where((tp+fn)>0, tp/(tp+fn), np.nan))
        fw_iou = np.nansum((cm.sum(1)/total) * iou)
        dice   = np.where((2*tp+fp+fn)>0, 2*tp/(2*tp+fp+fn), np.nan)
        return {
            "mIoU":           float(miou   * 100),
            "pixel_acc":      float(px_acc * 100),
            "mean_acc":       float(mn_acc * 100),
            "fw_iou":         float(fw_iou * 100),
            "mean_f1":        float(np.nanmean(f1)   * 100),
            "mean_dice":      float(np.nanmean(dice)  * 100),
            "per_class_iou":  (iou  * 100).tolist(),
            "per_class_f1":   (f1   * 100).tolist(),
            "per_class_dice": (dice * 100).tolist(),
            "confusion_matrix": cm.tolist(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# LR SCHEDULER — Cosine Annealing with linear warmup
# Replaces poly; converges better on clean synthetic data
# ─────────────────────────────────────────────────────────────────────────────
class CosineWarmupLR:
    def __init__(self, optimizer, total_epochs, warmup_epochs=8):
        self.optimizer = optimizer
        self.total     = total_epochs
        self.warmup    = warmup_epochs
        self.base_lrs  = [g['lr'] for g in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup:
            factor = (epoch + 1) / self.warmup
        else:
            progress = (epoch - self.warmup) / (self.total - self.warmup)
            factor   = 0.5 * (1.0 + np.cos(np.pi * progress))
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g['lr'] = base * factor
        return self.optimizer.param_groups[0]['lr']


# ─────────────────────────────────────────────────────────────────────────────
# TTA — multi-scale test-time augmentation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def tta_predict(model, imgs, scales=(0.75, 1.0, 1.25), num_classes=21):
    H, W  = imgs.shape[2:]
    total = torch.zeros(imgs.size(0), num_classes, H, W, device=imgs.device)
    for s in scales:
        x = F.interpolate(imgs, size=(int(H*s), int(W*s)),
                          mode='bilinear', align_corners=True) if s != 1.0 else imgs
        logit = model(x)
        logit = F.interpolate(logit, size=(H, W),
                              mode='bilinear', align_corners=True)
        total += F.softmax(logit, dim=1)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg,
                 class_weights=None):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg

        self._freeze_backbone()

        backbone_params = list(model.encoder.parameters())
        head_params     = (list(model.aspp.parameters()) +
                           list(model.decoder.parameters()))
        self.optimizer  = optim.AdamW([
            {"params": backbone_params, "lr": cfg["lr"] * 0.1},
            {"params": head_params,     "lr": cfg["lr"]},
        ], weight_decay=cfg["weight_decay"])
        self.scheduler  = CosineWarmupLR(
            self.optimizer, cfg["num_epochs"], cfg["warmup_epochs"])
        self.criterion  = CombinedLoss(
            num_classes=cfg["num_classes"],
            ignore_index=cfg["ignore_index"],
            class_weights=class_weights,
            label_smoothing=cfg["label_smoothing"],
        )
        self.metrics = SegmentationMetrics(cfg["num_classes"], cfg["ignore_index"])
        self.scaler  = GradScaler(enabled=cfg.get("use_amp", True))

        self.history = {
            "train_loss": [], "val_loss": [], "miou": [],
            "pixel_acc": [], "mean_acc": [], "fw_iou": [],
            "mean_f1": [], "mean_dice": [], "lr": [], "epoch_time": [],
            "best_miou": 0.0, "best_epoch": 0,
        }
        self.best_miou = 0.0

    # ── Progressive unfreezing ────────────────────────────────────────────────
    def _freeze_backbone(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        logger.info("  Backbone frozen. Progressive unfreezing scheduled.")

    def _apply_unfreeze(self, epoch):
        schedule = self.cfg.get("unfreeze_schedule", {})
        for ep_thresh in sorted(schedule.keys()):
            if epoch == int(ep_thresh):
                for name in schedule[ep_thresh]:
                    layer = getattr(self.model.encoder, name, None)
                    if layer:
                        for p in layer.parameters():
                            p.requires_grad = True
                logger.info(
                    f"  [Epoch {epoch+1}] Unfroze: {schedule[ep_thresh]}")
                # Rebuild optimizer with newly unfrozen params
                bb = [p for p in self.model.encoder.parameters()
                      if p.requires_grad]
                hd = (list(self.model.aspp.parameters()) +
                      list(self.model.decoder.parameters()))
                self.optimizer = optim.AdamW([
                    {"params": bb, "lr": self.cfg["lr"] * 0.1},
                    {"params": hd, "lr": self.cfg["lr"]},
                ], weight_decay=self.cfg["weight_decay"])
                self.scheduler = CosineWarmupLR(
                    self.optimizer, self.cfg["num_epochs"],
                    self.cfg["warmup_epochs"])

    # ── TRAIN ONE EPOCH ───────────────────────────────────────────────────────
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        t0    = time.time()
        accum = self.cfg.get("grad_accum", 1)
        amp   = self.cfg.get("use_amp", True)

        self.optimizer.zero_grad()
        for i, (imgs, masks) in enumerate(self.train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast(enabled=amp):
                preds = self.model(imgs)
                loss  = self.criterion(preds, masks) / accum
            self.scaler.scale(loss).backward()

            if (i + 1) % accum == 0 or (i + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accum
            if (i + 1) % self.cfg["log_interval"] == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{self.cfg['num_epochs']}] "
                    f"Step [{i+1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item()*accum:.4f}")

        return total_loss / len(self.train_loader), time.time() - t0

    # ── VALIDATE ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        use_tta = self.cfg.get("use_tta", False)
        scales  = self.cfg.get("tta_scales", [1.0])
        amp     = self.cfg.get("use_amp", True)

        for imgs, masks in self.val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast(enabled=amp):
                preds = self.model(imgs)
                loss  = self.criterion(preds, masks)
            total_loss += loss.item()

            if use_tta:
                prob        = tta_predict(self.model, imgs, scales,
                                          self.cfg["num_classes"])
                pred_labels = prob.argmax(dim=1)
            else:
                pred_labels = preds.argmax(dim=1)
            self.metrics.update(pred_labels, masks)

        return total_loss / len(self.val_loader), self.metrics.compute()

    # ── FULL TRAINING LOOP ────────────────────────────────────────────────────
    def train(self):
        logger.info("=" * 70)
        logger.info("  Training: ResNet50-ASPP-ResNet50Decoder | Syn-Mediverse")
        logger.info(f"  Target: >90% mIoU | AMP={self.cfg['use_amp']} "
                    f"| TTA={self.cfg['use_tta']}")
        logger.info("=" * 70)

        stats = {}
        for epoch in range(self.cfg["num_epochs"]):
            self._apply_unfreeze(epoch)
            lr = self.scheduler.step(epoch)
            train_loss, epoch_time = self.train_epoch(epoch)

            if (epoch + 1) % self.cfg["val_interval"] == 0:
                val_loss, stats = self.val_epoch()
                miou = stats["mIoU"]

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["miou"].append(miou)
                self.history["pixel_acc"].append(stats["pixel_acc"])
                self.history["mean_acc"].append(stats["mean_acc"])
                self.history["fw_iou"].append(stats["fw_iou"])
                self.history["mean_f1"].append(stats["mean_f1"])
                self.history["mean_dice"].append(stats["mean_dice"])
                self.history["lr"].append(lr)
                self.history["epoch_time"].append(epoch_time)

                logger.info(
                    f"\n{'─'*60}\n"
                    f"  Epoch {epoch+1:3d}/{self.cfg['num_epochs']} | "
                    f"LR: {lr:.6f} | Time: {epoch_time:.1f}s\n"
                    f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
                    f"  mIoU: {miou:.2f}% | PixAcc: {stats['pixel_acc']:.2f}% | "
                    f"MeanAcc: {stats['mean_acc']:.2f}%\n"
                    f"  FW-IoU: {stats['fw_iou']:.2f}% | "
                    f"F1: {stats['mean_f1']:.2f}% | Dice: {stats['mean_dice']:.2f}%\n"
                    f"{'─'*60}"
                )

                if miou > self.best_miou:
                    self.best_miou = miou
                    self.history["best_miou"]  = miou
                    self.history["best_epoch"] = epoch + 1
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "miou": miou, "stats": stats, "config": self.cfg,
                    }, os.path.join(self.cfg["save_dir"], "checkpoints",
                                    "best_model.pth"))
                    logger.info(f"  BEST mIoU: {miou:.2f}% — checkpoint saved!")

                torch.save({
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "history": self.history,
                }, os.path.join(self.cfg["save_dir"], "checkpoints",
                                "latest_model.pth"))

                with open(os.path.join(self.cfg["save_dir"], "logs",
                                       "val_stats.json"), "w") as f:
                    json.dump({
                        "epoch": epoch + 1, "val_loss": val_loss, **stats,
                        "per_class_iou_named":
                            dict(zip(SYN_CLASSES, stats["per_class_iou"]))
                    }, f, indent=2)

                if (epoch + 1) % 5 == 0 or epoch == self.cfg["num_epochs"] - 1:
                    self.plot_all(stats)

        logger.info(f"\n{'='*70}")
        logger.info(f"  Done! Best mIoU: {self.best_miou:.2f}% "
                    f"at Epoch {self.history['best_epoch']}")
        logger.info(f"{'='*70}")
        if stats:
            self.plot_all(stats, final=True)
        self.save_history()
        return self.history

    # ── GRAPHS (same structure as original, updated labels) ───────────────────
    def plot_all(self, last_stats, final=False):
        tag = "final" if final else f"ep{len(self.history['miou'])}"
        self._plot_loss_curves(tag)
        self._plot_metrics_dashboard(tag)
        self._plot_per_class_iou(last_stats, tag)
        self._plot_confusion_matrix(last_stats, tag)
        self._plot_lr_schedule(tag)
        if final:
            self._plot_research_summary(last_stats)

    def _plot_loss_curves(self, tag):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs = list(range(1, len(self.history["train_loss"]) + 1))
        axes[0].plot(epochs, self.history["train_loss"], 'b-o', ms=3, label="Train")
        axes[0].plot(epochs, self.history["val_loss"],   'r-s', ms=3, label="Val")
        axes[0].set_title("Loss Curves", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].plot(epochs, self.history["miou"], 'g-^', ms=3, label="mIoU (%)")
        axes[1].axhline(y=90, color='orange', linestyle='--', label='Target 90%')
        be, bm = self.history["best_epoch"], self.history["best_miou"]
        axes[1].scatter([be], [bm], color='red', zorder=5, s=100,
                        label=f'Best: {bm:.2f}% @Ep{be}')
        axes[1].set_title("mIoU over Training", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU (%)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_ylim(0, 105)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["save_dir"], "graphs",
                                 f"loss_miou_{tag}.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_metrics_dashboard(self, tag):
        epochs  = list(range(1, len(self.history["miou"]) + 1))
        metrics = {
            "mIoU (%)":      self.history["miou"],
            "Pixel Acc (%)": self.history["pixel_acc"],
            "Mean Acc (%)":  self.history["mean_acc"],
            "FW-IoU (%)":    self.history["fw_iou"],
            "Mean F1 (%)":   self.history["mean_f1"],
            "Mean Dice (%)": self.history["mean_dice"],
        }
        colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for ax, (name, vals), col in zip(axes.flat, metrics.items(), colors):
            ax.plot(epochs, vals, color=col, linewidth=2, marker='o', markersize=3)
            ax.fill_between(epochs, vals, alpha=0.15, color=col)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel("Epoch"); ax.set_ylabel(name)
            ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
            if "mIoU" in name:
                ax.axhline(y=90, color='red', linestyle='--',
                           alpha=0.7, label='Target 90%')
                ax.legend(fontsize=9)
        fig.suptitle("Validation Metrics — Syn-Mediverse Segmentation",
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["save_dir"], "graphs",
                                 f"metrics_dashboard_{tag}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_per_class_iou(self, stats, tag):
        ious   = [v if not np.isnan(v) else 0 for v in stats["per_class_iou"]]
        colors = ['#2ecc71' if v >= 90 else '#e74c3c' if v < 60
                  else '#f39c12' for v in ious]
        fig, ax = plt.subplots(figsize=(16, 8))
        bars = ax.barh(SYN_CLASSES, ious, color=colors,
                       edgecolor='white', linewidth=0.5)
        ax.axvline(x=stats["mIoU"], color='navy', linewidth=2,
                   linestyle='--', label=f'mIoU: {stats["mIoU"]:.2f}%')
        ax.axvline(x=90, color='orange', linewidth=1.5,
                   linestyle=':', label='Target: 90%')
        for bar, v in zip(bars, ious):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{v:.1f}%', va='center', ha='left', fontsize=9)
        ax.set_xlim(0, 115); ax.set_xlabel("IoU (%)", fontsize=12)
        ax.set_title("Per-Class IoU — Syn-Mediverse (21 Classes)",
                     fontsize=14, fontweight='bold')
        patches = [mpatches.Patch(color='#2ecc71', label='IoU >= 90%'),
                   mpatches.Patch(color='#f39c12', label='60% <= IoU < 90%'),
                   mpatches.Patch(color='#e74c3c', label='IoU < 60%')]
        ax.legend(handles=patches + ax.get_legend_handles_labels()[0][:2],
                  fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["save_dir"], "graphs",
                                 f"per_class_iou_{tag}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrix(self, stats, tag):
        cm = np.array(stats["confusion_matrix"], dtype=np.float64)
        rs = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(rs > 0, cm / rs, 0)
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=SYN_CLASSES, yticklabels=SYN_CLASSES,
                    ax=ax, linewidths=0.3, linecolor='gray',
                    annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True",      fontsize=12)
        ax.set_title("Normalized Confusion Matrix — Syn-Mediverse",
                     fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0,  fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["save_dir"], "graphs",
                                 f"confusion_matrix_{tag}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_lr_schedule(self, tag):
        epochs = list(range(1, len(self.history["lr"]) + 1))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, self.history["lr"], 'purple', linewidth=2)
        ax.fill_between(epochs, self.history["lr"], alpha=0.1, color='purple')
        ax.set_title("LR Schedule (Cosine Annealing + Warmup)",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)
        ax.axvline(x=CFG["warmup_epochs"], color='red', linestyle='--',
                   alpha=0.5, label=f'Warmup @Ep{CFG["warmup_epochs"]}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["save_dir"], "graphs",
                                 f"lr_schedule_{tag}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_research_summary(self, stats):
        fig = plt.figure(figsize=(20, 16))
        gs  = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        epochs = list(range(1, len(self.history["miou"]) + 1))

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.history["train_loss"], 'b-',  lw=2, label='Train')
        ax1.plot(epochs, self.history["val_loss"],   'r--', lw=2, label='Val')
        ax1.set_title("(a) Loss Curves", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.history["miou"],      'g-',  lw=2, label='mIoU')
        ax2.plot(epochs, self.history["pixel_acc"], 'b--', lw=2, label='PixAcc')
        ax2.plot(epochs, self.history["mean_f1"],   'm:',  lw=2, label='MeanF1')
        ax2.axhline(y=90, color='orange', linestyle='-.', lw=1.5, label='Target 90%')
        ax2.set_title("(b) Validation Metrics", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score (%)")
        ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 105)

        ax3 = fig.add_subplot(gs[1, 0])
        ious = [v if not np.isnan(v) else 0 for v in stats["per_class_iou"]]
        sidx = np.argsort(ious)[::-1]
        clrs = ['#2ecc71' if ious[i] >= 90 else
                '#e74c3c' if ious[i] < 60 else '#f39c12' for i in sidx]
        ax3.bar(range(len(sidx)), [ious[i] for i in sidx], color=clrs)
        ax3.set_xticks(range(len(sidx)))
        ax3.set_xticklabels([SYN_CLASSES[i] for i in sidx],
                             rotation=45, ha='right', fontsize=8)
        ax3.axhline(y=stats["mIoU"], color='navy', linestyle='--', lw=1.5,
                    label=f'mIoU={stats["mIoU"]:.1f}%')
        ax3.set_title("(c) Per-Class IoU (sorted)", fontsize=14, fontweight='bold')
        ax3.set_ylabel("IoU (%)"); ax3.legend(fontsize=10)
        ax3.grid(True, axis='y', alpha=0.3); ax3.set_ylim(0, 105)

        ax4 = fig.add_subplot(gs[1, 1]); ax4.axis('off')
        rows = [
            ["mIoU",           f"{stats['mIoU']:.2f}%"],
            ["Pixel Accuracy", f"{stats['pixel_acc']:.2f}%"],
            ["Mean Accuracy",  f"{stats['mean_acc']:.2f}%"],
            ["FW-IoU",         f"{stats['fw_iou']:.2f}%"],
            ["Mean F1",        f"{stats['mean_f1']:.2f}%"],
            ["Mean Dice",      f"{stats['mean_dice']:.2f}%"],
            ["Best Epoch",     str(self.history['best_epoch'])],
        ]
        table = ax4.table(cellText=rows, colLabels=["Metric", "Value"],
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False); table.set_fontsize(13)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 1:
                cell.set_facecolor('#ecf0f1')
        ax4.set_title("(d) Summary Metrics", fontsize=14, fontweight='bold')

        fig.suptitle(
            "ResNet50-ASPP-ResNet50Decoder | Syn-Mediverse Healthcare Segmentation",
            fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(self.cfg["save_dir"], "graphs",
                                 "research_summary_FINAL.png"),
                    dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info("  Research summary saved.")

    def save_history(self):
        safe = {k: v for k, v in self.history.items()
                if isinstance(v, (list, float, int, str))}
        with open(os.path.join(self.cfg["save_dir"], "logs",
                               "training_history.json"), "w") as f:
            json.dump(safe, f, indent=2)
        logger.info("  Training history saved.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 70)
    logger.info("  Syn-Mediverse Semantic Segmentation — S2P Pipeline")
    logger.info("  Architecture: ResNet50 -> ASPP -> ResNet50StyleDecoder")
    logger.info("=" * 70)

    train_ds = SynMediverseDataset(
        root=CFG["data_dir"], split='train', img_size=CFG["img_size"])
    val_ds   = SynMediverseDataset(
        root=CFG["data_dir"], split='val',   img_size=CFG["img_size"])

    class_weights = compute_class_weights(
        train_ds, CFG["num_classes"], CFG["ignore_index"],
        sample_frac=0.3, dev=device)

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(
        val_ds,   batch_size=4,
        shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    logger.info(f"Train batches: {len(train_loader)} | Val: {len(val_loader)}")

    model = SegmentationModel(num_classes=CFG["num_classes"], pretrained=True)
    n_tot = sum(p.numel() for p in model.parameters()) / 1e6
    n_tr  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Params — Total: {n_tot:.2f}M | Trainable: {n_tr:.2f}M")

    trainer = Trainer(model, train_loader, val_loader, CFG,
                      class_weights=class_weights)
    return trainer.train()


if __name__ == "__main__":
    main()