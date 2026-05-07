"""
Semantic Segmentation with ResNet50 Encoder-ASPP-ResNet50 Decoder on Pascal VOC
Author: Algorithm Developer
Architecture: ResNet50 Encoder → ASPP Bottleneck → ResNet50-style Decoder
Dataset: Pascal VOC 2012 + SBD Augmented = 10,582 training masks
Target mIoU: >85% on Pascal VOC 2012 val (1,449 masks)
"""

import os
import sys
import time
import json
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
import urllib.request
import tarfile
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms, models
from torchvision.datasets import VOCSegmentation, SBDataset
import torchvision.transforms.functional as TF
from PIL import Image
import random

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = {
    "num_classes": 21,          # Pascal VOC (20 + background)
    "ignore_index": 255,
    "img_size": 512,
    "batch_size": 8,
    "num_epochs": 100,           # More epochs for 10,582 samples
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "aspp_dilations": [1, 6, 12, 18],
    "aspp_out_channels": 256,
    "decoder_channels": 256,
    "dropout": 0.1,
    "seed": 42,
    "use_sbd": True,            # ← Use SBD augmented set (10,582 train masks)
    "save_dir": "output1/experiment",
    "data_dir": "output1/VOCdata",
    "sbd_dir":  "output1/SBDdata",
    "log_interval": 10,
    "val_interval": 1,
}

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]

VOC_COLORMAP = [
    [0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],
    [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
    [64,128,0],[192,128,0],[64,0,128],[192,0,128],[64,128,128],
    [192,128,128],[0,64,0],[128,64,0],[0,192,0],[128,192,0],[0,64,128]
]

os.makedirs(CFG["save_dir"], exist_ok=True)
os.makedirs(os.path.join(CFG["save_dir"], "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(CFG["save_dir"], "graphs"), exist_ok=True)
os.makedirs(os.path.join(CFG["save_dir"], "logs"), exist_ok=True)
os.makedirs(os.path.join(CFG["save_dir"], "visualizations"), exist_ok=True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(CFG["seed"])

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
log_path = os.path.join(CFG["save_dir"], "logs", "training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# DATASET & AUGMENTATIONS
# ─────────────────────────────────────────────

class Augmentations:
    """Shared joint augmentation logic for both VOC and SBD wrappers."""

    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train

    def __call__(self, img, mask):
        if self.is_train:
            return self._train_transform(img, mask)
        else:
            return self._val_transform(img, mask)

    def _train_transform(self, img, mask):
        # 1. Random horizontal flip
        if random.random() > 0.5:
            img  = TF.hflip(img)
            mask = TF.hflip(mask)

        # 2. Random scale [0.5, 2.0]
        scale = random.uniform(0.5, 2.0)
        w, h  = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img  = TF.resize(img,  (new_h, new_w), interpolation=Image.BILINEAR)
        mask = TF.resize(mask, (new_h, new_w), interpolation=Image.NEAREST)

        # 3. Pad then random crop to img_size × img_size
        img, mask = self._random_crop_or_pad(img, mask)

        # 4. Color jitter (image only)
        img = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )(img)

        # 5. Random rotation ±10°
        angle = random.uniform(-10, 10)
        img  = TF.rotate(img,  angle, interpolation=Image.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=Image.NEAREST, fill=255)

        # 6. Random Gaussian blur
        if random.random() > 0.5:
            img = img.filter(__import__('PIL').ImageFilter.GaussianBlur(
                radius=random.uniform(0.1, 1.5)))

        return img, mask

    def _val_transform(self, img, mask):
        s = self.img_size
        img  = TF.resize(img,  (s, s), interpolation=Image.BILINEAR)
        mask = TF.resize(mask, (s, s), interpolation=Image.NEAREST)
        return img, mask

    def _random_crop_or_pad(self, img, mask):
        s = self.img_size
        w, h = img.size
        pad_h = max(s - h, 0)
        pad_w = max(s - w, 0)
        if pad_h > 0 or pad_w > 0:
            img  = TF.pad(img,  (0, 0, pad_w, pad_h), fill=0)
            mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=255)
        w, h = img.size
        x = random.randint(0, w - s)
        y = random.randint(0, h - s)
        img  = TF.crop(img,  y, x, s, s)
        mask = TF.crop(mask, y, x, s, s)
        return img, mask


def to_tensor_normalize(img, mask):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    img  = TF.to_tensor(img)
    img  = TF.normalize(img, mean, std)
    mask = torch.from_numpy(np.array(mask)).long()
    return img, mask


# ── VOC Wrapper ──────────────────────────────
class VOCSegDataset(Dataset):
    """Pascal VOC 2012 Segmentation dataset."""

    def __init__(self, root, image_set='train', img_size=512):
        self.aug = Augmentations(img_size, is_train=(image_set == 'train'))
        self.dataset = VOCSegmentation(
            root=root, year='2012', image_set=image_set,
            download=True, transforms=None
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        img, mask = self.aug(img, mask)
        return to_tensor_normalize(img, mask)


# ── SBD Wrapper ──────────────────────────────
class SBDSegDataset(Dataset):
    """
    Semantic Boundaries Dataset (Hariharan et al.)
    Provides 8,498 extra annotated images for VOC classes.
    Combined with VOC train (1,464) → 10,582 unique training images
    (duplicates with VOC val are automatically removed).
    """

    def __init__(self, root, image_set='train', img_size=512):
        self.aug = Augmentations(img_size, is_train=True)
        try:
            self.dataset = SBDataset(
                root=root, image_set=image_set,
                mode='segmentation', download=True
            )
            self.valid = True
        except Exception as e:
            logger.warning(f"SBD load failed: {e}. Falling back to VOC-only.")
            self.valid = False
            self.dataset = []

    def __len__(self):
        return len(self.dataset) if self.valid else 0

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        # SBD masks come as numpy arrays — convert to PIL
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask.astype(np.uint8))
        img, mask = self.aug(img, mask)
        return to_tensor_normalize(img, mask)


# ── Build combined train dataset ─────────────
def build_datasets(cfg):
    """
    Returns:
        train_dataset : VOC train (1,464) + SBD train (8,498) = 10,582
                        OR VOC train only if SBD unavailable
        val_dataset   : VOC val  (1,449)  — never mixed with SBD
    """
    logger.info("=" * 60)
    logger.info("  Building datasets...")

    voc_train = VOCSegDataset(cfg["data_dir"], image_set='train', img_size=cfg["img_size"])
    voc_val   = VOCSegDataset(cfg["data_dir"], image_set='val',   img_size=cfg["img_size"])
    logger.info(f"  VOC train: {len(voc_train):,} masks")
    logger.info(f"  VOC val  : {len(voc_val):,}  masks")

    if cfg.get("use_sbd", True):
        sbd_train = SBDSegDataset(cfg["sbd_dir"], image_set='train', img_size=cfg["img_size"])
        if sbd_train.valid and len(sbd_train) > 0:
            train_dataset = ConcatDataset([voc_train, sbd_train])
            logger.info(f"  SBD train: {len(sbd_train):,} masks")
            logger.info(f"  ✓ Combined train: {len(train_dataset):,} masks (VOC + SBD)")
        else:
            train_dataset = voc_train
            logger.info("  ⚠ SBD unavailable — using VOC train only (1,464 masks)")
    else:
        train_dataset = voc_train
        logger.info("  Using VOC train only (use_sbd=False)")

    logger.info(f"  Val: {len(voc_val):,} masks (VOC val — ground truth available)")
    logger.info("=" * 60)
    return train_dataset, voc_val


# ─────────────────────────────────────────────
# MODEL COMPONENTS
# ─────────────────────────────────────────────
class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, stride, padding=dilation if dilation > 1 else padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    def __init__(self, in_channels=2048, out_channels=256, dilations=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        # 1×1 conv
        self.branches.append(ConvBNReLU(in_channels, out_channels, k=1, padding=0, dilation=1))
        # atrous convs
        for d in dilations[1:]:
            self.branches.append(ConvBNReLU(in_channels, out_channels, k=3, dilation=d))
        # global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_channels, out_channels, k=1, padding=0)
        )
        self.project = nn.Sequential(
            ConvBNReLU((len(dilations) + 1) * out_channels, out_channels, k=1, padding=0),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [b(x) for b in self.branches]
        gap_feat = F.interpolate(self.gap(x), size=(h, w), mode='bilinear', align_corners=True)
        feats.append(gap_feat)
        return self.project(torch.cat(feats, dim=1))


class ResNet50Encoder(nn.Module):
    """ResNet50 encoder with dilated convolutions for dense prediction."""
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # stride 4,  channels 256
        self.layer2 = backbone.layer2   # stride 8,  channels 512
        self.layer3 = backbone.layer3   # stride 16, channels 1024
        self.layer4 = backbone.layer4   # stride 32 → dilated to 16, channels 2048

        # Make layer3 & layer4 use dilation instead of stride
        self._make_dilated(self.layer3, stride=1, dilation=2)
        self._make_dilated(self.layer4, stride=1, dilation=4)

    def _make_dilated(self, layer, stride, dilation):
        for module in layer.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (stride, stride)
                if module.kernel_size == (3, 3):
                    module.dilation = (dilation, dilation)
                    module.padding  = (dilation, dilation)

    def forward(self, x):
        x0 = self.layer0(x)   # /4
        x1 = self.layer1(x0)  # /4,  256ch  — low-level feature
        x2 = self.layer2(x1)  # /8,  512ch
        x3 = self.layer3(x2)  # /16, 1024ch
        x4 = self.layer4(x3)  # /16, 2048ch (dilated)
        return x1, x2, x3, x4  # multi-scale features


class ResNet50StyleDecoder(nn.Module):
    """ResNet50-style decoder with skip connections (residual blocks)."""
    def __init__(self, aspp_channels=256, num_classes=21):
        super().__init__()

        # Low-level feature projection (from encoder layer1: 256ch)
        self.low_proj = ConvBNReLU(256, 48, k=1, padding=0)

        # Mid-level feature projection (from encoder layer2: 512ch)
        self.mid_proj = ConvBNReLU(512, 96, k=1, padding=0)

        # Fuse aspp + low + mid
        fuse_in = aspp_channels + 48 + 96   # 400

        self.fuse1 = self._make_resblock(fuse_in, 256)
        self.fuse2 = self._make_resblock(256, 256)
        self.fuse3 = self._make_resblock(256, 128)

        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Conv2d(128, num_classes, 1)

    def _make_resblock(self, in_c, out_c):
        """A simplified residual block (1×1 + 3×3 + 1×1 with BN/ReLU)."""
        layers = nn.Sequential(
            ConvBNReLU(in_c, out_c, k=1, padding=0),
            ConvBNReLU(out_c, out_c, k=3, padding=1),
        )
        shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ) if in_c != out_c else nn.Identity()

        class ResBlock(nn.Module):
            def forward(self_, x):
                return F.relu(layers(x) + shortcut(x), inplace=True)

        block = ResBlock()
        block.layers   = layers
        block.shortcut = shortcut
        return block

    def forward(self, low_feat, mid_feat, aspp_feat, target_size):
        # Upsample ASPP to low-level size
        aspp_up = F.interpolate(aspp_feat, size=low_feat.shape[2:],
                                mode='bilinear', align_corners=True)
        mid_up  = F.interpolate(self.mid_proj(mid_feat), size=low_feat.shape[2:],
                                mode='bilinear', align_corners=True)
        low_p   = self.low_proj(low_feat)

        x = torch.cat([aspp_up, low_p, mid_up], dim=1)
        x = self.fuse1(x)
        x = self.fuse2(x)
        x = self.fuse3(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        return x


class SegmentationModel(nn.Module):
    """
    Full model:
      Encoder  : ResNet50 (dilated, ImageNet pretrained)
      Bottleneck: ASPP (dilations 1, 6, 12, 18)
      Decoder  : ResNet50-style (residual blocks + skip connections)
    """
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        self.encoder  = ResNet50Encoder(pretrained=pretrained)
        self.aspp     = ASPPModule(in_channels=2048, out_channels=256,
                                    dilations=CFG["aspp_dilations"])
        self.decoder  = ResNet50StyleDecoder(aspp_channels=256, num_classes=num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        low, mid, _, high = self.encoder(x)
        aspp_out = self.aspp(high)
        out = self.decoder(low, mid, aspp_out, target_size=(h, w))
        return out


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────
class OHEMCrossEntropyLoss(nn.Module):
    """Online Hard Example Mining CE Loss."""
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000):
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = thresh
        self.min_kept = min_kept
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, pred, target):
        losses = self.ce(pred, target)
        mask = target != self.ignore_index
        losses_flat = losses[mask]
        if losses_flat.numel() == 0:
            return losses.mean()
        n_kept = max(self.min_kept, int(losses_flat.numel() * (1 - self.thresh)))
        n_kept = min(n_kept, losses_flat.numel())
        sorted_losses, _ = losses_flat.sort(descending=True)
        threshold = sorted_losses[n_kept - 1].item()
        hard_mask = (losses >= threshold) & mask
        return losses[hard_mask].mean()


class DiceLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        prob = F.softmax(pred, dim=1)
        valid = target != self.ignore_index
        total_loss = 0.0
        count = 0
        for c in range(self.num_classes):
            tgt_c = ((target == c) & valid).float()
            if tgt_c.sum() == 0:
                continue
            prd_c = prob[:, c][valid]
            tgt_c = tgt_c[valid]
            inter = (prd_c * tgt_c).sum()
            union = prd_c.sum() + tgt_c.sum()
            total_loss += 1.0 - (2 * inter + self.smooth) / (union + self.smooth)
            count += 1
        return total_loss / max(count, 1)


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255):
        super().__init__()
        self.ohem = OHEMCrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, pred, target):
        return 0.7 * self.ohem(pred, target) + 0.3 * self.dice(pred, target)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, target):
        pred   = pred.cpu().numpy()
        target = target.cpu().numpy()
        mask   = target != self.ignore_index
        pred   = pred[mask]
        target = target[mask]
        idx    = target * self.num_classes + pred
        bincount = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += bincount.reshape(self.num_classes, self.num_classes)

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
        f1    = np.where((prec + rec) > 0,
                          2 * prec * rec / (prec + rec), np.nan)

        total  = cm.sum()
        pixel_acc = tp.sum() / total if total > 0 else 0.0
        mean_acc  = np.nanmean(np.where((tp + fn) > 0, tp / (tp + fn), np.nan))

        freq    = cm.sum(1) / total
        fw_iou  = np.nansum(freq * iou)

        dice = np.where((2*tp + fp + fn) > 0,
                        2*tp / (2*tp + fp + fn), np.nan)

        return {
            "mIoU":           float(miou * 100),
            "pixel_acc":      float(pixel_acc * 100),
            "mean_acc":       float(mean_acc * 100),
            "fw_iou":         float(fw_iou * 100),
            "mean_f1":        float(np.nanmean(f1) * 100),
            "mean_dice":      float(np.nanmean(dice) * 100),
            "per_class_iou":  (iou * 100).tolist(),
            "per_class_f1":   (f1  * 100).tolist(),
            "per_class_dice": (dice* 100).tolist(),
            "confusion_matrix": cm.tolist(),
        }


# ─────────────────────────────────────────────
# LR SCHEDULER (Poly + Warmup)
# ─────────────────────────────────────────────
class PolyLRWithWarmup:
    def __init__(self, optimizer, total_epochs, warmup_epochs=5, power=0.9):
        self.optimizer     = optimizer
        self.total_epochs  = total_epochs
        self.warmup_epochs = warmup_epochs
        self.power         = power
        self.base_lrs      = [g['lr'] for g in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            e = epoch - self.warmup_epochs
            t = self.total_epochs - self.warmup_epochs
            factor = (1 - e / t) ** self.power
        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g['lr'] = base_lr * factor
        return self.optimizer.param_groups[0]['lr']


# ─────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────
class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg):
        self.model       = model.to(device)
        self.train_loader = train_loader
        self.val_loader  = val_loader
        self.cfg         = cfg

        # Different LR for backbone vs head
        backbone_params = list(model.encoder.parameters())
        head_params     = (list(model.aspp.parameters()) +
                           list(model.decoder.parameters()))
        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": cfg["lr"] * 0.1},
            {"params": head_params,     "lr": cfg["lr"]},
        ], weight_decay=cfg["weight_decay"])

        self.scheduler = PolyLRWithWarmup(
            self.optimizer, cfg["num_epochs"], cfg["warmup_epochs"]
        )
        self.criterion = CombinedLoss(cfg["num_classes"], cfg["ignore_index"])
        self.metrics   = SegmentationMetrics(cfg["num_classes"], cfg["ignore_index"])

        # History
        self.history = {
            "train_loss": [], "val_loss": [],
            "miou": [], "pixel_acc": [], "mean_acc": [],
            "fw_iou": [], "mean_f1": [], "mean_dice": [],
            "lr": [], "epoch_time": [],
            "best_miou": 0.0, "best_epoch": 0,
        }
        self.best_miou = 0.0

    # ── TRAIN ONE EPOCH ──────────────────────
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        t0 = time.time()

        for i, (imgs, masks) in enumerate(self.train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            self.optimizer.zero_grad()
            preds = self.model(imgs)
            loss  = self.criterion(preds, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()

            if (i + 1) % self.cfg["log_interval"] == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{self.cfg['num_epochs']}] "
                    f"Step [{i+1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss  = total_loss / len(self.train_loader)
        epoch_time = time.time() - t0
        return avg_loss, epoch_time

    # ── VALIDATE ─────────────────────────────
    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0

        for imgs, masks in self.val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = self.model(imgs)
            loss  = self.criterion(preds, masks)
            total_loss += loss.item()
            pred_labels = preds.argmax(dim=1)
            self.metrics.update(pred_labels, masks)

        avg_loss = total_loss / len(self.val_loader)
        stats    = self.metrics.compute()
        return avg_loss, stats

    # ── FULL TRAINING LOOP ───────────────────
    def train(self):
        logger.info("=" * 70)
        logger.info("  Starting Training: ResNet50-ASPP-ResNet50Decoder on Pascal VOC")
        logger.info("=" * 70)
        logger.info(f"  Config: {json.dumps({k:v for k,v in self.cfg.items() if k not in ['save_dir','data_dir']}, indent=2)}")

        for epoch in range(self.cfg["num_epochs"]):
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

                # Save best
                if miou > self.best_miou:
                    self.best_miou = miou
                    self.history["best_miou"]  = miou
                    self.history["best_epoch"] = epoch + 1
                    ckpt = {
                        "epoch": epoch + 1,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "miou": miou,
                        "stats": stats,
                        "config": self.cfg,
                    }
                    torch.save(ckpt, os.path.join(
                        self.cfg["save_dir"], "checkpoints", "best_model.pth"
                    ))
                    logger.info(f"  ★ NEW BEST mIoU: {miou:.2f}% — checkpoint saved!")

                # Save latest
                torch.save({
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "history": self.history,
                }, os.path.join(self.cfg["save_dir"], "checkpoints", "latest_model.pth"))

                # Save stats JSON
                with open(os.path.join(self.cfg["save_dir"], "logs", "val_stats.json"), "w") as f:
                    json.dump({
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        **stats,
                        "per_class_iou_named": dict(zip(VOC_CLASSES, stats["per_class_iou"]))
                    }, f, indent=2)

                # Plot graphs every 5 epochs
                if (epoch + 1) % 5 == 0 or epoch == self.cfg["num_epochs"] - 1:
                    self.plot_all(stats)

        logger.info(f"\n{'='*70}")
        logger.info(f"  Training Complete!")
        logger.info(f"  Best mIoU: {self.best_miou:.2f}% at Epoch {self.history['best_epoch']}")
        logger.info(f"{'='*70}\n")
        self.plot_all(stats, final=True)
        self.save_history()
        return self.history


    # ─────────────────────────────────────────
    # GRAPH GENERATION
    # ─────────────────────────────────────────
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
        axes[0].plot(epochs, self.history["train_loss"], 'b-o', ms=3, label="Train Loss")
        axes[0].plot(epochs, self.history["val_loss"],   'r-s', ms=3, label="Val Loss")
        axes[0].set_title("Loss Curves", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.history["miou"], 'g-^', ms=3, label="mIoU (%)")
        axes[1].axhline(y=85, color='orange', linestyle='--', label='Target 85%')
        best_ep = self.history["best_epoch"]
        best_miou = self.history["best_miou"]
        axes[1].scatter([best_ep], [best_miou], color='red', zorder=5, s=100,
                         label=f'Best: {best_miou:.2f}% @Ep{best_ep}')
        axes[1].set_title("mIoU over Training", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("mIoU (%)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["save_dir"], "graphs", f"loss_miou_{tag}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_metrics_dashboard(self, tag):
        epochs = list(range(1, len(self.history["miou"]) + 1))
        metrics = {
            "mIoU (%)":       self.history["miou"],
            "Pixel Acc (%)":  self.history["pixel_acc"],
            "Mean Acc (%)":   self.history["mean_acc"],
            "FW-IoU (%)":     self.history["fw_iou"],
            "Mean F1 (%)":    self.history["mean_f1"],
            "Mean Dice (%)":  self.history["mean_dice"],
        }
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for ax, (name, vals), col in zip(axes.flat, metrics.items(), colors):
            ax.plot(epochs, vals, color=col, linewidth=2, marker='o', markersize=3)
            ax.fill_between(epochs, vals, alpha=0.15, color=col)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel("Epoch"); ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)
            if "mIoU" in name:
                ax.axhline(y=85, color='red', linestyle='--', alpha=0.7, label='Target 85%')
                ax.legend(fontsize=9)
        fig.suptitle("Validation Metrics Dashboard — Pascal VOC Segmentation",
                     fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["save_dir"], "graphs", f"metrics_dashboard_{tag}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_per_class_iou(self, stats, tag):
        ious = [v if not np.isnan(v) else 0 for v in stats["per_class_iou"]]
        colors = ['#2ecc71' if v >= 85 else '#e74c3c' if v < 60 else '#f39c12' for v in ious]
        fig, ax = plt.subplots(figsize=(16, 7))
        bars = ax.barh(VOC_CLASSES, ious, color=colors, edgecolor='white', linewidth=0.5)
        ax.axvline(x=stats["mIoU"], color='navy', linewidth=2,
                   linestyle='--', label=f'mIoU: {stats["mIoU"]:.2f}%')
        ax.axvline(x=85, color='orange', linewidth=1.5,
                   linestyle=':', label='Target: 85%')
        for bar, v in zip(bars, ious):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{v:.1f}%', va='center', ha='left', fontsize=9)
        ax.set_xlim(0, 115)
        ax.set_xlabel("IoU (%)", fontsize=12)
        ax.set_title("Per-Class IoU — Pascal VOC (21 Classes)",
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)
        patches = [
            mpatches.Patch(color='#2ecc71', label='IoU ≥ 85%'),
            mpatches.Patch(color='#f39c12', label='60% ≤ IoU < 85%'),
            mpatches.Patch(color='#e74c3c', label='IoU < 60%'),
        ]
        ax.legend(handles=patches + ax.get_legend_handles_labels()[0][:2], fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["save_dir"], "graphs", f"per_class_iou_{tag}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrix(self, stats, tag):
        cm = np.array(stats["confusion_matrix"], dtype=np.float64)
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sum > 0, cm / row_sum, 0)
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=VOC_CLASSES, yticklabels=VOC_CLASSES,
                    ax=ax, linewidths=0.3, linecolor='gray',
                    annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Normalized Confusion Matrix — Pascal VOC",
                     fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["save_dir"], "graphs", f"confusion_matrix_{tag}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_lr_schedule(self, tag):
        epochs = list(range(1, len(self.history["lr"]) + 1))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, self.history["lr"], 'purple', linewidth=2)
        ax.fill_between(epochs, self.history["lr"], alpha=0.1, color='purple')
        ax.set_title("Learning Rate Schedule (Poly with Warmup)",
                     fontsize=13, fontweight='bold')
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)
        ax.axvline(x=CFG["warmup_epochs"], color='red', linestyle='--',
                   alpha=0.5, label=f'Warmup ends @Ep{CFG["warmup_epochs"]}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg["save_dir"], "graphs", f"lr_schedule_{tag}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_research_summary(self, stats):
        """Publication-quality summary figure (4-panel)."""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        epochs = list(range(1, len(self.history["miou"]) + 1))

        # Panel 1: Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.history["train_loss"], 'b-', lw=2, label='Train')
        ax1.plot(epochs, self.history["val_loss"],   'r--', lw=2, label='Validation')
        ax1.set_title("(a) Loss Curves", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.legend(fontsize=11); ax1.grid(True, alpha=0.3)

        # Panel 2: Metrics
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.history["miou"],      'g-',  lw=2, label='mIoU')
        ax2.plot(epochs, self.history["pixel_acc"], 'b--', lw=2, label='Pixel Acc')
        ax2.plot(epochs, self.history["mean_f1"],   'm:',  lw=2, label='Mean F1')
        ax2.axhline(y=85, color='orange', linestyle='-.', lw=1.5, label='Target 85%')
        ax2.set_title("(b) Validation Metrics", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Score (%)")
        ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 105)

        # Panel 3: Per-class IoU (top 10 + bottom 5)
        ax3 = fig.add_subplot(gs[1, 0])
        ious = [v if not np.isnan(v) else 0 for v in stats["per_class_iou"]]
        sorted_idx  = np.argsort(ious)[::-1]
        sorted_ious = [ious[i] for i in sorted_idx]
        sorted_names = [VOC_CLASSES[i] for i in sorted_idx]
        bar_colors   = ['#2ecc71' if v >= 85 else '#e74c3c' if v < 60 else '#f39c12'
                        for v in sorted_ious]
        ax3.bar(range(len(sorted_names)), sorted_ious, color=bar_colors)
        ax3.set_xticks(range(len(sorted_names)))
        ax3.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=9)
        ax3.axhline(y=stats["mIoU"], color='navy', linestyle='--', lw=1.5,
                    label=f'mIoU={stats["mIoU"]:.1f}%')
        ax3.set_title("(c) Per-Class IoU (sorted)", fontsize=14, fontweight='bold')
        ax3.set_ylabel("IoU (%)"); ax3.legend(fontsize=10)
        ax3.grid(True, axis='y', alpha=0.3); ax3.set_ylim(0, 105)

        # Panel 4: Final metrics table
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        metric_names  = ["mIoU", "Pixel Accuracy", "Mean Accuracy",
                         "FW-IoU", "Mean F1", "Mean Dice", "Best Epoch"]
        metric_values = [
            f"{stats['mIoU']:.2f}%",
            f"{stats['pixel_acc']:.2f}%",
            f"{stats['mean_acc']:.2f}%",
            f"{stats['fw_iou']:.2f}%",
            f"{stats['mean_f1']:.2f}%",
            f"{stats['mean_dice']:.2f}%",
            f"{self.history['best_epoch']}"
        ]
        table_data = list(zip(metric_names, metric_values))
        table = ax4.table(cellText=table_data,
                          colLabels=["Metric", "Value"],
                          cellLoc='center', loc='center',
                          bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 1:
                cell.set_facecolor('#ecf0f1')
        ax4.set_title("(d) Summary Metrics", fontsize=14, fontweight='bold')

        fig.suptitle(
            "ResNet50-ASPP-ResNet50Decoder Segmentation Results\n(Pascal VOC 2012)",
            fontsize=16, fontweight='bold', y=1.01
        )
        plt.savefig(os.path.join(self.cfg["save_dir"], "graphs", "research_summary_FINAL.png"),
                    dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info("  ✓ Research summary figure saved.")

    def save_history(self):
        safe = {k: v for k, v in self.history.items()
                if isinstance(v, (list, float, int, str))}
        with open(os.path.join(self.cfg["save_dir"], "logs", "training_history.json"), "w") as f:
            json.dump(safe, f, indent=2)
        logger.info("  ✓ Training history saved.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── Datasets ─────────────────────────────
    train_ds, val_ds = build_datasets(CFG)

    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"],
        shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=4,
        shuffle=False, num_workers=4,
        pin_memory=True
    )

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ────────────────────────────────
    logger.info("Building model: ResNet50 → ASPP → ResNet50Decoder")
    model = SegmentationModel(num_classes=CFG["num_classes"], pretrained=True)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"Total params: {n_params:.2f}M | Trainable: {n_train:.2f}M")

    # ── Train ────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, CFG)
    history = trainer.train()

    logger.info(f"\nAll outputs saved to: {CFG['save_dir']}")
    return history


if __name__ == "__main__":
    main()
