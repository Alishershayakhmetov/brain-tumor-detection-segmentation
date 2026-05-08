import os
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import config
from pathlib import Path
import random
from dataclasses import dataclass

# CONFIG
TARGET_SPACING = config.TARGET_SPACING
TARGET_SHAPE = config.TARGET_SHAPE
DATASET = config.DATASET
OUT_IMG_DIR = config.OUT_IMG_DIR
OUT_MASK_DIR = config.OUT_MASK_DIR
META_PATH_UNPROCESSED = config.META_PATH_UNPROCESSED
META_PATH = config.META_PATH
CHECKPOINT_DIR = config.CHECKPOINT_DIR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# torch.autograd.set_detect_anomaly(True)  # turn off after testing
SEED = 42

# All modalities in a fixed order. Channels are always kept in this order
# missing ones are zeroed out and flagged in the modality_mask
ALL_MODALITIES = ["t1_path", "t1c_path", "t2_path", "flair_path"]
NUM_MODALITIES = len(ALL_MODALITIES)

# Optimize SimpleITK
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(4)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Checkpoint helpers
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, best_loss=None):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_loss": best_loss,
    }, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    if scheduler and ckpt.get("scheduler_state"):
        scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_loss = ckpt.get("best_loss", None)
    return start_epoch, best_loss

def latest_checkpoint(ckpt_dir: Path):
    pts = sorted(ckpt_dir.glob("epoch_*.pt"))
    return pts[-1] if pts else None

# PREPROCESSING UTILS

def center_crop_or_pad(vol, target_shape):
    z, y, x = vol.shape
    tz, ty, tx = target_shape
    out = np.zeros(target_shape, dtype=vol.dtype)

    z0, y0, x0 = max((tz - z) // 2, 0), max((ty - y) // 2, 0), max((tx - x) // 2, 0)
    zs, ys, xs = max((z - tz) // 2, 0), max((y - ty) // 2, 0), max((x - tx) // 2, 0)
    zl, yl, xl = min(z, tz), min(y, ty), min(x, tx)

    out[z0:z0 + zl, y0:y0 + yl, x0:x0 + xl] = vol[zs:zs + zl, ys:ys + yl, xs:xs + xl]
    return out

def resample(img, is_label=False):
    res = sitk.ResampleImageFilter()
    res.SetOutputSpacing(TARGET_SPACING)
    res.SetSize([
        int(round(img.GetSize()[i] * img.GetSpacing()[i] / TARGET_SPACING[i]))
        for i in range(3)
    ])
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    res.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return res.Execute(img)

def process_single_case(row):
    """
    Robust path handling:
    - Accepts NaN/None/"" for any modality/seg path.
    - Resolves relative paths under DATASET.
    - If a path is already absolute, uses it as-is.
    - Saves a companion modality_mask .npy (shape [NUM_MODALITIES]) alongside each image
      so that the model knows which channels contain real data at runtime.
    """
    case_id = row["id"]
    img_out_path = os.path.join(OUT_IMG_DIR, f"{case_id}.npy")
    mask_out_path = os.path.join(OUT_MASK_DIR, f"{case_id}.npy")
    modality_mask_path = os.path.join(OUT_IMG_DIR, f"{case_id}_modmask.npy")

    if os.path.exists(img_out_path) and os.path.exists(mask_out_path) and os.path.exists(modality_mask_path):
        return {
            "image": img_out_path,
            "mask": mask_out_path,
            "modality_mask": modality_mask_path,
            "split": row["split"],
            "dataset": row["dataset"],
            "has_tumor": int(row.get("has_tumor", 0)),
            "tumor_type": str(row.get("tumor_type", "healthy")),
        }

    def _resolve_path(value):
        if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
            return None
        return Path(value)

    def _load_modality_or_zeros(p, name):
        """Load, resample, crop/pad, normalize. Returns (array, is_real: bool)."""
        if p is None:
            return np.zeros(TARGET_SHAPE, dtype=np.float16), False
        if not p.is_file():
            print(f"[WARN] Missing modality file ({name}): {p}")
            return np.zeros(TARGET_SHAPE, dtype=np.float16), False

        img = sitk.ReadImage(str(p))
        img = resample(img, is_label=False)
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        arr = center_crop_or_pad(arr, TARGET_SHAPE)

        p1, p99 = np.percentile(arr, (1, 99))
        arr = np.clip(arr, p1, p99)

        mean = arr.mean(dtype=np.float64)
        std = arr.std(dtype=np.float64)
        arr = (arr - mean) / (std + 1e-6)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(np.float16), True

    def _load_mask_or_zeros(p):
        if p is None or (not p.is_file()):
            if p is not None and (not p.is_file()):
                print(f"[WARN] Missing mask file (seg): {p}")
            return np.zeros((1, *TARGET_SHAPE), dtype=np.uint8)

        mask_img = sitk.ReadImage(str(p))
        mask_img = resample(mask_img, is_label=True)
        mask = sitk.GetArrayFromImage(mask_img)
        mask = (mask > 0).astype(np.uint8)
        mask = center_crop_or_pad(mask, TARGET_SHAPE)
        return mask[None, ...]  # (1,z,y,x)

    try:
        channels = []
        modality_present = []

        for m in ALL_MODALITIES:
            p = _resolve_path(row.get(m))
            arr, is_real = _load_modality_or_zeros(p, m)
            channels.append(arr)
            modality_present.append(float(is_real))

        image = np.stack(channels, axis=0).astype(np.float16)          # (4,z,y,x)
        modality_mask = np.array(modality_present, dtype=np.float32)   # (4,)

        seg_p = _resolve_path(row.get("seg_path"))
        mask = _load_mask_or_zeros(seg_p)

        np.save(img_out_path, image)
        np.save(mask_out_path, mask)
        np.save(modality_mask_path, modality_mask)

        return {
            "image": img_out_path,
            "mask": mask_out_path,
            "modality_mask": modality_mask_path,
            "split": row["split"],
            "dataset": row["dataset"],
            "has_tumor": int(row.get("has_tumor", 0)),
            "tumor_type": str(row.get("tumor_type", "healthy")),
        }

    except Exception as e:
        print(f"[ERROR] processing {case_id}: {e}")
        return None


# Tumor-type label helpers
def build_tumor_type_map(df: pd.DataFrame):
    """
    Returns a sorted list of unique tumor types and a dict mapping type->int.
    """
    types = sorted(df["tumor_type"].dropna().unique().tolist())
    if "healthy" in types:
        types.remove("healthy")
    types = ["healthy"] + types # healthy is always class 0
    type2idx = {t: i for i, t in enumerate(types)}
    return types, type2idx


def compute_class_weights(df: pd.DataFrame, type2idx: dict, device: str) -> torch.Tensor:
    """
    Weights are capped at max_weight (default 4.0)
    """
    num_classes = len(type2idx)
    labels = np.array([type2idx.get(str(t), 0) for t in df["tumor_type"]])
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    # Cap so no class exceeds 4× the average weight — prevents tiny classes
    # from monopolising the gradient signal
    weights = np.clip(weights, 0.0, 4.0)
    weights = weights / weights.sum() * num_classes  # re-normalise to keep scale ~ 1
    print(f"Class weights (capped): { {t: f'{weights[i]:.3f}' for t, i in type2idx.items()} }")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def make_weighted_sampler(df: pd.DataFrame, type2idx: dict) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so each mini-batch contains a
    balanced representation of all tumor types, including healthy
    """
    labels = np.array([type2idx.get(str(t), 0) for t in df["tumor_type"]])
    num_classes = len(type2idx)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    # Per-sample weight = inverse class frequency
    sample_weights = 1.0 / counts[labels]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )


# MODEL ARCHITECTURE
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        num_groups = min(out_ch // 4, 8)
        num_groups = max(1, num_groups)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, int_ch):
        super().__init__()
        self.Wg = nn.Conv3d(g_ch, int_ch, 1)
        self.Wx = nn.Conv3d(x_ch, int_ch, 1)
        self.psi = nn.Sequential(nn.Conv3d(int_ch, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.Wg(g) + self.Wx(x))
        return x * self.psi(psi)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.att = AttentionGate(out_ch, out_ch, out_ch // 2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.att(x1, x2)
        return self.conv(torch.cat([x2, x1], dim=1))


class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net with two heads:
      1. Segmentation head: (B, 1, D, H, W) - tumour voxel mask
      2. Classification head: (B, num_tumor_types) - tumour type logits

    Variable modality support:
    The model accepts a modality_mask tensor of shape (B, NUM_MODALITIES).
    Each entry is 1.0 if the corresponding MRI channel is present, 0.0 if missing.
    """

    def __init__(self, num_tumor_types: int = 2):
        super().__init__()

        # (1, NUM_MODALITIES, 1, 1, 1)
        # Stored as rank-5
        self.modality_embed = nn.Parameter(
            torch.zeros(1, NUM_MODALITIES, 1, 1, 1)
        )

        # Encoder
        self.inc = DoubleConv(NUM_MODALITIES, 16)
        self.d1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(16, 32))
        self.d2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(32, 64))
        self.d3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64, 128))
        self.d4 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(128, 256))

        # Decoder (segmentation)
        self.u1 = Up(256, 128)
        self.u2 = Up(128, 64)
        self.u3 = Up(64, 32)
        self.u4 = Up(32, 16)
        self.outc = nn.Conv3d(16, 1, 1)

        # Pools both bottleneck (256ch) and d3 (128ch) to (2,2,2) spatial,
        # then concatenates with modality_mask for full context.
        # pool to 2^3=8 spatial positions → 256*8 + 128*8 + NUM_MODALITIES features
        self.cls_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        cls_in_features = 256 * 8 + 128 * 8 + NUM_MODALITIES

        self.classifier = nn.Sequential(
            nn.Linear(cls_in_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_tumor_types),
        )

        # Output bias init: sigmoid(b) ~~ 6% fg voxels
        fg_ratio = 0.06
        nn.init.constant_(self.outc.bias, np.log(fg_ratio / (1.0 - fg_ratio)))
        nn.init.normal_(self.outc.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor):
        """
        x: (B, NUM_MODALITIES, D, H, W) float32
        modality_mask: (B, NUM_MODALITIES) 1=present, 0=absent

        Returns:
        seg_logits: (B, 1, D, H, W)
        cls_logits: (B, num_tumor_types)
        """
        mm5d = modality_mask[:, :, None, None, None] #(B,4,1,1,1)
        x = x * mm5d + self.modality_embed # embed always present

        # Encoder
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3) # (B,128,d,h,w)
        x5 = self.d4(x4) # bottleneck (B,256,d,h,w)

        # Segmentation decoder
        xd = self.u1(x5, x4)
        xd = self.u2(xd, x3)
        xd = self.u3(xd, x2)
        xd = self.u4(xd, x1)
        raw = self.outc(xd)
        seg_logits = 20.0 * torch.tanh(raw / 20.0) # soft clamp +/- 20

        gap_bottle = self.cls_pool(x5).flatten(1) # (B, 256*8)
        gap_mid    = self.cls_pool(x4).flatten(1) # (B, 128*8)
        cls_input  = torch.cat([gap_bottle, gap_mid, modality_mask], dim=1)
        cls_logits = self.classifier(cls_input) # (B, num_tumor_types)

        return seg_logits, cls_logits


# LOSSES
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5, pos_weight=3.0):
        super().__init__()
        self.smooth = smooth
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        logits = logits.float()
        target = target.float()

        probs = torch.sigmoid(logits)
        probs_clamped = probs.clamp(1e-4, 1 - 1e-4)
        dims = (2, 3, 4)

        intersection = (probs_clamped * target).sum(dims)
        p_sum = probs_clamped.sum(dims)
        t_sum = target.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (p_sum + t_sum + self.smooth)

        # Empty mask: reward near-zero prediction
        empty = (t_sum == 0)
        avg_prob = probs.mean(dim=dims)
        dice = torch.where(empty, 1.0 - avg_prob, dice)
        dice_loss = 1.0 - dice.mean()

        pw = torch.tensor(self.pos_weight, dtype=torch.float32, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="mean")

        return 0.8 * dice_loss + 0.2 * bce


class CombinedLoss(nn.Module):
    """
    Segmentation + Classification loss

    seg_weight: weight for the segmentation (DiceBCE) term
    cls_weight: weight for the classification (CrossEntropy) term
    cls_weights: (num_classes,) float tensor of inverse-frequency class weights
    """

    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 0.5,
                 seg_pos_weight: float = 3.0, cls_weights: torch.Tensor = None):
        super().__init__()
        self.seg_loss_fn = DiceBCELoss(pos_weight=seg_pos_weight)
        self.cls_loss_fn = nn.CrossEntropyLoss(
            weight=cls_weights,
            label_smoothing=0.1,
        )
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def forward(self, seg_logits, seg_target, cls_logits, cls_target):
        """
        seg_logits: (B, 1, D, H, W)
        seg_target: (B, 1, D, H, W) binary
        cls_logits: (B, num_types)
        cls_target: (B,) long class indices
        """
        l_seg = self.seg_loss_fn(seg_logits, seg_target)
        l_cls = self.cls_loss_fn(cls_logits, cls_target)
        return self.seg_weight * l_seg + self.cls_weight * l_cls, l_seg, l_cls


# DATASET

class MRIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, type2idx: dict, training: bool = False):
        self.df = df.reset_index(drop=True)
        self.type2idx = type2idx
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        img = np.load(row["image"]) # (4, D, H, W)  float16
        mask = np.load(row["mask"]) # (1, D, H, W)  uint8
        modality_mask = np.load(row["modality_mask"]) # (4,) float32

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        modality_mask = torch.from_numpy(modality_mask).float()

        if self.training:
            # With 30% probability, randomly zero out one of the present modalities
            if torch.rand(1).item() < 0.3:
                present_indices = modality_mask.nonzero(as_tuple=False).squeeze(1)
                if len(present_indices) > 1: # never drop the last remaining modality
                    drop_idx = present_indices[
                        torch.randint(len(present_indices), (1,)).item()
                    ]
                    img[drop_idx] = 0.0
                    modality_mask[drop_idx] = 0.0

            # Random horizontal flip
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[3])
                mask = torch.flip(mask, dims=[3])
            # Random vertical flip
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[2])
            # Per-channel intensity scale (only on present modalities)
            scale = torch.empty(NUM_MODALITIES, 1, 1, 1).uniform_(0.9, 1.1)
            # Do not amplify noise in zeroed-out missing channels
            scale = scale * modality_mask[:, None, None, None] + \
                    (1.0 - modality_mask[:, None, None, None])
            img = img * scale

        # Tumor type label (int)
        tumor_type_str = str(row.get("tumor_type", "healthy"))
        # Fall back to "healthy" if unseen type encountered at runtime
        cls_label = self.type2idx.get(tumor_type_str, 0)
        cls_label = torch.tensor(cls_label, dtype=torch.long)

        return img, mask, modality_mask, cls_label


# TRAINING UTILS

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, smooth=1e-5):
    probs = torch.sigmoid(logits)
    dims = (2, 3, 4)
    intersection = (probs * target).sum(dims)
    p_sum = probs.sum(dims)
    t_sum = target.sum(dims)
    dice = (2.0 * intersection + smooth) / (p_sum + t_sum + smooth)
    empty = (t_sum == 0)
    dice = torch.where(empty, (p_sum < 1.0).float(), dice)
    return dice.mean()

def grad_norm_l2(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


@torch.no_grad()
def evaluate(model, loader, criterion: CombinedLoss, device, max_batches=None):
    model.eval()
    total_loss = total_seg_loss = total_cls_loss = 0.0
    total_dice = 0.0
    total_fg_ratio = 0.0
    total_y_fg_ratio = 0.0
    total_cls_correct = total_cls_samples = 0
    logits_min = float("inf")
    logits_max = float("-inf")
    total_batches = 0

    for bi, (x, y, mm, cls_target) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        x = x.to(device, memory_format=torch.channels_last_3d, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mm = mm.to(device, non_blocking=True)
        cls_target = cls_target.to(device, non_blocking=True)

        seg_logits, cls_logits = model(x, mm)
        loss, l_seg, l_cls = criterion(seg_logits, y, cls_logits, cls_target)

        probs = torch.sigmoid(seg_logits)
        fg_ratio = (probs > 0.3).float().mean().item()
        y_fg_ratio = (y > 0.3).float().mean().item()
        d = dice_score_from_logits(seg_logits, y).item()

        cls_pred = cls_logits.argmax(dim=1)
        total_cls_correct += (cls_pred == cls_target).sum().item()
        total_cls_samples += cls_target.size(0)

        total_loss += loss.item()
        total_seg_loss += l_seg.item()
        total_cls_loss += l_cls.item()
        total_dice += d
        total_fg_ratio += fg_ratio
        total_y_fg_ratio += y_fg_ratio
        total_batches += 1

        logits_min = min(logits_min, float(seg_logits.min().item()))
        logits_max = max(logits_max, float(seg_logits.max().item()))

    if total_batches == 0:
        return {
            "loss": float("inf"), "seg_loss": float("inf"), "cls_loss": float("inf"),
            "dice": 0.0, "fg_ratio": 0.0, "y_fg_ratio": 0.0,
            "cls_acc": 0.0, "logits_min": 0.0, "logits_max": 0.0,
        }

    return {
        "loss": total_loss / total_batches,
        "seg_loss": total_seg_loss / total_batches,
        "cls_loss": total_cls_loss / total_batches,
        "dice": total_dice / total_batches,
        "fg_ratio": total_fg_ratio / total_batches,
        "y_fg_ratio": total_y_fg_ratio / total_batches,
        "cls_acc": total_cls_correct / max(1, total_cls_samples),
        "logits_min": logits_min,
        "logits_max": logits_max,
    }


@dataclass
class EarlyStopper:
    patience: int = 15
    min_delta: float = 0.0
    mode: str = "min"
    best: float = None
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False
        improved = (value < self.best - self.min_delta) if self.mode == "min" else (
            value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


if __name__ == "__main__":
    seed_everything(SEED)
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)

    raw_df = pd.read_csv(META_PATH_UNPROCESSED)

    ALL_DATASETS = config.DATASET_LIST
    rows = [row for _, row in raw_df.iterrows() if row["dataset"] in ALL_DATASETS]

    print(f"Parallel Preprocessing: {len(rows)} cases...")
    workers = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(process_single_case, rows), total=len(rows)))

    valid_results = [r for r in results if r is not None]
    meta = pd.DataFrame(valid_results)
    meta.to_csv(META_PATH, index=False)

    # Quick sanity checks
    sample = np.load(config.VALIDATE_SAMPLE)
    print("Sample shape:", sample.shape)
    print("Channel sums:", [sample[i].sum(dtype=np.float64) for i in range(sample.shape[0])])

    meta = pd.read_csv(META_PATH)

    # Build tumor-type label map from the processed metadata
    tumor_types, type2idx = build_tumor_type_map(meta)
    num_tumor_types = len(tumor_types)
    print(f"Tumor types ({num_tumor_types}): {tumor_types}")
    print(f"Label map: {type2idx}")

    df = meta[meta.split == "train"].reset_index(drop=True)

    img_sums, mask_sums, empty_masks = [], [], 0
    for i in range(min(len(df), 200)):
        img = np.load(df.loc[i, "image"])
        msk = np.load(df.loc[i, "mask"])
        img_sums.append(float(np.abs(img).mean()))
        s = float(msk.sum(dtype=np.float64))
        mask_sums.append(s)
        if s == 0:
            empty_masks += 1

    print("Image abs-mean (sample): min/avg/max =",
          min(img_sums), sum(img_sums) / len(img_sums), max(img_sums))
    print("Mask sum (sample): min/avg/max =",
          min(mask_sums), sum(mask_sums) / len(mask_sums), max(mask_sums))
    print(f"Empty masks in sample: {empty_masks}/{len(mask_sums)} ({empty_masks / len(mask_sums) * 100:.1f}%)")

    # Setup Training
    g = torch.Generator()
    g.manual_seed(SEED)

    train_df = meta[meta.split == "train"].copy()
    val_df = meta[meta.split == "val"].copy()

    # For segmentation: filter to cases with at least 1 positive voxel in mask.
    # Healthy / no-tumor cases are still included for classification training.

    # filter empty masks out of training entirely
    # keep = []
    # for p in tqdm(train_df["mask"].values, desc="Filtering empty masks"):
    #     m = np.load(p, mmap_mode="r")
    #     keep.append(m.sum() > 0)
    # train_df = train_df[keep].reset_index(drop=True)

    # keep all val cases
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Train cases: {len(train_df)} | Val cases: {len(val_df)}")

    # Mask stats
    empty = sum(
        1 for p in train_df["mask"].values if np.load(p, mmap_mode="r").sum() == 0
    )
    print(f"Empty masks in train: {empty}/{len(train_df)} ({empty / len(train_df) * 100:.2f}%)")

    # Tumor type distribution
    print("Train tumor type distribution:")
    print(train_df["tumor_type"].value_counts().to_string())

    cls_weights_tensor = compute_class_weights(train_df, type2idx, DEVICE)

    train_sampler = make_weighted_sampler(train_df, type2idx)

    train_loader = DataLoader(
        MRIDataset(train_df, type2idx=type2idx, training=True),
        batch_size=2,
        sampler=train_sampler,    # replaces shuffle=True; balances class distribution
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        MRIDataset(val_df, type2idx=type2idx, training=False),
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )

    max_epochs = config.EPOCH_NUMBER

    model = AttentionUNet3D(num_tumor_types=num_tumor_types).to(
        DEVICE, memory_format=torch.channels_last_3d
    )

    # seg_pos_weight raised to 8.0 as 85% of train masks are empty
    # the segmentation loss needs a strong positive-voxel incentive or the
    # model will learn to predict all-background on every case
    criterion = CombinedLoss(
        seg_weight=1.0,
        cls_weight=0.5,
        seg_pos_weight=8.0,
        cls_weights=cls_weights_tensor,
    )

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "outc" not in n and "classifier" not in n],
         "weight_decay": 1e-5},
        {"params": model.outc.parameters(), "weight_decay": 1e-3},
        {"params": model.classifier.parameters(), "weight_decay": 1e-4},
    ], lr=config.LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.EPOCH_SCHEDULE_RESTART,
        T_mult=1,
        eta_min=config.LEARNING_RATE_ETA_MIN,
    )

    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 1
    best_loss = float("inf")

    ckpt_path = latest_checkpoint(CHECKPOINT_DIR)
    if ckpt_path is not None:
        start_epoch, best_loss_loaded = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, scaler, map_location=DEVICE
        )
        if best_loss_loaded is not None:
            best_loss = best_loss_loaded
        print(f"Resumed from {ckpt_path.name} at epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from scratch.")

    early = EarlyStopper(patience=config.EARLY_STOPPING_EPOCH, min_delta=1e-4, mode="min")
    print(f"Starting Training on {DEVICE}...")

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        model.train()

        epoch_loss = epoch_seg_loss = epoch_cls_loss = 0.0
        epoch_dice = 0.0
        epoch_fg_ratio = epoch_y_fg_ratio = 0.0
        epoch_cls_correct = epoch_cls_samples = 0
        batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y, mm, cls_target in pbar:
            x = x.to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            mm = mm.to(DEVICE, non_blocking=True)
            cls_target = cls_target.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                seg_logits, cls_logits = model(x, mm)
                loss, l_seg, l_cls = criterion(seg_logits, y, cls_logits, cls_target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = grad_norm_l2(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Logit range warnings
            with torch.no_grad():
                lmax = seg_logits.abs().max().item()
                lmin = seg_logits.min().item()
                if lmax > 15.0:
                    print(f"\n[WARN] seg logits abs-max={lmax:.1f} at batch {batches}")
                if lmin < -15.0:
                    print(f"\n[WARN] seg logits abs-min={lmin:.1f} at batch {batches}")

            with torch.no_grad():
                d = dice_score_from_logits(seg_logits, y).item()
                probs = torch.sigmoid(seg_logits)
                fg_ratio = (probs > 0.3).float().mean().item()
                y_fg_ratio = (y > 0.3).float().mean().item()
                cls_pred = cls_logits.argmax(dim=1)
                correct = (cls_pred == cls_target).sum().item()

            epoch_loss += loss.item()
            epoch_seg_loss += l_seg.item()
            epoch_cls_loss += l_cls.item()
            epoch_dice += d
            epoch_fg_ratio += fg_ratio
            epoch_y_fg_ratio += y_fg_ratio
            epoch_cls_correct += correct
            epoch_cls_samples += cls_target.size(0)
            batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "seg": f"{l_seg.item():.4f}",
                "cls": f"{l_cls.item():.4f}",
                "dice": f"{d:.4f}",
                "cls_acc": f"{correct / cls_target.size(0):.2f}",
                "gn": f"{gn:.2f}",
            })

        n = max(1, batches)
        train_avg = {
            "loss": epoch_loss / n,
            "seg": epoch_seg_loss / n,
            "cls": epoch_cls_loss / n,
            "dice": epoch_dice / n,
            "fg": epoch_fg_ratio / n,
            "yfg": epoch_y_fg_ratio / n,
            "cls_acc": epoch_cls_correct / max(1, epoch_cls_samples),
        }

        # Validation
        if len(val_df) > 0:
            val_stats = evaluate(model, val_loader, criterion, DEVICE, max_batches=100)
        else:
            val_stats = {
                "loss": float("inf"), "seg_loss": float("inf"), "cls_loss": float("inf"),
                "dice": 0.0, "fg_ratio": 0.0, "y_fg_ratio": 0.0,
                "cls_acc": 0.0, "logits_min": 0.0, "logits_max": 0.0,
            }

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print("\n" + "=" * 90)
        print(f"Epoch {epoch} finished in {dt:.1f}s | lr={lr:g}")
        print(
            f"[TRAIN] loss={train_avg['loss']:.5f} | seg={train_avg['seg']:.5f} | "
            f"cls={train_avg['cls']:.5f} | dice={train_avg['dice']:.5f} | "
            f"cls_acc={train_avg['cls_acc']:.4f} | fg={train_avg['fg']:.6f}"
        )
        print(
            f"[VAL  ] loss={val_stats['loss']:.5f} | seg={val_stats['seg_loss']:.5f} | "
            f"cls={val_stats['cls_loss']:.5f} | dice={val_stats['dice']:.5f} | "
            f"cls_acc={val_stats['cls_acc']:.4f} | fg={val_stats['fg_ratio']:.6f}"
        )
        print(f"[VAL  ] seg logits min/max = {val_stats['logits_min']:.3f} / {val_stats['logits_max']:.3f}")

        if len(val_df) > 0:
            gap = val_stats["loss"] - train_avg["loss"]
            print(f"[GAP  ] val_loss - train_loss = {gap:.5f}")
        print("=" * 90 + "\n")

        current_metric = val_stats["loss"] if len(val_df) > 0 else train_avg["loss"]
        if current_metric < best_loss:
            best_loss = current_metric
            save_checkpoint(CHECKPOINT_DIR / "best.pt",
                            model, optimizer, scaler, scheduler, epoch, best_loss)
            print(f"[BEST] New best metric: {best_loss:.5f} -> saved best.pt")

        save_checkpoint(CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt",
                        model, optimizer, scaler, scheduler, epoch, best_loss)

        if len(val_df) > 0:
            should_stop = early.step(val_stats["loss"])
            print(f"[EARLY] best_val_loss={early.best:.5f} | bad_epochs={early.bad_epochs}/{early.patience}")
            if should_stop:
                print(f"[EARLY STOP] No val improvement for {early.patience} epochs. Stopping at epoch {epoch}.")
                break
