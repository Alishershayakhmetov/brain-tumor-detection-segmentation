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
from torch.utils.data import Dataset, DataLoader
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
torch.autograd.set_detect_anomaly(True)
SEED = 42

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

# Checkpoint
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(path, model, optimizer, scaler, epoch, best_loss=None):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_loss": best_loss,
    }, path)

def load_checkpoint(path, model, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=True) # weights_only=False
    model.load_state_dict(ckpt["model_state"])

    if optimizer is not None and "optimizer_state" in ckpt and ckpt["optimizer_state"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])

    if scheduler and ckpt.get("scheduler_state"):  # ← add this
        scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_loss = ckpt.get("best_loss", None)
    return start_epoch, best_loss

def latest_checkpoint(ckpt_dir: Path):
    # expects files like epoch_001.pt, epoch_002.pt...
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
    - If file is missing, fills that channel / mask with zeros (and logs a warning).
    """
    case_id = row["id"]
    img_out_path = os.path.join(OUT_IMG_DIR, f"{case_id}.npy")
    mask_out_path = os.path.join(OUT_MASK_DIR, f"{case_id}.npy")

    if os.path.exists(img_out_path) and os.path.exists(mask_out_path):
        return {"image": img_out_path, "mask": mask_out_path, "split": row["split"], "dataset": row["dataset"]}

    def _resolve_path(value):
        """Return a Path if valid string, else None. Handles relative->DATASET, absolute stays absolute."""
        # if pd.isna(value) or not isinstance(value, str) or value.strip() == "":
        #     return None
        # p = Path(value)
        # return p

        p = Path(value)
        if not p.is_absolute():
            p = DATASET / p
        return p

    def _load_modality_or_zeros(p: Path | None, name: str):
        """Load, resample, crop/pad, normalize. If missing -> zeros."""
        if p is None:
            return np.zeros(TARGET_SHAPE, dtype=np.float16)

        if not p.is_file():
            print(f"[WARN] Missing modality file ({name}): {p}")
            return np.zeros(TARGET_SHAPE, dtype=np.float16)

        img = sitk.ReadImage(str(p))
        img = resample(img, is_label=False)
        arr = sitk.GetArrayFromImage(img)  # (z,y,x)
        arr = center_crop_or_pad(arr, TARGET_SHAPE)

        arr = arr.astype(np.float32)
        p1, p99 = np.percentile(arr, (1, 99))
        arr = np.clip(arr, p1, p99)

        mean = arr.mean(dtype=np.float64)
        std = arr.std(dtype=np.float64)
        arr = (arr - mean) / (std + 1e-6)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(np.float16)

    def _load_mask_or_zeros(p: Path | None):
        """Load seg, resample with NN, crop/pad, binarize (WT), add channel dim. If missing -> zeros."""
        if p is None or (not p.is_file()):
            if p is not None and (not p.is_file()):
                print(f"[WARN] Missing mask file (seg): {p}")
            return np.zeros((1, *TARGET_SHAPE), dtype=np.uint8)

        mask_img = sitk.ReadImage(str(p))
        mask_img = resample(mask_img, is_label=True)  # NN
        mask = sitk.GetArrayFromImage(mask_img)       # (z,y,x)

        # Whole Tumor for BraTS: any label > 0 (labels often 0,1,2,4)
        mask = (mask > 0).astype(np.uint8)

        mask = center_crop_or_pad(mask, TARGET_SHAPE)
        return mask[None, ...]  # (1,z,y,x)

    try:
        # Images (4 modalities)
        channels = []
        modalities = ["t1_path", "t1c_path", "t2_path", "flair_path"]

        for m in modalities:
            p = _resolve_path(row.get(m))
            channels.append(_load_modality_or_zeros(p, m))

        image = np.stack(channels, axis=0).astype(np.float16)  # (4,z,y,x)

        # Mask (seg)
        seg_p = _resolve_path(row.get("seg_path"))
        mask = _load_mask_or_zeros(seg_p)

        # print(f"[DEBUG] {case_id} seg:", seg_p, "exists:", (seg_p.is_file() if seg_p else None),
        #       "| mask sum:", mask.sum(dtype=np.float64), "| image shape:", image.shape)

        np.save(img_out_path, image)
        np.save(mask_out_path, mask)

        return {"image": img_out_path, "mask": mask_out_path, "split": row["split"], "dataset": row["dataset"]}

    except Exception as e:
        print(f"[ERROR] processing {case_id}: {e}")
        return None


# MODEL ARCHITECTURE

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # GroupNorm instead of InstanceNorm - stable at small spatial dims
        # num_groups = min(8, out_ch)  # groups must divide out_ch
        num_groups = min(out_ch // 4, 8)  # ensure group_size >= 4
        num_groups = max(1, num_groups)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            # nn.InstanceNorm3d(out_ch),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            # nn.InstanceNorm3d(out_ch),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.conv(x)


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

# with 3D attention gate
# class Up(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
#         self.att = AttentionGate(out_ch, out_ch, out_ch // 2)
#         self.conv = DoubleConv(in_ch, out_ch)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x2 = self.att(x1, x2)
#         return self.conv(torch.cat([x2, x1], dim=1))

# without 3D attention gate
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))

class AttentionUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(4, 16)
        self.d1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(16, 32))
        self.d2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(32, 64))
        self.d3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64, 128))
        self.d4 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(128, 256))

        self.u1 = Up(256, 128)
        self.u2 = Up(128, 64)
        self.u3 = Up(64, 32)
        self.u4 = Up(32, 16)
        self.outc = nn.Conv3d(16, 1, 1)

        # Initialize output bias: sigmoid(b) = fg_ratio -> b = log(fg / (1 - fg))
        # appx 6% of voxels are tumor across the dataset
        fg_ratio = 0.06
        nn.init.constant_(self.outc.bias, np.log(fg_ratio / (1.0 - fg_ratio)))  # appx -2.75

        nn.init.normal_(self.outc.weight, mean=0.0, std=0.01)  # smaller than kaiming default

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        # return torch.clamp(self.outc(x), -20, 20)
        raw = self.outc(x)
        return 20.0 * torch.tanh(raw / 20.0)  # soft clamp, gradient never dies


# TRAINING UTILS

class MRIDataset(Dataset):
    def __init__(self, df, training=False):
        self.df = df.reset_index(drop=True)
        self.training = training

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = np.load(row["image"])   # (4, D, H, W)
        mask = np.load(row["mask"])   # (1, D, H, W)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        if self.training:
            # --- augmentation (training only, so pass a flag or check split) ---
            # 1. Random horizontal flip
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[3])  # flip W
                mask = torch.flip(mask, dims=[3])

            # 2. Random vertical flip
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[2])  # flip H
                mask = torch.flip(mask, dims=[2])

            # 3. Random intensity scale (per-channel, MRI-safe)
            scale = torch.empty(4, 1, 1, 1).uniform_(0.9, 1.1)
            img = img * scale

        return img, mask

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5, pos_weight=3.0):
        super().__init__()
        self.smooth = smooth
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        logits = logits.float()  # force fp32
        target = target.float()
        # logits = torch.clamp(logits, -30, 30)

        probs = torch.sigmoid(logits)
        # Clamp probs to avoid gradient vanishing at extremes
        probs_clamped = probs.clamp(1e-4, 1 - 1e-4)
        dims = (2, 3, 4)

        intersection = (probs_clamped * target).sum(dims)
        p_sum = probs_clamped.sum(dims)
        t_sum = target.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (p_sum + t_sum + self.smooth)

        # # Correct empty mask handling: reward near-zero prediction probability
        # empty = (t_sum == 0)
        # # avg_prob = probs.mean(dim=dims)  # mean sigmoid, not sum
        # weight = 1.0 / (t_sum + 1.0)  # inverse-frequency weighting
        # # dice = torch.where(empty, 1.0 - avg_prob, dice)
        # dice_weighted = (dice * weight).sum() / weight.sum()
        #
        # dice_loss = 1.0 - dice_weighted

        # empty mask handling
        empty = (t_sum == 0)
        avg_prob = probs.mean(dim=dims)
        dice = torch.where(empty, 1.0 - avg_prob, dice)
        dice_loss = 1.0 - dice.mean()

        # BCE with pos_weight to counteract class imbalance
        pw = torch.tensor(self.pos_weight, dtype=torch.float32, device=logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=pw,
            reduction="mean"
        )

        return 0.8 * dice_loss + 0.2 * bce


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# early stop

def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, smooth=1e-5) -> torch.Tensor:
    """
    logits/target: [B,1,D,H,W] -> returns mean dice over batch (scalar tensor)
    """
    probs = torch.sigmoid(logits)
    dims = (2, 3, 4)
    intersection = (probs * target).sum(dims)
    p_sum = probs.sum(dims)
    t_sum = target.sum(dims)
    dice = (2.0 * intersection + smooth) / (p_sum + t_sum + smooth)

    # handle empty GT masks: if GT empty, dice = 1 if prediction is (almost) empty, else 0
    empty = (t_sum == 0)
    dice = torch.where(empty, (p_sum < 1.0).float(), dice)
    return dice.mean()

@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_fg_ratio = 0.0
    total_y_fg_ratio = 0.0
    total_batches = 0

    logits_min = float("inf")
    logits_max = float("-inf")

    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        x = x.to(device, memory_format=torch.channels_last_3d, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        fg_ratio = (probs > 0.3).float().mean().item()
        y_fg_ratio = (y > 0.3).float().mean().item()

        d = dice_score_from_logits(logits, y).item()

        total_loss += loss.item()
        total_dice += d
        total_fg_ratio += fg_ratio
        total_y_fg_ratio += y_fg_ratio
        total_batches += 1

        logits_min = min(logits_min, float(logits.min().item()))
        logits_max = max(logits_max, float(logits.max().item()))

    if total_batches == 0:
        return {
            "loss": float("inf"),
            "dice": 0.0,
            "fg_ratio": 0.0,
            "y_fg_ratio": 0.0,
            "logits_min": 0.0,
            "logits_max": 0.0,
        }

    return {
        "loss": total_loss / total_batches,
        "dice": total_dice / total_batches,
        "fg_ratio": total_fg_ratio / total_batches,
        "y_fg_ratio": total_y_fg_ratio / total_batches,
        "logits_min": logits_min,
        "logits_max": logits_max,
    }

def grad_norm_l2(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.pow(2).sum().item())
    return total ** 0.5

@dataclass
class EarlyStopper:
    patience: int = 15
    min_delta: float = 0.0
    mode: str = "min"  # "min" for loss, "max" for dice
    best: float = None
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        """
        Returns True if should stop.
        """
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False

        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)

        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

# MAIN EXECUTION

if __name__ == "__main__":
    seed_everything(SEED)
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)

    raw_df = pd.read_csv(META_PATH_UNPROCESSED)

    ALL_DATASETS = config.DATASET_LIST
    rows = [row for _, row in raw_df.iterrows() if row['dataset'] in ALL_DATASETS]

    print(f"Parallel Preprocessing: {len(rows)} cases...")
    workers = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(process_single_case, rows), total=len(rows)))

    valid_results = [r for r in results if r is not None]
    meta = pd.DataFrame(valid_results)
    meta.to_csv(META_PATH, index=False)

    sample = np.load(config.VALIDATE_SAMPLE)
    print(sample.shape)
    print(sample[0].sum(dtype=np.float64), sample[1].sum(dtype=np.float64), sample[2].sum(dtype=np.float64), sample[3].sum(dtype=np.float64))  # Should be 0.0 (because t1c was missing)

    meta = pd.read_csv(META_PATH)
    df = meta[meta.split == "train"].reset_index(drop=True)

    img_sums = []
    mask_sums = []
    empty_masks = 0

    for i in range(min(len(df), 200)):  # sample 200 cases
        img = np.load(df.loc[i, "image"])  # (4,z,y,x)
        msk = np.load(df.loc[i, "mask"])  # (1,z,y,x)

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
    printed_stats = False
    g = torch.Generator()
    g.manual_seed(SEED)
    train_df = meta[meta.split == "train"]
    # keep only cases where mask has at least 1 positive voxel
    keep = []
    for p in tqdm(train_df["mask"].values, desc="Filtering empty masks"):
        m = np.load(p, mmap_mode="r")
        keep.append(m.sum() > 0)

    train_df = train_df.loc[keep].reset_index(drop=True)
    val_df = meta[meta.split == "val"]
    # Filter val to only non-empty masks too
    keep_val = []
    for p in tqdm(val_df["mask"].values, desc="Filtering val empty masks"):
        m = np.load(p, mmap_mode="r")
        keep_val.append(m.sum() > 0)
    val_df = val_df.loc[keep_val].reset_index(drop=True)
    print(f"Train cases: {len(train_df)} | Val cases (non-empty only): {len(val_df)}")
    print(f"Train cases: {len(train_df)} | Val cases: {len(val_df)}")

    empty = 0
    total = 0
    for p in train_df["mask"].values:
        m = np.load(p)
        if m.sum(dtype=np.float64) == 0:
            empty += 1
        total += 1
    print(f"Empty masks: {empty}/{total} ({empty / total * 100:.2f}%)")
    mask_paths = train_df["mask"].values[:5]
    for p in mask_paths:
        m = np.load(p)
        print(p, "sum=", float(m.sum(dtype=np.float64)), "min/max=", float(m.min()), float(m.max()), "unique_count=", len(np.unique(m)))

    def count_nonempty(df):
        n = 0
        for p in df["mask"].values:
            m = np.load(p, mmap_mode="r")
            if m.sum() > 0:
                n += 1
        return n, len(df)

    vn, vt = count_nonempty(val_df)
    tn, tt = count_nonempty(train_df)
    print(f"Train non-empty: {tn}/{tt}")
    print(f"Val non-empty:   {vn}/{vt}")

    train_loader = DataLoader(
        MRIDataset(train_df, training=True),
        batch_size=2,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        MRIDataset(val_df, training=False),
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )

    max_epochs = config.EPOCH_NUMBER

    model = AttentionUNet3D().to(DEVICE, memory_format=torch.channels_last_3d)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "outc" not in n], "weight_decay": 1e-5},
        {"params": model.outc.parameters(), "weight_decay": 1e-3},  # 100x stronger on output layer
    ], lr=config.LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=max_epochs,
    #     eta_min=1e-6
    # )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=max_epochs,
        pct_start=0.05  # 5% of training for warmup
    )

    scaler = torch.amp.GradScaler("cuda")


    start_epoch = 1
    best_loss = float("inf")

    # resume if checkpoint exists
    ckpt_path = latest_checkpoint(CHECKPOINT_DIR)
    if ckpt_path is not None:
        start_epoch, best_loss_loaded = load_checkpoint(
            ckpt_path, model, optimizer, scaler, map_location=DEVICE
        )
        if best_loss_loaded is not None:
            best_loss = best_loss_loaded
        print(f"Resumed from {ckpt_path.name} at epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from scratch.")

    # Training Loop
    early = EarlyStopper(patience=config.EARLY_STOPPING_EPOCH, min_delta=1e-4, mode="min")
    print(f"Starting Training on {DEVICE}...")
    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_fg_ratio = 0.0
        epoch_y_fg_ratio = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x = x.to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
                print(f"logit mean={logits.mean():.3f}, std={logits.std():.3f}, max={logits.max():.3f}")

                # # (logit regularization)
                # logit_reg = 1e-5 * torch.mean(logits ** 2)
                # loss = loss_main + logit_reg

            scaler.scale(loss).backward()

            # gradient norm
            scaler.unscale_(optimizer)
            gn = grad_norm_l2(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # print if logits are still escaping
            with torch.no_grad():
                lmax = logits.abs().max().item()
                lmin = logits.abs().min().item()
                if lmax > 15.0:
                    print(f"\n[WARN] logits abs-max={lmax:.1f} at batch {batches} — check initialization")
                if lmin < -15.0:
                    print(f"\n[WARN] logits abs-min={lmin:.1f} at batch {batches} — check initialization")

            with torch.no_grad():
                d = dice_score_from_logits(logits, y).item()
                probs = torch.sigmoid(logits)
                fg_ratio = (probs > 0.3).float().mean().item()
                y_fg_ratio = (y > 0.3).float().mean().item()

            epoch_loss += loss.item()
            epoch_dice += d
            epoch_fg_ratio += fg_ratio
            epoch_y_fg_ratio += y_fg_ratio
            batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dice": f"{d:.4f}",
                "fg>0.3": f"{fg_ratio:.4f}",
                "gn": f"{gn:.2f}",
            })

        train_avg_loss = epoch_loss / max(1, batches)
        train_avg_dice = epoch_dice / max(1, batches)
        train_avg_fg = epoch_fg_ratio / max(1, batches)
        train_avg_yfg = epoch_y_fg_ratio / max(1, batches)

        # Validation
        if len(val_df) > 0:
            # Limit val batches during early epochs
            val_stats = evaluate(model, val_loader, criterion, DEVICE, max_batches=100)
        else:
            val_stats = {"loss": float("inf"), "dice": 0.0, "fg_ratio": 0.0, "y_fg_ratio": 0.0, "logits_min": 0.0,
                         "logits_max": 0.0}

        # Debug summary (per epoch)
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print("\n" + "=" * 80)
        print(f"Epoch {epoch} finished in {dt:.1f}s | lr={lr:g}")
        print(
            f"[TRAIN] loss={train_avg_loss:.5f} | dice={train_avg_dice:.5f} | fg>0.3={train_avg_fg:.6f} | y_fg={train_avg_yfg:.6f}")
        print(
            f"[VAL  ] loss={val_stats['loss']:.5f} | dice={val_stats['dice']:.5f} | fg>0.3={val_stats['fg_ratio']:.6f} | y_fg={val_stats['y_fg_ratio']:.6f}")
        print(f"[VAL  ] logits min/max = {val_stats['logits_min']:.3f} / {val_stats['logits_max']:.3f}")

        # Simple “overfit signal”
        if len(val_df) > 0:
            gap = val_stats["loss"] - train_avg_loss
            print(f"[GAP  ] val_loss - train_loss = {gap:.5f} (positive & growing often = overfitting)")
        print("=" * 80 + "\n")

        # Save best (based on val loss)
        current_metric = val_stats["loss"] if len(val_df) > 0 else train_avg_loss
        if current_metric < best_loss:
            best_loss = current_metric
            save_checkpoint(CHECKPOINT_DIR / "best.pt",
                            model, optimizer, scaler, epoch, best_loss)
            print(f"[BEST] New best metric: {best_loss:.5f} -> saved best.pt")

        save_checkpoint(CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt",
                        model, optimizer, scaler, epoch, best_loss)

        # Early stopping (uses val loss)
        if len(val_df) > 0:
            should_stop = early.step(val_stats["loss"])
            print(f"[EARLY] best_val_loss={early.best:.5f} | bad_epochs={early.bad_epochs}/{early.patience}")
            if should_stop:
                print(f"[EARLY STOP] No val improvement for {early.patience} epochs. Stopping at epoch {epoch}.")
                break
