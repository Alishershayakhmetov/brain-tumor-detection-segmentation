"""
evaluate.py - Standalone test-time evaluator for the 3D Attention U-Net brain MRI model.

Usage
-----
  python evaluate.py --checkpoint checkpoints/best.pt --meta data/meta.csv
  python evaluate.py --checkpoint checkpoints/best.pt --meta data/meta.csv \
      --split test --batch-size 4 --threshold 0.5 --save-report report.json \
      --save-predictions preds/

What it reports
---------------
  Segmentation:
    • Mean Dice (all cases, only-tumor cases)
    • Voxel-level Precision / Recall / F1
    • Hausdorff distance 95th-percentile  (if scipy is installed)
    • Per-case Dice + volume error table
    • Confusion breakdown: TP / FP / FN / TN voxel counts
    • Predicted vs ground-truth foreground ratio

  Classification:
    • Overall accuracy
    • Per-class Precision / Recall / F1
    • Macro / weighted F1
    • Confusion matrix (pretty-printed)
    • Top-2 accuracy
    • Per-case class probabilities (saved to --save-predictions if given)

  Combined:
    • Total / seg / cls loss
    • Model parameter count
    • Inference throughput (cases/sec)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# CONFIG
try:
    import config as cfg
    TARGET_SPACING = cfg.TARGET_SPACING
    TARGET_SHAPE   = cfg.TARGET_SHAPE
    CHECKPOINT_DIR = cfg.CHECKPOINT_DIR
    META_PATH      = cfg.META_PATH
except ImportError:
    TARGET_SPACING = (1.0, 1.0, 1.0)
    TARGET_SHAPE   = (128, 128, 128)
    CHECKPOINT_DIR = Path("checkpoints")
    META_PATH      = Path("data/meta.csv")

NUM_MODALITIES = 4
ALL_MODALITIES = ["t1_path", "t1c_path", "t2_path", "flair_path"]
SEED = 42

# MODEL

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        num_groups = max(1, min(out_ch // 4, 8))
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
        self.Wg  = nn.Conv3d(g_ch, int_ch, 1)
        self.Wx  = nn.Conv3d(x_ch, int_ch, 1)
        self.psi = nn.Sequential(nn.Conv3d(int_ch, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.Wg(g) + self.Wx(x)))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up  = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.att = AttentionGate(out_ch, out_ch, out_ch // 2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.att(x1, x2)
        return self.conv(torch.cat([x2, x1], dim=1))


class AttentionUNet3D(nn.Module):
    def __init__(self, num_tumor_types: int = 2):
        super().__init__()
        self.modality_embed = nn.Parameter(torch.zeros(1, NUM_MODALITIES, 1, 1, 1))

        self.inc = DoubleConv(NUM_MODALITIES, 16)
        self.d1  = nn.Sequential(nn.MaxPool3d(2), DoubleConv(16, 32))
        self.d2  = nn.Sequential(nn.MaxPool3d(2), DoubleConv(32, 64))
        self.d3  = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64, 128))
        self.d4  = nn.Sequential(nn.MaxPool3d(2), DoubleConv(128, 256))

        self.u1 = Up(256, 128)
        self.u2 = Up(128, 64)
        self.u3 = Up(64, 32)
        self.u4 = Up(32, 16)
        self.outc = nn.Conv3d(16, 1, 1)

        self.cls_pool = nn.AdaptiveAvgPool3d((2, 2, 2))

        stats_in = NUM_MODALITIES * 4 + NUM_MODALITIES   # mean+std+min+max per ch + mm = 20
        self.stats_proj = nn.Sequential(
            nn.Linear(stats_in, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(),
        )

        cls_in_features = 256 * 8 + 128 * 8 + NUM_MODALITIES + 64
        self.classifier = nn.Sequential(
            nn.Linear(cls_in_features, 512), nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_tumor_types),
        )

        fg_ratio = 0.06
        nn.init.constant_(self.outc.bias, np.log(fg_ratio / (1.0 - fg_ratio)))
        nn.init.normal_(self.outc.weight, 0.0, 0.01)

    def forward(self, x, modality_mask):
        mm5d = modality_mask[:, :, None, None, None]

        x_masked = x * mm5d
        x_flat   = x_masked.flatten(2)
        ch_mean  = x_flat.mean(dim=2)
        ch_std   = x_flat.std(dim=2)
        ch_min   = x_flat.min(dim=2).values
        ch_max   = x_flat.max(dim=2).values
        stats      = torch.cat([ch_mean, ch_std, ch_min, ch_max, modality_mask], dim=1)
        stats_feat = self.stats_proj(stats)

        x_enc = x * mm5d + self.modality_embed
        x1 = self.inc(x_enc)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)

        xd = self.u1(x5, x4)
        xd = self.u2(xd, x3)
        xd = self.u3(xd, x2)
        xd = self.u4(xd, x1)
        raw = self.outc(xd)
        seg_logits = 8.0 * torch.tanh(raw / 8.0)

        gap_bottle = self.cls_pool(x5).flatten(1)
        gap_mid    = self.cls_pool(x4).flatten(1)
        cls_input  = torch.cat([gap_bottle, gap_mid, modality_mask, stats_feat], dim=1)
        cls_logits = self.classifier(cls_input)

        return seg_logits, cls_logits


# DATASET

class MRITestDataset(Dataset):
    """
    Reads pre-processed .npy files produced by main.py's process_single_case().
    Returns (image, mask, modality_mask, cls_label, seg_available, case_id).
    """
    def __init__(self, df: pd.DataFrame, type2idx: dict):
        self.df       = df.reset_index(drop=True)
        self.type2idx = type2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        img           = torch.from_numpy(np.load(row["image"]).astype(np.float32))
        mask          = torch.from_numpy(np.load(row["mask"]).astype(np.float32))
        modality_mask = torch.from_numpy(np.load(row["modality_mask"]).astype(np.float32))

        if "seg_available" in row.index:
            seg_avail = bool(row["seg_available"])
        else:
            seg_avail = bool(mask.sum() > 0)
        seg_available = torch.tensor(seg_avail, dtype=torch.bool)

        tumor_type_str = str(row.get("tumor_type", "healthy"))
        cls_label = self.type2idx.get(tumor_type_str, 0)
        cls_label = torch.tensor(cls_label, dtype=torch.long)

        case_id = str(row.get("case_id", idx))
        return img, mask, modality_mask, cls_label, seg_available, case_id

def dice_np(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    """Binary Dice on numpy arrays."""
    inter = (pred * gt).sum()
    return float((2.0 * inter + smooth) / (pred.sum() + gt.sum() + smooth))


def hausdorff95(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    95th-percentile Hausdorff distance between two binary 3-D volumes.

    Uses distance_transform_edt (O(N) in volume size) — safe on 128^3 inputs.
    Returns NaN if either mask is empty.
    """
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")

    from scipy.ndimage import distance_transform_edt as edt

    dist_pred = edt(pred == 0)
    dist_gt   = edt(gt == 0)

    # Surface voxels = foreground voxels (close enough for 3D MRI volumes)
    d_pred_to_gt = dist_gt[pred > 0]    # shape: (n_pred_voxels,)
    d_gt_to_pred = dist_pred[gt > 0]    # shape: (n_gt_voxels,)

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def keep_largest_component(pred: np.ndarray, n_components: int = 1) -> np.ndarray:
    """
    Post-processing: keep only the N largest connected components in a binary
    3-D prediction mask. Removes isolated false-positive islands.

    Requires scipy. If unavailable, returns pred unchanged.
    """
    if pred.sum() == 0:
        return pred

    from scipy.ndimage import label

    labeled, num_features = label(pred)
    if num_features <= n_components:
        return pred   # nothing to remove

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    top_labels = np.argsort(sizes)[::-1][:n_components]
    out = np.zeros_like(pred)
    for lbl in top_labels:
        out[labeled == lbl] = 1
    return out.astype(pred.dtype)


@torch.no_grad()
def tta_predict(model: nn.Module, imgs: torch.Tensor, mm: torch.Tensor) -> tuple:
    """
    Test-Time Augmentation: average sigmoid/softmax outputs over a set of
    spatial flip augmentations.

    Flip axes (in 3D spatial dims = [2, 3, 4]):
      - flip W  (axis 4)

    Returns (avg_seg_sigmoid, avg_cls_softmax) as plain tensors on CPU.
    """
    # Minimal TTA - just LR flip, preserves spatial meaning
    flip_combos = [[], [4]]
    seg_sum = None
    cls_sum = None

    for axes in flip_combos:
        x_aug = torch.flip(imgs, axes) if axes else imgs
        seg_logits, cls_logits = model(x_aug, mm)

        # Undo spatial flip on seg output
        seg_prob = torch.sigmoid(seg_logits)
        if axes:
            seg_prob = torch.flip(seg_prob, axes)

        cls_prob = torch.softmax(cls_logits, dim=1)

        seg_sum = seg_prob if seg_sum is None else seg_sum + seg_prob
        cls_sum = cls_prob if cls_sum is None else cls_sum + cls_prob

    n = len(flip_combos)
    return seg_sum / n, cls_sum / n


def volume_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Relative volume error (signed). Positive = over-segmentation."""
    gt_vol = int(gt.sum())
    pred_vol = int(pred.sum())
    if gt_vol == 0:
        return float("nan")
    return float((pred_vol - gt_vol) / gt_vol)


def per_class_prf(tp, fp, fn, num_classes):
    precision = np.zeros(num_classes)
    recall    = np.zeros(num_classes)
    f1        = np.zeros(num_classes)
    for c in range(num_classes):
        p = tp[c] / max(tp[c] + fp[c], 1)
        r = tp[c] / max(tp[c] + fn[c], 1)
        precision[c] = p
        recall[c]    = r
        f1[c]        = 2 * p * r / max(p + r, 1e-8)
    return precision, recall, f1


def top_k_accuracy(all_probs: np.ndarray, all_labels: np.ndarray, k: int) -> float:
    top_k = np.argsort(all_probs, axis=1)[:, -k:]
    return float(np.mean([all_labels[i] in top_k[i] for i in range(len(all_labels))]))


def confusion_matrix_str(cm: np.ndarray, class_names: list) -> str:
    n = len(class_names)
    col_w = max(max(len(c) for c in class_names), 6)
    header = " " * (col_w + 2) + "  ".join(f"{c:>{col_w}}" for c in class_names)
    lines  = [header, " " * (col_w + 2) + "-" * (col_w * n + 2 * (n - 1))]
    for i, row_name in enumerate(class_names):
        row = f"{row_name:>{col_w}} |  " + "  ".join(f"{cm[i, j]:>{col_w}}" for j in range(n))
        lines.append(row)
    return "\n".join(lines)

def remove_small_components(pred: np.ndarray, min_voxels: int = 50) -> np.ndarray:
    from scipy.ndimage import label
    if pred.sum() == 0:
        return pred
    labeled, _ = label(pred)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    out = np.zeros_like(pred)
    for lbl, size in enumerate(sizes):
        if lbl > 0 and size >= min_voxels:
            out[labeled == lbl] = 1
    return out.astype(pred.dtype)

# CORE EVALUATION LOOP
@torch.no_grad()
def run_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float,
    tumor_types: list,
    save_pred_dir: Path | None,
    use_tta: bool = False,
    use_cc_filter: bool = False,
) -> dict:
    """
    Full evaluation pass. Returns a results dict with all metrics plus
    per-case records (for the case table).
    """
    num_classes = len(tumor_types)

    # Accumulators
    cls_tp = np.zeros(num_classes, dtype=np.int64)
    cls_fp = np.zeros(num_classes, dtype=np.int64)
    cls_fn = np.zeros(num_classes, dtype=np.int64)
    cls_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    seg_tp = seg_fp = seg_fn = seg_tn = 0

    total_dice = total_dice_tumor_only = 0.0
    n_dice = n_dice_tumor = 0
    total_dice_annotated = 0.0
    n_dice_annotated = 0
    total_hd95 = 0.0
    n_hd = 0
    vol_errors = []

    total_cls_correct = total_cls_top2 = total_cls_samples = 0
    all_probs  = []
    all_labels = []
    per_case_records = []

    t_start = time.perf_counter()

    for batch in tqdm(loader, desc="Evaluating", unit="batch"):
        imgs, masks, mm, cls_targets, seg_avail, case_ids = batch
        imgs       = imgs.to(device, non_blocking=True)
        masks      = masks.to(device, non_blocking=True)
        mm         = mm.to(device, non_blocking=True)
        cls_targets = cls_targets.to(device, non_blocking=True)

        # Forward pass
        if use_tta:
            probs_seg, probs_cls = tta_predict(model, imgs, mm)
            probs_seg = probs_seg.to(device)
            probs_cls = probs_cls.to(device)
        else:
            seg_logits, cls_logits = model(imgs, mm)
            probs_seg = torch.sigmoid(seg_logits)
            probs_cls = torch.softmax(cls_logits, dim=1)

        seg_preds = (probs_seg > threshold).long()
        seg_gt    = masks.long()
        cls_preds = probs_cls.argmax(dim=1)

        # Per-batch accumulation
        seg_tp += int((seg_preds * seg_gt).sum())
        seg_fp += int((seg_preds * (1 - seg_gt)).sum())
        seg_fn += int(((1 - seg_preds) * seg_gt).sum())
        seg_tn += int(((1 - seg_preds) * (1 - seg_gt)).sum())

        total_cls_correct += int((cls_preds == cls_targets).sum())
        total_cls_samples += cls_targets.size(0)

        for c in range(num_classes):
            pred_c = (cls_preds == c)
            gt_c   = (cls_targets == c)
            cls_tp[c] += int((pred_c & gt_c).sum())
            cls_fp[c] += int((pred_c & ~gt_c).sum())
            cls_fn[c] += int((~pred_c & gt_c).sum())

        for gt_c, pr_c in zip(cls_targets.cpu().numpy(), cls_preds.cpu().numpy()):
            cls_conf_matrix[int(gt_c), int(pr_c)] += 1

        # Top-2 accuracy
        all_probs.append(probs_cls.cpu().numpy())
        all_labels.append(cls_targets.cpu().numpy())

        # Per-case metrics
        for i, case_id in enumerate(case_ids):
            pred_np = seg_preds[i, 0].cpu().numpy().astype(np.uint8)
            gt_np   = seg_gt[i, 0].cpu().numpy().astype(np.uint8)

            if use_cc_filter:
                pred_np = remove_small_components(pred_np)

            is_annotated = bool(seg_avail[i].item()) if hasattr(seg_avail[i], 'item') else bool(seg_avail[i])

            d = dice_np(pred_np, gt_np)
            total_dice += d
            n_dice += 1
            if gt_np.sum() > 0:
                total_dice_tumor_only += d
                n_dice_tumor += 1
            if is_annotated:
                total_dice_annotated += d
                n_dice_annotated += 1

            ve = volume_error(pred_np, gt_np)
            if not np.isnan(ve):
                vol_errors.append(ve)

            hd = hausdorff95(pred_np, gt_np)
            if not np.isnan(hd):
                total_hd95 += hd
                n_hd += 1

            gt_cls  = int(cls_targets[i].item())
            pr_cls  = int(cls_preds[i].item())
            pr_name = tumor_types[pr_cls]
            gt_name = tumor_types[gt_cls]
            top_probs = {tumor_types[c]: round(float(probs_cls[i, c]), 4)
                         for c in range(num_classes)}

            record = {
                "case_id":   case_id,
                "gt_class":  gt_name,
                "pred_class": pr_name,
                "cls_correct": gt_cls == pr_cls,
                "seg_dice":  round(d, 4),
                "vol_error": round(ve, 4) if not np.isnan(ve) else None,
                "hd95":      round(hd, 4) if not np.isnan(hd) else None,
                "probs":     top_probs,
            }
            per_case_records.append(record)

            if save_pred_dir is not None:
                save_pred_dir.mkdir(parents=True, exist_ok=True)
                np.save(save_pred_dir / f"{case_id}_seg_pred.npy", pred_np)
                with open(save_pred_dir / f"{case_id}_cls.json", "w") as f:
                    json.dump(record, f, indent=2)

    elapsed = time.perf_counter() - t_start
    total_cases = total_cls_samples

    # Aggregate
    all_probs_np  = np.vstack(all_probs)
    all_labels_np = np.concatenate(all_labels)

    cls_precision, cls_recall, cls_f1 = per_class_prf(cls_tp, cls_fp, cls_fn, num_classes)
    macro_f1    = float(cls_f1.mean())
    support     = cls_tp + cls_fn
    weighted_f1 = float(np.average(cls_f1, weights=np.where(support > 0, support, 1e-8)))

    seg_prec = seg_tp / max(seg_tp + seg_fp, 1)
    seg_rec  = seg_tp / max(seg_tp + seg_fn, 1)
    seg_f1v  = 2 * seg_prec * seg_rec / max(seg_prec + seg_rec, 1e-8)

    top2_acc = top_k_accuracy(all_probs_np, all_labels_np, k=min(2, num_classes))

    results = {
        # --- overview ---
        "total_cases":       total_cases,
        "inference_sec":     round(elapsed, 2),
        "cases_per_sec":     round(total_cases / max(elapsed, 1e-6), 2),
        "threshold":         threshold,
        "tta_enabled":       use_tta,
        "cc_filter_enabled": use_cc_filter,
        # --- segmentation ---
        "seg_dice_mean":          round(total_dice / max(n_dice, 1), 4),
        "seg_dice_tumor_only":    round(total_dice_tumor_only / max(n_dice_tumor, 1), 4),
        "seg_dice_annotated_only": round(total_dice_annotated / max(n_dice_annotated, 1), 4),
        "seg_dice_annotated_n":   n_dice_annotated,
        "seg_precision":     round(float(seg_prec), 4),
        "seg_recall":        round(float(seg_rec), 4),
        "seg_f1":            round(float(seg_f1v), 4),
        "seg_hd95_mean":     round(total_hd95 / max(n_hd, 1), 3),
        "seg_volume_error_mean": round(float(np.mean(vol_errors)) if vol_errors else float("nan"), 4),
        "seg_volume_error_std":  round(float(np.std(vol_errors))  if vol_errors else float("nan"), 4),
        "seg_tp":            seg_tp, "seg_fp": seg_fp,
        "seg_fn":            seg_fn, "seg_tn": seg_tn,
        # --- classification ---
        "cls_accuracy":      round(total_cls_correct / max(total_cls_samples, 1), 4),
        "cls_top2_accuracy": round(top2_acc, 4),
        "cls_macro_f1":      round(macro_f1, 4),
        "cls_weighted_f1":   round(weighted_f1, 4),
        "cls_per_class": {
            tumor_types[c]: {
                "precision": round(float(cls_precision[c]), 4),
                "recall":    round(float(cls_recall[c]), 4),
                "f1":        round(float(cls_f1[c]), 4),
                "support":   int(support[c]),
            }
            for c in range(num_classes)
        },
        "confusion_matrix":  cls_conf_matrix.tolist(),
        # --- per-case list ---
        "per_case": per_case_records,
    }
    return results

def print_report(results: dict, tumor_types: list):
    print(f"  TEST EVALUATION REPORT")

    print(f"\n  Cases evaluated  : {results['total_cases']}")
    print(f"  Inference time   : {results['inference_sec']:.1f}s  "
          f"({results['cases_per_sec']:.2f} cases/sec)")
    print(f"  Seg threshold    : {results['threshold']}")
    print(f"  TTA              : {'ON (8 flips)' if results.get('tta_enabled') else 'OFF'}")
    print(f"  CC filter        : {'ON (largest component)' if results.get('cc_filter_enabled') else 'OFF'}")

    print(f"\n{'─'*90}")
    print("  SEGMENTATION")
    print(f"{'─'*90}")
    print(f"  Mean Dice (all)              : {results['seg_dice_mean']:.4f}")
    print(f"  Mean Dice (tumor only)       : {results['seg_dice_tumor_only']:.4f}")
    n_ann = results.get('seg_dice_annotated_n', 0)
    print(f"  Mean Dice (annotated only)   : {results['seg_dice_annotated_only']:.4f}"
          f"  [{n_ann} cases with seg_available=True]")
    print(f"  Precision                    : {results['seg_precision']:.4f}")
    print(f"  Recall                       : {results['seg_recall']:.4f}")
    print(f"  F1 (voxel)                   : {results['seg_f1']:.4f}")
    if not np.isnan(results['seg_hd95_mean']):
        print(f"  Hausdorff-95 (mm, mean)      : {results['seg_hd95_mean']:.3f}")
    if not np.isnan(results['seg_volume_error_mean']):
        print(f"  Volume Error (mean±std)      : "
              f"{results['seg_volume_error_mean']:+.4f} ± {results['seg_volume_error_std']:.4f}")
    print(f"  Voxel counts  TP={results['seg_tp']:,}  "
          f"FP={results['seg_fp']:,}  FN={results['seg_fn']:,}  TN={results['seg_tn']:,}")

    print(f"\n{'─'*90}")
    print("  CLASSIFICATION")
    print(f"{'─'*90}")
    print(f"  Accuracy          : {results['cls_accuracy']:.4f}")
    print(f"  Top-2 Accuracy    : {results['cls_top2_accuracy']:.4f}")
    print(f"  Macro F1          : {results['cls_macro_f1']:.4f}")
    print(f"  Weighted F1       : {results['cls_weighted_f1']:.4f}")

    print(f"\n  {'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*58}")
    for name, m in results["cls_per_class"].items():
        flag = "  ← LOW" if m["f1"] < 0.3 and m["support"] > 0 else ""
        print(f"  {name:<16} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>10}{flag}")

    print(f"\n  Confusion Matrix (rows = ground truth, cols = predicted)")
    cm = np.array(results["confusion_matrix"])
    print(confusion_matrix_str(cm, tumor_types))

    print(f"\n{'─'*90}")
    print("  PER-CASE SUMMARY  (worst 10 Dice cases)")
    print(f"{'─'*90}")
    per_case = results["per_case"]
    # Sort by Dice ascending (worst first), show top 10
    sorted_cases = sorted(
        [c for c in per_case if c["seg_dice"] is not None],
        key=lambda c: c["seg_dice"]
    )[:10]
    print(f"  {'CaseID':<24} {'GT':>14} {'Pred':>14} {'Dice':>8} {'VolErr':>9} {'Correct':>8}")
    print(f"  {'-'*82}")
    for c in sorted_cases:
        correct_mark = "✓" if c["cls_correct"] else "✗"
        ve = f"{c['vol_error']:+.3f}" if c["vol_error"] is not None else "  n/a "
        print(f"  {c['case_id']:<24} {c['gt_class']:>14} {c['pred_class']:>14} "
              f"{c['seg_dice']:>8.4f} {ve:>9} {correct_mark:>8}")

def load_model(checkpoint_path: Path, device: str) -> tuple[AttentionUNet3D, dict, list]:
    """
    Load AttentionUNet3D from a checkpoint.
    Returns (model, ckpt_meta, tumor_types).
    Infers num_tumor_types from the checkpoint's classifier output layer weight.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt["model_state"]

    # Infer num_tumor_types from the saved weights
    last_cls_key = [k for k in state if "classifier" in k and "weight" in k][-1]
    num_tumor_types = state[last_cls_key].shape[0]

    model = AttentionUNet3D(num_tumor_types=num_tumor_types).to(device)
    for module in model.modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            module.to(memory_format=torch.channels_last_3d)

    model.load_state_dict(state)
    model.eval()

    meta = {
        "epoch":      ckpt.get("epoch", "?"),
        "best_loss":  ckpt.get("best_loss", None),
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    return model, meta, num_tumor_types


# BUILD TUMOR TYPES from metadata

def build_tumor_type_map(df: pd.DataFrame):
    types = sorted(df["tumor_type"].dropna().unique().tolist())
    if "healthy" in types:
        types.remove("healthy")
    types = ["healthy"] + types
    type2idx = {t: i for i, t in enumerate(types)}
    return types, type2idx


# FIGURE GENERATION

def generate_figures(results: dict, tumor_types: list, out_dir: Path) -> list[Path]:
    """
    Generate one PNG per metric group and save to out_dir.
    Returns a list of saved file paths.

    Figures produced
    ----------------
    1. seg_overview.png      - Dice / Precision / Recall / F1 bar chart
    2. seg_dice_dist.png     - Per-case Dice histogram + KDE
    3. seg_volume_error.png  - Volume error distribution (signed histogram)
    4. seg_hd95_dist.png     - Hausdorff-95 distribution (when available)
    5. cls_metrics.png       - Per-class Precision / Recall / F1 grouped bar chart
    6. cls_confusion.png     - Confusion matrix heatmap (normalised + raw counts)
    7. cls_accuracy.png      - Overall accuracy / Top-2 / Macro-F1 / Weighted-F1
    8. per_case_dice.png     - Sorted per-case Dice strip (all cases, coloured by class)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Light theme
    DARK_BG = "#ffffff"
    PANEL_BG = "#ffffff"
    GRID_CLR = "#d0d0d0"
    TEXT_CLR = "#111111"

    ACCENT = "#2563eb"
    GREEN = "#16a34a"
    ORANGE = "#ea580c"
    RED = "#dc2626"
    PURPLE = "#7c3aed"
    YELLOW = "#ca8a04"

    CLASS_COLORS = [ACCENT, GREEN, ORANGE, RED, PURPLE, YELLOW,
                    "#e879a0", "#66d9e8", "#a9e34b", "#ffa94d"]

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "axes.edgecolor": "#cccccc",
        "axes.labelcolor": "#111111",
        "axes.titlecolor": "#111111",

        "xtick.color": "#111111",
        "ytick.color": "#111111",

        "text.color": "#111111",

        "grid.color": "#dddddd",
        "grid.linewidth": 0.8,

        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",

        "figure.dpi": 140,
    })

    saved = []

    # Helper
    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        saved.append(p)
        print(f"  [FIG] saved {p}")

    def _bar(ax, labels, values, colors, title, ylabel, adaptive=True):
        values = np.array(values, dtype=float)

        bars = ax.bar(
            labels,
            values,
            color=colors,
            width=0.55,
            edgecolor="#bbbbbb",
            linewidth=0.8,
            zorder=3
        )

        # Adaptive zoom for close values
        if adaptive:
            vmin = values.min()
            vmax = values.max()

            margin = max(0.01, (vmax - vmin) * 0.35)

            low = max(0, vmin - margin)
            high = min(1.0, vmax + margin)

            # Prevent ultra-flat charts
            if high - low < 0.08:
                center = (high + low) / 2
                low = max(0, center - 0.04)
                high = min(1.0, center + 0.04)

            ax.set_ylim(low, high)
        else:
            ax.set_ylim(0, 1)

        ax.set_title(title)
        ax.set_ylabel(ylabel)

        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=TEXT_CLR,
                fontweight="bold"
            )

        return bars

    # 1. Segmentation overview bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Segmentation — Overview", fontsize=14, fontweight="bold", y=1.01)

    # Left: Dice variants
    dice_labels = ["Mean Dice\n(all)", "Mean Dice\n(tumor only)", "Mean Dice\n(annotated)"]
    dice_vals   = [results["seg_dice_mean"],
                   results["seg_dice_tumor_only"],
                   results["seg_dice_annotated_only"]]
    _bar(axes[0], dice_labels, dice_vals,
         [ACCENT, GREEN, ORANGE], "Dice Scores", "Dice")

    # Right: Precision / Recall / F1
    prf_labels = ["Precision", "Recall", "F1 (voxel)"]
    prf_vals   = [results["seg_precision"], results["seg_recall"], results["seg_f1"]]
    _bar(axes[1], prf_labels, prf_vals,
         [PURPLE, YELLOW, RED], "Voxel-Level Seg Metrics", "Score")

    fig.tight_layout()
    _save(fig, "seg_overview.png")

    # 2. Per-case Dice distribution
    per_case = results.get("per_case", [])
    dice_vals_all = [c["seg_dice"] for c in per_case if c["seg_dice"] is not None]

    if dice_vals_all:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        n_bins = min(40, max(10, len(dice_vals_all) // 5))
        ax.hist(dice_vals_all, bins=n_bins, color=ACCENT, edgecolor="#ffffff",
                linewidth=0.5, alpha=0.85, zorder=3)
        mean_d  = np.mean(dice_vals_all)
        median_d = np.median(dice_vals_all)
        ax.axvline(mean_d,   color=ORANGE, lw=1.8, linestyle="--", label=f"Mean {mean_d:.4f}")
        ax.axvline(median_d, color=GREEN,  lw=1.8, linestyle=":",  label=f"Median {median_d:.4f}")
        ax.set_xlabel("Dice Score")
        ax.set_ylabel("Number of Cases")
        ax.set_title("Per-Case Dice Distribution")
        ax.legend(framealpha=0.2, facecolor=PANEL_BG, edgecolor=GRID_CLR)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        fig.tight_layout()
        _save(fig, "seg_dice_dist.png")

    # 3. Volume error distribution
    vol_errs = [c["vol_error"] for c in per_case
                if c.get("vol_error") is not None and not np.isnan(c["vol_error"])]
    if vol_errs:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        n_bins = min(40, max(10, len(vol_errs) // 5))
        pos_errs = [v for v in vol_errs if v >= 0]
        neg_errs = [v for v in vol_errs if v < 0]
        if neg_errs:
            ax.hist(neg_errs, bins=n_bins // 2 or 5, color=RED,   edgecolor=DARK_BG,
                    linewidth=0.5, alpha=0.8, label="Under-seg (neg)", zorder=3)
        if pos_errs:
            ax.hist(pos_errs, bins=n_bins // 2 or 5, color=ORANGE, edgecolor=DARK_BG,
                    linewidth=0.5, alpha=0.8, label="Over-seg (pos)", zorder=3)
        ax.axvline(0, color=TEXT_CLR, lw=1.2, linestyle="-")
        mean_ve = np.mean(vol_errs)
        ax.axvline(mean_ve, color=YELLOW, lw=1.8, linestyle="--",
                   label=f"Mean {mean_ve:+.4f}")
        ax.set_xlabel("Relative Volume Error  (pred − gt) / gt")
        ax.set_ylabel("Number of Cases")
        ax.set_title("Volume Error Distribution")
        ax.legend(framealpha=0.2, facecolor=PANEL_BG, edgecolor=GRID_CLR)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        fig.tight_layout()
        _save(fig, "seg_volume_error.png")

    # 4. HD95 distribution
    hd95_vals = [c["hd95"] for c in per_case
                 if c.get("hd95") is not None and not np.isnan(c["hd95"])]
    if hd95_vals:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        n_bins = min(40, max(10, len(hd95_vals) // 5))
        ax.hist(hd95_vals, bins=n_bins, color=PURPLE, edgecolor=DARK_BG,
                linewidth=0.5, alpha=0.85, zorder=3)
        mean_hd = np.mean(hd95_vals)
        ax.axvline(mean_hd, color=YELLOW, lw=1.8, linestyle="--",
                   label=f"Mean {mean_hd:.2f} mm")
        ax.set_xlabel("Hausdorff-95 Distance (mm)")
        ax.set_ylabel("Number of Cases")
        ax.set_title("Hausdorff-95 Distribution")
        ax.legend(framealpha=0.2, facecolor=PANEL_BG, edgecolor=GRID_CLR)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        fig.tight_layout()
        _save(fig, "seg_hd95_dist.png")

    # 5. Per-class classification metrics
    cls_pc = results.get("cls_per_class", {})
    if cls_pc:
        classes = list(cls_pc.keys())
        n_cls   = len(classes)
        prec    = [cls_pc[c]["precision"] for c in classes]
        rec     = [cls_pc[c]["recall"]    for c in classes]
        f1s     = [cls_pc[c]["f1"]        for c in classes]
        sup     = [cls_pc[c]["support"]   for c in classes]

        x = np.arange(n_cls)
        w = 0.26
        fig, ax = plt.subplots(figsize=(max(9, n_cls * 1.4), 5))
        b1 = ax.bar(x - w, prec, w, label="Precision", color=ACCENT,   edgecolor=DARK_BG, zorder=3)
        b2 = ax.bar(x,     rec,  w, label="Recall",    color=GREEN,    edgecolor=DARK_BG, zorder=3)
        b3 = ax.bar(x + w, f1s,  w, label="F1",        color=ORANGE,   edgecolor=DARK_BG, zorder=3)

        # Value labels
        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8,
                        color=TEXT_CLR)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(classes, sup)], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Classification Metrics")
        ax.legend(framealpha=0.2, facecolor=PANEL_BG, edgecolor=GRID_CLR)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        fig.tight_layout()
        _save(fig, "cls_metrics.png")

    # 6. Confusion matrix heatmap
    cm_raw = np.array(results.get("confusion_matrix", []))
    if cm_raw.ndim == 2 and cm_raw.shape[0] > 0:
        n_cls  = cm_raw.shape[0]
        labels = tumor_types[:n_cls]

        # Normalise row-wise (recall per class)
        row_sums = cm_raw.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm  = cm_raw / row_sums

        fig, axes = plt.subplots(1, 2, figsize=(max(10, n_cls * 2.5), max(5, n_cls * 1.8)))

        for ax, data, title, fmt in [
            (axes[0], cm_norm, "Confusion Matrix (Row-Normalised)", ".2f"),
            (axes[1], cm_raw,  "Confusion Matrix (Raw Counts)",     "d"),
        ]:
            im = ax.imshow(data, cmap="viridis", vmin=0,
                           vmax=1 if fmt == ".2f" else cm_raw.max())
            ax.set_xticks(range(n_cls)); ax.set_yticks(range(n_cls))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
            ax.set_title(title)
            thresh = (data.max() / 2) if fmt == ".2f" else (cm_raw.max() / 2)
            for i in range(n_cls):
                for j in range(n_cls):
                    val = data[i, j]
                    txt = f"{val:{fmt}}"
                    color = "black" if val > thresh else TEXT_CLR
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=8, color=color)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        _save(fig, "cls_confusion.png")

    # 7. Overall classification accuracy summary
    fig, ax = plt.subplots(figsize=(7, 4))
    summary_labels = ["Accuracy", "Top-2 Acc", "Macro F1", "Weighted F1"]
    summary_vals   = [
        results.get("cls_accuracy",      0),
        results.get("cls_top2_accuracy", 0),
        results.get("cls_macro_f1",      0),
        results.get("cls_weighted_f1",   0),
    ]
    colors_sum = [ACCENT, GREEN, ORANGE, PURPLE]
    _bar(ax, summary_labels, summary_vals, colors_sum,
         "Classification — Summary Metrics", "Score")
    fig.tight_layout()
    _save(fig, "cls_accuracy.png")

    # 8. Per-case Dice strip
    if per_case:
        sorted_pc = sorted(per_case, key=lambda c: c["seg_dice"])
        dice_sorted  = [c["seg_dice"]  for c in sorted_pc]
        class_sorted = [c["gt_class"]  for c in sorted_pc]
        correct_sorted = [c["cls_correct"] for c in sorted_pc]

        unique_classes = sorted(set(class_sorted))
        color_map = {cls: CLASS_COLORS[i % len(CLASS_COLORS)]
                     for i, cls in enumerate(unique_classes)}
        bar_colors = [color_map[cls] for cls in class_sorted]

        fig, ax = plt.subplots(figsize=(max(12, len(dice_sorted) * 0.04 + 2), 5))
        xs = np.arange(len(dice_sorted))
        ax.bar(xs, dice_sorted, color=bar_colors, width=1.0,
               edgecolor="none", alpha=0.9, zorder=3)

        # Mark misclassified cases with a red X at the top
        for xi, (d, ok) in enumerate(zip(dice_sorted, correct_sorted)):
            if not ok:
                ax.scatter(xi, min(d + 0.04, 0.98), marker="x",
                           color=RED, s=14, linewidths=0.8, zorder=5)

        ax.axhline(np.mean(dice_sorted), color=YELLOW, lw=1.5,
                   linestyle="--", label=f"Mean Dice {np.mean(dice_sorted):.4f}")
        ax.set_xlim(-1, len(dice_sorted))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Cases (sorted by Dice, ascending)")
        ax.set_ylabel("Dice Score")
        ax.set_title("Per-Case Dice — All Cases  (✗ = misclassified)")
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)

        patches = [mpatches.Patch(color=color_map[cls], label=cls)
                   for cls in unique_classes]
        ax.legend(handles=patches, framealpha=0.2,
                  facecolor=PANEL_BG, edgecolor=GRID_CLR,
                  loc="upper left", fontsize=8)
        fig.tight_layout()
        _save(fig, "per_case_dice.png")

    print(f"\n[INFO] {len(saved)} figure(s) saved to {out_dir}/")
    return saved


# ENTRY POINT

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 3D Attention U-Net on brain MRI test data"
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--meta", type=Path, default=META_PATH)
    parser.add_argument("--split", type=str, default="test", help="test | val | train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-report", type=Path, default=None, help="Save JSON results to this path")
    parser.add_argument("--save-predictions", type=Path, default=None, help="Dir to save per-case seg predictions & cls probs as .npy/.json")
    parser.add_argument("--tta", action="store_true", default=False)
    parser.add_argument("--cc-filter", action="store_true", default=False)
    parser.add_argument("--save-figures", type=Path, default=None, help="Directory to save metric visualisation PNGs")
    args = parser.parse_args()

    print(f"\n[INFO] Device       : {args.device}")
    print(f"[INFO] Checkpoint   : {args.checkpoint}")
    print(f"[INFO] Meta CSV     : {args.meta}")
    print(f"[INFO] Split        : {args.split}")
    print(f"[INFO] Threshold    : {args.threshold}")

    # Load metadata
    if not args.meta.exists():
        sys.exit(f"[ERROR] Meta CSV not found: {args.meta}")
    meta = pd.read_csv(args.meta)

    split_df = meta[meta["split"] == args.split].copy()
    if len(split_df) == 0:
        sys.exit(f"[ERROR] No rows found for split='{args.split}' in {args.meta}")
    print(f"[INFO] {args.split} cases : {len(split_df)}")

    tumor_types, type2idx = build_tumor_type_map(meta)   # use full meta for consistent label map
    print(f"[INFO] Tumor types  : {tumor_types}")

    # Tumor type distribution for the split
    print("\n[INFO] Tumor type distribution in split:")
    for tt, cnt in split_df["tumor_type"].value_counts().items():
        print(f"       {tt:<16}: {cnt}")

    # Load model
    if not args.checkpoint.exists():
        sys.exit(f"[ERROR] Checkpoint not found: {args.checkpoint}")

    model, ckpt_meta, num_tumor_types = load_model(args.checkpoint, args.device)
    print(f"\n[INFO] Model loaded from epoch {ckpt_meta['epoch']} | "
          f"best_loss={ckpt_meta['best_loss']} | "
          f"params={ckpt_meta['num_params']:,}")

    if len(tumor_types) != num_tumor_types:
        print(f"[WARN] Meta has {len(tumor_types)} classes but checkpoint has "
              f"{num_tumor_types} output neurons — using checkpoint count.")
        # Rebuild label map to match checkpoint
        tumor_types = tumor_types[:num_tumor_types]
        type2idx    = {t: i for i, t in enumerate(tumor_types)}

    # DataLoader
    dataset = MRITestDataset(split_df, type2idx)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda") and args.num_workers > 0),
        persistent_workers=False,
    )

    # Run evaluation
    results = run_evaluation(
        model=model,
        loader=loader,
        device=args.device,
        threshold=args.threshold,
        tumor_types=tumor_types,
        save_pred_dir=args.save_predictions,
        use_tta=args.tta,
        use_cc_filter=args.cc_filter,
    )

    # Print report
    print_report(results, tumor_types)

    # Threshold sensitivity
    print("  Threshold sensitivity (Dice on first 50 cases, seg only):")
    print(f"  {'Threshold':>10} {'Dice':>10} {'Precision':>12} {'Recall':>10}")
    print(f"  {'-'*46}")
    for t, (p, r) in [
        (0.10, (results["seg_precision"], results["seg_recall"])),
        (0.20, (results["seg_precision"], results["seg_recall"])),
        (0.30, (results["seg_precision"], results["seg_recall"])),
        (0.50, (results["seg_precision"], results["seg_recall"])),
    ]:
        if t == args.threshold:
            dice_at_t = results["seg_dice_mean"]
            print(f"  {t:>10.2f} {dice_at_t:>10.4f} {p:>12.4f} {r:>10.4f}  ← current")
        else:
            print(f"  {t:>10.2f}  (run again with --threshold {t} for exact values)")
    print()

    if args.save_report:
        def _convert(o):
            if isinstance(o, (np.integer,)):   return int(o)
            if isinstance(o, (np.floating,)):  return float(o)
            if isinstance(o, np.ndarray):      return o.tolist()
            raise TypeError(f"Unserializable type: {type(o)}")

        with open(args.save_report, "w") as f:
            json.dump(results, f, indent=2, default=_convert)
        print(f"[INFO] Report saved to {args.save_report}")

    if args.save_predictions:
        print(f"[INFO] Per-case predictions saved to {args.save_predictions}/")

    if args.save_figures:
        generate_figures(results, tumor_types, args.save_figures)


if __name__ == "__main__":
    main()