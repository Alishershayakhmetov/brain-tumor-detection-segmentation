"""
Single-patient inference with explainable AI


python predict.py \
    --checkpoint checkpoints/best.pt \
    --t1    patient/t1.nii.gz \
    --t1c   patient/t1c.nii.gz \
    --t2    patient/t2.nii.gz \
    --flair patient/flair.nii.gz \
    --out   results/patient_001

"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

TARGET_SPACING = config.TARGET_SPACING
TARGET_SHAPE = config.TARGET_SHAPE
NUM_MODALITIES = 4
ALL_MODALITIES = ["t1", "t1c", "t2", "flair"]
MODALITY_LABELS = ["T1", "T1c", "T2", "FLAIR"]
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(4)


# Model

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
    def __init__(self, num_tumor_types: int = 2):
        super().__init__()
        self.modality_embed = nn.Parameter(torch.zeros(1, NUM_MODALITIES, 1, 1, 1))
        self.inc = DoubleConv(NUM_MODALITIES, 16)
        self.d1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(16, 32))
        self.d2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(32, 64))
        self.d3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64, 128))
        self.d4 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(128, 256))
        self.u1 = Up(256, 128)
        self.u2 = Up(128, 64)
        self.u3 = Up(64, 32)
        self.u4 = Up(32, 16)
        self.outc = nn.Conv3d(16, 1, 1)
        self.cls_pool = nn.AdaptiveAvgPool3d((2, 2, 2))

        stats_in = NUM_MODALITIES * 4 + NUM_MODALITIES
        self.stats_proj = nn.Sequential(
            nn.Linear(stats_in, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(),
        )
        cls_in = 256 * 8 + 128 * 8 + NUM_MODALITIES + 64
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_tumor_types),
        )
        fg_ratio = 0.06
        nn.init.constant_(self.outc.bias, np.log(fg_ratio / (1.0 - fg_ratio)))
        nn.init.normal_(self.outc.weight, mean=0.0, std=0.01)

    def forward(self, x, modality_mask):
        mm5d = modality_mask[:, :, None, None, None]
        x_masked = x * mm5d
        x_flat = x_masked.flatten(2)
        stats = torch.cat([
            x_flat.mean(dim=2), x_flat.std(dim=2),
            x_flat.min(dim=2).values, x_flat.max(dim=2).values,
            modality_mask,
        ], dim=1)
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
        seg_logits = 8.0 * torch.tanh(self.outc(xd) / 8.0)

        gap_bottle = self.cls_pool(x5).flatten(1)
        gap_mid = self.cls_pool(x4).flatten(1)
        cls_logits = self.classifier(
            torch.cat([gap_bottle, gap_mid, modality_mask, stats_feat], dim=1)
        )
        return seg_logits, cls_logits


# Preprocessing

def _resample(img: sitk.Image) -> sitk.Image:
    res = sitk.ResampleImageFilter()
    res.SetOutputSpacing(TARGET_SPACING)
    res.SetSize([
        int(round(img.GetSize()[i] * img.GetSpacing()[i] / TARGET_SPACING[i]))
        for i in range(3)
    ])
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    res.SetInterpolator(sitk.sitkLinear)
    return res.Execute(img)


def _center_crop_or_pad(vol: np.ndarray, target: tuple) -> np.ndarray:
    z, y, x = vol.shape
    tz, ty, tx = target
    out = np.zeros(target, dtype=vol.dtype)
    z0 = max((tz - z) // 2, 0); zs = max((z - tz) // 2, 0); zl = min(z, tz)
    y0 = max((ty - y) // 2, 0); ys = max((y - ty) // 2, 0); yl = min(y, ty)
    x0 = max((tx - x) // 2, 0); xs = max((x - tx) // 2, 0); xl = min(x, tx)
    out[z0:z0+zl, y0:y0+yl, x0:x0+xl] = vol[zs:zs+zl, ys:ys+yl, xs:xs+xl]
    return out


def _load_modality(path):
    if path is None:
        return np.zeros(TARGET_SHAPE, dtype=np.float32), False, None
    p = Path(path)
    if not p.is_file():
        print(f"  [WARN] Not found: {p}  → zeroed")
        return np.zeros(TARGET_SHAPE, dtype=np.float32), False, None

    img = sitk.ReadImage(str(p))
    img = _resample(img)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = _center_crop_or_pad(arr, TARGET_SHAPE)

    p1, p99 = np.percentile(arr, (1, 99))
    arr = np.clip(arr, p1, p99)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32), True, img


def load_patient(paths: dict, device: str):
    """Returns x(1,4,D,H,W), mm(1,4), reference sitk.Image."""
    channels, mask_vals, ref_sitk = [], [], None
    for mod in ALL_MODALITIES:
        arr, present, sitk_img = _load_modality(paths.get(mod))
        channels.append(arr)
        mask_vals.append(float(present))
        if present and ref_sitk is None:
            ref_sitk = sitk_img
        label = MODALITY_LABELS[ALL_MODALITIES.index(mod)]
        print(f"  {label:<6} {'ok' if present else 'missing (zeroed)'}")

    x  = torch.from_numpy(np.stack(channels)).unsqueeze(0).to(device)
    mm = torch.tensor([mask_vals], dtype=torch.float32).to(device)
    return x, mm, ref_sitk

def _save_nifti(arr: np.ndarray, ref: sitk.Image, path: Path):
    out = sitk.GetImageFromArray(arr.astype(np.float32))
    out.SetSpacing(ref.GetSpacing())
    out.SetOrigin(ref.GetOrigin())
    out.SetDirection(ref.GetDirection())
    sitk.WriteImage(out, str(path))
    print(f" saved {path.name}")

def _best_slice(volume: np.ndarray, guide: np.ndarray = None) -> int:
    """Axial index with the highest mean value in `guide` (or volume itself)."""
    src = guide if guide is not None else volume
    per_slice = np.array([src[i].mean() for i in range(src.shape[0])])
    smoothed = np.convolve(per_slice, np.ones(3) / 3.0, mode="same")
    return int(np.argmax(smoothed))

# XAI

class _GradCAMHook:
    def __init__(self, layer: nn.Module):
        self._acts = self._grads = None
        self._fh = layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o.detach()))
        self._bh = layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0].detach()))

    def remove(self):
        self._fh.remove(); self._bh.remove()


def _run_gradcam(model, hook, x, mm, cls_idx) -> np.ndarray:
    with torch.enable_grad():
        _, cls_logits = model(x, mm)
        model.zero_grad()
        cls_logits[0, cls_idx].backward()

    weights = hook._grads.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((weights * hook._acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    vmin, vmax = cam.min(), cam.max()
    if vmax - vmin > 1e-8:
        cam = (cam - vmin) / (vmax - vmin)
    return cam.astype(np.float32)


def _extract_attention_maps(model, x, mm) -> dict:
    maps, hooks = {}, []
    for name in ["u1", "u2", "u3", "u4"]:
        def _hook(module, inp, out, _n=name):
            maps[_n] = out.detach()
        hooks.append(getattr(model, name).att.psi.register_forward_hook(_hook))

    with torch.no_grad():
        model(x, mm)
    for h in hooks:
        h.remove()

    result = {}
    for name, psi in maps.items():
        up = F.interpolate(psi[:1].float(), size=x.shape[2:],
                           mode="trilinear", align_corners=False)
        result[name] = up.squeeze().cpu().numpy().astype(np.float32)
    return result


def _occlusion_sensitivity(model, x, mm, seg_ref, pred_class):
    def _dice(logits, y, smooth=1e-5):
        p = torch.sigmoid(logits)
        i = (p * y).sum((2, 3, 4))
        ps = p.sum((2, 3, 4)); ts = y.sum((2, 3, 4))
        d = (2 * i + smooth) / (ps + ts + smooth)
        return float(torch.where(ts == 0, (ps < 1.0).float(), d).mean().item())

    with torch.no_grad():
        seg_bl, cls_bl = model(x, mm)
    base_conf = float(torch.softmax(cls_bl, dim=1)[0, pred_class].item())
    base_dice = _dice(seg_bl, seg_ref)

    rows = []
    for i in range(NUM_MODALITIES):
        x_o = x.clone(); x_o[:, i] = 0.0
        m_o = mm.clone(); m_o[:, i] = 0.0
        with torch.no_grad():
            seg_o, cls_o = model(x_o, m_o)
        occ_conf = float(torch.softmax(cls_o, dim=1)[0, pred_class].item())
        rows.append({
            "modality": MODALITY_LABELS[i],
            "present": bool(mm[0, i].item() > 0.5),
            "cls_conf_drop": round(base_conf - occ_conf, 4),
            "dice_drop": round(base_dice - _dice(seg_o, seg_ref), 4),
        })
    return base_conf, base_dice, rows


def _save_seg_overlay(x_np, seg_np, out_path):
    bg = x_np[0];  mask = seg_np[0]
    sl = _best_slice(bg, mask)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Segmentation — best axial slice {sl}", fontsize=11)
    axes[0].imshow(bg[sl], cmap="gray"); axes[0].set_title("T1 input"); axes[0].axis("off")
    axes[1].imshow(bg[sl], cmap="gray")
    axes[1].imshow(mask[sl], cmap="Reds", alpha=0.55)
    axes[1].set_title("Tumour mask"); axes[1].axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _save_gradcam(x_np, cam, seg_np, out_path):
    bg = x_np[0]; mask = seg_np[0]
    sl = _best_slice(cam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Grad-CAM — best activation slice {sl}", fontsize=11)
    axes[0].imshow(bg[sl],  cmap="gray");  axes[0].set_title("T1 input"); axes[0].axis("off")
    im = axes[1].imshow(cam[sl], cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(bg[sl],  cmap="gray")
    axes[2].imshow(cam[sl], cmap="jet", alpha=0.55, vmin=0, vmax=1)
    if mask[sl].any():
        axes[2].contour(mask[sl], levels=[0.5], colors="lime", linewidths=1.2)
    axes[2].set_title("CAM + seg contour"); axes[2].axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _save_attention_grid(x_np, attn_maps, cam, out_path):
    bg = x_np[0]
    sl = _best_slice(cam)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(f"Attention gates — axial slice {sl}  (guided by Grad-CAM peak)", fontsize=12)

    # Top row: all 4 gates at the CAM-best slice
    for ax, (name, arr) in zip(axes[0], attn_maps.items()):
        vmax = float(np.percentile(arr, 99)) or 1e-6   # suppress uniform background
        ax.imshow(bg[sl], cmap="gray")
        im = ax.imshow(arr[sl], cmap="hot", alpha=0.7, vmin=0, vmax=vmax)
        ax.set_title(f"{name}  (vmax={vmax:.3f})"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom row: u4 alone at 3 neighbouring slices for context
    u4 = attn_maps["u4"]
    vmax_u4 = float(np.percentile(u4, 99)) or 1e-6
    offsets = [-5, 0, +5]
    D = bg.shape[0]
    for ax, offset in zip(axes[1, :3], offsets):
        s = min(max(sl + offset, 0), D - 1)
        ax.imshow(bg[s], cmap="gray")
        im = ax.imshow(u4[s], cmap="hot", alpha=0.7, vmin=0, vmax=vmax_u4)
        ax.set_title(f"u4  [sl {s}]"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[1, 3].axis("off")   # unused cell

    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _save_occlusion_bar(rows, out_path):
    labels = [r["modality"] + ("\n(absent)" if not r["present"] else "") for r in rows]
    conf_drop = [r["cls_conf_drop"] for r in rows]
    dice_drop = [r["dice_drop"]     for r in rows]
    x_pos = np.arange(len(labels))
    colors = lambda vals: ["steelblue" if v >= 0 else "salmon" for v in vals]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, vals, title, ylabel in zip(
        axes,
        [conf_drop, dice_drop],
        ["Classification confidence drop\n(positive = modality mattered)",
         "Dice drop\n(positive = modality mattered for segmentation)"],
        ["Δ confidence", "Δ Dice"],
    ):
        ax.bar(x_pos, vals, color=colors(vals))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x_pos); ax.set_xticklabels(labels)
        ax.set_title(title); ax.set_ylabel(ylabel)

    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _print_report(tumor_types, pred_class, probs, base_conf, base_dice, occ_rows):
    print("\n" + "-" * 70)
    print("  PREDICTION REPORT")
    print("-" * 70)
    print(f"  Predicted class  : {tumor_types[pred_class]}  (index {pred_class})")
    print(f"  Confidence       : {base_conf:.1%}")
    print()
    for i, name in enumerate(tumor_types):
        print(f"    {name:<16} {probs[i]:>6.1%}  {'█' * int(probs[i] * 30)}")
    print()
    print("  Occlusion sensitivity:")
    print(f"    {'Modality':<8} {'Present':<9} {'Cls conf Δ':>12} {'Dice Δ':>10}")
    print(f"    {'-'*42}")
    for r in occ_rows:
        print(f"    {r['modality']:<8} {'yes' if r['present'] else 'no':<9}"
              f" {r['cls_conf_drop']:>+12.4f} {r['dice_drop']:>+10.4f}")
    print("-" * 70)


# Main

def predict(
    checkpoint, t1=None, t1c=None, t2=None, flair=None,
    out_dir="xai_results", device=None, tumor_types=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(checkpoint)

    if not ckpt_path.is_file():
        sys.exit(f"[ERROR] Checkpoint not found: {ckpt_path}")

    # Load checkpoint
    print(f"\n[1/5] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    cls_keys = sorted(
        [k for k in ckpt["model_state"] if k.startswith("classifier.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    if not cls_keys:
        sys.exit("[ERROR] No classifier weights found in checkpoint.")
    num_tumor_types = ckpt["model_state"][cls_keys[-1]].shape[0]

    if tumor_types is None:
        tumor_types = ckpt.get("tumor_types", None)
    if tumor_types is None:
        tumor_types = ["healthy", "glioma", "meningioma", "metastasis"]
        print(f"  [WARN] tumor_types not in checkpoint — using: {tumor_types}")

    model = AttentionUNet3D(num_tumor_types=num_tumor_types).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    print(f"  Classes ({num_tumor_types}): {tumor_types}")

    # Preprocess
    print("\n[2/5] Loading & preprocessing …")
    x, mm, ref_sitk = load_patient({"t1": t1, "t1c": t1c, "t2": t2, "flair": flair}, device)
    if ref_sitk is None:
        sys.exit("[ERROR] No valid modality files found.")

    # Inference
    print("\n[3/5] Running inference …")
    with torch.no_grad():
        seg_logits, cls_logits = model(x, mm)

    probs_t    = torch.softmax(cls_logits, dim=1)[0]
    probs      = probs_t.cpu().numpy()
    pred_class = int(probs_t.argmax().item())
    confidence = float(probs[pred_class])

    if pred_class == 0:
        seg_logits = seg_logits.masked_fill(
            torch.ones_like(seg_logits, dtype=torch.bool), -10.0)

    seg_mask = (torch.sigmoid(seg_logits) > 0.5).float()
    seg_np   = seg_mask.squeeze(0).cpu().numpy()   # (1,D,H,W)
    x_np     = x.squeeze(0).cpu().numpy()          # (4,D,H,W)
    # Use the baseline seg_mask as occlusion reference
    seg_ref = seg_mask.detach()

    # XAI
    print("\n[4/5] Running XAI …")

    print("  • Grad-CAM …")
    hook = _GradCAMHook(model.d4[1])  # hook DoubleConv inside Sequential, not MaxPool
    cam  = _run_gradcam(model, hook, x, mm, pred_class)
    hook.remove()

    print("  • Attention maps …")
    attn_maps = _extract_attention_maps(model, x, mm)

    print("  • Occlusion sensitivity …")
    base_conf, base_dice, occ_rows = _occlusion_sensitivity(
        model, x, mm, seg_ref, pred_class)

    # Save
    print("\n[5/5] Saving outputs …")

    # NIfTI (3D Slicer)
    bg_arr = x_np[0]
    bg_arr = (bg_arr - bg_arr.min()) / (bg_arr.max() - bg_arr.min() + 1e-8)
    _save_nifti(bg_arr, ref_sitk, out_dir / "t1_input.nii.gz")
    _save_nifti(seg_np[0], ref_sitk, out_dir / "seg_mask.nii.gz")
    _save_nifti(cam, ref_sitk, out_dir / "gradcam.nii.gz")
    for name, arr in attn_maps.items():
        _save_nifti(arr, ref_sitk, out_dir / f"attn_{name}.nii.gz")

    _save_seg_overlay(x_np, seg_np, out_dir / "seg_overlay.png")
    _save_gradcam(x_np, cam, seg_np, out_dir / "gradcam.png")
    _save_attention_grid(x_np, attn_maps, cam, out_dir / "attention_maps.png")
    _save_occlusion_bar(occ_rows, out_dir / "occlusion.png")

    _print_report(tumor_types, pred_class, probs, base_conf, base_dice, occ_rows)

    print(f"\n  All outputs → {out_dir.resolve()}")
    print()
    print("  3D Slicer (File → Add Data, select all .nii.gz):")
    print("    t1_input.nii.gz     background scan")
    print("    seg_mask.nii.gz     tumour mask  → set as LabelMap, opacity 0.5")
    print("    gradcam.nii.gz      Grad-CAM     → colourmap Hot")
    print("    attn_u1..u4.nii.gz  attention    → colourmap Hot")

    return {
        "pred_class":  pred_class,
        "pred_name":   tumor_types[pred_class],
        "confidence":  confidence,
        "probs":       probs,
        "tumor_types": tumor_types,
        "seg_mask":    seg_np,
        "cam":         cam,
        "attn_maps":   attn_maps,
        "occlusion":   occ_rows,
    }


# CLI

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Patient inference + XAI. Saves NIfTI for 3D Slicer.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--t1",    default=None)
    p.add_argument("--t1c",   default=None)
    p.add_argument("--t2",    default=None)
    p.add_argument("--flair", default=None)
    p.add_argument("--out",   default="xai_results")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if not any([args.t1, args.t1c, args.t2, args.flair]):
        sys.exit("[ERROR] Provide at least one modality.")

    predict(
        checkpoint=args.checkpoint,
        t1=args.t1, t1c=args.t1c, t2=args.t2, flair=args.flair,
        out_dir=args.out,
        device=args.device,
    )
