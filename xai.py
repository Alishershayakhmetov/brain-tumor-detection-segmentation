"""
Three techniques, all inference-only (no retraining):

  1. GradCAM3D - class activation map from the bottleneck (x5) for the classification head.
  2. AttentionMapExtractor - collects the 4 attention gate psi maps from the segmentation decoder.
  3. OcclusionSensitivity - drops each MRI modality one-at-a-time and measures the change in cls confidence and Dice.

Usage:

    from xai import GradCAM3D, AttentionMapExtractor, OcclusionSensitivity, run_xai_on_sample

    run_xai_on_sample(
        model=test_model,
        sample=next(iter(test_loader)),   # one batch
        tumor_types=tumor_types,
        device=DEVICE,
        out_dir=Path("xai_outputs"),
    )
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional


# Grad-CAM (classification head, bottleneck feature map x5)

class GradCAM3D:
    """
    Target layer: the bottleneck (d4 output = x5, shape B×256×d×h×w).
    Produces a 3D heatmap in the SAME spatial resolution as the input scan
    by trilinear upsampling.

    Parameters
    model: AttentionUNet3D instance (eval mode, on correct device).
    target_layer: nn.Module to hook — pass model.d4 (the bottleneck block).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._acts: Optional[torch.Tensor] = None   # saved forward activations
        self._grads: Optional[torch.Tensor] = None  # saved backward gradients

        # Register hooks on the target layer
        self._fwd_hook = target_layer.register_forward_hook(self._save_acts)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, module, inp, out):
        self._acts = out.detach()          # (B, C, d, h, w)

    def _save_grads(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach() # (B, C, d, h, w)

    def generate(
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        x: (1, 4, D, H, W) input scan tensor, on model device.
        modality_mask: (1, 4) float tensor.
        target_class: class index to explain (e.g. 0=healthy, 1=glioma…).

        Returns
        -------
        cam: np.ndarray, shape (D, H, W), values in [0, 1].
        """
        self.model.eval()

        # Need gradients for Grad-CAM
        x = x.requires_grad_(False)
        with torch.enable_grad():
            seg_logits, cls_logits = self.model(x, modality_mask)

            self.model.zero_grad()
            score = cls_logits[0, target_class]  # scalar for class of interest
            score.backward()

        # Global-average-pool gradients over spatial dims -> channel weights
        # acts/grads shape: (1, C, d, h, w)
        weights = self._grads.mean(dim=(2, 3, 4), keepdim=True)  # (1,C,1,1,1)
        cam = (weights * self._acts).sum(dim=1, keepdim=True)     # (1,1,d,h,w)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(
            cam, size=x.shape[2:], mode="trilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()   # (D, H, W)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.astype(np.float32)

    def remove_hooks(self):
        """Call when done to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# 2.  Attention-map extractor (decoder gates u1…u4)

class AttentionMapExtractor:
    """
    Extracts the spatial attention weights (psi) from all 4 AttentionGates
    in the decoder (u1, u2, u3, u4).

    The psi maps have shape (B, 1, d_l, h_l, w_l) at each decoder level l.
    They are upsampled to the input resolution for easy comparison.

    Parameters
    ----------
    model : AttentionUNet3D instance.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._maps: dict[str, torch.Tensor] = {}
        self._hooks: list = []

        # Hook each AttentionGate.psi sigmoid output
        for name in ["u1", "u2", "u3", "u4"]:
            up_block = getattr(model, name)
            hook = up_block.att.psi.register_forward_hook(
                self._make_hook(name)
            )
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def _hook(module, inp, out):
            # out: (B, 1, d, h, w) — sigmoid attention weights
            self._maps[name] = out.detach()
        return _hook

    def extract(
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """
        Forward pass and return attention maps for all 4 decoder stages.

        Returns
        -------
        maps : dict with keys "u1"…"u4".
                Each value is np.ndarray of shape (D, H, W), values in [0, 1].
        """
        self.model.eval()
        self._maps.clear()

        with torch.no_grad():
            _ = self.model(x, modality_mask)

        input_size = x.shape[2:]  # (D, H, W)
        result: dict[str, np.ndarray] = {}

        for name, psi in self._maps.items():
            # psi: (B, 1, d, h, w) — take batch index 0
            psi_up = F.interpolate(
                psi[:1].float(),
                size=input_size,
                mode="trilinear",
                align_corners=False,
            )
            arr = psi_up.squeeze().cpu().numpy()   # (D, H, W)
            # Already in [0,1] (sigmoid output), just cast
            result[name] = arr.astype(np.float32)

        return result

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


# 3.  Occlusion sensitivity (modality-level)

MODALITY_NAMES = ["T1", "T1c", "T2", "FLAIR"]


class OcclusionSensitivity:
    """
    Modality-level occlusion: zeroes one MRI channel at a time and measures
    the drop in (a) classification confidence for the predicted class and
    (b) Dice score for the segmentation.

    Parameters
    ----------
    model  : AttentionUNet3D, eval mode.
    device : "cuda" or "cpu".
    """

    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device

    @staticmethod
    def _dice(seg_logits: torch.Tensor, y: torch.Tensor, smooth: float = 1e-5) -> float:
        probs = torch.sigmoid(seg_logits)
        dims = (2, 3, 4)
        intersection = (probs * y).sum(dims)
        p_sum = probs.sum(dims)
        t_sum = y.sum(dims)
        dice = (2.0 * intersection + smooth) / (p_sum + t_sum + smooth)
        empty = (t_sum == 0)
        dice = torch.where(empty, (p_sum < 1.0).float(), dice)
        return float(dice.mean().item())

    def run(
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor,
        y: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        x              : (1, 4, D, H, W)
        modality_mask  : (1, 4)
        y              : (1, 1, D, H, W) ground-truth seg mask

        Returns
        -------
        results : dict with keys:
            "baseline_cls_conf"    : float — softmax prob of predicted class, baseline
            "baseline_dice"        : float
            "per_modality" : list of dicts, one per modality:
                {
                  "name"         : str,
                  "present"      : bool,   # was this modality in the original scan?
                  "cls_conf_drop": float,  # baseline_conf − occluded_conf
                  "dice_drop"    : float,  # baseline_dice − occluded_dice
                }
        """
        self.model.eval()

        with torch.no_grad():
            seg_logits, cls_logits = self.model(x, modality_mask)

        probs_cls = torch.softmax(cls_logits, dim=1)
        pred_class = int(cls_logits.argmax(dim=1).item())
        baseline_conf = float(probs_cls[0, pred_class].item())
        baseline_dice = self._dice(seg_logits, y)

        per_modality = []
        num_modalities = x.shape[1]

        for i in range(num_modalities):
            x_occ = x.clone()
            mm_occ = modality_mask.clone()

            x_occ[:, i] = 0.0    # zero out the channel
            mm_occ[:, i] = 0.0   # tell the model it's absent

            with torch.no_grad():
                seg_occ, cls_occ = self.model(x_occ, mm_occ)

            occ_conf = float(torch.softmax(cls_occ, dim=1)[0, pred_class].item())
            occ_dice = self._dice(seg_occ, y)

            per_modality.append({
                "name":          MODALITY_NAMES[i],
                "present":       bool(modality_mask[0, i].item() > 0.5),
                "cls_conf_drop": round(baseline_conf - occ_conf, 4),
                "dice_drop":     round(baseline_dice - occ_dice, 4),
            })

        return {
            "baseline_cls_conf": round(baseline_conf, 4),
            "baseline_dice":     round(baseline_dice, 4),
            "predicted_class":   pred_class,
            "per_modality":      per_modality,
        }


# Convenience runner — call this on a single sample after training

def run_xai_on_sample(
    model: nn.Module,
    sample: tuple,
    tumor_types: list[str],
    device: str,
    out_dir: Path,
):
    """
    Runs all three XAI techniques on the first item in a DataLoader batch
    and saves results as .npy files + prints a text report.

    Parameters
    ----------
    model       : AttentionUNet3D loaded from best.pt, already on `device`.
    sample      : one batch from DataLoader — tuple (x, y, mm, cls_target, seg_available).
    tumor_types : ordered list of class names, e.g. ["healthy","glioma","meningioma",…]
    device      : "cuda" or "cpu"
    out_dir     : directory to save output .npy files

    Output files
    ------------
    gradcam.npy          — (D, H, W) float32 in [0,1]
    attn_u1.npy … u4     — (D, H, W) float32 in [0,1]
    occlusion_report.txt — human-readable table
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y, mm, cls_target, *_ = sample

    # Take only the first sample in the batch
    x   = x[:1].to(device)
    y   = y[:1].to(device)
    mm  = mm[:1].to(device)
    true_class = int(cls_target[0].item())

    model.eval()

    # ── 1. Grad-CAM ─────────────────────────────────────────────────────────
    print("[XAI] Running Grad-CAM …")
    with torch.no_grad():
        _, cls_logits_check = model(x, mm)
    pred_class = int(cls_logits_check.argmax(dim=1).item())
    pred_name  = tumor_types[pred_class] if pred_class < len(tumor_types) else str(pred_class)
    true_name  = tumor_types[true_class] if true_class < len(tumor_types) else str(true_class)

    gradcam = GradCAM3D(model, target_layer=model.d4)
    cam = gradcam.generate(x, mm, target_class=pred_class)
    gradcam.remove_hooks()

    np.save(out_dir / "gradcam.npy", cam)
    print(f"    Saved gradcam.npy  (shape {cam.shape}, min={cam.min():.3f}, max={cam.max():.3f})")
    print(f"    Explained class: {pred_name} (pred) | true: {true_name}")

    # ── 2. Attention maps ────────────────────────────────────────────────────
    print("[XAI] Extracting attention maps …")
    extractor = AttentionMapExtractor(model)
    attn_maps = extractor.extract(x, mm)
    extractor.remove_hooks()

    for name, arr in attn_maps.items():
        path = out_dir / f"attn_{name}.npy"
        np.save(path, arr)
        print(f"    Saved attn_{name}.npy  (shape {arr.shape}, "
              f"mean={arr.mean():.3f}, max={arr.max():.3f})")

    # ── 3. Occlusion sensitivity ─────────────────────────────────────────────
    print("[XAI] Running occlusion sensitivity …")
    occlusion = OcclusionSensitivity(model, device)
    occ_result = occlusion.run(x, mm, y)

    report_lines = [
        "=" * 60,
        "  Occlusion Sensitivity Report",
        "=" * 60,
        f"  Predicted class : {tumor_types[occ_result['predicted_class']]}",
        f"  Baseline cls conf: {occ_result['baseline_cls_conf']:.4f}",
        f"  Baseline Dice    : {occ_result['baseline_dice']:.4f}",
        "",
        f"  {'Modality':<8} {'Present':<9} {'Cls conf drop':>14} {'Dice drop':>11}",
        f"  {'-'*46}",
    ]
    for entry in occ_result["per_modality"]:
        report_lines.append(
            f"  {entry['name']:<8} {'yes' if entry['present'] else 'no':<9}"
            f" {entry['cls_conf_drop']:>+14.4f} {entry['dice_drop']:>+11.4f}"
        )
    report_lines += ["=" * 60, ""]

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = out_dir / "occlusion_report.txt"
    report_path.write_text(report_text)
    print(f"    Saved occlusion_report.txt")

    print(f"\n[XAI] All outputs written to: {out_dir.resolve()}")
    return {
        "gradcam": cam,
        "attention_maps": attn_maps,
        "occlusion": occ_result,
    }