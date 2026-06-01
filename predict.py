import argparse
import torch
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch.nn as nn
import matplotlib.pyplot as plt

r"""
CLI Use case:
python predict.py --t1   .\path\to\t1n.nii.gz \
                  --t1ce .\path\to\t1c.nii.gz \
                  --t2   .\path\to\t2w.nii.gz \
                  --flair .\path\to\t2f.nii.gz \
                  --model ./best.pt
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_SPACING = (1.0, 1.0, 1.0)
TARGET_SHAPE = (128, 128, 128)

NUM_MODALITIES = 4
ALL_MODALITIES = ["t1", "t1ce", "t2", "flair"]
NUM_TUMOR_TYPES = 4
TUMOR_TYPE_NAMES = ["healthy", "glioma", "meningioma", "metastasis"]


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
        self.Wg = nn.Conv3d(g_ch, int_ch, 1)
        self.Wx = nn.Conv3d(x_ch, int_ch, 1)
        self.psi = nn.Sequential(nn.Conv3d(int_ch, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.Wg(g) + self.Wx(x)))


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
    def __init__(self, num_tumor_types: int = NUM_TUMOR_TYPES):
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

        stats_in = NUM_MODALITIES * 4 + NUM_MODALITIES  # = 20
        self.stats_proj = nn.Sequential(
            nn.Linear(stats_in, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(),
        )

        cls_in = 256 * 8 + 128 * 8 + NUM_MODALITIES + 64  # = 3140
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, 512), nn.LayerNorm(512), nn.GELU(),
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
        x_flat = x_masked.flatten(2)
        ch_mean = x_flat.mean(dim=2)
        ch_std = x_flat.std(dim=2)
        ch_min = x_flat.min(dim=2).values
        ch_max = x_flat.max(dim=2).values
        stats = torch.cat([ch_mean, ch_std, ch_min, ch_max, modality_mask], dim=1)
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
        seg_logits = 20.0 * torch.tanh(raw / 20.0)

        gap_bottle = self.cls_pool(x5).flatten(1)
        gap_mid = self.cls_pool(x4).flatten(1)
        cls_input = torch.cat([gap_bottle, gap_mid, modality_mask, stats_feat], dim=1)
        cls_logits = self.classifier(cls_input)

        return seg_logits, cls_logits

def _resample_sitk(img: sitk.Image, target_spacing: tuple) -> sitk.Image:
    """Resample to isotropic target_spacing using linear interpolation."""
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()

    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    res = sitk.ResampleImageFilter()
    res.SetOutputSpacing(target_spacing)
    res.SetSize(new_size)
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    res.SetInterpolator(sitk.sitkLinear)
    return res.Execute(img)


def _center_crop_or_pad(vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Symmetric center-crop or zero-pad to target_shape — matches training code."""
    z, y, x = vol.shape
    tz, ty, tx = target_shape
    out = np.zeros(target_shape, dtype=vol.dtype)

    z0 = max((tz - z) // 2, 0); y0 = max((ty - y) // 2, 0); x0 = max((tx - x) // 2, 0)
    zs = max((z - tz) // 2, 0); ys = max((y - ty) // 2, 0); xs = max((x - tx) // 2, 0)
    zl = min(z, tz); yl = min(y, ty); xl = min(x, tx)

    out[z0:z0+zl, y0:y0+yl, x0:x0+xl] = vol[zs:zs+zl, ys:ys+yl, xs:xs+xl]
    return out


def _itk_affine(img: sitk.Image) -> np.ndarray:
    """Convert a SimpleITK image's spatial metadata to a nibabel-style 4x4 affine."""
    spacing = np.array(img.GetSpacing())       # (sx, sy, sz) in x,y,z order
    origin = np.array(img.GetOrigin())         # (ox, oy, oz)
    direction = np.array(img.GetDirection()).reshape(3, 3)  # row-major, x,y,z

    lps_to_ras = np.diag([-1, -1, 1])

    affine = np.eye(4)
    affine[:3, :3] = lps_to_ras @ direction @ np.diag(spacing)
    affine[:3,  3] = lps_to_ras @ origin

    return affine


def _load_and_preprocess_with_affine(path, target_spacing, target_shape):
    img = sitk.ReadImage(path)
    img = _resample_sitk(img, target_spacing)

    affine = _itk_affine(img)

    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = _center_crop_or_pad(arr, target_shape)

    # Adjust affine origin for the center crop offset
    z, y, x = sitk.GetArrayFromImage(img).shape  # pre-crop shape
    tz, ty, tx = target_shape
    dz = max((z - tz) // 2, 0)
    dy = max((y - ty) // 2, 0)
    dx = max((x - tx) // 2, 0)
    crop_offset_ras = affine[:3, :3] @ np.array([dx, dy, dz])
    affine[:3, 3] += crop_offset_ras

    p1, p99 = np.percentile(arr, (1, 99))
    arr = np.clip(arr, p1, p99)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), affine


def preprocess(modality_paths: dict,
               target_spacing: tuple = TARGET_SPACING,
               target_shape:   tuple = TARGET_SHAPE):
    """
    modality_paths: {mod_name: path_or_None} for each modality in ALL_MODALITIES.

    Returns
    -------
    image_tensor  : (1, 4, D, H, W) float32 torch.Tensor
    mm_tensor     : (1, 4)          float32 torch.Tensor  (1=present, 0=absent)
    affine        : numpy affine from nibabel (for saving output mask)
    """
    channels         = []
    modality_present = []
    affine           = None

    for mod in ALL_MODALITIES:
        path = modality_paths.get(mod)
        if path is not None:
            arr, aff = _load_and_preprocess_with_affine(path, target_spacing, target_shape)
            channels.append(arr)
            modality_present.append(1.0)
            if affine is None:
                affine = aff
        else:
            channels.append(np.zeros(target_shape, dtype=np.float32))
            modality_present.append(0.0)

    image = np.stack(channels, axis=0)[None, ...]        # (1, 4, D, H, W)
    mm = np.array(modality_present, dtype=np.float32)[None, :]  # (1, 4)

    return torch.tensor(image).float(), torch.tensor(mm).float(), affine

def visualize_single_slice(mri: np.ndarray, mask: np.ndarray,
                           slice_idx: int, title: str = ""):
    plt.figure(figsize=(6, 6))
    plt.imshow(mri[slice_idx], cmap="gray")
    plt.imshow(mask[slice_idx], cmap="Reds", alpha=0.4)
    plt.title(f"{title} – slice {slice_idx}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def save_mask(mask: np.ndarray, affine: np.ndarray, output_path: str):
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(nii, output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", default=None)
    parser.add_argument("--t1ce", default=None)
    parser.add_argument("--t2", default=None)
    parser.add_argument("--flair", default=None)
    parser.add_argument("--model")
    parser.add_argument("--output", default="prediction.nii.gz")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--slice", type=int, default=48,
                        help="Axial slice index to visualize (in resampled space, 0-based)")
    args = parser.parse_args()

    modality_paths = {
        "t1":    args.t1,
        "t1ce":  args.t1ce,
        "t2":    args.t2,
        "flair": args.flair,
    }
    present = [k for k, v in modality_paths.items() if v is not None]
    if not present:
        raise ValueError("Provide at least one modality path.")

    print(f"Present modalities : {present}")
    print(f"Target spacing : {TARGET_SPACING}")
    print(f"Target shape : {TARGET_SHAPE}")

    # Load model
    model = AttentionUNet3D(num_tumor_types=NUM_TUMOR_TYPES).to(device)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint  : epoch {ckpt.get('epoch', '?')} | "
          f"best_loss={ckpt.get('best_loss', 'n/a')}")

    # Preprocess
    print("Preprocessing (resample → center crop/pad → normalize)...")
    image_tensor, mm_tensor, affine = preprocess(modality_paths)
    image_tensor = image_tensor.to(device)
    mm_tensor    = mm_tensor.to(device)

    print("Running inference...")
    with torch.no_grad():
        seg_logits, cls_logits = model(image_tensor, mm_tensor)
        prob_map = torch.sigmoid(seg_logits).cpu().numpy()[0, 0]
        tumor_type = cls_logits.argmax(dim=1).item()
        cls_probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]

    mask = (prob_map > args.threshold).astype(np.uint8)
    tumor_present = bool(mask.sum() > 0)
    confidence = float(prob_map[mask == 1].mean()) if tumor_present else 0.0

    print(f"Tumor detected: {tumor_present}")
    print(f"Seg confidence: {confidence:.4f}  (mean prob in predicted region)")
    print(f"Predicted type: {TUMOR_TYPE_NAMES[tumor_type]} (class {tumor_type})")
    print("Class probs:", {TUMOR_TYPE_NAMES[i]: f"{p:.3f}" for i, p in enumerate(cls_probs)})

    slice_idx = min(args.slice, prob_map.shape[0] - 1)
    if args.t1ce:
        t1ce_arr = _load_and_preprocess_with_affine(args.t1ce, TARGET_SPACING, TARGET_SHAPE)
        visualize_single_slice(t1ce_arr[0], mask, slice_idx, title="T1ce")
    if args.flair:
        flair_arr = _load_and_preprocess_with_affine(args.flair, TARGET_SPACING, TARGET_SHAPE)
        visualize_single_slice(flair_arr[0], mask, slice_idx, title="FLAIR")

    save_mask(mask, affine, args.output)
    print(f"\nSaved mask to      : {args.output}")
    print(f"  (mask is in resampled {TARGET_SHAPE} space)")


if __name__ == "__main__":
    main()