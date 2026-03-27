import argparse
import torch
import nibabel as nib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

"""
CLI Use case
python predict.py --t1 .\merged_dataset\glioma_brats2023\BraTS2023-GLI-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t1n.nii.gz --t1ce .\merged_dataset\glioma_brats2023\BraTS2023-GLI-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t1c.nii.gz --t2 .\merged_dataset\glioma_brats2023\BraTS2023-GLI-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t2w.nii.gz --flair .\merged_dataset\glioma_brats2023\BraTS2023-GLI-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t2f.nii.gz

python predict.py --t1 .\merged_dataset\glioma_brats2023\BraTS2023-GLI-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t1n.nii.gz --t1ce .\merged_dataset\glioma_brats2023\BraTS2023-GLI-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t1c.nii.gz --t2 .\merged_dataset\glioma_brats2023\BraTS2023-GLI-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t2w.nii.gz --flair .\merged_dataset\glioma_brats2023\BraTS2023-GLI-TrainingData\BraTS-GLI-00000-000\BraTS-GLI-00000-000-t2f.nii.gz --model ./epoch_008.pt
"""

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# UTILS
# =========================

def visualize_single_slice(mri, mask, slice_idx):
    plt.figure(figsize=(6, 6))
    plt.imshow(mri[slice_idx], cmap="gray")
    plt.imshow(mask[slice_idx], cmap="Reds", alpha=0.4)
    plt.title(f"Slice {slice_idx}")
    plt.axis("off")
    plt.show()

def visualize_slices(mri, mask, axis=0):
    """
    mri: (D, H, W)
    mask: (D, H, W)
    axis: 0=depth, 1=height, 2=width
    """

    if axis == 1:
        mri = np.transpose(mri, (1, 0, 2))
        mask = np.transpose(mask, (1, 0, 2))
    elif axis == 2:
        mri = np.transpose(mri, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

    num_slices = mri.shape[0]

    for i in range(num_slices):
        if mask[i].sum() == 0:
            continue
        plt.figure(figsize=(6, 6))

        plt.imshow(mri[i], cmap="gray")

        # overlay mask (red)
        plt.imshow(mask[i], cmap="Reds", alpha=0.4)

        plt.title(f"Slice {i}")
        plt.axis("off")

        plt.show()

def center_crop(tensor, target_shape):
    _, _, d, h, w = tensor.shape
    td, th, tw = target_shape

    d1 = (d - td) // 2
    h1 = (h - th) // 2
    w1 = (w - tw) // 2

    return tensor[:, :, d1:d1+td, h1:h1+th, w1:w1+tw]

def pad_to_multiple(img, multiple=16):
    _, d, h, w = img.shape

    pad_d = (multiple - d % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    img = np.pad(
        img,
        ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
        mode="constant"
    )

    return img

# =========================
# MODEL
# =========================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        num_groups = min(8, out_ch)
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

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = center_crop(x2, x1.shape[2:])
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
        return self.outc(x)

# =========================
# PREPROCESS
# =========================

def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine

def normalize(img):
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    mean = img.mean()
    std = img.std()
    img = (img - mean) / (std + 1e-6)
    return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

def preprocess(t1, t1ce, t2, flair):
    t1, affine = load_nifti(t1)
    t1ce, _ = load_nifti(t1ce)
    t2, _ = load_nifti(t2)
    flair, _ = load_nifti(flair)

    original_shape = t1.shape

    t1 = normalize(t1)
    t1ce = normalize(t1ce)
    t2 = normalize(t2)
    flair = normalize(flair)

    img = np.stack([t1, t1ce, t2, flair], axis=0)
    img = pad_to_multiple(img)
    img = np.expand_dims(img, axis=0)

    return img, affine, original_shape

# =========================
# SAVE
# =========================

def save_mask(mask, affine, output_path):
    nii = nib.Nifti1Image(mask.astype(np.uint8), affine)
    nib.save(nii, output_path)

# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1", required=True)
    parser.add_argument("--t1ce", required=True)
    parser.add_argument("--t2", required=True)
    parser.add_argument("--flair", required=True)
    parser.add_argument("--model", default="best.pt")
    parser.add_argument("--output", default="prediction.nii.gz")

    args = parser.parse_args()

    # Load model
    model = AttentionUNet3D().to(device)

    checkpoint = torch.load(args.model, map_location=device)
    print(checkpoint.keys())

    model.load_state_dict(checkpoint["model_state"])  # adjust if needed
    model.eval()

    # Preprocess
    image, affine, original_shape = preprocess(
        args.t1, args.t1ce, args.t2, args.flair
    )

    image_tensor = torch.tensor(image).float().to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]

    d, h, w = original_shape

    # Crop BOTH
    prob_map = prob_map[:d, :h, :w]
    mask = (prob_map > 0.3).astype(np.uint8)

    # Now shapes match ✅
    confidence = prob_map[mask == 1].mean() if mask.sum() > 0 else 0

    tumor_present = mask.sum() > 0

    # after inference
    # Load one modality for visualization (FLAIR is best for tumors)
    flair_img, _ = load_nifti(args.flair)
    t1ce_img, _ = load_nifti(args.t1ce)

    # Normalize same way
    flair_img = normalize(flair_img)
    t1ce_img = normalize(t1ce_img)

    # Ensure same crop as mask
    d, h, w = original_shape
    flair_img = flair_img[:d, :h, :w]
    t1ce_img = t1ce_img[:d, :h, :w]

    # visualize_slices(t1ce_img, mask)
    # visualize_slices(flair_img, mask, axis=0)

    visualize_single_slice(t1ce_img, mask, slice_idx=80)
    visualize_single_slice(flair_img, mask, slice_idx=80)

    print("==== RESULT ====")
    print("Tumor detected:", tumor_present)
    print("Confidence:", float(confidence))

    save_mask(mask, affine, args.output)
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()