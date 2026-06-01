"""
analyze.py — Research-grade analysis for BraTS dataset + AttentionUNet3D model
Produces publication-ready figures and printed statistics for a diploma paper.

Usage:
    # Dataset analysis only (no model needed):
    python analyze.py --dataset_root ./merged_dataset --mode dataset

    # Model analysis only:
    python analyze.py --model ./best.pt --mode model

    # Full pipeline (dataset + model + inference on one sample):
    python analyze.py \
        --dataset_root ./merged_dataset \
        --model ./best.pt \
        --t1    ./sample/t1n.nii.gz \
        --t1ce  ./sample/t1c.nii.gz \
        --t2    ./sample/t2w.nii.gz \
        --flair ./sample/t2f.nii.gz \
        --mode all

"""

import argparse
import os
import glob
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import nibabel as nib

PALETTE = {
    "bg":       "#0D1117",
    "surface":  "#161B22",
    "border":   "#30363D",
    "accent1":  "#58A6FF",   # blue
    "accent2":  "#F78166",   # orange-red
    "accent3":  "#3FB950",   # green
    "accent4":  "#D2A8FF",   # purple
    "text":     "#E6EDF3",
    "subtext":  "#8B949E",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["surface"],
    "axes.edgecolor":    PALETTE["border"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["subtext"],
    "ytick.color":       PALETTE["subtext"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["border"],
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": PALETTE["bg"],
})

TUMOR_CMAP = LinearSegmentedColormap.from_list(
    "tumor", ["#00000000", "#F78166CC", "#F7816699"], N=256
)


# MODEL

def center_crop(tensor, target_shape):
    _, _, d, h, w = tensor.shape
    td, th, tw = target_shape
    d1, h1, w1 = (d - td) // 2, (h - th) // 2, (w - tw) // 2
    return tensor[:, :, d1:d1+td, h1:h1+th, w1:w1+tw]


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        ng = min(8, out_ch)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(ng, out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(ng, out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = center_crop(x2, x1.shape[2:])
        return self.conv(torch.cat([x2, x1], dim=1))


class AttentionUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(4, 16)
        self.d1  = nn.Sequential(nn.MaxPool3d(2), DoubleConv(16,  32))
        self.d2  = nn.Sequential(nn.MaxPool3d(2), DoubleConv(32,  64))
        self.d3  = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64,  128))
        self.d4  = nn.Sequential(nn.MaxPool3d(2), DoubleConv(128, 256))
        self.u1  = Up(256, 128)
        self.u2  = Up(128,  64)
        self.u3  = Up(64,   32)
        self.u4  = Up(32,   16)
        self.outc = nn.Conv3d(16, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1); x3 = self.d2(x2)
        x4 = self.d3(x3); x5 = self.d4(x4)
        x  = self.u1(x5, x4); x = self.u2(x, x3)
        x  = self.u3(x, x2);  x = self.u4(x, x1)
        return self.outc(x)

def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata().astype(np.float32), nii.affine, nii.header

def normalize(img):
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    img = (img - img.mean()) / (img.std() + 1e-6)
    return np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

def pad_to_multiple(img, m=16):
    _, d, h, w = img.shape
    pd = (m - d % m) % m; ph = (m - h % m) % m; pw = (m - w % m) % m
    return np.pad(img, ((0,0),(0,pd),(0,ph),(0,pw)), mode="constant")

def title_box(ax, text):
    ax.set_title(text, pad=8, fontsize=12,
                 bbox=dict(facecolor=PALETTE["border"], edgecolor="none",
                           boxstyle="round,pad=0.3", alpha=0.8))

def count_parameters(model):
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def layer_breakdown(model):
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d,
                               nn.GroupNorm, nn.MaxPool3d)):
            params = sum(p.numel() for p in module.parameters())
            rows.append((name or "root", type(module).__name__, params))
    return rows

def plot_model_architecture(model, save_path="fig_model_architecture.png"):
    """
    Two-panel figure:
      Left  – encoder/decoder parameter distribution (horizontal bars)
      Right – layer-type breakdown (donut chart)
    """
    device = next(model.parameters()).device

    blocks = {
        "Encoder\nBlock 0 (inc)": model.inc,
        "Encoder\nBlock 1 (d1)": model.d1,
        "Encoder\nBlock 2 (d2)": model.d2,
        "Encoder\nBlock 3 (d3)": model.d3,
        "Bottleneck\n(d4)":       model.d4,
        "Decoder\nBlock 1 (u1)": model.u1,
        "Decoder\nBlock 2 (u2)": model.u2,
        "Decoder\nBlock 3 (u3)": model.u3,
        "Decoder\nBlock 4 (u4)": model.u4,
        "Output\nConv (outc)":   model.outc,
    }
    labels = list(blocks.keys())
    counts = [sum(p.numel() for p in b.parameters()) for b in blocks.values()]

    # encoder / bottleneck / decoder colours
    colors = (
        [PALETTE["accent1"]] * 4 +
        [PALETTE["accent2"]] +
        [PALETTE["accent3"]] * 4 +
        [PALETTE["accent4"]]
    )

    # layer type counts
    type_counts = {}
    for _, m in model.named_modules():
        t = type(m).__name__
        if t not in ("Sequential", "AttentionUNet3D", "DoubleConv", "Up"):
            type_counts[t] = type_counts.get(t, 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             gridspec_kw={"width_ratios": [1.6, 1]})
    fig.suptitle("AttentionUNet3D — Model Architecture Analysis",
                 fontsize=15, fontweight="bold", y=1.01)

    ax = axes[0]
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, counts, color=colors, height=0.6,
                   edgecolor=PALETTE["border"], linewidth=0.5)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Parameters")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.grid(axis="x", alpha=0.4)
    ax.set_axisbelow(True)
    title_box(ax, "Parameter Count per Block")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_width() + max(counts)*0.01,
                bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=8,
                color=PALETTE["subtext"])

    # legend patches
    from matplotlib.patches import Patch
    legend_items = [
        Patch(color=PALETTE["accent1"], label="Encoder"),
        Patch(color=PALETTE["accent2"], label="Bottleneck"),
        Patch(color=PALETTE["accent3"], label="Decoder"),
        Patch(color=PALETTE["accent4"], label="Output"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              framealpha=0.3, fontsize=9)

    ax2 = axes[1]
    donut_labels = list(type_counts.keys())
    donut_vals   = list(type_counts.values())
    donut_colors = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
                    PALETTE["accent4"], PALETTE["subtext"],
                    "#FFA657", "#79C0FF"][:len(donut_labels)]

    wedges, texts, autotexts = ax2.pie(
        donut_vals, labels=None, autopct="%1.0f%%",
        colors=donut_colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"], linewidth=1.5),
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8); at.set_color(PALETTE["bg"])
    ax2.legend(wedges, donut_labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8,
               framealpha=0.2)
    title_box(ax2, "Layer Type Distribution")

    # total params annotation
    total, trainable = count_parameters(model)
    fig.text(0.5, -0.03,
             f"Total parameters: {total:,}  |  Trainable: {trainable:,}  "
             f"|  Size ≈ {total*4/1024**2:.1f} MB (float32)",
             ha="center", fontsize=10, color=PALETTE["subtext"])

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f" Saved: {save_path}")
    return total, trainable


def plot_model_summary_table(model, save_path="fig_model_summary_table.png"):
    """Renders a clean table of all layers with shapes & param counts."""
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d,
                               nn.GroupNorm, nn.MaxPool3d, nn.ReLU)):
            params = sum(p.numel() for p in module.parameters())
            # kernel / groups info
            extra = ""
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                extra = f"k={module.kernel_size[0]}, pad={module.padding[0]}"
            elif isinstance(module, nn.GroupNorm):
                extra = f"G={module.num_groups}, C={module.num_channels}"
            rows.append([name or "(root)", type(module).__name__,
                         extra, f"{params:,}" if params else "—"])

    col_labels = ["Layer Name", "Type", "Config", "Params"]
    col_widths  = [0.38, 0.22, 0.22, 0.13]

    fig, ax = plt.subplots(figsize=(14, max(6, len(rows)*0.32 + 1.5)))
    ax.axis("off")
    title_box(ax, "AttentionUNet3D — Full Layer Summary")

    row_colors = []
    for i in range(len(rows)):
        c = PALETTE["surface"] if i % 2 == 0 else PALETTE["bg"]
        row_colors.append([c]*4)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.25)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["border"])
        if r == 0:
            cell.set_facecolor(PALETTE["border"])
            cell.set_text_props(color=PALETTE["text"], fontweight="bold")
        else:
            cell.set_facecolor(row_colors[r-1][c])
            cell.set_text_props(color=PALETTE["text"])

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f" Saved: {save_path}")


def plot_receptive_field(save_path="fig_receptive_field.png"):
    """
    Analytically computes the theoretical receptive field at each encoder depth.
    """
    labels_d  = ["Depth 0\n(inc)", "Depth 1\n(d1)", "Depth 2\n(d2)",
                 "Depth 3\n(d3)", "Depth 4\n(bottleneck)"]
    strides   = [1, 2, 4, 8, 16]          # after MaxPool cascades
    # RF = 1 + sum over layers of (kernel-1)*effective_stride
    # For 2×Conv(k=3) per block at stride s: delta = 2*(3-1)*s = 4s
    rf = [1 + 4*1,
          1 + 4*1 + 4*2,
          1 + 4*1 + 4*2 + 4*4,
          1 + 4*1 + 4*2 + 4*4 + 4*8,
          1 + 4*1 + 4*2 + 4*4 + 4*8 + 4*16]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Theoretical Receptive Field Analysis", fontsize=14, fontweight="bold")

    # bar chart
    ax = axes[0]
    bars = ax.bar(labels_d, rf, color=PALETTE["accent1"],
                  edgecolor=PALETTE["border"], width=0.5)
    for bar, val in zip(bars, rf):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val}", ha="center", fontsize=9, color=PALETTE["text"])
    ax.set_ylabel("Receptive Field (voxels)")
    ax.set_ylim(0, max(rf)*1.15)
    ax.grid(axis="y", alpha=0.4); ax.set_axisbelow(True)
    title_box(ax, "RF vs Encoder Depth")

    # line + feature map size
    ax2 = axes[1]
    INPUT = np.array([240, 240, 160])
    fm_sizes = [INPUT // s for s in strides]
    fm_vol   = [int(np.prod(s)) for s in fm_sizes]

    ax2.plot(labels_d, fm_vol, "o-", color=PALETTE["accent2"],
             lw=2, ms=8, markeredgewidth=1.5,
             markerfacecolor=PALETTE["bg"])
    for i, (s, v) in enumerate(zip(fm_sizes, fm_vol)):
        ax2.annotate(f"{s[0]}×{s[1]}×{s[2]}\n({v:,} vox)",
                     (i, v), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=7.5,
                     color=PALETTE["subtext"])
    ax2.set_ylabel("Feature Map Volume (voxels)")
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax2.grid(alpha=0.4); ax2.set_axisbelow(True)
    title_box(ax2, "Feature Map Size vs Encoder Depth")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f" Saved: {save_path}")


def plot_channel_progression(save_path="fig_channel_progression.png"):
    enc_channels = [4, 16, 32, 64, 128, 256]
    dec_channels = [256, 128, 64, 32, 16, 1]
    enc_labels   = ["Input", "inc→16", "d1→32", "d2→64", "d3→128", "d4→256"]
    dec_labels   = ["256 (bottleneck)", "u1→128", "u2→64", "u3→32", "u4→16", "outc→1"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x_enc = np.arange(len(enc_channels))
    x_dec = x_enc + len(enc_channels) - 1 + 0.5

    ax.plot(x_enc, enc_channels, "o-", color=PALETTE["accent1"],
            lw=2.5, ms=9, label="Encoder path",
            markerfacecolor=PALETTE["bg"], markeredgewidth=2)
    ax.plot(np.arange(len(dec_channels)) + len(enc_channels) - 1,
            dec_channels, "s-", color=PALETTE["accent3"],
            lw=2.5, ms=9, label="Decoder path",
            markerfacecolor=PALETTE["bg"], markeredgewidth=2)

    # annotations
    for i, (x, y) in enumerate(zip(x_enc, enc_channels)):
        ax.text(x, y+10, str(y), ha="center", fontsize=9,
                color=PALETTE["accent1"])
    for i, (y) in enumerate(dec_channels):
        ax.text(i + len(enc_channels)-1, y+10, str(y), ha="center",
                fontsize=9, color=PALETTE["accent3"])

    ax.axvline(len(enc_channels)-1, color=PALETTE["accent2"],
               linestyle="--", lw=1.2, label="Bottleneck")
    ax.set_ylabel("Number of Channels")
    ax.set_xticks(list(x_enc) + [len(enc_channels)-1+i for i in range(1, len(dec_channels))])
    labels_all = enc_labels + dec_labels[1:]
    ax.set_xticklabels(labels_all, rotation=30, ha="right", fontsize=8)
    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(alpha=0.35); ax.set_axisbelow(True)
    ax.legend(framealpha=0.25)
    title_box(ax, "Channel Progression — Encoder / Decoder")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f" Saved: {save_path}")


def model_analysis(model_path):
    print("  MODEL ANALYSIS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AttentionUNet3D().to(device)

    ckpt = torch.load(model_path, map_location=device)
    # handle various checkpoint formats
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        extra_keys = {k: v for k, v in ckpt.items() if k != "model_state"}
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        extra_keys = {k: v for k, v in ckpt.items() if k != "state_dict"}
    else:
        model.load_state_dict(ckpt)
        extra_keys = {}

    model.eval()

    total, trainable = count_parameters(model)
    print(f"  Total parameters   : {total:,}")
    print(f"  Trainable params   : {trainable:,}")
    print(f"  Model size (fp32)  : {total*4/1024**2:.2f} MB")
    if extra_keys:
        print(f"  Checkpoint extras  : {list(extra_keys.keys())}")
        for k, v in extra_keys.items():
            if isinstance(v, (int, float, str)):
                print(f"    {k}: {v}")

    # throughput benchmark
    x = torch.randn(1, 4, 32, 32, 32).to(device)
    # warmup
    with torch.no_grad():
        for _ in range(3): model(x)
    t0 = time.time()
    REPS = 10
    with torch.no_grad():
        for _ in range(REPS): model(x)
    elapsed = (time.time()-t0)/REPS*1000
    print(f" Inference (32³): {elapsed:.1f} ms  (avg over {REPS} runs, {device})")

    plot_model_architecture(model)
    plot_model_summary_table(model)
    plot_receptive_field()
    plot_channel_progression()

    return model


# DATASET ANALYSIS

MODALITIES = {
    "t1n": ("T1",     PALETTE["accent1"]),
    "t1c": ("T1ce",   PALETTE["accent2"]),
    "t2w": ("T2",     PALETTE["accent3"]),
    "t2f": ("FLAIR",  PALETTE["accent4"]),
}
SEG_SUFFIX = "seg"

def discover_cases(dataset_root):
    """
    Walks dataset_root recursively and returns a list of dicts:
      { "t1n": path, "t1c": path, "t2w": path, "t2f": path, "seg": path }
    Any case with at least one modality is included (missing ones → None).
    """
    cases = {}
    for path in sorted(glob.glob(os.path.join(dataset_root, "**", "*.nii.gz"),
                                  recursive=True)):
        basename = os.path.basename(path)
        # identify modality from filename
        for suffix in list(MODALITIES.keys()) + [SEG_SUFFIX]:
            if f"-{suffix}." in basename or f"_{suffix}." in basename:
                case_id = os.path.basename(os.path.dirname(path))
                if case_id not in cases:
                    cases[case_id] = {}
                cases[case_id][suffix] = path
                break
    return list(cases.values())


def collect_stats(cases, max_cases=200):
    """
    Samples up to max_cases for intensity/shape statistics.
    Returns dict of lists.
    """
    stats = {k: {"mean": [], "std": [], "min": [], "max": [], "p1": [], "p99": []}
             for k in MODALITIES}
    shapes = []
    seg_vols = []       # tumour voxel counts
    seg_present = 0

    sampled = cases[:max_cases]
    print(f" Collecting stats from {len(sampled)} / {len(cases)} cases …")

    for case in sampled:
        # shape from first available modality
        for key in MODALITIES:
            if key in case:
                d, _ , _ = load_nifti(case[key])
                shapes.append(d.shape)
                break

        for key in MODALITIES:
            if key not in case:
                continue
            vol, _, _ = load_nifti(case[key])
            vol = vol.astype(np.float32)
            stats[key]["mean"].append(float(np.mean(vol)))
            stats[key]["std"].append(float(np.std(vol)))
            stats[key]["min"].append(float(np.min(vol)))
            stats[key]["max"].append(float(np.max(vol)))
            p1, p99 = np.percentile(vol, (1, 99))
            stats[key]["p1"].append(float(p1))
            stats[key]["p99"].append(float(p99))

        if SEG_SUFFIX in case:
            seg, _, _ = load_nifti(case[SEG_SUFFIX])
            tv = int((seg > 0).sum())
            seg_vols.append(tv)
            seg_present += (tv > 0)

    return stats, shapes, seg_vols, seg_present


def plot_dataset_overview(cases, stats, shapes, seg_vols, seg_present,
                          save_path="fig_dataset_overview.png"):
    """4-panel figure: case count, shape distribution, seg presence, modality coverage."""

    # counts
    mod_present = {k: sum(1 for c in cases if k in c) for k in MODALITIES}
    n_with_seg  = sum(1 for c in cases if SEG_SUFFIX in c)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle("BraTS Dataset — Overview Statistics", fontsize=15, fontweight="bold")

    # panel 1: modality coverage bar
    ax1 = fig.add_subplot(gs[0, 0])
    names  = [MODALITIES[k][0] for k in MODALITIES] + ["Seg"]
    counts = [mod_present[k]   for k in MODALITIES] + [n_with_seg]
    colors = [MODALITIES[k][1] for k in MODALITIES] + [PALETTE["subtext"]]
    bars   = ax1.bar(names, counts, color=colors,
                     edgecolor=PALETTE["border"], width=0.55)
    for bar, val in zip(bars, counts):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+len(cases)*0.01,
                 str(val), ha="center", fontsize=9)
    ax1.set_ylabel("Cases with file")
    ax1.set_ylim(0, len(cases)*1.15)
    ax1.axhline(len(cases), color=PALETTE["accent2"], lw=1, ls="--", label=f"Total: {len(cases)}")
    ax1.legend(fontsize=8, framealpha=0.2)
    ax1.grid(axis="y", alpha=0.4); ax1.set_axisbelow(True)
    title_box(ax1, "Modality & Segmentation Coverage")

    # panel 2: spatial shape
    ax2 = fig.add_subplot(gs[0, 1])
    if shapes:
        dims = np.array(shapes)
        axes_labels = ["Depth (D)", "Height (H)", "Width (W)"]
        bps = ax2.boxplot(
            [dims[:, i] for i in range(min(3, dims.shape[1]))],
            patch_artist=True, widths=0.45,
            boxprops=dict(facecolor=PALETTE["surface"], edgecolor=PALETTE["accent1"]),
            medianprops=dict(color=PALETTE["accent2"], lw=2),
            whiskerprops=dict(color=PALETTE["subtext"]),
            capprops=dict(color=PALETTE["subtext"]),
            flierprops=dict(marker="o", markerfacecolor=PALETTE["accent4"],
                            ms=4, alpha=0.5),
        )
        ax2.set_xticklabels(axes_labels[:dims.shape[1]])
        ax2.set_ylabel("Voxels")
        ax2.grid(axis="y", alpha=0.4); ax2.set_axisbelow(True)
    title_box(ax2, "Volume Shape Distribution")

    # panel 3: tumour voxel volume histogram
    ax3 = fig.add_subplot(gs[0, 2])
    if seg_vols:
        nz = [v for v in seg_vols if v > 0]
        ax3.hist(nz, bins=30, color=PALETTE["accent2"],
                 edgecolor=PALETTE["border"], alpha=0.85)
        ax3.axvline(np.median(nz), color=PALETTE["accent3"], lw=1.5,
                    ls="--", label=f"Median: {int(np.median(nz)):,}")
        ax3.set_xlabel("Tumour Voxels")
        ax3.set_ylabel("Cases")
        ax3.legend(fontsize=8, framealpha=0.2)
        ax3.grid(alpha=0.4); ax3.set_axisbelow(True)
        pct = 100*seg_present/len(seg_vols) if seg_vols else 0
        ax3.set_title(f"Tumour Volume Distribution\n({pct:.0f}% cases with tumour)",
                      fontsize=11, fontweight="bold")
    else:
        ax3.text(0.5, 0.5, "No segmentation\nfiles found",
                 ha="center", va="center", transform=ax3.transAxes,
                 color=PALETTE["subtext"], fontsize=11)
        title_box(ax3, "Tumour Volume Distribution")

    # panel 4: per-modality mean intensity box
    ax4 = fig.add_subplot(gs[1, :2])
    plot_data  = []
    plot_labels = []
    plot_colors = []
    for k in MODALITIES:
        if stats[k]["mean"]:
            plot_data.append(stats[k]["mean"])
            plot_labels.append(MODALITIES[k][0])
            plot_colors.append(MODALITIES[k][1])
    if plot_data:
        bps2 = ax4.boxplot(
            plot_data, patch_artist=True, widths=0.5,
            medianprops=dict(color=PALETTE["bg"], lw=2.5),
            whiskerprops=dict(color=PALETTE["subtext"]),
            capprops=dict(color=PALETTE["subtext"]),
            flierprops=dict(marker="o", ms=3, alpha=0.4),
        )
        for patch, col in zip(bps2["boxes"], plot_colors):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        ax4.set_xticklabels(plot_labels)
        ax4.set_ylabel("Mean Voxel Intensity (raw)")
        ax4.grid(axis="y", alpha=0.4); ax4.set_axisbelow(True)
    title_box(ax4, "Per-Modality Intensity Distribution (sampled cases)")

    # panel 5: std distribution
    ax5 = fig.add_subplot(gs[1, 2])
    for k in MODALITIES:
        if stats[k]["std"]:
            ax5.hist(stats[k]["std"], bins=25, alpha=0.55,
                     label=MODALITIES[k][0], color=MODALITIES[k][1],
                     edgecolor="none")
    ax5.set_xlabel("Std Dev of Voxel Intensity")
    ax5.set_ylabel("Cases")
    ax5.legend(fontsize=8, framealpha=0.2)
    ax5.grid(alpha=0.4); ax5.set_axisbelow(True)
    title_box(ax5, "Intensity Std Distribution")

    plt.savefig(save_path)
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


def plot_intensity_histograms(cases, save_path="fig_intensity_histograms.png",
                               max_cases=50):
    """One histogram per modality, overlaid across sampled cases."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Voxel Intensity Histograms per Modality\n"
                 f"(sampled {min(max_cases, len(cases))} cases)",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for idx, (key, (name, color)) in enumerate(MODALITIES.items()):
        ax = axes[idx]
        sampled = [c for c in cases if key in c][:max_cases]
        all_vals = []
        for c in sampled:
            vol, _, _ = load_nifti(c[key])
            brain_mask = vol > 0
            if brain_mask.sum() > 1000:
                all_vals.append(vol[brain_mask])
        if all_vals:
            combined = np.concatenate(all_vals)
            # clip for display
            p1, p99 = np.percentile(combined, (0.5, 99.5))
            combined = combined[(combined >= p1) & (combined <= p99)]
            ax.hist(combined, bins=100, color=color,
                    edgecolor="none", alpha=0.85)
            ax.axvline(np.mean(combined), color=PALETTE["text"], lw=1.5,
                       ls="--", label=f"Mean: {np.mean(combined):.1f}")
            ax.axvline(np.median(combined), color=PALETTE["subtext"], lw=1.5,
                       ls=":", label=f"Median: {np.median(combined):.1f}")
            ax.set_xlabel("Intensity"); ax.set_ylabel("Voxel Count")
            ax.legend(fontsize=8, framealpha=0.2)
            ax.grid(alpha=0.35); ax.set_axisbelow(True)
        title_box(ax, f"{name} Intensity Histogram")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f" Saved: {save_path}")


def plot_slice_montage(cases, save_path="fig_slice_montage.png", n_cases=6):
    """
    Shows a grid: rows = random cases, cols = T1/T1ce/T2/FLAIR/Seg.
    """
    import random
    complete = [c for c in cases
                if all(k in c for k in MODALITIES)
                and SEG_SUFFIX in c]
    if not complete:
        print(" No complete cases (all modalities + seg) — skipping montage.")
        return
    random.seed(42)
    sampled = random.sample(complete, min(n_cases, len(complete)))

    n_cols = len(MODALITIES) + 1   # modalities + seg
    n_rows = len(sampled)
    col_titles = [MODALITIES[k][0] for k in MODALITIES] + ["Seg Overlay"]
    col_colors = [MODALITIES[k][1] for k in MODALITIES] + [PALETTE["accent2"]]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_rows == 1: axes = axes[np.newaxis, :]
    fig.suptitle("BraTS Dataset — Multi-Modality Slice Montage",
                 fontsize=14, fontweight="bold")

    for row, case in enumerate(sampled):
        # pick middle axial slice
        ref_vol, _, _ = load_nifti(case[list(MODALITIES.keys())[0]])
        mid = ref_vol.shape[2] // 2

        seg_vol, _, _ = load_nifti(case[SEG_SUFFIX]) if SEG_SUFFIX in case else (None, None, None)

        for col, key in enumerate(MODALITIES):
            ax = axes[row, col]
            ax.axis("off")
            if key in case:
                vol, _, _ = load_nifti(case[key])
                slc = np.rot90(vol[:, :, mid])
                p1, p99 = np.percentile(slc, (1, 99))
                ax.imshow(np.clip(slc, p1, p99), cmap="gray", interpolation="bilinear")
            if row == 0:
                ax.set_title(col_titles[col], color=col_colors[col],
                             fontsize=10, fontweight="bold")

        # last column: FLAIR + seg overlay
        ax = axes[row, -1]
        ax.axis("off")
        flair_key = "t2f"
        if flair_key in case:
            flair_vol, _, _ = load_nifti(case[flair_key])
            slc = np.rot90(flair_vol[:, :, mid])
            p1, p99 = np.percentile(slc, (1, 99))
            ax.imshow(np.clip(slc, p1, p99), cmap="gray", interpolation="bilinear")
            if seg_vol is not None:
                seg_slc = np.rot90(seg_vol[:, :, mid])
                ax.imshow(np.ma.masked_where(seg_slc == 0, seg_slc),
                          cmap=TUMOR_CMAP, alpha=0.7, interpolation="nearest")
        if row == 0:
            ax.set_title(col_titles[-1], color=col_colors[-1],
                         fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f" Saved: {save_path}")


def dataset_analysis(dataset_root):
    print("DATASET ANALYSIS")

    cases = discover_cases(dataset_root)
    print(f"  Cases discovered : {len(cases)}")
    if not cases:
        print(" No .nii.gz files found — check --dataset_root path.")
        return

    stats, shapes, seg_vols, seg_present = collect_stats(cases)

    if shapes:
        sh = np.array(shapes)
        print(f"  Volume shapes    : D={sh[:,0].min()}–{sh[:,0].max()}  "
              f"H={sh[:,1].min()}–{sh[:,1].max()}  "
              f"W={sh[:,2].min()}–{sh[:,2].max()}")
    if seg_vols:
        nz = [v for v in seg_vols if v > 0]
        print(f"  Cases with tumour: {seg_present} / {len(seg_vols)}")
        if nz:
            print(f"  Tumour voxels    : median={int(np.median(nz)):,}  "
                  f"mean={int(np.mean(nz)):,}  "
                  f"max={int(np.max(nz)):,}")

    plot_dataset_overview(cases, stats, shapes, seg_vols, seg_present)
    plot_intensity_histograms(cases)
    plot_slice_montage(cases)


# INFERENCE VISUALISATION

def plot_inference_result(t1_path, t1ce_path, t2_path, flair_path,
                          model, save_path="fig_inference_result.png"):
    """
    Multi-panel figure showing model output on one patient:
      Row 1 – 3 axial slices (low/mid/high) of all 4 modalities
      Row 2 – probability map + binary mask overlaid on T1ce and FLAIR
    """
    device = next(model.parameters()).device

    # --- load & preprocess ---
    def load_norm(p):
        v, _, _ = load_nifti(p)
        return normalize(v)

    t1   = load_norm(t1_path)
    t1ce = load_norm(t1ce_path)
    t2   = load_norm(t2_path)
    flair= load_norm(flair_path)
    original_shape = t1.shape

    img = np.stack([t1, t1ce, t2, flair], axis=0)
    img = pad_to_multiple(img)
    tensor = torch.tensor(img[np.newaxis]).float().to(device)

    with torch.no_grad():
        out = model(tensor)
        prob = torch.sigmoid(out).cpu().numpy()[0, 0]

    d, h, w = original_shape
    prob  = prob[:d, :h, :w]
    mask  = (prob > 0.3).astype(np.uint8)

    n_tumor = mask.sum()
    conf    = float(prob[mask==1].mean()) if n_tumor > 0 else 0.0

    # pick slices with most tumour (or evenly spaced)
    per_slice = mask.sum(axis=(1,2))
    if per_slice.max() > 0:
        best = int(per_slice.argmax())
        low  = max(0, best - 20)
        high = min(d-1, best + 20)
        slice_idxs = [low, best, high]
    else:
        slice_idxs = [d//4, d//2, 3*d//4]

    n_rows = 3

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Inference Result — BraTS Patient",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(n_rows, len(slice_idxs), figure=fig,
                           hspace=0.35, wspace=0.08)

    # Row 0: T1ce slices
    for col, si in enumerate(slice_idxs):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(np.rot90(t1ce[:, :, si]), cmap="gray")
        ax.axis("off")
        title_box(ax, f"T1ce — Slice {si}")

    # Row 1: FLAIR + prob overlay
    for col, si in enumerate(slice_idxs):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(np.rot90(flair[:, :, si]), cmap="gray")
        p_slc = np.rot90(prob[:, :, si])
        im = ax.imshow(p_slc, cmap="hot", alpha=0.55,
                       vmin=0, vmax=1, interpolation="bilinear")
        ax.axis("off")
        title_box(ax, f"FLAIR + Prob Map — Slice {si}")
    # colorbar on last
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label="P(tumour)")
    cax.yaxis.label.set_color(PALETTE["text"])
    cax.tick_params(colors=PALETTE["subtext"])
    cax.set_facecolor(PALETTE["surface"])

    # Row 2: T1ce + binary mask overlay
    for col, si in enumerate(slice_idxs):
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(np.rot90(t1ce[:, :, si]), cmap="gray")
        m_slc = np.rot90(mask[:, :, si])
        ax.imshow(np.ma.masked_where(m_slc == 0, m_slc),
                  cmap=TUMOR_CMAP, alpha=0.75, interpolation="nearest")
        ax.axis("off")
        title_box(ax, f"T1ce + Mask — Slice {si}")

    # stats text box
    stats_txt = (f"Tumour detected: {'Yes' if n_tumor>0 else 'No'}   "
                 f"Voxels: {n_tumor:,}   "
                 f"Confidence: {conf:.3f}   "
                 f"Volume ≈ {n_tumor*1e-3:.1f}k vox")
    fig.text(0.5, 0.01, stats_txt, ha="center", fontsize=10,
             color=PALETTE["subtext"],
             bbox=dict(facecolor=PALETTE["border"], edgecolor="none",
                       boxstyle="round,pad=0.4", alpha=0.7))

    plt.savefig(save_path)
    plt.close(fig)
    print(f" Saved: {save_path}")
    print(f" Tumour voxels: {n_tumor:,}   Confidence: {conf:.4f}")


# ENTRY POINT

def main():
    global matplotlib

    parser = argparse.ArgumentParser(
        description="Research-grade analysis for BraTS + AttentionUNet3D"
    )
    parser.add_argument("--mode", default="all",
                        choices=["model", "dataset", "inference", "all"],
                        help="What to analyse")
    parser.add_argument("--model",        default="best.pt",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--dataset_root", default="./merged_dataset",
                        help="Root folder of BraTS dataset")
    parser.add_argument("--t1",    help="T1 .nii.gz for inference")
    parser.add_argument("--t1ce", help="T1ce .nii.gz for inference")
    parser.add_argument("--t2",    help="T2 .nii.gz for inference")
    parser.add_argument("--flair", help="FLAIR .nii.gz for inference")
    args = parser.parse_args()

    print("AttentionUNet3D  —  Research Analysis Tool")

    loaded_model = None

    if args.mode in ("model", "all", "inference"):
        if os.path.isfile(args.model):
            loaded_model = model_analysis(args.model)
        else:
            print(f"\n Model file not found: {args.model}")
            print(" Running architecture-only analysis …")
            loaded_model = AttentionUNet3D()
            plot_model_architecture(loaded_model)
            plot_model_summary_table(loaded_model)
            plot_receptive_field()
            plot_channel_progression()

    if args.mode in ("dataset", "all"):
        if os.path.isdir(args.dataset_root):
            dataset_analysis(args.dataset_root)
        else:
            print(f"\n Dataset root not found: {args.dataset_root}")

    if args.mode in ("inference", "all"):
        if all([args.t1, args.t1ce, args.t2, args.flair]):
            if loaded_model is None:
                loaded_model = AttentionUNet3D()
            print("  INFERENCE VISUALISATION")
            plot_inference_result(
                args.t1, args.t1ce, args.t2, args.flair, loaded_model
            )
        elif args.mode == "inference":
            print(" --t1 / --t1ce / --t2 / --flair required for inference mode.")

    print("\n All figures saved to current directory.")
    print("  Files produced:")
    for f in sorted(glob.glob("fig_*.png")):
        sz = os.path.getsize(f) / 1024
        print(f"    {f:45s}  {sz:6.1f} KB")


if __name__ == "__main__":
    main()