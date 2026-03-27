from pathlib import Path
import pandas as pd

DATASET = Path("merged_dataset")
META_PATH = DATASET / "processed/metadata.csv"

meta = pd.read_csv(META_PATH)

def seg_exists(seg_path):
    if pd.isna(seg_path) or not isinstance(seg_path, str) or seg_path.strip() == "":
        return False
    p = Path(seg_path)
    return p.is_file()

# boolean column
meta["train_missing_seg"] = (meta["split"] == "train") & (~meta["mask"].apply(seg_exists))

total_train = (meta["split"] == "train").sum()
missing_train_seg = meta["train_missing_seg"].sum()

print("\n===== TRAIN SEGMENTATION INTEGRITY =====")
print(f"Total train rows: {total_train}")
print(f"Train rows missing seg file: {missing_train_seg} ({missing_train_seg/ max(total_train,1)*100:.2f}%)")
