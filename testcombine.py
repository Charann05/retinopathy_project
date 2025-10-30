import os
import pandas as pd
from pathlib import Path
from shutil import copy2

old_img_dir = Path(r"C:\VS Code\retinopathy_project\data\images")
old_csv = Path(r"C:\VS Code\retinopathy_project\data\labels.csv")

new_img_dir = Path(r"C:\VS Code\retinopathy_project\data\images1")
new_csv = Path(r"C:\VS Code\retinopathy_project\data\train.csv")

output_img_dir = Path("data/combined/images")
output_img_dir.mkdir(parents=True, exist_ok=True)

combined_csv = Path("data/combined/labels_combined.csv")

def copy_and_prefix_images(img_dir, csv_file, prefix):
    df = pd.read_csv(csv_file, names=['image', 'label'])
    renamed_rows = []

    for _, row in df.iterrows():
        old_name = row['image']
        new_name = f"{prefix}_{old_name}"
        src = img_dir / f"{old_name}.png"
        dst = output_img_dir / new_name

        if src.exists():
            copy2(src, dst)
            renamed_rows.append({'image': new_name, 'label': row['label']})
        else:
            print(f"Missing: {src}")

    return pd.DataFrame(renamed_rows)

df1 = copy_and_prefix_images(old_img_dir, old_csv, "gu")
df2 = copy_and_prefix_images(new_img_dir, new_csv, "co")

combined = pd.concat([df1, df2], ignore_index=True).sample(frac=1, random_state=42)
combined.to_csv(combined_csv, index=False)

print(f"Combined dataset saved: {combined_csv}")
print(f"Total images: {len(combined)}")
