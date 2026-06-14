import os
import cv2
import json
import pandas as pd
import numpy as np

# 🔥 gambar hasil augment
image_folder = "../dataset_90_aug"

# 🔥 JSON & CSV tetap dari dataset asli
data_folder = "../dataset"

output_folder = "../dataset_slot"

config = {
    "json": "all_slots_points-2.json",
    "csv": "labels_90.csv"
}

def crop_polygon_bbox(image, points):
    pts = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    return image[y:y+h, x:x+w]

os.makedirs(output_folder, exist_ok=True)

total_saved = 0
total_failed = 0

print("\n=== Proses angle 90 (AUGMENTED) ===")
print("Folder gambar:", os.path.abspath(image_folder))

if not os.path.exists(image_folder):
    print("Folder tidak ditemukan:", image_folder)
    exit()

json_path = os.path.join(data_folder, config["json"])
csv_path = os.path.join(data_folder, config["csv"])

if not os.path.exists(json_path):
    print("JSON tidak ditemukan:", json_path)
    exit()

if not os.path.exists(csv_path):
    print("CSV tidak ditemukan:", csv_path)
    exit()

with open(json_path, "r") as f:
    coords = json.load(f)

df = pd.read_csv(csv_path)
slot_columns = [col for col in df.columns if col.startswith("slot")]

# buat folder label
all_labels = set()
for col in slot_columns:
    all_labels.update(df[col].dropna().astype(str).unique())

for label in all_labels:
    os.makedirs(os.path.join(output_folder, label), exist_ok=True)

# 🔥 loop CSV (tetap)
for _, row in df.iterrows():
    filename = str(row["filename"]).strip()

    # 🔥 handle file AUGMENT (karena nama berubah!)
    base_name = filename.rsplit(".", 1)[0]

    # cari semua file yg cocok (termasuk _aug)
    matched_files = [
        f for f in os.listdir(image_folder)
        if f.startswith(base_name)
    ]

    if not matched_files:
        print(f"[TIDAK ADA FILE MATCH] {base_name}")
        total_failed += 1
        continue

    for file_match in matched_files:
        image_path = os.path.join(image_folder, file_match)

        image = cv2.imread(image_path)
        if image is None:
            print(f"[GAGAL BACA] {image_path}")
            total_failed += 1
            continue

        for slot_name in slot_columns:
            if slot_name not in coords:
                continue

            points = coords[slot_name]
            label = str(row[slot_name]).strip()

            if label == "" or label.lower() == "nan":
                continue

            cropped = crop_polygon_bbox(image, points)

            if cropped is None or cropped.size == 0:
                continue

            name_no_ext = file_match.rsplit(".", 1)[0]
            save_name = f"{name_no_ext}_{slot_name}.jpg"
            save_path = os.path.join(output_folder, label, save_name)

            ok = cv2.imwrite(save_path, cropped)
            if ok:
                total_saved += 1

    print(f"Selesai: {filename}")

print("\n=== RINGKASAN ===")
print("Total tersimpan:", total_saved)
print("Total gagal:", total_failed)