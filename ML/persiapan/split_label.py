import os
import pandas as pd

# =========================
# PATH
# =========================
csv_all_path = "../dataset/label_all.csv"   # CSV gabungan kamu
train_folder = "../dataset"            # folder utama gambar

# folder angle yang dipakai
angle_folders = ["0", "90", "180", "270"]

# =========================
# BACA CSV GABUNGAN
# =========================
df = pd.read_csv(csv_all_path)

# pastikan ada kolom filename
if "filename" not in df.columns:
    print("ERROR: kolom 'filename' tidak ditemukan di CSV.")
    exit()

# rapikan nama file di CSV
df["filename"] = df["filename"].astype(str).str.strip()

# =========================
# PROSES PER FOLDER ANGLE
# =========================
for angle in angle_folders:
    folder_path = os.path.join(train_folder, angle)

    if not os.path.exists(folder_path):
        print(f"Folder tidak ditemukan: {folder_path}")
        continue

    # ambil semua nama file di folder angle
    files_in_folder = []
    for fname in os.listdir(folder_path):
        full_path = os.path.join(folder_path, fname)
        if os.path.isfile(full_path):
            files_in_folder.append(fname.strip())

    # filter CSV berdasarkan nama file yang ada di folder
    df_angle = df[df["filename"].isin(files_in_folder)].copy()

    # simpan hasil
    output_csv = f"labels_{angle}.csv"
    df_angle.to_csv(output_csv, index=False)

    print(f"{output_csv} berhasil dibuat. Jumlah baris: {len(df_angle)}")

print("Selesai.")