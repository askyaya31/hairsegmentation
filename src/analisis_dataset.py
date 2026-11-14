
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# Lokasi proyek
ROOT_DIR = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT_DIR / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT_DIR / "dataset" / "hair_mask"
OUTPUT_DIR = ROOT_DIR / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ambang untuk menandai citra yang perlu diperiksa manual.
# Ini hanya penanda awal, bukan standar kualitas mask.
MIN_HAIR_PERCENT = 0.1
MAX_HAIR_PERCENT = 30.0

records = []

image_paths = sorted(IMAGE_DIR.glob("*.png"))

for image_path in image_paths:

    mask_path = MASK_DIR / image_path.name

    if not mask_path.exists():
        records.append({
            "filename": image_path.name,
            "status": "MASK_TIDAK_DITEMUKAN"
        })
        continue

    image = cv2.imread(str(image_path))
    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        records.append({
            "filename": image_path.name,
            "status": "FILE_TIDAK_TERBACA"
        })
        continue

    if image.shape[:2] != mask.shape:
        records.append({
            "filename": image_path.name,
            "status": "UKURAN_TIDAK_SAMA"
        })
        continue

    # Periksa nilai piksel mask
    unique_values = np.unique(mask)

    # Area rambut adalah piksel putih (255)
    hair_pixels = np.count_nonzero(mask == 255)
    total_pixels = mask.size

    hair_percentage = (
        hair_pixels / total_pixels
    ) * 100

    # Tandai citra yang perlu diperiksa manual
    if hair_pixels == 0:
        status = "MASK_KOSONG"
    elif hair_percentage < MIN_HAIR_PERCENT:
        status = "AREA_RAMBUT_SANGAT_SEDIKIT"
    elif hair_percentage > MAX_HAIR_PERCENT:
        status = "AREA_RAMBUT SANGAT_BESAR"
    else:
        status = "NORMAL"

    records.append({
        "filename": image_path.name,
        "width": image.shape[1],
        "height": image.shape[0],
        "hair_pixels": hair_pixels,
        "total_pixels": total_pixels,
        "hair_percentage": round(hair_percentage, 4),
        "unique_mask_values": str(unique_values.tolist()),
        "status": status
    })

# Simpan hasil pemeriksaan
df = pd.DataFrame(records)

output_csv = OUTPUT_DIR / "analisis_dataset.csv"
df.to_csv(output_csv, index=False)

# Ringkasan
print("=" * 50)
print("HASIL ANALISIS DATASET")
print("=" * 50)

print("Total citra:", len(image_paths))
print("Total berhasil diperiksa:",
      (df["status"].isin([
          "NORMAL",
          "MASK_KOSONG",
          "AREA_RAMBUT_SANGAT_SEDIKIT",
          "AREA_RAMBUT SANGAT_BESAR"
      ])).sum())

if "hair_percentage" in df.columns:

    valid = df.dropna(subset=["hair_percentage"])

    print("\nStatistik persentase area rambut:")
    print(valid["hair_percentage"].describe())

    print("\nJumlah berdasarkan status:")
    print(df["status"].value_counts())

    print("\nLima citra dengan area rambut terkecil:")
    print(
        valid.sort_values("hair_percentage")[
            ["filename", "hair_percentage", "status"]
        ].head(5).to_string(index=False)
    )

    print("\nLima citra dengan area rambut terbesar:")
    print(
        valid.sort_values(
            "hair_percentage", ascending=False
        )[
            ["filename", "hair_percentage", "status"]
        ].head(5).to_string(index=False)
    )

print("\nHasil CSV disimpan di:")
print(output_csv)
