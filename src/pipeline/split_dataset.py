from pathlib import Path

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

# Lokasi dataset
ROOT_DIR = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT_DIR / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT_DIR / "dataset" / "hair_mask"
OUTPUT_DIR = ROOT_DIR / "results"

RANDOM_SEED = 42
TEST_SIZE = 0.30

# Citra yang sebelumnya digunakan untuk eksplorasi
EXPLORATORY_IDS = {
    "ISIC_0000042",
    "ISIC_0000043",
    "ISIC_0000004",
    "ISIC_0000022",
    "ISIC_0000032"
}

data = []

# Membaca seluruh citra
image_paths = sorted([
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
])

for image_path in image_paths:
    mask_path = MASK_DIR / image_path.name

    if not mask_path.exists():
        print(f"Mask tidak ditemukan: {image_path.name}")
        continue

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Mask gagal dibaca: {image_path.name}")
        continue

    hair_percentage = (mask > 127).sum() / mask.size * 100

    if hair_percentage == 0:
        category = "KOSONG"
    elif hair_percentage < 0.1:
        category = "SANGAT_SEDIKIT"
    elif hair_percentage <= 30:
        category = "NORMAL"
    else:
        category = "SANGAT_BESAR"

    data.append({
        "filename": image_path.name,
        "hair_percentage": hair_percentage,
        "category": category
    })

df = pd.DataFrame(data)

if len(df) != 500:
    raise ValueError(
        f"Jumlah data {len(df)}, bukan 500."
    )

# Memisahkan lima citra eksplorasi
df["image_id"] = df["filename"].apply(
    lambda x: Path(x).stem
)

exploratory = df[
    df["image_id"].isin(EXPLORATORY_IDS)
].copy()

if len(exploratory) != len(EXPLORATORY_IDS):
    raise ValueError("Ada citra eksplorasi yang tidak ditemukan.")

remaining = df[
    ~df["image_id"].isin(EXPLORATORY_IDS)
].copy()

# Target: 350 tuning dan 150 testing.
# Lima citra eksplorasi sudah ditetapkan sebagai tuning.
tuning_remaining, testing = train_test_split(
    remaining,
    test_size=150,
    random_state=RANDOM_SEED,
    stratify=remaining["category"]
)

tuning = pd.concat(
    [exploratory, tuning_remaining],
    ignore_index=True
)

# Menambahkan label kelompok
tuning["split"] = "TUNING"
testing["split"] = "PENGUJIAN"

# Menggabungkan hasil
result = pd.concat(
    [tuning, testing],
    ignore_index=True
)

result = result.drop(columns=["image_id"])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUTPUT_DIR / "pembagian_dataset.csv"
result.to_csv(output_path, index=False)

# Menampilkan hasil
print("\n=== HASIL PEMBAGIAN DATASET ===")
print(f"Total citra    : {len(result)}")
print(f"Data tuning    : {len(tuning)}")
print(f"Data pengujian : {len(testing)}")

print("\n=== DISTRIBUSI KATEGORI ===")
print(pd.crosstab(result["category"], result["split"]))

print("\n=== PEMERIKSAAN CITRA EKSPLORASI ===")
print(
    result[
        result["filename"].apply(
            lambda x: Path(x).stem in EXPLORATORY_IDS
        )
    ][["filename", "split"]].to_string(index=False)
)

print(f"\nFile tersimpan: {output_path}")
