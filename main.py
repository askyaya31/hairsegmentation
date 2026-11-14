
from pathlib import Path
import cv2
import numpy as np

# Sesuaikan dengan lokasi dataset di komputer
DATASET_DIR = Path("dataset")

IMAGE_DIR = DATASET_DIR / "dermoscopic_image"
MASK_DIR = DATASET_DIR / "hair_mask"

# Periksa folder
if not IMAGE_DIR.exists() or not MASK_DIR.exists():
    raise FileNotFoundError(
        "Folder gambar atau hair mask tidak ditemukan."
    )

images = sorted(IMAGE_DIR.glob("*.png"))
masks = sorted(MASK_DIR.glob("*.png"))

print("Jumlah citra:", len(images))
print("Jumlah mask:", len(masks))

# Cocokkan nama file
image_names = {p.stem for p in images}
mask_names = {p.stem for p in masks}

print("Citra tanpa pasangan mask:",
      len(image_names - mask_names))

print("Mask tanpa pasangan citra:",
      len(mask_names - image_names))

# Periksa beberapa pasangan
for image_path in images[:5]:

    mask_path = MASK_DIR / image_path.name

    if not mask_path.exists():
        continue

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    print("\nNama:", image_path.name)
    print("Ukuran citra:", image.shape)
    print("Ukuran mask:", mask.shape)
    print("Nilai piksel mask:", np.unique(mask))