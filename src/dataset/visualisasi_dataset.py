
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import random

# Lokasi proyek
ROOT_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT_DIR / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT_DIR / "dataset" / "hair_mask"

# Pilih 5 citra secara acak
image_paths = sorted(IMAGE_DIR.glob("*.png"))

random.seed(42)
selected_images = random.sample(image_paths, 5)

# Buat visualisasi
fig, axes = plt.subplots(5, 3, figsize=(12, 18))

for i, image_path in enumerate(selected_images):

    mask_path = MASK_DIR / image_path.name

    # Baca citra dan mask
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    # Buat overlay merah pada area rambut
    overlay = image.copy()

    red = np.zeros_like(image)
    red[:, :, 0] = 255

    hair_region = mask > 0

    overlay[hair_region] = (
        0.5 * image[hair_region] +
        0.5 * red[hair_region]
    ).astype(np.uint8)

    # Hitung persentase piksel rambut
    hair_percentage = np.mean(hair_region) * 100

    # Tampilkan hasil
    axes[i, 0].imshow(image)
    axes[i, 0].set_title(f"Original: {image_path.name}")

    axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[i, 1].set_title("Ground Truth Mask")

    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title(
        f"Overlay (Hair: {hair_percentage:.2f}%)"
    )

    for j in range(3):
        axes[i, j].axis("off")

plt.tight_layout()

# Simpan hasil visualisasi
OUTPUT_DIR = ROOT_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUTPUT_DIR / "visualisasi_dataset.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("Visualisasi berhasil disimpan:", output_path)