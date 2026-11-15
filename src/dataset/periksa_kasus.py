
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT_DIR / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT_DIR / "dataset" / "hair_mask"
OUTPUT_DIR = ROOT_DIR / "results"

df = pd.read_csv(OUTPUT_DIR / "analisis_dataset.csv")

# Ambil kasus yang perlu diperiksa
cases = df[
    df["status"].isin([
        "MASK_KOSONG",
        "AREA_RAMBUT_SANGAT_BESAR"
    ])
].copy()

# Urutkan berdasarkan persentase rambut
cases = cases.sort_values("hair_percentage")

# Tampilkan 38 kasus dalam beberapa halaman
page_size = 10

for page_number, start in enumerate(
    range(0, len(cases), page_size), start=1
):

    subset = cases.iloc[start:start + page_size]

    fig, axes = plt.subplots(
        len(subset), 3,
        figsize=(12, 3 * len(subset)),
        squeeze=False
    )

    for row, (_, item) in enumerate(subset.iterrows()):

        filename = item["filename"]

        image = cv2.imread(
            str(IMAGE_DIR / filename)
        )
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        mask = cv2.imread(
            str(MASK_DIR / filename),
            cv2.IMREAD_GRAYSCALE
        )

        overlay = image.copy()
        overlay[mask == 255] = (
            0.5 * image[mask == 255].astype(float) +
            0.5 * np.array([255, 0, 0])
        ).astype("uint8")

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(filename)

        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title(
            f"Mask: {item['hair_percentage']:.2f}%"
        )

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")

        for col in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / (
        f"kasus_pemeriksaan_{page_number}.png"
    )

    plt.savefig(output_path, dpi=120)
    plt.close()

    print("Disimpan:", output_path)
