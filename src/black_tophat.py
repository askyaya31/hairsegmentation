import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# Lokasi dataset dan hasil
ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
OUTPUT_DIR = ROOT / "results" / "black_tophat"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_line_kernel(length, angle):
    """
    Membuat structuring element berbentuk garis
    dengan panjang dan arah tertentu.
    """
    kernel = np.zeros((length, length), dtype=np.uint8)

    center = (length // 2, length // 2)
    radians = np.deg2rad(angle)

    dx = int((length // 2) * np.cos(radians))
    dy = int((length // 2) * np.sin(radians))

    pt1 = (center[0] - dx, center[1] - dy)
    pt2 = (center[0] + dx, center[1] + dy)

    cv2.line(kernel, pt1, pt2, 1, thickness=1)

    return kernel


def multi_directional_black_tophat(gray, length=15):
    """
    Menghitung Black Top-Hat dari beberapa arah,
    kemudian mengambil respons maksimum.
    """
    responses = []

    angles = [0, 30, 60, 90, 120, 150]

    for angle in angles:
        kernel = create_line_kernel(length, angle)

        closing = cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            kernel
        )

        black_tophat = cv2.subtract(closing, gray)
        responses.append(black_tophat)

    combined = np.maximum.reduce(responses)

    return combined


def main():
    # Ambil lima citra untuk pengujian awal
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]

    image_paths = []

    for ext in extensions:
        image_paths.extend(IMAGE_DIR.glob(ext))

    image_paths = sorted(image_paths)[:5]

    if not image_paths:
        print("Tidak ditemukan citra JPG di folder dataset.")
        return

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Gagal membaca: {image_path.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        response = multi_directional_black_tophat(
            gray,
            length=15
        )

        # Visualisasi
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Citra Asli")

        axes[1].imshow(gray, cmap="gray")
        axes[1].set_title("Grayscale")

        axes[2].imshow(response, cmap="gray")
        axes[2].set_title("Multi-Directional Black Top-Hat")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{image_path.stem}_result.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Hasil disimpan: {output_path}")


if __name__ == "__main__":
    main()
