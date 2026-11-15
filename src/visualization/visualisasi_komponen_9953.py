
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"
OUTPUT_DIR = ROOT / "results" / "06_error_analysis" / "analisis_kasus" / "komponen_9953"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_ID = "ISIC_0009953"
LOW = 15
HIGH = 40

def find_image(folder):
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        path = folder / (IMAGE_ID + ext)
        if path.exists():
            return path
    raise FileNotFoundError(IMAGE_ID)

def create_line_kernel(length, angle):
    kernel = np.zeros((length, length), dtype=np.uint8)
    center = (length // 2, length // 2)

    radians = np.deg2rad(angle)
    dx = int((length // 2) * np.cos(radians))
    dy = int((length // 2) * np.sin(radians))

    cv2.line(
        kernel,
        (center[0] - dx, center[1] - dy),
        (center[0] + dx, center[1] + dy),
        1,
        thickness=1
    )
    return kernel

def get_response(gray):
    responses = []

    for angle in [0, 30, 60, 90, 120, 150]:
        kernel = create_line_kernel(15, angle)
        closing = cv2.morphologyEx(
            gray, cv2.MORPH_CLOSE, kernel
        )
        responses.append(cv2.subtract(closing, gray))

    return np.maximum.reduce(responses)

def main():
    image = cv2.imread(str(find_image(IMAGE_DIR)))
    mask = cv2.imread(
        str(find_image(MASK_DIR)),
        cv2.IMREAD_GRAYSCALE
    )

    original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gt = mask > 127

    response = get_response(gray)
    weak = response >= LOW
    strong = response >= HIGH

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        weak.astype(np.uint8),
        connectivity=8
    )

    # Komponen terbesar yang dipertahankan
    retained_labels = np.unique(labels[strong])
    retained_labels = retained_labels[retained_labels != 0]

    largest_label = max(
        retained_labels,
        key=lambda label: stats[label, cv2.CC_STAT_AREA]
    )

    largest = labels == largest_label
    largest_tp = largest & gt
    largest_fp = largest & ~gt

    # Overlay komponen terbesar:
    # Hijau = beririsan dengan ground truth
    # Merah = di luar ground truth
    overlay = original.copy().astype(np.float32)
    overlay[largest_tp] = (
        0.45 * overlay[largest_tp] +
        0.55 * np.array([0, 255, 0])
    )
    overlay[largest_fp] = (
        0.45 * overlay[largest_fp] +
        0.55 * np.array([255, 0, 0])
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Semua komponen yang dipertahankan
    all_retained = np.isin(labels, retained_labels)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    panels = [
        (original, "Citra Asli"),
        (gt, "Ground Truth"),
        (response, "Respons Black Top-Hat"),
        (largest, f"Komponen Terbesar (Label {largest_label})"),
        (overlay, "Overlay: Hijau TP, Merah FP"),
        (all_retained, "Semua Komponen Dipertahankan")
    ]

    for ax, (data, title) in zip(axes.flat, panels):
        if title == "Ground Truth" or data.ndim == 2:
            ax.imshow(data, cmap="gray")
        else:
            ax.imshow(data)

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "visualisasi_komponen_terbesar.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Label komponen terbesar: {largest_label}")
    print(f"Luas komponen: {largest.sum()}")
    print(f"TP komponen: {largest_tp.sum()}")
    print(f"FP komponen: {largest_fp.sum()}")
    print(f"Visualisasi: {output_path}")

if __name__ == "__main__":
    main()
