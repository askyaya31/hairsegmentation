
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================
# 1. KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

CASE_FILE = (
    ROOT / "results" / "06_error_analysis" / "analisis_kasus"
    / "kasus_representatif_setiap_pola.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "06_error_analysis" / "analisis_kasus"
    / "visualisasi_pola_representatif"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_LENGTH = 15
ANGLES = [0, 30, 60, 90, 120, 150]

BASELINE_THRESHOLD = 20
HYSTERESIS_LOW = 15
HYSTERESIS_HIGH = 40


# ==========================================
# 2. BLACK TOP-HAT
# ==========================================

def create_line_kernel(length, angle):
    kernel = np.zeros((length, length), dtype=np.uint8)

    center = (length // 2, length // 2)
    radians = np.deg2rad(angle)

    dx = int((length // 2) * np.cos(radians))
    dy = int((length // 2) * np.sin(radians))

    pt1 = (center[0] - dx, center[1] - dy)
    pt2 = (center[0] + dx, center[1] + dy)

    cv2.line(kernel, pt1, pt2, 1, thickness=1)

    return kernel


def multi_directional_black_tophat(gray):
    responses = []

    for angle in ANGLES:
        kernel = create_line_kernel(KERNEL_LENGTH, angle)

        closing = cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            kernel
        )

        response = cv2.subtract(closing, gray)
        responses.append(response)

    return np.maximum.reduce(responses)


# ==========================================
# 3. HYSTERESIS THRESHOLDING
# ==========================================

def hysteresis_threshold(response, low, high):
    weak = response >= low
    strong = response >= high

    num_labels, labels = cv2.connectedComponents(
        weak.astype(np.uint8),
        connectivity=8
    )

    strong_labels = np.unique(labels[strong])
    strong_labels = strong_labels[strong_labels != 0]

    return np.isin(labels, strong_labels)


# ==========================================
# 4. ERROR MAP
# ==========================================

def error_map(prediction, ground_truth):
    result = np.zeros(
        (*ground_truth.shape, 3),
        dtype=np.uint8
    )

    tp = prediction & ground_truth
    fp = prediction & ~ground_truth
    fn = ~prediction & ground_truth

    # RGB
    result[tp] = [255, 255, 255]
    result[fp] = [255, 0, 0]
    result[fn] = [0, 100, 255]

    return result


# ==========================================
# 5. VISUALISASI KASUS
# ==========================================

def find_image(directory, image_name):
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        path = directory / f"{image_name}{ext}"

        if path.exists():
            return path

    raise FileNotFoundError(
        f"Citra {image_name} tidak ditemukan di {directory}"
    )


def create_case_visualization(row):

    image_name = row["image"]

    image_path = find_image(IMAGE_DIR, image_name)
    mask_path = find_image(MASK_DIR, image_name)

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError(f"Gagal membaca file: {image_name}")

    if image.shape[:2] != mask.shape:
        raise ValueError(f"Ukuran citra dan mask berbeda: {image_name}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ground_truth = mask > 127

    # Respons BTH
    response = multi_directional_black_tophat(gray)

    # Baseline
    prediction_baseline = (
        response >= BASELINE_THRESHOLD
    )

    # Hysteresis
    prediction_hysteresis = hysteresis_threshold(
        response,
        HYSTERESIS_LOW,
        HYSTERESIS_HIGH
    )

    # Error map kedua metode
    baseline_error = error_map(
        prediction_baseline,
        ground_truth
    )

    hysteresis_error = error_map(
        prediction_hysteresis,
        ground_truth
    )

    # Panel visualisasi
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    panels = [
        (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "Citra Asli",
            None
        ),
        (
            ground_truth,
            "Ground Truth",
            "gray"
        ),
        (
            response,
            "Respons Black Top-Hat",
            "gray"
        ),
        (
            prediction_baseline,
            f"Baseline (Threshold {BASELINE_THRESHOLD})",
            "gray"
        ),
        (
            prediction_hysteresis,
            f"Hysteresis (Low {HYSTERESIS_LOW}, High {HYSTERESIS_HIGH})",
            "gray"
        ),
        (
            hysteresis_error,
            "Kesalahan Hysteresis",
            None
        )
    ]

    for ax, (data, title, cmap) in zip(axes.flat, panels):

        if cmap == "gray":
            ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(data)

        ax.set_title(title, fontsize=11)
        ax.axis("off")

    kelompok = row["pola"]

    dice_baseline = row["baseline_dice"]
    dice_hysteresis = row["hysteresis_dice"]
    difference = row["delta_dice"]

    fig.suptitle(
        f"{image_name} | {kelompok}\n"
        f"Baseline Dice: {dice_baseline:.4f} | "
        f"Hysteresis Dice: {dice_hysteresis:.4f} | "
        f"Selisih: {difference:+.4f}",
        fontsize=13
    )

    fig.text(
        0.5,
        0.025,
        "Putih: TP | Merah: FP | Biru: FN",
        ha="center",
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

    output_path = OUTPUT_DIR / f"{image_name}_representatif.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Tersimpan: {output_path}")


# ==========================================
# 6. MAIN
# ==========================================

def main():

    df = pd.read_csv(CASE_FILE)

    print(f"Jumlah kasus: {len(df)}")

    for _, row in df.iterrows():
        create_case_visualization(row)

    print("\nSemua visualisasi selesai.")
    print(f"Folder hasil: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
