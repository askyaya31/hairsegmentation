
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# ==========================================
# 1. KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

SPLIT_CSV = ROOT / "results" / "pembagian_dataset.csv"

OUTPUT_DIR = ROOT / "results" / "03_evaluation" / "hysteresis_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_LENGTH = 15
ANGLES = [0, 30, 60, 90, 120, 150]

THRESHOLD_PAIRS = [
    (10, 20),
    (10, 30),
    (10, 40),
    (15, 20),
    (15, 30),
    (15, 40)
]


# ==========================================
# 2. MULTI-DIRECTIONAL BLACK TOP-HAT
# ==========================================

def create_line_kernel(length, angle):

    kernel = np.zeros(
        (length, length),
        dtype=np.uint8
    )

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

        kernel = create_line_kernel(
            KERNEL_LENGTH,
            angle
        )

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

    # Piksel dengan respons minimal T_low
    weak = response >= low

    # Piksel dengan respons minimal T_high
    strong = response >= high

    # Mencari komponen terhubung pada area weak
    num_labels, labels = cv2.connectedComponents(
        weak.astype(np.uint8),
        connectivity=8
    )

    # Mengambil label yang memiliki piksel strong
    strong_labels = np.unique(labels[strong])

    # Menghilangkan label background (0)
    strong_labels = strong_labels[strong_labels != 0]

    # Mempertahankan komponen yang terhubung
    prediction = np.isin(labels, strong_labels)

    return prediction


# ==========================================
# 4. PERHITUNGAN METRIK
# ==========================================

def calculate_metrics(prediction, ground_truth):

    tp = np.logical_and(prediction, ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()

    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else np.nan
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else np.nan
    )

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn > 0 else 1.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0 else 1.0
    )

    return precision, recall, dice, iou


# ==========================================
# 5. EVALUASI DATA TUNING
# ==========================================

def main():

    df_split = pd.read_csv(SPLIT_CSV)

    df_tuning = df_split[
        df_split["split"] == "TUNING"
    ].copy()

    print("=== INFORMASI DATA TUNING ===")
    print(f"Jumlah citra: {len(df_tuning)}")

    if len(df_tuning) != 350:
        raise ValueError(
            "Jumlah data tuning harus 350 citra."
        )

    results = []

    for index, row in df_tuning.iterrows():

        filename = row["filename"]

        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / filename

        image = cv2.imread(str(image_path))
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None or mask is None:
            raise FileNotFoundError(
                f"Citra atau mask gagal dibaca: {filename}"
            )

        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Ukuran citra dan mask berbeda: {filename}"
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Respons Black Top-Hat dihitung satu kali
        response = multi_directional_black_tophat(gray)

        ground_truth = mask > 127

        for low, high in THRESHOLD_PAIRS:

            prediction = hysteresis_threshold(
                response,
                low,
                high
            )

            precision, recall, dice, iou = calculate_metrics(
                prediction,
                ground_truth
            )

            results.append({
                "image": Path(filename).stem,
                "low": low,
                "high": high,
                "precision": precision,
                "recall": recall,
                "dice": dice,
                "iou": iou
            })

        processed = (len(results) // len(THRESHOLD_PAIRS))

        print(
            f"[{processed}/{len(df_tuning)}] "
            f"Selesai: {Path(filename).stem}"
        )

    # ==========================================
    # 6. MENYIMPAN HASIL
    # ==========================================

    df_results = pd.DataFrame(results)

    detail_path = OUTPUT_DIR / "hasil_detail.csv"

    df_results.to_csv(
        detail_path,
        index=False
    )

    df_summary = (
        df_results
        .groupby(["low", "high"])[
            ["precision", "recall", "dice", "iou"]
        ]
        .mean()
        .reset_index()
    )

    summary_path = OUTPUT_DIR / "hasil_ringkasan.csv"

    df_summary.to_csv(
        summary_path,
        index=False
    )

    # ==========================================
    # 7. MENAMPILKAN HASIL
    # ==========================================

    print("\n=== RATA-RATA HASIL HYSTERESIS ===")

    print(
        df_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    best_row = df_summary.loc[
        df_summary["dice"].idxmax()
    ]

    print("\n=== KOMBINASI DICE TERTINGGI ===")
    print(f"T_low  : {int(best_row['low'])}")
    print(f"T_high : {int(best_row['high'])}")
    print(f"Dice   : {best_row['dice']:.4f}")
    print(f"IoU    : {best_row['iou']:.4f}")

    print("\n=== FILE HASIL ===")
    print(f"Detail    : {detail_path}")
    print(f"Ringkasan : {summary_path}")


if __name__ == "__main__":
    main()
