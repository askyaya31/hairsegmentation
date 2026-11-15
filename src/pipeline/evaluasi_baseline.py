
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from pathlib import Path


# ==========================================
# 1. KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

SPLIT_CSV = ROOT / "results" / "pembagian_dataset.csv"

OUTPUT_DIR = ROOT / "results" / "evaluasi_baseline_tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [10, 20, 30, 40, 50]

KERNEL_LENGTH = 15
ANGLES = [0, 30, 60, 90, 120, 150]


# ==========================================
# 2. MULTI-DIRECTIONAL BLACK TOP-HAT
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


def multi_directional_black_tophat(gray, length=15):
    responses = []

    for angle in ANGLES:
        kernel = create_line_kernel(length, angle)

        closing = cv2.morphologyEx(
            gray,
            cv2.MORPH_CLOSE,
            kernel
        )

        response = cv2.subtract(closing, gray)
        responses.append(response)

    combined = np.maximum.reduce(responses)

    return combined


# ==========================================
# 3. PERHITUNGAN METRIK
# ==========================================

def calculate_metrics(prediction, ground_truth):

    tp = np.logical_and(prediction, ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()

    # Precision
    precision = (
        tp / (tp + fp)
        if tp + fp > 0 else np.nan
    )

    # Recall
    recall = (
        tp / (tp + fn)
        if tp + fn > 0 else np.nan
    )

    # Dice
    if 2 * tp + fp + fn == 0:
        dice = 1.0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)

    # IoU
    if tp + fp + fn == 0:
        iou = 1.0
    else:
        iou = tp / (tp + fp + fn)

    return precision, recall, dice, iou


# ==========================================
# 4. EVALUASI DATA TUNING
# ==========================================

def main():

    # Membaca pembagian dataset
    df_split = pd.read_csv(SPLIT_CSV)

    # Mengambil hanya data tuning
    df_tuning = df_split[
        df_split["split"] == "TUNING"
    ].copy()

    print("=== INFORMASI DATA TUNING ===")
    print(f"Jumlah citra: {len(df_tuning)}")

    if len(df_tuning) != 350:
        raise ValueError(
            "Jumlah data tuning tidak sesuai. "
            "Seharusnya terdapat 350 citra."
        )

    results = []

    # Memproses seluruh citra tuning
    for index, row in df_tuning.iterrows():

        filename = row["filename"]
        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / filename

        image = cv2.imread(str(image_path))
        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"Citra gagal dibaca: {image_path}"
            )

        if mask is None:
            raise FileNotFoundError(
                f"Mask gagal dibaca: {mask_path}"
            )

        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Ukuran citra dan mask berbeda: {filename}"
            )

        # Konversi citra ke grayscale
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Menghitung respons Black Top-Hat
        response = multi_directional_black_tophat(
            gray,
            length=KERNEL_LENGTH
        )

        # Ground truth: piksel > 127 adalah rambut
        ground_truth = mask > 127

        # Menguji setiap threshold
        for threshold in THRESHOLDS:

            prediction = response >= threshold

            precision, recall, dice, iou = calculate_metrics(
                prediction,
                ground_truth
            )

            results.append({
                "image": Path(filename).stem,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "dice": dice,
                "iou": iou
            })

        processed = len(results) // len(THRESHOLDS)

        print(
            f"[{processed}/{len(df_tuning)}] "
            f"Selesai: {Path(filename).stem}"
        )

    # ==========================================
    # 5. MENYIMPAN HASIL
    # ==========================================

    df_results = pd.DataFrame(results)

    # Hasil seluruh citra dan threshold
    detail_path = OUTPUT_DIR / "hasil_detail.csv"

    df_results.to_csv(
        detail_path,
        index=False
    )

    # Rata-rata metrik untuk setiap threshold
    df_summary = (
        df_results
        .groupby("threshold")[
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
    # 6. MENAMPILKAN HASIL
    # ==========================================

    print("\n=== RATA-RATA HASIL EVALUASI BASELINE ===")
    print(
        df_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    best_row = df_summary.loc[
        df_summary["dice"].idxmax()
    ]

    print("\n=== THRESHOLD DENGAN RATA-RATA DICE TERTINGGI ===")
    print(f"Threshold : {int(best_row['threshold'])}")
    print(f"Dice      : {best_row['dice']:.4f}")
    print(f"IoU       : {best_row['iou']:.4f}")

    print("\n=== FILE HASIL ===")
    print(f"Detail    : {detail_path}")
    print(f"Ringkasan : {summary_path}")


if __name__ == "__main__":
    main()
