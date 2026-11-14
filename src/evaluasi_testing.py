
import cv2
import numpy as np
import pandas as pd
import time
from pathlib import Path


# ==========================================
# 1. KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

SPLIT_CSV = ROOT / "results" / "pembagian_dataset.csv"

OUTPUT_DIR = ROOT / "results" / "evaluasi_testing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_LENGTH = 15
ANGLES = [0, 30, 60, 90, 120, 150]

BASELINE_THRESHOLD = 20

HYSTERESIS_LOW = 15
HYSTERESIS_HIGH = 40


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

    weak = response >= low
    strong = response >= high

    num_labels, labels = cv2.connectedComponents(
        weak.astype(np.uint8),
        connectivity=8
    )

    strong_labels = np.unique(labels[strong])
    strong_labels = strong_labels[strong_labels != 0]

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
# 5. EVALUASI DATA PENGUJIAN
# ==========================================

def main():

    df_split = pd.read_csv(SPLIT_CSV)

    df_testing = df_split[
        df_split["split"] == "PENGUJIAN"
    ].copy()

    print("=== INFORMASI DATA PENGUJIAN ===")
    print(f"Jumlah citra: {len(df_testing)}")

    if len(df_testing) != 150:
        raise ValueError(
            "Jumlah data pengujian harus 150 citra."
        )

    results = []

    for index, row in df_testing.iterrows():

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

        ground_truth = mask > 127

        # Black Top-Hat dihitung satu kali
        start_time = time.perf_counter()

        response = multi_directional_black_tophat(gray)

        preprocessing_time = time.perf_counter() - start_time

        # --------------------------------------
        # METODE A: BASELINE
        # --------------------------------------

        start_time = time.perf_counter()

        prediction_baseline = (
            response >= BASELINE_THRESHOLD
        )

        baseline_time = time.perf_counter() - start_time

        baseline_metrics = calculate_metrics(
            prediction_baseline,
            ground_truth
        )

        # --------------------------------------
        # METODE B: HYSTERESIS
        # --------------------------------------

        start_time = time.perf_counter()

        prediction_hysteresis = hysteresis_threshold(
            response,
            HYSTERESIS_LOW,
            HYSTERESIS_HIGH
        )

        hysteresis_time = time.perf_counter() - start_time

        hysteresis_metrics = calculate_metrics(
            prediction_hysteresis,
            ground_truth
        )

        # --------------------------------------
        # MENYIMPAN HASIL
        # --------------------------------------

        results.append({
            "image": Path(filename).stem,

            "baseline_precision": baseline_metrics[0],
            "baseline_recall": baseline_metrics[1],
            "baseline_dice": baseline_metrics[2],
            "baseline_iou": baseline_metrics[3],

            "hysteresis_precision": hysteresis_metrics[0],
            "hysteresis_recall": hysteresis_metrics[1],
            "hysteresis_dice": hysteresis_metrics[2],
            "hysteresis_iou": hysteresis_metrics[3],

            "bth_time_seconds": preprocessing_time,
            "baseline_time_seconds": baseline_time,
            "hysteresis_time_seconds": hysteresis_time
        })

        processed = len(results)

        print(
            f"[{processed}/{len(df_testing)}] "
            f"Selesai: {Path(filename).stem}"
        )

    # ==========================================
    # 6. MENYIMPAN HASIL DETAIL
    # ==========================================

    df_results = pd.DataFrame(results)

    detail_path = OUTPUT_DIR / "hasil_detail.csv"

    df_results.to_csv(
        detail_path,
        index=False
    )

    # ==========================================
    # 7. RINGKASAN METRIK
    # ==========================================

    summary = pd.DataFrame([
        {
            "metode": "Baseline",
            "precision": df_results["baseline_precision"].mean(),
            "recall": df_results["baseline_recall"].mean(),
            "dice": df_results["baseline_dice"].mean(),
            "iou": df_results["baseline_iou"].mean(),
            "waktu_threshold_seconds": df_results["baseline_time_seconds"].mean()
        },
        {
            "metode": "Hysteresis",
            "precision": df_results["hysteresis_precision"].mean(),
            "recall": df_results["hysteresis_recall"].mean(),
            "dice": df_results["hysteresis_dice"].mean(),
            "iou": df_results["hysteresis_iou"].mean(),
            "waktu_threshold_seconds": df_results["hysteresis_time_seconds"].mean()
        }
    ])

    summary_path = OUTPUT_DIR / "hasil_ringkasan.csv"

    summary.to_csv(
        summary_path,
        index=False
    )

    # ==========================================
    # 8. SELISIH HASIL
    # ==========================================

    metrics = [
        "precision",
        "recall",
        "dice",
        "iou"
    ]

    comparison = []

    for metric in metrics:

        baseline_value = summary.loc[
            summary["metode"] == "Baseline",
            metric
        ].iloc[0]

        hysteresis_value = summary.loc[
            summary["metode"] == "Hysteresis",
            metric
        ].iloc[0]

        comparison.append({
            "metric": metric,
            "baseline": baseline_value,
            "hysteresis": hysteresis_value,
            "difference": hysteresis_value - baseline_value
        })

    df_comparison = pd.DataFrame(comparison)

    comparison_path = OUTPUT_DIR / "hasil_perbandingan.csv"

    df_comparison.to_csv(
        comparison_path,
        index=False
    )

    # ==========================================
    # 9. MENAMPILKAN HASIL
    # ==========================================

    print("\n=== RATA-RATA HASIL PENGUJIAN ===")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n=== PERBANDINGAN METRIK ===")

    print(
        df_comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n=== RATA-RATA WAKTU ===")

    print(
        f"Black Top-Hat : "
        f"{df_results['bth_time_seconds'].mean():.6f} detik"
    )

    print(
        f"Baseline      : "
        f"{df_results['baseline_time_seconds'].mean():.6f} detik"
    )

    print(
        f"Hysteresis    : "
        f"{df_results['hysteresis_time_seconds'].mean():.6f} detik"
    )

    print("\n=== FILE HASIL ===")
    print(f"Detail      : {detail_path}")
    print(f"Ringkasan   : {summary_path}")
    print(f"Perbandingan: {comparison_path}")


if __name__ == "__main__":
    main()
