
import cv2
import numpy as np
import pandas as pd
import time

from pathlib import Path
from src.core.black_tophat import multi_directional_black_tophat


# ==================================================
# KONFIGURASI FINAL
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"
SPLIT_FILE = ROOT / "results" / "pembagian_dataset.csv"

OUTPUT_DIR = ROOT / "results" / "03_evaluation" / "final_testing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameter hasil tuning
BASELINE_KERNEL = 11
BASELINE_THRESHOLD = 20

HYSTERESIS_KERNEL = 9
LOW_THRESHOLD = 10
HIGH_THRESHOLD = 30


# ==================================================
# HYSTERESIS THRESHOLDING
# ==================================================

def hysteresis_threshold(response, low, high):

    weak = (response >= low).astype(np.uint8)
    strong = (response >= high).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(
        weak,
        connectivity=8
    )

    strong_labels = np.unique(labels[strong == 1])
    valid_labels = strong_labels[strong_labels != 0]

    result = np.isin(labels, valid_labels).astype(np.uint8)

    return result


# ==================================================
# EVALUASI METRIK
# ==================================================

def calculate_metrics(prediction, ground_truth):

    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    dice_denominator = 2 * tp + fp + fn

    dice = (
        1.0 if dice_denominator == 0
        else (2 * tp) / dice_denominator
    )

    iou_denominator = tp + fp + fn

    iou = (
        1.0 if iou_denominator == 0
        else tp / iou_denominator
    )

    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou
    }


# ==================================================
# MEMBACA DATA TESTING
# ==================================================

def load_testing_data():

    split_df = pd.read_csv(SPLIT_FILE)

    testing_df = split_df[
        split_df["split"].str.upper() == "PENGUJIAN"
    ].copy()

    if len(testing_df) != 150:
        raise ValueError(
            f"Jumlah citra pengujian tidak sesuai: {len(testing_df)}"
        )

    return testing_df


# ==================================================
# EVALUASI FINAL
# ==================================================

def main():

    testing_df = load_testing_data()

    print("=" * 70)
    print("EVALUASI FINAL SEGMENTASI RAMBUT CITRA DERMOSKOPI")
    print("=" * 70)

    print(f"Jumlah citra testing : {len(testing_df)}")
    print(
        f"Baseline             : kernel {BASELINE_KERNEL}, "
        f"threshold {BASELINE_THRESHOLD}"
    )
    print(
        f"Hysteresis           : kernel {HYSTERESIS_KERNEL}, "
        f"low {LOW_THRESHOLD}, high {HIGH_THRESHOLD}"
    )

    all_results = []

    for index, row in testing_df.iterrows():

        filename = row["filename"]

        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / filename

        image = cv2.imread(str(image_path))

        ground_truth = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(image_path)

        if ground_truth is None:
            raise FileNotFoundError(mask_path)

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gt = ground_truth > 0

        # ------------------------------------------
        # BASELINE
        # ------------------------------------------

        start = time.perf_counter()

        baseline_response = multi_directional_black_tophat(
            gray,
            length=BASELINE_KERNEL
        )

        baseline_bth_time = time.perf_counter() - start

        start = time.perf_counter()

        baseline_prediction = (
            baseline_response >= BASELINE_THRESHOLD
        ).astype(np.uint8)

        baseline_threshold_time = time.perf_counter() - start

        baseline_metrics = calculate_metrics(
            baseline_prediction,
            gt
        )

        # ------------------------------------------
        # HYSTERESIS
        # ------------------------------------------

        start = time.perf_counter()

        hysteresis_response = multi_directional_black_tophat(
            gray,
            length=HYSTERESIS_KERNEL
        )

        hysteresis_bth_time = time.perf_counter() - start

        start = time.perf_counter()

        hysteresis_prediction = hysteresis_threshold(
            hysteresis_response,
            low=LOW_THRESHOLD,
            high=HIGH_THRESHOLD
        )

        hysteresis_threshold_time = time.perf_counter() - start

        hysteresis_metrics = calculate_metrics(
            hysteresis_prediction,
            gt
        )

        # ------------------------------------------
        # SIMPAN HASIL PER CITRA
        # ------------------------------------------

        result = {
            "filename": filename,

            "baseline_precision": baseline_metrics["precision"],
            "baseline_recall": baseline_metrics["recall"],
            "baseline_dice": baseline_metrics["dice"],
            "baseline_iou": baseline_metrics["iou"],

            "hysteresis_precision": hysteresis_metrics["precision"],
            "hysteresis_recall": hysteresis_metrics["recall"],
            "hysteresis_dice": hysteresis_metrics["dice"],
            "hysteresis_iou": hysteresis_metrics["iou"],

            "baseline_bth_time": baseline_bth_time,
            "baseline_threshold_time": baseline_threshold_time,

            "hysteresis_bth_time": hysteresis_bth_time,
            "hysteresis_threshold_time": hysteresis_threshold_time
        }

        all_results.append(result)

        if len(all_results) % 25 == 0:
            print(
                f"Progres: {len(all_results)}/{len(testing_df)} citra"
            )

    # ==================================================
    # SIMPAN HASIL DETAIL
    # ==================================================

    detail_df = pd.DataFrame(all_results)

    detail_path = OUTPUT_DIR / "hasil_detail.csv"
    detail_df.to_csv(detail_path, index=False)

    # ==================================================
    # RINGKASAN METRIK
    # ==================================================

    metric_columns = [
        "baseline_precision",
        "baseline_recall",
        "baseline_dice",
        "baseline_iou",
        "hysteresis_precision",
        "hysteresis_recall",
        "hysteresis_dice",
        "hysteresis_iou"
    ]

    summary_rows = []

    for method, columns in {
        "Baseline": metric_columns[:4],
        "Hysteresis": metric_columns[4:]
    }.items():

        for metric in ["precision", "recall", "dice", "iou"]:

            column = f"{method.lower()}_{metric}"

            summary_rows.append({
                "method": method,
                "metric": metric,
                "mean": detail_df[column].mean(),
                "std": detail_df[column].std()
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = OUTPUT_DIR / "hasil_ringkasan.csv"
    summary_df.to_csv(summary_path, index=False)

    # ==================================================
    # ANALISIS PERUBAHAN DICE
    # ==================================================

    dice_difference = (
        detail_df["hysteresis_dice"]
        - detail_df["baseline_dice"]
    )

    print("\n" + "=" * 70)
    print("ANALISIS PERUBAHAN DICE")
    print("=" * 70)

    print(f"Dice meningkat : {(dice_difference > 0).sum()}")
    print(f"Dice menurun   : {(dice_difference < 0).sum()}")
    print(f"Dice tetap     : {(dice_difference == 0).sum()}")

    print(f"Mean perubahan : {dice_difference.mean():.6f}")
    print(f"Median perubahan: {dice_difference.median():.6f}")

    print("\n" + "=" * 70)
    print("RINGKASAN HASIL TESTING")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # ==================================================
    # RINGKASAN WAKTU EKSEKUSI
    # ==================================================

    print("\n" + "=" * 70)
    print("RATA-RATA WAKTU EKSEKUSI")
    print("=" * 70)

    print(
        f"Baseline BTH       : "
        f"{detail_df['baseline_bth_time'].mean():.6f} detik"
    )

    print(
        f"Baseline threshold : "
        f"{detail_df['baseline_threshold_time'].mean():.6f} detik"
    )

    print(
        f"Hysteresis BTH     : "
        f"{detail_df['hysteresis_bth_time'].mean():.6f} detik"
    )

    print(
        f"Hysteresis         : "
        f"{detail_df['hysteresis_threshold_time'].mean():.6f} detik"
    )

    print("\nFile hasil tersimpan di:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
