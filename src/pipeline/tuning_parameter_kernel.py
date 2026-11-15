
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from src.core.black_tophat import multi_directional_black_tophat


# ==================================================
# KONFIGURASI
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"
SPLIT_FILE = ROOT / "results" / "pembagian_dataset.csv"

OUTPUT_DIR = ROOT / "results" / "tuning_parameter_kernel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_LENGTHS = [9, 11, 13, 15, 17, 19, 21]

SINGLE_THRESHOLDS = [10, 20, 30, 40, 50]

HYSTERESIS_PAIRS = [
    (10, 20),
    (10, 30),
    (10, 40),
    (15, 20),
    (15, 30),
    (15, 40)
]


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
# EVALUASI
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

    return precision, recall, dice, iou


# ==================================================
# MEMBACA DATA TUNING
# ==================================================

def load_tuning_data():

    split_df = pd.read_csv(SPLIT_FILE)

    tuning_df = split_df[
        split_df["split"].str.upper() == "TUNING"
    ].copy()

    if len(tuning_df) != 350:
        raise ValueError(
            f"Jumlah citra tuning tidak sesuai: {len(tuning_df)}"
        )

    return tuning_df


# ==================================================
# MENYIMPAN HASIL METRIK
# ==================================================

def add_result(
    results,
    filename,
    method,
    kernel_length,
    metrics,
    single_threshold=None,
    low_threshold=None,
    high_threshold=None
):

    precision, recall, dice, iou = metrics

    results.append({
        "filename": filename,
        "method": method,
        "kernel_length": kernel_length,
        "single_threshold": single_threshold,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou
    })


# ==================================================
# EKSPERIMEN TUNING
# ==================================================

def main():

    tuning_df = load_tuning_data()

    print("=" * 70)
    print("TUNING PARAMETER KERNEL DAN THRESHOLD")
    print("=" * 70)

    print(f"Jumlah citra tuning : {len(tuning_df)}")
    print(f"Variasi kernel      : {KERNEL_LENGTHS}")
    print(f"Single threshold    : {SINGLE_THRESHOLDS}")
    print(f"Hysteresis pairs    : {HYSTERESIS_PAIRS}")

    all_results = []

    for length in KERNEL_LENGTHS:

        print(f"\nMenguji kernel length = {length}")

        for _, row in tuning_df.iterrows():

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

            # Hitung BTH satu kali untuk setiap kernel dan citra
            response = multi_directional_black_tophat(
                gray,
                length=length
            )

            # ------------------------------------------
            # BASELINE: SINGLE THRESHOLD
            # ------------------------------------------

            for threshold in SINGLE_THRESHOLDS:

                prediction = (
                    response >= threshold
                ).astype(np.uint8)

                metrics = calculate_metrics(
                    prediction,
                    gt
                )

                add_result(
                    all_results,
                    filename,
                    "Baseline",
                    length,
                    metrics,
                    single_threshold=threshold
                )

            # ------------------------------------------
            # HYSTERESIS THRESHOLDING
            # ------------------------------------------

            for low, high in HYSTERESIS_PAIRS:

                prediction = hysteresis_threshold(
                    response,
                    low=low,
                    high=high
                )

                metrics = calculate_metrics(
                    prediction,
                    gt
                )

                add_result(
                    all_results,
                    filename,
                    "Hysteresis",
                    length,
                    metrics,
                    low_threshold=low,
                    high_threshold=high
                )

        print(f"Selesai kernel {length}")

    # ==================================================
    # HASIL DETAIL
    # ==================================================

    detail_df = pd.DataFrame(all_results)

    detail_path = OUTPUT_DIR / "hasil_detail.csv"
    detail_df.to_csv(detail_path, index=False)

    # ==================================================
    # RINGKASAN SETIAP KONFIGURASI
    # ==================================================

    group_columns = [
        "method",
        "kernel_length",
        "single_threshold",
        "low_threshold",
        "high_threshold"
    ]

    summary_df = (
        detail_df
        .groupby(group_columns, dropna=False)
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            dice=("dice", "mean"),
            iou=("iou", "mean")
        )
        .reset_index()
    )

    summary_path = OUTPUT_DIR / "hasil_ringkasan.csv"
    summary_df.to_csv(summary_path, index=False)

    # ==================================================
    # KONFIGURASI TERBAIK SETIAP METODE
    # ==================================================

    best_results = []

    for method in ["Baseline", "Hysteresis"]:

        subset = summary_df[
            summary_df["method"] == method
        ]

        best_idx = subset["dice"].idxmax()
        best = subset.loc[best_idx]

        best_results.append(best)

    best_df = pd.DataFrame(best_results)

    best_path = OUTPUT_DIR / "parameter_terbaik.csv"
    best_df.to_csv(best_path, index=False)

    # ==================================================
    # TAMPILKAN HASIL
    # ==================================================

    print("\n" + "=" * 85)
    print("KONFIGURASI TERBAIK BERDASARKAN MEAN DICE")
    print("=" * 85)

    print(
        best_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nFile hasil tersimpan di:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
