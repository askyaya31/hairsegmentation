
import cv2
import numpy as np
import pandas as pd
import time

from pathlib import Path
from black_tophat import multi_directional_black_tophat


# ============================================================
# KONFIGURASI DIREKTORI
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

SPLIT_FILE = ROOT / "results" / "pembagian_dataset.csv"

OUTPUT_DIR = ROOT / "results" / "evaluasi_kernel_terkontrol"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PARAMETER EMPAT KONFIGURASI
# ============================================================

CONFIGURATIONS = [
    {
        "configuration": "A",
        "method": "Baseline",
        "kernel": 9,
        "single_threshold": 20,
        "low_threshold": None,
        "high_threshold": None
    },
    {
        "configuration": "B",
        "method": "Hysteresis",
        "kernel": 9,
        "single_threshold": None,
        "low_threshold": 10,
        "high_threshold": 30
    },
    {
        "configuration": "C",
        "method": "Baseline",
        "kernel": 11,
        "single_threshold": 20,
        "low_threshold": None,
        "high_threshold": None
    },
    {
        "configuration": "D",
        "method": "Hysteresis",
        "kernel": 11,
        "single_threshold": None,
        "low_threshold": 15,
        "high_threshold": 40
    }
]


# ============================================================
# FUNGSI HYSTERESIS THRESHOLDING
# ============================================================

def hysteresis_threshold(response, low, high):

    weak = (response >= low).astype(np.uint8)
    strong = (response >= high).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(
        weak,
        connectivity=8
    )

    strong_labels = np.unique(labels[strong == 1])

    valid_labels = strong_labels[strong_labels != 0]

    result = np.isin(
        labels,
        valid_labels
    ).astype(np.uint8)

    return result


# ============================================================
# FUNGSI EVALUASI METRIK
# ============================================================

def calculate_metrics(prediction, ground_truth):

    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else np.nan
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else np.nan
    )

    dice = (
        1.0
        if 2 * tp + fp + fn == 0
        else (2 * tp) / (2 * tp + fp + fn)
    )

    iou = (
        1.0
        if tp + fp + fn == 0
        else tp / (tp + fp + fn)
    )

    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn)
    }


# ============================================================
# MEMBACA DATA PENGUJIAN
# ============================================================

def load_testing_data():

    split_df = pd.read_csv(SPLIT_FILE)

    testing_df = split_df[
        split_df["split"].str.upper() == "PENGUJIAN"
    ].copy()

    if len(testing_df) != 150:
        raise ValueError(
            f"Jumlah data pengujian tidak sesuai: {len(testing_df)}"
        )

    return testing_df


# ============================================================
# EVALUASI EMPAT KONFIGURASI
# ============================================================

def main():

    testing_df = load_testing_data()

    results = []

    print("=" * 65)
    print("EVALUASI EMPAT KONFIGURASI KERNEL TERKONTROL")
    print("=" * 65)

    print(f"Jumlah citra pengujian : {len(testing_df)}")
    print("Konfigurasi            : A, B, C, D")
    print()

    for index, row in testing_df.iterrows():

        filename = row["filename"]
        category = row["category"]

        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / filename

        image = cv2.imread(str(image_path))
        ground_truth = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"Citra tidak ditemukan: {image_path}"
            )

        if ground_truth is None:
            raise FileNotFoundError(
                f"Mask tidak ditemukan: {mask_path}"
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        ground_truth = (
            ground_truth > 0
        ).astype(np.uint8)

        # Hitung BTH untuk setiap kernel satu kali
        responses = {}
        bth_times = {}

        for kernel in [9, 11]:

            start = time.perf_counter()

            response = multi_directional_black_tophat(
                gray,
                length=kernel
            )

            elapsed = time.perf_counter() - start

            responses[kernel] = response
            bth_times[kernel] = elapsed

        # Evaluasi empat konfigurasi
        for config in CONFIGURATIONS:

            kernel = config["kernel"]
            response = responses[kernel]

            start = time.perf_counter()

            if config["method"] == "Baseline":

                prediction = (
                    response >= config["single_threshold"]
                ).astype(np.uint8)

            else:

                prediction = hysteresis_threshold(
                    response,
                    config["low_threshold"],
                    config["high_threshold"]
                )

            threshold_time = time.perf_counter() - start

            metrics = calculate_metrics(
                prediction,
                ground_truth
            )

            bth_time = bth_times[kernel]

            results.append({
                "filename": filename,
                "category": category,
                "configuration": config["configuration"],
                "method": config["method"],
                "kernel_length": kernel,
                "single_threshold": config["single_threshold"],
                "low_threshold": config["low_threshold"],
                "high_threshold": config["high_threshold"],
                **metrics,
                "bth_time": bth_time,
                "threshold_time": threshold_time,
                "total_time": bth_time + threshold_time
            })

        print(
            f"[{len(results) // 4:3d}/{len(testing_df)}] "
            f"{filename} selesai"
        )

    # ========================================================
    # SIMPAN HASIL DETAIL
    # ========================================================

    detail_df = pd.DataFrame(results)

    detail_path = OUTPUT_DIR / "hasil_detail.csv"

    detail_df.to_csv(
        detail_path,
        index=False
    )

    # ========================================================
    # RINGKASAN METRIK
    # ========================================================

    summary_df = (
        detail_df
        .groupby(
            [
                "configuration",
                "method",
                "kernel_length",
                "single_threshold",
                "low_threshold",
                "high_threshold"
            ],
            dropna=False
        )
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            iou_mean=("iou", "mean"),
            iou_std=("iou", "std"),
            bth_time_mean=("bth_time", "mean"),
            threshold_time_mean=("threshold_time", "mean"),
            total_time_mean=("total_time", "mean")
        )
        .reset_index()
    )

    summary_path = OUTPUT_DIR / "hasil_ringkasan.csv"

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # TAMPILKAN HASIL
    # ========================================================

    print("\n" + "=" * 65)
    print("RINGKASAN HASIL EVALUASI")
    print("=" * 65)

    columns = [
        "configuration",
        "method",
        "kernel_length",
        "precision_mean",
        "recall_mean",
        "dice_mean",
        "iou_mean",
        "total_time_mean"
    ]

    print(
        summary_df[columns].to_string(index=False)
    )

    print("\nHasil detail   :", detail_path)
    print("Hasil ringkasan:", summary_path)

    print("\nEvaluasi selesai.")


if __name__ == "__main__":
    main()
