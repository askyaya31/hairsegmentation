
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from src.core.black_tophat import multi_directional_black_tophat


# ==================================================
# KONFIGURASI
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

SPLIT_FILE = ROOT / "results" / "pembagian_dataset.csv"

OUTPUT_DIR = ROOT / "results" / "eksperimen_variasi_kernel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_LENGTHS = [9, 11, 13, 15, 17, 19, 21]

SINGLE_THRESHOLD = 20
LOW_THRESHOLD = 15
HIGH_THRESHOLD = 40


# ==================================================
# HYSTERESIS THRESHOLDING
# ==================================================

def hysteresis_threshold(response, low=15, high=40):

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

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (2 * tp) / dice_denominator

    iou_denominator = tp + fp + fn

    if iou_denominator == 0:
        iou = 1.0
    else:
        iou = tp / iou_denominator

    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou
    }


# ==================================================
# MEMBACA DATA TUNING
# ==================================================

def load_tuning_data():

    split_df = pd.read_csv(SPLIT_FILE)

    tuning_df = split_df[
        split_df["split"].str.upper() == "TUNING"
    ]

    if len(tuning_df) != 350:
        raise ValueError(
            f"Jumlah citra tuning tidak sesuai: {len(tuning_df)}"
        )

    return tuning_df


# ==================================================
# EKSPERIMEN VARIASI KERNEL
# ==================================================

def main():

    tuning_df = load_tuning_data()

    print("=" * 60)
    print("EKSPERIMEN VARIASI PANJANG KERNEL BLACK TOP-HAT")
    print("=" * 60)

    print(f"Jumlah citra tuning : {len(tuning_df)}")
    print(f"Variasi kernel      : {KERNEL_LENGTHS}")
    print(f"Single threshold    : {SINGLE_THRESHOLD}")
    print(f"Hysteresis          : {LOW_THRESHOLD}, {HIGH_THRESHOLD}")

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

            # Multi-Directional Black Top-Hat
            response = multi_directional_black_tophat(
                gray,
                length=length
            )

            # Ground truth biner
            gt = ground_truth > 0

            # Metode baseline
            baseline_mask = (
                response >= SINGLE_THRESHOLD
            ).astype(np.uint8)

            # Metode hysteresis
            hysteresis_mask = hysteresis_threshold(
                response,
                low=LOW_THRESHOLD,
                high=HIGH_THRESHOLD
            )

            # Evaluasi baseline
            baseline_metrics = calculate_metrics(
                baseline_mask,
                gt
            )

            # Evaluasi hysteresis
            hysteresis_metrics = calculate_metrics(
                hysteresis_mask,
                gt
            )

            result = {
                "filename": filename,
                "kernel_length": length,

                "baseline_precision": baseline_metrics["precision"],
                "baseline_recall": baseline_metrics["recall"],
                "baseline_dice": baseline_metrics["dice"],
                "baseline_iou": baseline_metrics["iou"],

                "hysteresis_precision": hysteresis_metrics["precision"],
                "hysteresis_recall": hysteresis_metrics["recall"],
                "hysteresis_dice": hysteresis_metrics["dice"],
                "hysteresis_iou": hysteresis_metrics["iou"]
            }

            all_results.append(result)

        print(f"Selesai kernel {length}")

    # ==================================================
    # SIMPAN HASIL DETAIL
    # ==================================================

    detail_df = pd.DataFrame(all_results)

    detail_path = OUTPUT_DIR / "hasil_detail.csv"

    detail_df.to_csv(detail_path, index=False)

    # ==================================================
    # RINGKASAN HASIL
    # ==================================================

    summary_rows = []

    for length in KERNEL_LENGTHS:

        subset = detail_df[
            detail_df["kernel_length"] == length
        ]

        summary_rows.append({
            "kernel_length": length,

            "baseline_precision": subset["baseline_precision"].mean(),
            "baseline_recall": subset["baseline_recall"].mean(),
            "baseline_dice": subset["baseline_dice"].mean(),
            "baseline_iou": subset["baseline_iou"].mean(),

            "hysteresis_precision": subset["hysteresis_precision"].mean(),
            "hysteresis_recall": subset["hysteresis_recall"].mean(),
            "hysteresis_dice": subset["hysteresis_dice"].mean(),
            "hysteresis_iou": subset["hysteresis_iou"].mean()
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = OUTPUT_DIR / "hasil_ringkasan.csv"

    summary_df.to_csv(summary_path, index=False)

    # ==================================================
    # GRAFIK DICE
    # ==================================================

    plt.figure(figsize=(9, 5))

    plt.plot(
        summary_df["kernel_length"],
        summary_df["baseline_dice"],
        marker="o",
        label="Single Threshold"
    )

    plt.plot(
        summary_df["kernel_length"],
        summary_df["hysteresis_dice"],
        marker="s",
        label="Hysteresis Thresholding"
    )

    plt.xlabel("Panjang Kernel")
    plt.ylabel("Mean Dice")
    plt.title("Pengaruh Panjang Kernel terhadap Dice")

    plt.xticks(KERNEL_LENGTHS)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "grafik_dice.png",
        dpi=300
    )

    plt.close()

    # ==================================================
    # GRAFIK IOU
    # ==================================================

    plt.figure(figsize=(9, 5))

    plt.plot(
        summary_df["kernel_length"],
        summary_df["baseline_iou"],
        marker="o",
        label="Single Threshold"
    )

    plt.plot(
        summary_df["kernel_length"],
        summary_df["hysteresis_iou"],
        marker="s",
        label="Hysteresis Thresholding"
    )

    plt.xlabel("Panjang Kernel")
    plt.ylabel("Mean IoU")
    plt.title("Pengaruh Panjang Kernel terhadap IoU")

    plt.xticks(KERNEL_LENGTHS)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "grafik_iou.png",
        dpi=300
    )

    plt.close()

    # ==================================================
    # TAMPILKAN HASIL
    # ==================================================

    print("\n" + "=" * 75)
    print("RINGKASAN HASIL EKSPERIMEN")
    print("=" * 75)

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    best_idx = summary_df["hysteresis_dice"].idxmax()

    best_kernel = summary_df.loc[
        best_idx,
        "kernel_length"
    ]

    print("\nKernel dengan Dice hysteresis tertinggi:")
    print(f"Kernel length = {best_kernel}")

    print("\nFile hasil tersimpan di:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
