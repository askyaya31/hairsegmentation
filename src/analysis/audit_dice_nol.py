
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.core.black_tophat import multi_directional_black_tophat


# ============================================================
# KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

SPLIT_FILE = ROOT / "results" / "pembagian_dataset.csv"
DETAIL_FILE = (
    ROOT / "results" / "03_evaluation" / "final_testing" / "hasil_detail.csv"
)

OUTPUT_DIR = ROOT / "results" / "06_error_analysis" / "audit_dice_nol"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "hasil_audit_dice_nol.csv"

# Parameter final, tidak diubah
BASELINE_KERNEL = 11
BASELINE_THRESHOLD = 20

HYSTERESIS_KERNEL = 9
LOW_THRESHOLD = 10
HIGH_THRESHOLD = 30


# ============================================================
# HYSTERESIS THRESHOLDING
# ============================================================

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

    return prediction.astype(np.uint8)


# ============================================================
# PERHITUNGAN METRIK
# ============================================================

def calculate_counts(prediction, ground_truth):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = int(np.logical_and(prediction, ground_truth).sum())
    fp = int(np.logical_and(prediction, ~ground_truth).sum())
    fn = int(np.logical_and(~prediction, ground_truth).sum())

    dice_denominator = 2 * tp + fp + fn

    if dice_denominator == 0:
        dice = 1.0
    else:
        dice = (2 * tp) / dice_denominator

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_pixels": int(ground_truth.sum()),
        "prediction_pixels": int(prediction.sum()),
        "dice_recomputed": dice
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("AUDIT CITRA DENGAN DICE NOL")
    print("=" * 65)

    split_df = pd.read_csv(SPLIT_FILE)
    detail_df = pd.read_csv(DETAIL_FILE)

    testing_df = split_df[
        split_df["split"].astype(str).str.upper() == "PENGUJIAN"
    ].copy()

    df = testing_df.merge(
        detail_df,
        on="filename",
        how="inner",
        validate="one_to_one"
    )

    zero_cases = df[
        (df["baseline_dice"] == 0) |
        (df["hysteresis_dice"] == 0)
    ].copy()

    print(f"\nJumlah citra pengujian : {len(df)}")
    print(f"Citra dengan Dice nol  : {len(zero_cases)}")

    results = []

    for _, row in zero_cases.iterrows():
        filename = row["filename"]
        category = row["category"]

        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / filename

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(image_path)

        if mask is None:
            raise FileNotFoundError(mask_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ground truth mengikuti evaluasi final
        gt = (mask > 127).astype(np.uint8)

        # Baseline
        baseline_response = multi_directional_black_tophat(
            gray,
            length=BASELINE_KERNEL
        )

        baseline_prediction = (
            baseline_response >= BASELINE_THRESHOLD
        ).astype(np.uint8)

        # Hysteresis
        hysteresis_response = multi_directional_black_tophat(
            gray,
            length=HYSTERESIS_KERNEL
        )

        hysteresis_prediction = hysteresis_threshold(
            hysteresis_response,
            LOW_THRESHOLD,
            HIGH_THRESHOLD
        )

        baseline = calculate_counts(baseline_prediction, gt)
        hysteresis = calculate_counts(hysteresis_prediction, gt)

        results.append({
            "filename": filename,
            "category": category,

            "baseline_dice_original": row["baseline_dice"],
            "baseline_dice_recomputed": baseline["dice_recomputed"],
            "baseline_gt_pixels": baseline["gt_pixels"],
            "baseline_prediction_pixels": baseline["prediction_pixels"],
            "baseline_tp": baseline["tp"],
            "baseline_fp": baseline["fp"],
            "baseline_fn": baseline["fn"],

            "hysteresis_dice_original": row["hysteresis_dice"],
            "hysteresis_dice_recomputed": hysteresis["dice_recomputed"],
            "hysteresis_gt_pixels": hysteresis["gt_pixels"],
            "hysteresis_prediction_pixels": hysteresis["prediction_pixels"],
            "hysteresis_tp": hysteresis["tp"],
            "hysteresis_fp": hysteresis["fp"],
            "hysteresis_fn": hysteresis["fn"],
        })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df["baseline_match"] = np.isclose(
            result_df["baseline_dice_original"],
            result_df["baseline_dice_recomputed"],
            atol=1e-9
        )

        result_df["hysteresis_match"] = np.isclose(
            result_df["hysteresis_dice_original"],
            result_df["hysteresis_dice_recomputed"],
            atol=1e-9
        )

        result_df.to_csv(OUTPUT_FILE, index=False)

        print("\nHASIL AUDIT")
        print("=" * 65)

        for _, row in result_df.iterrows():
            print(f"\nCitra   : {row['filename']}")
            print(f"Kategori: {row['category']}")

            print("\nGround truth:")
            print(f"Jumlah piksel rambut: {row['baseline_gt_pixels']}")

            print("\nBaseline:")
            print(f"Prediksi : {row['baseline_prediction_pixels']}")
            print(f"TP       : {row['baseline_tp']}")
            print(f"FP       : {row['baseline_fp']}")
            print(f"FN       : {row['baseline_fn']}")
            print(f"Dice     : {row['baseline_dice_recomputed']:.6f}")
            print(f"Sesuai hasil awal: {row['baseline_match']}")

            print("\nHysteresis:")
            print(f"Prediksi : {row['hysteresis_prediction_pixels']}")
            print(f"TP       : {row['hysteresis_tp']}")
            print(f"FP       : {row['hysteresis_fp']}")
            print(f"FN       : {row['hysteresis_fn']}")
            print(f"Dice     : {row['hysteresis_dice_recomputed']:.6f}")
            print(f"Sesuai hasil awal: {row['hysteresis_match']}")

        print("\n" + "=" * 65)
        print(f"Hasil disimpan: {OUTPUT_FILE}")

        if not (
            result_df["baseline_match"].all()
            and result_df["hysteresis_match"].all()
        ):
            print("PERINGATAN: Ada Dice yang tidak cocok dengan hasil awal.")
        else:
            print("Semua Dice hasil audit cocok dengan hasil evaluasi awal.")

    else:
        print("\nTidak ditemukan citra dengan Dice nol.")

    print("=" * 65)


if __name__ == "__main__":
    main()
