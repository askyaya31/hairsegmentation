
import cv2
import numpy as np
import pandas as pd

from pathlib import Path

from src.core.black_tophat import multi_directional_black_tophat
from src.pipeline.evaluasi_final_testing import (
    IMAGE_DIR,
    MASK_DIR,
    BASELINE_KERNEL,
    BASELINE_THRESHOLD,
    HYSTERESIS_KERNEL,
    LOW_THRESHOLD,
    HIGH_THRESHOLD,
    hysteresis_threshold
)

ROOT = Path(__file__).resolve().parents[2]

SPLIT_FILE = ROOT / "results" / "pembagian_dataset.csv"

OUTPUT_DIR = ROOT / "results" / "06_error_analysis" / "overlap_prediksi"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "hasil_overlap_prediksi.csv"

TARGET_IMAGES = [
    "ISIC_0013713.png",
    "ISIC_0012879.png",
    "ISIC_0013320.png",
    "ISIC_0013601.png",
    "ISIC_0012089.png",
    "ISIC_0013554.png",
    "ISIC_0012966.png"
]


# ============================================================
# ANALISIS OVERLAP
# ============================================================

def analyze_overlap(filename, category):

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

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_response = multi_directional_black_tophat(
        gray,
        length=BASELINE_KERNEL
    )

    baseline = (
        baseline_response >= BASELINE_THRESHOLD
    )

    # --------------------------------------------------------
    # HYSTERESIS
    # --------------------------------------------------------

    hysteresis_response = multi_directional_black_tophat(
        gray,
        length=HYSTERESIS_KERNEL
    )

    hysteresis = hysteresis_threshold(
        hysteresis_response,
        low=LOW_THRESHOLD,
        high=HIGH_THRESHOLD
    ).astype(bool)

    # --------------------------------------------------------
    # HUBUNGAN KEDUA PREDIKSI
    # --------------------------------------------------------

    intersection = baseline & hysteresis

    baseline_only = baseline & ~hysteresis

    hysteresis_only = hysteresis & ~baseline

    union = baseline | hysteresis

    baseline_pixels = int(baseline.sum())
    hysteresis_pixels = int(hysteresis.sum())

    intersection_pixels = int(intersection.sum())
    baseline_only_pixels = int(baseline_only.sum())
    hysteresis_only_pixels = int(hysteresis_only.sum())
    union_pixels = int(union.sum())

    # --------------------------------------------------------
    # OVERLAP METRICS
    # --------------------------------------------------------

    prediction_iou = (
        intersection_pixels / union_pixels
        if union_pixels > 0 else 1.0
    )

    retained_baseline = (
        intersection_pixels / baseline_pixels * 100
        if baseline_pixels > 0 else np.nan
    )

    shared_hysteresis = (
        intersection_pixels / hysteresis_pixels * 100
        if hysteresis_pixels > 0 else np.nan
    )

    # --------------------------------------------------------
    # HUBUNGAN DENGAN GROUND TRUTH
    # --------------------------------------------------------

    baseline_tp = int((baseline & gt).sum())
    hysteresis_tp = int((hysteresis & gt).sum())

    baseline_fp = int((baseline & ~gt).sum())
    hysteresis_fp = int((hysteresis & ~gt).sum())

    baseline_fn = int((~baseline & gt).sum())
    hysteresis_fn = int((~hysteresis & gt).sum())

    return {
        "filename": filename,
        "category": category,

        "baseline_pixels": baseline_pixels,
        "hysteresis_pixels": hysteresis_pixels,

        "intersection_pixels": intersection_pixels,
        "baseline_only_pixels": baseline_only_pixels,
        "hysteresis_only_pixels": hysteresis_only_pixels,
        "union_pixels": union_pixels,

        "prediction_iou": prediction_iou,
        "retained_baseline_percent": retained_baseline,
        "shared_hysteresis_percent": shared_hysteresis,

        "baseline_tp": baseline_tp,
        "hysteresis_tp": hysteresis_tp,

        "baseline_fp": baseline_fp,
        "hysteresis_fp": hysteresis_fp,

        "baseline_fn": baseline_fn,
        "hysteresis_fn": hysteresis_fn
    }


# ============================================================
# MAIN
# ============================================================

def main():

    split_df = pd.read_csv(SPLIT_FILE)

    category_map = (
        split_df.set_index("filename")["category"].to_dict()
    )

    print("=" * 70)
    print("ANALISIS OVERLAP PREDIKSI BASELINE DAN HYSTERESIS")
    print("=" * 70)

    print(
        f"Baseline   : kernel {BASELINE_KERNEL}, "
        f"threshold {BASELINE_THRESHOLD}"
    )

    print(
        f"Hysteresis : kernel {HYSTERESIS_KERNEL}, "
        f"low {LOW_THRESHOLD}, high {HIGH_THRESHOLD}"
    )

    results = []

    for filename in TARGET_IMAGES:

        if filename not in category_map:
            raise ValueError(
                f"Kategori tidak ditemukan: {filename}"
            )

        result = analyze_overlap(
            filename,
            category_map[filename]
        )

        results.append(result)

        print(f"\nCitra: {filename}")
        print(f"Kategori: {result['category']}")

        print(
            f"Baseline pixels       : "
            f"{result['baseline_pixels']}"
        )

        print(
            f"Hysteresis pixels     : "
            f"{result['hysteresis_pixels']}"
        )

        print(
            f"Intersection          : "
            f"{result['intersection_pixels']}"
        )

        print(
            f"Baseline only         : "
            f"{result['baseline_only_pixels']}"
        )

        print(
            f"Hysteresis only       : "
            f"{result['hysteresis_only_pixels']}"
        )

        print(
            f"Prediction IoU        : "
            f"{result['prediction_iou']:.6f}"
        )

        print(
            f"Retained Baseline     : "
            f"{result['retained_baseline_percent']:.2f}%"
        )

        print(
            f"Shared Hysteresis     : "
            f"{result['shared_hysteresis_percent']:.2f}%"
        )

        print(
            f"TP Baseline/Hysteresis: "
            f"{result['baseline_tp']} / "
            f"{result['hysteresis_tp']}"
        )

    # --------------------------------------------------------
    # SIMPAN CSV
    # --------------------------------------------------------

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("RINGKASAN OVERLAP")
    print("=" * 70)

    columns = [
        "filename",
        "baseline_pixels",
        "hysteresis_pixels",
        "intersection_pixels",
        "baseline_only_pixels",
        "hysteresis_only_pixels",
        "prediction_iou"
    ]

    print(
        result_df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nHasil disimpan:")
    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()
