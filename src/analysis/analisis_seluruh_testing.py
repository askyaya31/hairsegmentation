
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
OLD_RESULT = ROOT / "results" / "evaluasi_testing" / "hasil_detail.csv"

OUTPUT_DIR = ROOT / "results" / "analisis_kasus"
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
# 3. HYSTERESIS
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
# 4. HITUNG TP, FP, FN
# ==========================================

def calculate_counts(prediction, ground_truth):
    tp = np.logical_and(prediction, ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()

    return int(tp), int(fp), int(fn)


# ==========================================
# 5. ANALISIS SELURUH DATA TESTING
# ==========================================

def main():

    df_split = pd.read_csv(SPLIT_CSV)

    df_testing = df_split[
        df_split["split"] == "PENGUJIAN"
    ].copy()

    if len(df_testing) != 150:
        raise ValueError(
            f"Jumlah data testing tidak sesuai: {len(df_testing)}"
        )

    old_results = pd.read_csv(OLD_RESULT)

    old_results["image"] = old_results["image"].astype(str)

    results = []

    for index, row in df_testing.iterrows():

        filename = row["filename"]
        image_name = Path(filename).stem

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

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ground_truth = mask > 127

        response = multi_directional_black_tophat(gray)

        prediction_baseline = (
            response >= BASELINE_THRESHOLD
        )

        prediction_hysteresis = hysteresis_threshold(
            response,
            HYSTERESIS_LOW,
            HYSTERESIS_HIGH
        )

        baseline_tp, baseline_fp, baseline_fn = (
            calculate_counts(
                prediction_baseline,
                ground_truth
            )
        )

        hysteresis_tp, hysteresis_fp, hysteresis_fn = (
            calculate_counts(
                prediction_hysteresis,
                ground_truth
            )
        )

        results.append({
            "image": image_name,

            "baseline_TP": baseline_tp,
            "baseline_FP": baseline_fp,
            "baseline_FN": baseline_fn,

            "hysteresis_TP": hysteresis_tp,
            "hysteresis_FP": hysteresis_fp,
            "hysteresis_FN": hysteresis_fn,

            "delta_TP": hysteresis_tp - baseline_tp,
            "delta_FP": hysteresis_fp - baseline_fp,
            "delta_FN": hysteresis_fn - baseline_fn
        })

        if len(results) % 25 == 0:
            print(f"Proses: {len(results)}/150")

    df = pd.DataFrame(results)

    # ======================================
    # 6. VALIDASI DENGAN HASIL EVALUASI LAMA
    # ======================================

    merged = df.merge(
        old_results[
            ["image", "baseline_dice", "hysteresis_dice"]
        ],
        on="image",
        how="left",
        validate="one_to_one"
    )

    if merged["baseline_dice"].isna().any():
        raise ValueError(
            "Ada citra yang tidak cocok dengan hasil evaluasi lama."
        )

    def calculate_dice(tp, fp, fn):
        denominator = 2 * tp + fp + fn
        return 2 * tp / denominator if denominator > 0 else 1.0

    merged["calculated_baseline_dice"] = [
        calculate_dice(tp, fp, fn)
        for tp, fp, fn in zip(
            merged["baseline_TP"],
            merged["baseline_FP"],
            merged["baseline_FN"]
        )
    ]

    merged["calculated_hysteresis_dice"] = [
        calculate_dice(tp, fp, fn)
        for tp, fp, fn in zip(
            merged["hysteresis_TP"],
            merged["hysteresis_FP"],
            merged["hysteresis_FN"]
        )
    ]

    baseline_match = np.allclose(
        merged["baseline_dice"],
        merged["calculated_baseline_dice"],
        atol=1e-12
    )

    hysteresis_match = np.allclose(
        merged["hysteresis_dice"],
        merged["calculated_hysteresis_dice"],
        atol=1e-12
    )

    print("\n=== VALIDASI DICE ===")
    print(f"Baseline sesuai  : {baseline_match}")
    print(f"Hysteresis sesuai: {hysteresis_match}")

    if not baseline_match or not hysteresis_match:
        raise ValueError(
            "Hasil Dice tidak cocok. Periksa kembali data dan implementasi."
        )

    # ======================================
    # 7. STATISTIK PERUBAHAN
    # ======================================

    components = ["TP", "FP", "FN"]

    summary = []

    for component in components:

        delta = merged[f"delta_{component}"]

        if component == "FP":
            decreased = int((delta < 0).sum())
            increased = int((delta > 0).sum())
        else:
            increased = int((delta > 0).sum())
            decreased = int((delta < 0).sum())

        unchanged = int((delta == 0).sum())

        summary.append({
            "komponen": component,
            "meningkat": increased,
            "menurun": decreased,
            "tetap": unchanged,
            "rata_rata_perubahan": delta.mean(),
            "median_perubahan": delta.median()
        })

    df_summary = pd.DataFrame(summary)

    # ======================================
    # 8. SIMPAN HASIL
    # ======================================

    detail_path = (
        OUTPUT_DIR / "analisis_seluruh_testing_detail.csv"
    )

    summary_path = (
        OUTPUT_DIR / "analisis_seluruh_testing_ringkasan.csv"
    )

    merged.to_csv(detail_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    # ======================================
    # 9. TAMPILKAN HASIL
    # ======================================

    print("\n=== STATISTIK PERUBAHAN TP, FP, FN ===")

    print(
        df_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("\n=== RATA-RATA JUMLAH PIKSEL ===")

    for component in components:
        baseline_mean = merged[
            f"baseline_{component}"
        ].mean()

        hysteresis_mean = merged[
            f"hysteresis_{component}"
        ].mean()

        print(
            f"{component}: "
            f"Baseline={baseline_mean:.2f}, "
            f"Hysteresis={hysteresis_mean:.2f}"
        )

    print("\n=== FILE HASIL ===")
    print(f"Detail    : {detail_path}")
    print(f"Ringkasan : {summary_path}")


if __name__ == "__main__":
    main()
