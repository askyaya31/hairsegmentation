
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

CASE_FILE = ROOT / "results" / "06_error_analysis" / "analisis_kasus" / "10_kasus_terpilih.csv"

OUTPUT_DIR = ROOT / "results" / "06_error_analysis" / "analisis_kasus"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "analisis_tp_fp_fn.csv"

KERNEL_LENGTH = 15
ANGLES = [0, 30, 60, 90, 120, 150]

BASELINE_THRESHOLD = 20
HYSTERESIS_LOW = 15
HYSTERESIS_HIGH = 40


# ==========================================
# 2. FUNGSI PENCARIAN FILE
# ==========================================

def find_image(directory, stem):
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path

    raise FileNotFoundError(f"File tidak ditemukan: {stem}")


# ==========================================
# 3. BLACK TOP-HAT
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
# 4. HYSTERESIS
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
# 5. PERHITUNGAN TP, FP, FN
# ==========================================

def calculate_confusion(prediction, ground_truth):
    tp = np.logical_and(prediction, ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()

    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn > 0 else 1.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0 else 1.0
    )

    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou
    }


# ==========================================
# 6. ANALISIS 10 KASUS
# ==========================================

def main():
    df_cases = pd.read_csv(CASE_FILE)

    results = []

    for _, row in df_cases.iterrows():
        name = row["image"]

        image_path = find_image(IMAGE_DIR, name)
        mask_path = find_image(MASK_DIR, name)

        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Gagal membaca citra atau mask: {name}")

        if image.shape[:2] != mask.shape:
            raise ValueError(f"Ukuran citra dan mask berbeda: {name}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ground_truth = mask > 127

        response = multi_directional_black_tophat(gray)

        prediction_baseline = response >= BASELINE_THRESHOLD

        prediction_hysteresis = hysteresis_threshold(
            response,
            HYSTERESIS_LOW,
            HYSTERESIS_HIGH
        )

        baseline = calculate_confusion(
            prediction_baseline,
            ground_truth
        )

        hysteresis = calculate_confusion(
            prediction_hysteresis,
            ground_truth
        )

        results.append({
            "image": name,
            "kategori": row["kategori"],

            "baseline_TP": baseline["TP"],
            "baseline_FP": baseline["FP"],
            "baseline_FN": baseline["FN"],
            "baseline_precision": baseline["precision"],
            "baseline_recall": baseline["recall"],
            "baseline_dice": baseline["dice"],
            "baseline_iou": baseline["iou"],

            "hysteresis_TP": hysteresis["TP"],
            "hysteresis_FP": hysteresis["FP"],
            "hysteresis_FN": hysteresis["FN"],
            "hysteresis_precision": hysteresis["precision"],
            "hysteresis_recall": hysteresis["recall"],
            "hysteresis_dice": hysteresis["dice"],
            "hysteresis_iou": hysteresis["iou"],

            "delta_TP": hysteresis["TP"] - baseline["TP"],
            "delta_FP": hysteresis["FP"] - baseline["FP"],
            "delta_FN": hysteresis["FN"] - baseline["FN"],
            "delta_dice": hysteresis["dice"] - baseline["dice"]
        })

        print(f"Selesai: {name}")

    df_results = pd.DataFrame(results)

    df_results.to_csv(OUTPUT_FILE, index=False)

    # ======================================
    # 7. TAMPILKAN RINGKASAN
    # ======================================

    columns = [
        "image", "kategori",
        "baseline_TP", "baseline_FP", "baseline_FN",
        "hysteresis_TP", "hysteresis_FP", "hysteresis_FN",
        "delta_TP", "delta_FP", "delta_FN", "delta_dice"
    ]

    print("\n=== HASIL ANALISIS TP, FP, FN ===")

    print(
        df_results[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print(f"\nFile tersimpan: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
