
import cv2
import numpy as np
import pandas as pd

from pathlib import Path


# ==========================================
# 1. KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

RESULT_FILE = (
    ROOT / "results" / "analisis_kasus"
    / "analisis_seluruh_testing_detail.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "analisis_kasus"
    / "analisis_weak_strong_testing"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_LENGTH = 15
ANGLES = [0, 30, 60, 90, 120, 150]

LOW = 15
HIGH = 40


# ==========================================
# 2. BLACK TOP-HAT
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

    pt1 = (
        center[0] - dx,
        center[1] - dy
    )

    pt2 = (
        center[0] + dx,
        center[1] + dy
    )

    cv2.line(
        kernel,
        pt1,
        pt2,
        1,
        thickness=1
    )

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

        response = cv2.subtract(
            closing,
            gray
        )

        responses.append(response)

    return np.maximum.reduce(responses)


# ==========================================
# 3. PENCARIAN FILE
# ==========================================

def find_image(directory, image_name):

    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:

        path = directory / f"{image_name}{ext}"

        if path.exists():
            return path

    raise FileNotFoundError(
        f"File tidak ditemukan: {image_name}"
    )


# ==========================================
# 4. ANALISIS SATU CITRA
# ==========================================

def analyze_image(image_name, dice_group):

    image_path = find_image(
        IMAGE_DIR,
        image_name
    )

    mask_path = find_image(
        MASK_DIR,
        image_name
    )

    image = cv2.imread(str(image_path))

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        raise ValueError(
            f"Gagal membaca citra: {image_name}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gt = mask > 127

    # Respons BTH
    response = multi_directional_black_tophat(gray)

    # Weak dan strong
    weak = response >= LOW
    strong = response >= HIGH

    # Connected components
    num_labels, labels = cv2.connectedComponents(
        weak.astype(np.uint8),
        connectivity=8
    )

    # Komponen dengan minimal satu strong pixel
    strong_labels = np.unique(labels[strong])
    strong_labels = strong_labels[strong_labels != 0]

    retained = np.isin(
        labels,
        strong_labels
    )

    discarded = weak & ~retained
    below_low = ~weak

    # ======================================
    # KLASIFIKASI GROUND TRUTH
    # ======================================

    gt_total = int(gt.sum())

    retained_gt = int(
        (retained & gt).sum()
    )

    discarded_gt = int(
        (discarded & gt).sum()
    )

    below_low_gt = int(
        (below_low & gt).sum()
    )

    classified_gt = (
        retained_gt
        + discarded_gt
        + below_low_gt
    )

    if classified_gt != gt_total:

        raise ValueError(
            f"Validasi GT gagal pada {image_name}: "
            f"{classified_gt} != {gt_total}"
        )

    def percentage(value):

        if gt_total == 0:
            return np.nan

        return value / gt_total * 100

    # ======================================
    # HASIL PER CITRA
    # ======================================

    return {

        "image": image_name,
        "kelompok_dice": dice_group,

        "gt_pixels": gt_total,

        "weak_pixels": int(weak.sum()),
        "strong_pixels": int(strong.sum()),

        "strong_TP": int((strong & gt).sum()),
        "strong_FP": int((strong & ~gt).sum()),

        "total_components": num_labels - 1,
        "retained_components": len(strong_labels),
        "discarded_components": (
            num_labels - 1 - len(strong_labels)
        ),

        "retained_GT_pixels": retained_gt,
        "weak_discarded_GT_pixels": discarded_gt,
        "below_low_GT_pixels": below_low_gt,

        "retained_GT_percent": percentage(retained_gt),
        "weak_discarded_GT_percent": percentage(discarded_gt),
        "below_low_GT_percent": percentage(below_low_gt),

        "retained_FP": int((retained & ~gt).sum()),

        "classification_valid": (
            classified_gt == gt_total
        ),
    }


# ==========================================
# 5. RINGKASAN KELOMPOK
# ==========================================

def summarize_groups(df):

    rows = []

    groups = [
        "DICE MENINGKAT",
        "DICE MENURUN",
        "DICE TETAP"
    ]

    for group in groups:

        subset = df[
            df["kelompok_dice"] == group
        ]

        total_gt = int(
            subset["gt_pixels"].sum()
        )

        retained = int(
            subset["retained_GT_pixels"].sum()
        )

        discarded = int(
            subset["weak_discarded_GT_pixels"].sum()
        )

        below_low = int(
            subset["below_low_GT_pixels"].sum()
        )

        def percentage(value):

            if total_gt == 0:
                return np.nan

            return value / total_gt * 100

        rows.append({

            "kelompok_dice": group,
            "jumlah_citra": len(subset),

            "citra_GT_kosong": int(
                (subset["gt_pixels"] == 0).sum()
            ),

            "total_GT_pixels": total_gt,

            "retained_GT_pixels": retained,
            "weak_discarded_GT_pixels": discarded,
            "below_low_GT_pixels": below_low,

            "retained_GT_percent": percentage(retained),
            "weak_discarded_GT_percent": percentage(discarded),
            "below_low_GT_percent": percentage(below_low),

            "mean_retained_percent": (
                subset["retained_GT_percent"].mean()
            ),

            "mean_weak_discarded_percent": (
                subset["weak_discarded_GT_percent"].mean()
            ),

            "mean_below_low_percent": (
                subset["below_low_GT_percent"].mean()
            ),

            "mean_retained_FP": (
                subset["retained_FP"].mean()
            ),

            "mean_strong_TP": (
                subset["strong_TP"].mean()
            ),

            "mean_strong_FP": (
                subset["strong_FP"].mean()
            ),
        })

    return pd.DataFrame(rows)


# ==========================================
# 6. MAIN
# ==========================================

def main():

    df = pd.read_csv(RESULT_FILE)

    # Validasi kolom
    required_columns = [
        "image",
        "baseline_dice",
        "hysteresis_dice"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Kolom tidak ditemukan: {column}"
            )

    # Kelompok berdasarkan perubahan Dice
    df["delta_dice"] = (
        df["hysteresis_dice"]
        - df["baseline_dice"]
    )

    def assign_group(delta):

        if delta > 1e-12:
            return "DICE MENINGKAT"

        elif delta < -1e-12:
            return "DICE MENURUN"

        return "DICE TETAP"

    df["kelompok_dice"] = df["delta_dice"].apply(
        assign_group
    )

    print("Jumlah citra testing:", len(df))

    print("\n=== JUMLAH CITRA PER KELOMPOK ===")

    print(
        df["kelompok_dice"]
        .value_counts()
        .reindex(
            [
                "DICE MENINGKAT",
                "DICE MENURUN",
                "DICE TETAP"
            ],
            fill_value=0
        )
        .to_string()
    )

    # Analisis setiap citra
    results = []

    for index, row in df.iterrows():

        result = analyze_image(
            row["image"],
            row["kelompok_dice"]
        )

        results.append(result)

        if (index + 1) % 25 == 0:
            print(
                f"Progress: {index + 1}/{len(df)}"
            )

    detail = pd.DataFrame(results)

    # Validasi seluruh citra
    if not detail["classification_valid"].all():

        raise ValueError(
            "Ada citra dengan klasifikasi GT tidak valid."
        )

    # Simpan detail
    detail.to_csv(
        OUTPUT_DIR / "detail_weak_strong_150.csv",
        index=False
    )

    # Ringkasan per kelompok
    summary = summarize_groups(detail)

    summary.to_csv(
        OUTPUT_DIR / "ringkasan_per_kelompok.csv",
        index=False
    )

    # Ringkasan keseluruhan
    total_gt = int(detail["gt_pixels"].sum())

    retained = int(detail["retained_GT_pixels"].sum())
    discarded = int(detail["weak_discarded_GT_pixels"].sum())
    below_low = int(detail["below_low_GT_pixels"].sum())

    def overall_percentage(value):
        if total_gt == 0:
            return np.nan
        return value / total_gt * 100

    overall = pd.DataFrame([{
        "kelompok_dice": "SELURUH TESTING",
        "jumlah_citra": len(detail),
        "citra_GT_kosong": int((detail["gt_pixels"] == 0).sum()),
        "total_GT_pixels": total_gt,

        "retained_GT_pixels": retained,
        "weak_discarded_GT_pixels": discarded,
        "below_low_GT_pixels": below_low,

        "retained_GT_percent": overall_percentage(retained),
        "weak_discarded_GT_percent": overall_percentage(discarded),
        "below_low_GT_percent": overall_percentage(below_low),

        "mean_retained_percent": detail["retained_GT_percent"].mean(),
        "mean_weak_discarded_percent": detail["weak_discarded_GT_percent"].mean(),
        "mean_below_low_percent": detail["below_low_GT_percent"].mean(),

        "mean_retained_FP": detail["retained_FP"].mean(),
        "mean_strong_TP": detail["strong_TP"].mean(),
        "mean_strong_FP": detail["strong_FP"].mean()
    }])

    overall.to_csv(
        OUTPUT_DIR / "ringkasan_keseluruhan.csv",
        index=False
    )

    # ======================================
    # OUTPUT
    # ======================================

    print("\n=== VALIDASI KLASIFIKASI GT ===")

    print(
        "Citra valid:",
        int(detail["classification_valid"].sum()),
        "/",
        len(detail)
    )

    print(
        "Total GT:",
        int(detail["gt_pixels"].sum())
    )

    print(
        "Retained GT:",
        int(detail["retained_GT_pixels"].sum())
    )

    print(
        "Weak discarded GT:",
        int(detail["weak_discarded_GT_pixels"].sum())
    )

    print(
        "Below low GT:",
        int(detail["below_low_GT_pixels"].sum())
    )

    print("\n=== RINGKASAN KELOMPOK DICE ===")

    print(
        summary.round(2).to_string(index=False)
    )

    print("\n=== RINGKASAN SELURUH TESTING ===")

    print(
        overall.round(2).to_string(index=False)
    )

    print("\nAnalisis selesai.")
    print("Folder hasil:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
