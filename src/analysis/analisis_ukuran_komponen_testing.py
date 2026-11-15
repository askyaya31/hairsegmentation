
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import spearmanr

from src.core.black_tophat import multi_directional_black_tophat


# ==========================================
# 1. KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

RESULT_FILE = (
    ROOT / "results" / "analisis_kasus"
    / "analisis_seluruh_testing_detail.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "analisis_kasus"
    / "analisis_ukuran_komponen"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOW = 15
HIGH = 40
KERNEL_LENGTH = 15

SIZE_BINS = [
    0, 10, 50, 200, 1000, np.inf
]

SIZE_LABELS = [
    "1-10",
    "11-50",
    "51-200",
    "201-1000",
    ">1000"
]


# ==========================================
# 2. PENCARIAN FILE
# ==========================================

def find_file(directory, image_name):

    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        path = directory / f"{image_name}{ext}"

        if path.exists():
            return path

    raise FileNotFoundError(image_name)


# ==========================================
# 3. ANALISIS KOMPONEN SATU CITRA
# ==========================================

def analyze_image(image_name, dice_group):

    image_path = find_file(
        IMAGE_DIR, image_name
    )

    mask_path = find_file(
        MASK_DIR, image_name
    )

    image = cv2.imread(str(image_path))

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        raise ValueError(
            f"Gagal membaca {image_name}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gt = mask > 127

    # Gunakan implementasi BTH utama
    response = multi_directional_black_tophat(
        gray,
        length=KERNEL_LENGTH
    )

    weak = response >= LOW
    strong = response >= HIGH

    num_labels, labels = cv2.connectedComponents(
        weak.astype(np.uint8),
        connectivity=8
    )

    component_rows = []

    for label in range(1, num_labels):

        component = labels == label

        area = int(component.sum())

        strong_count = int(
            (component & strong).sum()
        )

        gt_count = int(
            (component & gt).sum()
        )

        non_gt_count = area - gt_count

        retained = strong_count > 0

        # Precision hanya bermakna untuk
        # komponen yang benar-benar dipertahankan
        precision = (
            gt_count / area * 100
            if retained and area > 0
            else np.nan
        )

        component_rows.append({

            "image": image_name,
            "kelompok_dice": dice_group,

            "component_label": label,
            "area": area,

            "strong_pixels": strong_count,

            "gt_pixels": gt_count,
            "non_gt_pixels": non_gt_count,

            "retained": retained,

            "precision_percent": precision
        })

    return component_rows


# ==========================================
# 4. MAIN
# ==========================================

def main():

    df = pd.read_csv(RESULT_FILE)

    required = [
        "image",
        "baseline_dice",
        "hysteresis_dice"
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Kolom tidak ditemukan: {column}"
            )

    delta = (
        df["hysteresis_dice"]
        - df["baseline_dice"]
    )

    df["kelompok_dice"] = np.select(
        [
            delta > 1e-12,
            delta < -1e-12
        ],
        [
            "DICE MENINGKAT",
            "DICE MENURUN"
        ],
        default="DICE TETAP"
    )

    all_components = []

    print("Jumlah citra testing:", len(df))

    for index, row in df.iterrows():

        components = analyze_image(
            row["image"],
            row["kelompok_dice"]
        )

        all_components.extend(components)

        if (index + 1) % 25 == 0:
            print(
                f"Progress: {index + 1}/{len(df)}"
            )

    detail = pd.DataFrame(all_components)

    if detail.empty:
        raise ValueError(
            "Tidak ditemukan connected component."
        )

    # ======================================
    # 5. KATEGORI UKURAN
    # ======================================

    detail["size_category"] = pd.cut(
        detail["area"],
        bins=SIZE_BINS,
        labels=SIZE_LABELS
    )

    # Simpan detail seluruh komponen
    detail.to_csv(
        OUTPUT_DIR / "detail_seluruh_komponen.csv",
        index=False
    )

    # ======================================
    # 6. RINGKASAN BERDASARKAN UKURAN
    # ======================================

    summary_size = (
        detail.groupby(
            ["kelompok_dice", "retained", "size_category"],
            observed=False
        )
        .agg(
            jumlah_komponen=("area", "size"),
            total_area=("area", "sum"),
            total_strong=("strong_pixels", "sum"),
            total_GT=("gt_pixels", "sum"),
            total_non_GT=("non_gt_pixels", "sum"),
            mean_area=("area", "mean"),
            mean_strong=("strong_pixels", "mean"),
            mean_precision=("precision_percent", "mean")
        )
        .reset_index()
    )

    summary_size.to_csv(
        OUTPUT_DIR / "ringkasan_ukuran_komponen.csv",
        index=False
    )

    # ======================================
    # 7. RINGKASAN KOMPONEN DIPERTAHANKAN
    #    VS DIBUANG
    # ======================================

    summary_status = (
        detail.groupby(
            ["kelompok_dice", "retained"],
            observed=False
        )
        .agg(
            jumlah_komponen=("area", "size"),
            total_area=("area", "sum"),
            mean_area=("area", "mean"),
            median_area=("area", "median"),
            max_area=("area", "max"),
            total_strong=("strong_pixels", "sum"),
            total_GT=("gt_pixels", "sum"),
            total_non_GT=("non_gt_pixels", "sum"),
            mean_precision=("precision_percent", "mean")
        )
        .reset_index()
    )

    summary_status.to_csv(
        OUTPUT_DIR / "ringkasan_status_komponen.csv",
        index=False
    )

    # ======================================
    # 8. KORELASI KOMPONEN DIPERTAHANKAN
    # ======================================

    retained_components = detail[
        detail["retained"]
    ].copy()

    correlation_pairs = [
        ("area", "precision_percent"),
        ("area", "non_gt_pixels"),
        ("strong_pixels", "non_gt_pixels"),
        ("strong_pixels", "precision_percent")
    ]

    correlation_rows = []

    for x_col, y_col in correlation_pairs:

        valid = retained_components[
            [x_col, y_col]
        ].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if (
            len(valid) < 3
            or valid[x_col].nunique() < 2
            or valid[y_col].nunique() < 2
        ):
            rho = np.nan
            p_value = np.nan
        else:
            result = spearmanr(
                valid[x_col],
                valid[y_col]
            )

            rho = result.statistic
            p_value = result.pvalue

        correlation_rows.append({

            "hubungan": f"{x_col} vs {y_col}",
            "jumlah_komponen": len(valid),
            "spearman_rho": rho,
            "p_value": p_value
        })

    correlation = pd.DataFrame(
        correlation_rows
    )

    correlation.to_csv(
        OUTPUT_DIR / "korelasi_komponen.csv",
        index=False
    )

    # ======================================
    # 9. OUTPUT
    # ======================================

    print("\n=== JUMLAH KOMPONEN ===")
    print("Total:", len(detail))

    print(
        "Dipertahankan:",
        int(detail["retained"].sum())
    )

    print(
        "Dibuang:",
        int((~detail["retained"]).sum())
    )

    print("\n=== RINGKASAN STATUS KOMPONEN ===")
    print(
        summary_status.round(2).to_string(
            index=False
        )
    )

    print("\n=== RINGKASAN BERDASARKAN UKURAN ===")
    print(
        summary_size.round(2).to_string(
            index=False
        )
    )

    print("\n=== KORELASI KOMPONEN DIPERTAHANKAN ===")
    print(
        correlation.round(5).to_string(
            index=False
        )
    )

    print("\nAnalisis selesai.")
    print("Folder hasil:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
