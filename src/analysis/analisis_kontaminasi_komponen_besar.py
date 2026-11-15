
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import spearmanr


# ==========================================
# 1. KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

COMPONENT_FILE = (
    ROOT / "results" / "06_error_analysis" / "analisis_kasus"
    / "analisis_ukuran_komponen"
    / "detail_seluruh_komponen.csv"
)

IMAGE_FILE = (
    ROOT / "results" / "06_error_analysis" / "analisis_kasus"
    / "analisis_seluruh_testing_detail.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "06_error_analysis" / "analisis_kasus"
    / "analisis_kontaminasi_komponen_besar"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LARGE_THRESHOLD = 1000


# ==========================================
# 2. FUNGSI KORELASI
# ==========================================

def calculate_spearman(df, x_col, y_col):

    valid = df[[x_col, y_col]].replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if (
        len(valid) < 3
        or valid[x_col].nunique() < 2
        or valid[y_col].nunique() < 2
    ):
        return {
            "hubungan": f"{x_col} vs {y_col}",
            "n": len(valid),
            "rho": np.nan,
            "p_value": np.nan
        }

    result = spearmanr(
        valid[x_col],
        valid[y_col]
    )

    return {
        "hubungan": f"{x_col} vs {y_col}",
        "n": len(valid),
        "rho": result.statistic,
        "p_value": result.pvalue
    }


# ==========================================
# 3. MEMBACA DATA
# ==========================================

def main():

    components = pd.read_csv(COMPONENT_FILE)
    images = pd.read_csv(IMAGE_FILE)

    required_component = [
        "image",
        "area",
        "gt_pixels",
        "non_gt_pixels",
        "retained"
    ]

    required_image = [
        "image",
        "baseline_dice",
        "hysteresis_dice"
    ]

    for col in required_component:
        if col not in components.columns:
            raise ValueError(
                f"Kolom komponen tidak ditemukan: {col}"
            )

    for col in required_image:
        if col not in images.columns:
            raise ValueError(
                f"Kolom citra tidak ditemukan: {col}"
            )

    # Hanya komponen yang benar-benar dipertahankan
    retained = components[
        components["retained"] == True
    ].copy()

    retained["is_large"] = (
        retained["area"] > LARGE_THRESHOLD
    )

    # ======================================
    # 4. AGREGASI PER CITRA
    # ======================================

    # Seluruh komponen yang dipertahankan
    all_summary = (
        retained.groupby("image")
        .agg(
            retained_component_count=("area", "size"),
            retained_area=("area", "sum"),
            retained_GT=("gt_pixels", "sum"),
            retained_FP=("non_gt_pixels", "sum")
        )
        .reset_index()
    )

    # Komponen besar saja
    large = retained[
        retained["is_large"]
    ].copy()

    large_summary = (
        large.groupby("image")
        .agg(
            large_component_count=("area", "size"),
            large_area=("area", "sum"),
            large_GT=("gt_pixels", "sum"),
            large_FP=("non_gt_pixels", "sum")
        )
        .reset_index()
    )

    # Gabungkan seluruh citra testing
    result = images[
        required_image
    ].copy()

    result = result.merge(
        all_summary,
        on="image",
        how="left"
    )

    result = result.merge(
        large_summary,
        on="image",
        how="left"
    )

    count_columns = [
        "retained_component_count",
        "retained_area",
        "retained_GT",
        "retained_FP",
        "large_component_count",
        "large_area",
        "large_GT",
        "large_FP"
    ]

    result[count_columns] = (
        result[count_columns].fillna(0)
    )

    # ======================================
    # 5. MENGHITUNG PROPORSI KONTAMINASI
    # ======================================

    result["delta_dice"] = (
        result["hysteresis_dice"]
        - result["baseline_dice"]
    )

    # Proporsi non-GT dalam komponen besar.
    # NaN jika citra tidak memiliki komponen besar.
    result["large_contamination_percent"] = (
        result["large_FP"]
        / result["large_area"].replace(0, np.nan)
        * 100
    )

    # Proporsi non-GT pada seluruh komponen retained
    result["overall_contamination_percent"] = (
        result["retained_FP"]
        / result["retained_area"].replace(0, np.nan)
        * 100
    )

    # Bagian seluruh retained FP yang berasal
    # dari komponen besar
    result["large_FP_share_percent"] = (
        result["large_FP"]
        / result["retained_FP"].replace(0, np.nan)
        * 100
    )

    # Bagian area retained yang berasal
    # dari komponen besar
    result["large_area_share_percent"] = (
        result["large_area"]
        / result["retained_area"].replace(0, np.nan)
        * 100
    )

    # Apakah citra memiliki komponen besar?
    result["has_large_component"] = (
        result["large_component_count"] > 0
    )

    result.to_csv(
        OUTPUT_DIR / "detail_kontaminasi_per_citra.csv",
        index=False
    )

    # ======================================
    # 6. RINGKASAN BERDASARKAN KELOMPOK DICE
    # ======================================

    result["kelompok_dice"] = np.select(
        [
            result["delta_dice"] > 1e-12,
            result["delta_dice"] < -1e-12
        ],
        [
            "DICE MENINGKAT",
            "DICE MENURUN"
        ],
        default="DICE TETAP"
    )

    summary = (
        result.groupby("kelompok_dice")
        .agg(
            jumlah_citra=("image", "size"),

            citra_dengan_komponen_besar=(
                "has_large_component",
                "sum"
            ),

            mean_delta_dice=("delta_dice", "mean"),
            median_delta_dice=("delta_dice", "median"),

            mean_large_count=(
                "large_component_count", "mean"
            ),

            mean_large_area=(
                "large_area", "mean"
            ),

            mean_large_FP=(
                "large_FP", "mean"
            ),

            mean_large_contamination=(
                "large_contamination_percent", "mean"
            ),

            median_large_contamination=(
                "large_contamination_percent", "median"
            ),

            mean_large_FP_share=(
                "large_FP_share_percent", "mean"
            ),

            mean_large_area_share=(
                "large_area_share_percent", "mean"
            ),

            mean_overall_contamination=(
                "overall_contamination_percent", "mean"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "ringkasan_per_kelompok.csv",
        index=False
    )

    # ======================================
    # 7. KORELASI SELURUH CITRA
    # ======================================

    correlation_pairs = [

        (
            "large_contamination_percent",
            "delta_dice"
        ),

        (
            "large_FP",
            "delta_dice"
        ),

        (
            "large_FP_share_percent",
            "delta_dice"
        ),

        (
            "large_area_share_percent",
            "delta_dice"
        ),

        (
            "overall_contamination_percent",
            "delta_dice"
        )
    ]

    correlation_rows = []

    for x_col, y_col in correlation_pairs:

        correlation_rows.append(
            calculate_spearman(
                result,
                x_col,
                y_col
            )
        )

    correlation = pd.DataFrame(
        correlation_rows
    )

    correlation.to_csv(
        OUTPUT_DIR / "korelasi_seluruh_citra.csv",
        index=False
    )

    # ======================================
    # 8. KORELASI KHUSUS CITRA YANG
    #    MEMILIKI KOMPONEN BESAR
    # ======================================

    has_large = result[
        result["has_large_component"]
    ].copy()

    correlation_large_rows = []

    for x_col, y_col in correlation_pairs:

        correlation_large_rows.append(
            calculate_spearman(
                has_large,
                x_col,
                y_col
            )
        )

    correlation_large = pd.DataFrame(
        correlation_large_rows
    )

    correlation_large.to_csv(
        OUTPUT_DIR / "korelasi_citra_dengan_komponen_besar.csv",
        index=False
    )

    # ======================================
    # 9. KASUS DENGAN KONTAMINASI TERTINGGI
    # ======================================

    highest_contamination = (
        has_large.sort_values(
            "large_contamination_percent",
            ascending=False
        )
        .head(15)
    )

    highest_contamination.to_csv(
        OUTPUT_DIR / "15_kasus_kontaminasi_tertinggi.csv",
        index=False
    )

    # ======================================
    # 10. OUTPUT
    # ======================================

    print("\n=== RINGKASAN PER KELOMPOK ===")
    print(
        summary.round(3).to_string(index=False)
    )

    print("\n=== KORELASI SELURUH CITRA ===")
    print(
        correlation.round(5).to_string(index=False)
    )

    print(
        "\nJumlah citra dengan komponen besar:",
        len(has_large)
    )

    print(
        "\n=== KORELASI CITRA DENGAN KOMPONEN BESAR ==="
    )
    print(
        correlation_large.round(5).to_string(
            index=False
        )
    )

    print(
        "\n=== 15 KASUS KONTAMINASI TERTINGGI ==="
    )

    display_columns = [
        "image",
        "kelompok_dice",
        "delta_dice",
        "large_component_count",
        "large_area",
        "large_FP",
        "large_contamination_percent",
        "large_FP_share_percent"
    ]

    print(
        highest_contamination[
            display_columns
        ].round(3).to_string(index=False)
    )

    print("\nAnalisis selesai.")
    print("Folder hasil:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
