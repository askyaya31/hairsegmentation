import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import spearmanr


# ============================================================
# KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

EVAL_FILE = (
    ROOT / "results" /
    "evaluasi_kernel_terkontrol" /
    "hasil_detail.csv"
)

COMPONENT_FILE = (
    ROOT / "results" /
    "analisis_komponen_150_testing_gray_converted" /
    "detail_per_komponen.csv"
)

COMPONENT_SUMMARY_FILE = (
    ROOT / "results" /
    "analisis_komponen_150_testing_gray_converted" /
    "ringkasan_per_citra.csv"
)

OUTPUT_DIR = (
    ROOT / "results" /
    "analisis_gabungan_komponen_dice_gray_converted"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CITRA DIAGNOSTIK
# ============================================================

SELECTED_IMAGES = [
    "ISIC_0000456.png",
    "ISIC_0009945.png",
    "ISIC_0014806.png",
    "ISIC_0014968.png",
    "ISIC_0015418.png",
    "ISIC_0016019.png"
]


# ============================================================
# FUNGSI VALIDASI
# ============================================================

def validate_files():

    files = [
        EVAL_FILE,
        COMPONENT_FILE,
        COMPONENT_SUMMARY_FILE
    ]

    for file_path in files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"File tidak ditemukan: {file_path}"
            )


def validate_unique(df, columns, description):

    duplicated = df.duplicated(columns)

    if duplicated.any():
        raise ValueError(
            f"Ditemukan {duplicated.sum()} duplikasi "
            f"pada {description}: {columns}"
        )


# ============================================================
# MEMBACA DATA
# ============================================================

def load_data():

    evaluation = pd.read_csv(EVAL_FILE)
    components = pd.read_csv(COMPONENT_FILE)
    component_reference = pd.read_csv(COMPONENT_SUMMARY_FILE)

    print("=" * 70)
    print("ANALISIS GABUNGAN KOMPONEN DAN HASIL EVALUASI")
    print("=" * 70)

    print("Jumlah baris evaluasi:", len(evaluation))
    print("Jumlah baris detail komponen:", len(components))
    print("Jumlah baris ringkasan komponen:", len(component_reference))

    return evaluation, components, component_reference


# ============================================================
# FILTER ENAM CITRA
# ============================================================

def filter_selected_images(
    evaluation,
    components,
    component_reference
):

    evaluation_6 = evaluation[
        evaluation["filename"].isin(SELECTED_IMAGES)
    ].copy()

    components_6 = components[
        components["filename"].isin(SELECTED_IMAGES)
    ].copy()

    reference_6 = component_reference[
        component_reference["filename"].isin(SELECTED_IMAGES)
    ].copy()

    # Validasi jumlah citra
    for name, df in [
        ("evaluasi", evaluation_6),
        ("detail komponen", components_6),
        ("ringkasan komponen", reference_6)
    ]:

        found = set(df["filename"].unique())
        missing = set(SELECTED_IMAGES) - found

        if missing:
            raise ValueError(
                f"Citra pada {name} tidak lengkap: {missing}"
            )

    # Validasi jumlah konfigurasi
    validate_unique(
        evaluation_6,
        ["filename", "configuration"],
        "evaluasi"
    )

    validate_unique(
        components_6,
        ["filename", "configuration", "component_id"],
        "detail komponen"
    )

    validate_unique(
        reference_6,
        ["filename", "configuration"],
        "ringkasan komponen"
    )

    print("\nEnam citra berhasil ditemukan pada ketiga data.")

    print(
        "Jumlah baris evaluasi terpilih:",
        len(evaluation_6)
    )

    print(
        "Jumlah baris komponen terpilih:",
        len(components_6)
    )

    return evaluation_6, components_6, reference_6


# ============================================================
# 1. RINGKASAN KOMPONEN PER CITRA
# ============================================================

def calculate_component_summary(components_6):

    component_rows = []

    for (filename, config), group in components_6.groupby(
        ["filename", "configuration"]
    ):

        retained = group[
            group["status"] == "DIPERTAHANKAN"
        ]

        total_components = len(group)
        retained_components = len(retained)

        total_gt = int(group["gt_pixels"].sum())
        total_non_gt = int(group["non_gt_pixels"].sum())

        retained_gt = int(retained["gt_pixels"].sum())
        retained_non_gt = int(retained["non_gt_pixels"].sum())

        total_weak = int(group["component_size"].sum())
        retained_pixels = int(retained["component_size"].sum())

        component_rows.append({

            "filename": filename,
            "configuration": config,

            "total_components": total_components,
            "retained_components": retained_components,

            "component_retention_pct":
                100 * retained_components / total_components
                if total_components else np.nan,

            "total_weak_pixels": total_weak,
            "retained_pixels": retained_pixels,

            "pixel_retention_pct":
                100 * retained_pixels / total_weak
                if total_weak else np.nan,

            "total_gt_in_weak": total_gt,
            "retained_gt": retained_gt,

            "gt_retention_pct":
                100 * retained_gt / total_gt
                if total_gt else np.nan,

            "total_non_gt_in_weak": total_non_gt,
            "retained_non_gt": retained_non_gt,

            "non_gt_retention_pct":
                100 * retained_non_gt / total_non_gt
                if total_non_gt else np.nan
        })

    return pd.DataFrame(component_rows)


# ============================================================
# 2. VALIDASI RINGKASAN KOMPONEN
# ============================================================

def validate_component_summary(
    component_summary,
    reference_6
):

    numeric_columns = [
        "total_components",
        "retained_components",
        "total_weak_pixels",
        "retained_pixels",
        "total_gt_in_weak",
        "retained_gt",
        "total_non_gt_in_weak",
        "retained_non_gt"
    ]

    comparison = component_summary.merge(
        reference_6,
        on=["filename", "configuration"],
        how="outer",
        suffixes=("_calculated", "_reference"),
        validate="one_to_one",
        indicator=True
    )

    if not (comparison["_merge"] == "both").all():
        raise ValueError(
            "Pasangan data ringkasan komponen tidak lengkap."
        )

    mismatch_summary = {}

    for column in numeric_columns:

        calculated = comparison[
            f"{column}_calculated"
        ]

        reference = comparison[
            f"{column}_reference"
        ]

        mismatch_summary[column] = int(
            (calculated != reference).sum()
        )

    print("\nVALIDASI RINGKASAN KOMPONEN")
    print("-" * 55)

    for column, count in mismatch_summary.items():
        print(f"{column}: {count} berbeda")

    if any(mismatch_summary.values()):
        raise ValueError(
            "Ringkasan komponen hasil perhitungan "
            "tidak cocok dengan CSV referensi."
        )

    print("Seluruh ringkasan komponen cocok.")

    return comparison


# ============================================================
# 3. PIVOT HASIL EVALUASI
# ============================================================

def create_evaluation_pivot(evaluation_6):

    metrics = [
        "precision",
        "recall",
        "dice",
        "iou",
        "tp",
        "fp",
        "fn"
    ]

    evaluation_pivot = evaluation_6.pivot(
        index="filename",
        columns="configuration",
        values=metrics
    )

    evaluation_pivot.columns = [
        f"{metric}_{config}"
        for metric, config in evaluation_pivot.columns
    ]

    return evaluation_pivot.reset_index(), metrics


# ============================================================
# 4. PIVOT KARAKTERISTIK KOMPONEN
# ============================================================

def create_component_pivot(component_summary):

    component_metrics = [
        "total_components",
        "retained_components",
        "component_retention_pct",
        "total_weak_pixels",
        "retained_pixels",
        "pixel_retention_pct",
        "total_gt_in_weak",
        "retained_gt",
        "gt_retention_pct",
        "total_non_gt_in_weak",
        "retained_non_gt",
        "non_gt_retention_pct"
    ]

    component_pivot = component_summary.pivot(
        index="filename",
        columns="configuration",
        values=component_metrics
    )

    component_pivot.columns = [
        f"{metric}_{config}"
        for metric, config in component_pivot.columns
    ]

    return component_pivot.reset_index()


# ============================================================
# 5. GABUNGKAN EVALUASI DAN KOMPONEN
# ============================================================

def merge_analysis(
    evaluation_pivot,
    component_pivot
):

    combined = evaluation_pivot.merge(
        component_pivot,
        on="filename",
        how="inner",
        validate="one_to_one"
    )

    if len(combined) != len(SELECTED_IMAGES):
        raise ValueError(
            f"Jumlah citra gabungan tidak sesuai: {len(combined)}"
        )

    return combined


# ============================================================
# 6. VALIDASI TP, FP, FN DENGAN KOMPONEN
# ============================================================

def validate_performance_components(combined):

    # Total GT = TP + FN dari evaluasi utama.
    # FN = total GT - retained GT.

    checks = {
        "TP_K9": (
            combined["tp_B"],
            combined["retained_gt_B"]
        ),

        "FP_K9": (
            combined["fp_B"],
            combined["retained_non_gt_B"]
        ),

        "FN_K9": (
            combined["fn_B"],
            combined["tp_B"] + combined["fn_B"]
            - combined["retained_gt_B"]
        ),

        "TP_K11": (
            combined["tp_D"],
            combined["retained_gt_D"]
        ),

        "FP_K11": (
            combined["fp_D"],
            combined["retained_non_gt_D"]
        ),

        "FN_K11": (
            combined["fn_D"],
            combined["tp_D"] + combined["fn_D"]
            - combined["retained_gt_D"]
        )
    }

    print("\nVALIDASI EVALUASI DAN KOMPONEN")
    print("-" * 55)

    for name, (actual, expected) in checks.items():

        mismatch = int(
            (actual != expected).sum()
        )

        print(
            f"{name}: {mismatch}/{len(combined)} berbeda"
        )

        if mismatch:
            raise ValueError(
                f"Validasi {name} gagal."
            )

    print("TP, FP, dan FN seluruhnya konsisten.")



# ============================================================
# 7. HITUNG PERUBAHAN BASELINE KE HYSTERESIS
# ============================================================

def calculate_performance_changes(combined, metrics):

    # K9: B dibandingkan A
    # K11: D dibandingkan C

    for metric in metrics:

        combined[f"delta_{metric}_K9"] = (
            combined[f"{metric}_B"] -
            combined[f"{metric}_A"]
        )

        combined[f"delta_{metric}_K11"] = (
            combined[f"{metric}_D"] -
            combined[f"{metric}_C"]
        )

    combined["delta_fp_pct_K9"] = (
        combined["delta_fp_K9"] /
        combined["fp_A"].replace(0, np.nan) * 100
    )

    combined["delta_fp_pct_K11"] = (
        combined["delta_fp_K11"] /
        combined["fp_C"].replace(0, np.nan) * 100
    )

    return combined


# ============================================================
# 8. KORELASI RETENSI DENGAN PERUBAHAN PERFORMA
# ============================================================

def calculate_correlations(combined):

    correlation_pairs = [
        (
            "non_gt_retention_pct_B",
            "delta_fp_K9",
            "K9: Retensi non-GT vs perubahan FP"
        ),
        (
            "non_gt_retention_pct_D",
            "delta_fp_K11",
            "K11: Retensi non-GT vs perubahan FP"
        ),
        (
            "non_gt_retention_pct_B",
            "delta_dice_K9",
            "K9: Retensi non-GT vs perubahan Dice"
        ),
        (
            "non_gt_retention_pct_D",
            "delta_dice_K11",
            "K11: Retensi non-GT vs perubahan Dice"
        ),
        (
            "gt_retention_pct_B",
            "delta_dice_K9",
            "K9: Retensi GT vs perubahan Dice"
        ),
        (
            "gt_retention_pct_D",
            "delta_dice_K11",
            "K11: Retensi GT vs perubahan Dice"
        )
    ]

    correlation_rows = []

    for x_col, y_col, description in correlation_pairs:

        valid = combined[[x_col, y_col]].dropna()

        if len(valid) < 3:
            continue

        rho, p_value = spearmanr(
            valid[x_col],
            valid[y_col]
        )

        correlation_rows.append({

            "analysis": description,
            "x": x_col,
            "y": y_col,
            "n": len(valid),
            "spearman_rho": rho,
            "p_value": p_value
        })

    return pd.DataFrame(correlation_rows)


# ============================================================
# 9. RINGKASAN PERUBAHAN PERFORMA
# ============================================================

def calculate_performance_summary(combined):

    performance_rows = []

    for config_name in ["K9", "K11"]:

        for metric in [
            "dice",
            "precision",
            "recall",
            "iou",
            "fp",
            "fn"
        ]:

            column = f"delta_{metric}_{config_name}"

            values = combined[column]

            performance_rows.append({

                "configuration": config_name,
                "metric": metric,
                "mean_change": values.mean(),
                "median_change": values.median(),
                "increase": int((values > 0).sum()),
                "decrease": int((values < 0).sum()),
                "unchanged": int((values == 0).sum())
            })

    return pd.DataFrame(performance_rows)


# ============================================================
# 10. SIMPAN HASIL
# ============================================================

def save_results(
    combined,
    component_summary,
    correlation_df,
    performance_summary,
    validation_comparison
):

    combined.to_csv(
        OUTPUT_DIR / "data_gabungan_enam_citra.csv",
        index=False
    )

    component_summary.to_csv(
        OUTPUT_DIR / "ringkasan_komponen_enam_citra.csv",
        index=False
    )

    correlation_df.to_csv(
        OUTPUT_DIR / "korelasi_retensi_performa.csv",
        index=False
    )

    performance_summary.to_csv(
        OUTPUT_DIR / "ringkasan_perubahan_performa.csv",
        index=False
    )

    validation_comparison.to_csv(
        OUTPUT_DIR / "validasi_ringkasan_komponen.csv",
        index=False
    )


# ============================================================
# 11. TAMPILKAN HASIL
# ============================================================

def display_results(
    combined,
    component_summary,
    correlation_df,
    performance_summary
):

    print("\n" + "=" * 70)
    print("A. PERUBAHAN PERFORMA BASELINE KE HYSTERESIS")
    print("=" * 70)

    print(
        performance_summary.round(5).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("B. RETENSI KOMPONEN PER CITRA")
    print("=" * 70)

    print(
        component_summary[
            [
                "filename",
                "configuration",
                "component_retention_pct",
                "pixel_retention_pct",
                "gt_retention_pct",
                "non_gt_retention_pct"
            ]
        ].round(3).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("C. KORELASI RETENSI DAN PERFORMA")
    print("=" * 70)

    print(
        correlation_df.round(5).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("D. DATA GABUNGAN")
    print("=" * 70)

    print(
        combined[
            [
                "filename",
                "dice_A",
                "dice_B",
                "delta_dice_K9",
                "dice_C",
                "dice_D",
                "delta_dice_K11",
                "delta_fp_K9",
                "delta_fp_K11"
            ]
        ].round(5).to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    validate_files()

    evaluation, components, component_reference = load_data()

    (
        evaluation_6,
        components_6,
        reference_6
    ) = filter_selected_images(
        evaluation,
        components,
        component_reference
    )

    component_summary = calculate_component_summary(
        components_6
    )

    validation_comparison = validate_component_summary(
        component_summary,
        reference_6
    )

    evaluation_pivot, metrics = create_evaluation_pivot(
        evaluation_6
    )

    component_pivot = create_component_pivot(
        component_summary
    )

    combined = merge_analysis(
        evaluation_pivot,
        component_pivot
    )

    validate_performance_components(combined)

    combined = calculate_performance_changes(
        combined,
        metrics
    )

    correlation_df = calculate_correlations(
        combined
    )

    performance_summary = calculate_performance_summary(
        combined
    )

    save_results(
        combined,
        component_summary,
        correlation_df,
        performance_summary,
        validation_comparison
    )

    display_results(
        combined,
        component_summary,
        correlation_df,
        performance_summary
    )

    print("\n" + "=" * 70)
    print("ANALISIS SELESAI")
    print("=" * 70)

    print("Lokasi hasil:", OUTPUT_DIR)

    for path in sorted(OUTPUT_DIR.iterdir()):
        print("-", path.name)


if __name__ == "__main__":
    main()
