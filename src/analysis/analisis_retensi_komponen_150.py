
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVALUATION_FILE = os.path.join(
    BASE_DIR,
    "results",
    "03_evaluation/kernel_terkontrol",
    "hasil_detail.csv"
)

COMPONENT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "05_component_analysis/komponen_150_gray",
    "detail_per_komponen.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "05_component_analysis/retensi_150"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FUNGSI VALIDASI
# ============================================================

def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")


def check_columns(df, required, name):
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Kolom berikut tidak ditemukan pada {name}: {missing}\n"
            f"Kolom tersedia: {list(df.columns)}"
        )


# ============================================================
# MEMBACA DATA
# ============================================================

def load_data():

    check_file(EVALUATION_FILE)
    check_file(COMPONENT_FILE)

    evaluation = pd.read_csv(EVALUATION_FILE)
    components = pd.read_csv(COMPONENT_FILE)

    check_columns(
        evaluation,
        [
            "filename",
            "configuration",
            "tp",
            "fp",
            "fn",
            "dice"
        ],
        "hasil_detail.csv"
    )

    check_columns(
        components,
        [
            "filename",
            "configuration",
            "component_id",
            "component_size",
            "gt_pixels",
            "non_gt_pixels",
            "status"
        ],
        "detail_per_komponen.csv"
    )

    print("=" * 70)
    print("ANALISIS RETENSI KOMPONEN PADA 150 CITRA PENGUJIAN")
    print("=" * 70)

    print(f"Jumlah baris evaluasi : {len(evaluation)}")
    print(f"Jumlah baris komponen : {len(components)}")

    return evaluation, components


# ============================================================
# MENGHITUNG RETENSI KOMPONEN
# ============================================================

def calculate_retention(components):

    retained = components[
        components["status"] == "DIPERTAHANKAN"
    ].copy()

    components = components.copy()

    summary = components.groupby(
        ["filename", "configuration"]
    ).agg(
        total_components=("component_id", "count"),
        total_weak_pixels=("component_size", "sum"),
        total_gt_in_weak=("gt_pixels", "sum"),
        total_non_gt_in_weak=("non_gt_pixels", "sum")
    ).reset_index()

    retained_summary = retained.groupby(
        ["filename", "configuration"]
    ).agg(
        retained_components=("component_id", "count"),
        retained_pixels=("component_size", "sum"),
        retained_gt=("gt_pixels", "sum"),
        retained_non_gt=("non_gt_pixels", "sum")
    ).reset_index()

    summary = summary.merge(
        retained_summary,
        on=["filename", "configuration"],
        how="left",
        validate="one_to_one"
    )

    count_columns = [
        "retained_components",
        "retained_pixels",
        "retained_gt",
        "retained_non_gt"
    ]

    summary[count_columns] = (
        summary[count_columns].fillna(0)
    )

    # Persentase retensi komponen
    summary["component_retention_pct"] = (
        summary["retained_components"]
        / summary["total_components"].replace(0, np.nan)
        * 100
    )

    # Persentase retensi seluruh piksel weak mask
    summary["pixel_retention_pct"] = (
        summary["retained_pixels"]
        / summary["total_weak_pixels"].replace(0, np.nan)
        * 100
    )

    # Persentase GT yang dipertahankan dari GT dalam weak mask
    summary["gt_retention_pct"] = (
        summary["retained_gt"]
        / summary["total_gt_in_weak"].replace(0, np.nan)
        * 100
    )

    # Persentase non-GT yang dipertahankan dari non-GT weak mask
    summary["non_gt_retention_pct"] = (
        summary["retained_non_gt"]
        / summary["total_non_gt_in_weak"].replace(0, np.nan)
        * 100
    )

    return summary


# ============================================================
# MENGHITUNG PERUBAHAN PERFORMA
# ============================================================

def calculate_performance_changes(evaluation):

    metrics = ["tp", "fp", "fn", "dice"]

    pivot = evaluation.pivot(
        index="filename",
        columns="configuration",
        values=metrics
    )

    pivot.columns = [
        f"{metric}_{config}"
        for metric, config in pivot.columns
    ]

    pivot = pivot.reset_index()

    # K9: B - A
    pivot["delta_dice_K9"] = (
        pivot["dice_B"] - pivot["dice_A"]
    )

    pivot["delta_fp_K9"] = (
        pivot["fp_B"] - pivot["fp_A"]
    )

    pivot["delta_fn_K9"] = (
        pivot["fn_B"] - pivot["fn_A"]
    )

    # K11: D - C
    pivot["delta_dice_K11"] = (
        pivot["dice_D"] - pivot["dice_C"]
    )

    pivot["delta_fp_K11"] = (
        pivot["fp_D"] - pivot["fp_C"]
    )

    pivot["delta_fn_K11"] = (
        pivot["fn_D"] - pivot["fn_C"]
    )

    return pivot


# ============================================================
# PIVOT RETENSI
# ============================================================

def create_retention_pivot(retention):

    metrics = [
        "total_components",
        "retained_components",
        "total_weak_pixels",
        "retained_pixels",
        "total_gt_in_weak",
        "retained_gt",
        "total_non_gt_in_weak",
        "retained_non_gt",
        "component_retention_pct",
        "pixel_retention_pct",
        "gt_retention_pct",
        "non_gt_retention_pct"
    ]

    pivot = retention.pivot(
        index="filename",
        columns="configuration",
        values=metrics
    )

    pivot.columns = [
        f"{metric}_{config}"
        for metric, config in pivot.columns
    ]

    return pivot.reset_index()


# ============================================================
# VALIDASI KOMPONEN DAN EVALUASI
# ============================================================

def validate_data(combined):

    print("\nVALIDASI KOMPONEN DAN EVALUASI")
    print("-" * 60)

    checks = {
        "TP K9": (
            combined["tp_B"],
            combined["retained_gt_B"]
        ),
        "FP K9": (
            combined["fp_B"],
            combined["retained_non_gt_B"]
        ),
        "TP K11": (
            combined["tp_D"],
            combined["retained_gt_D"]
        ),
        "FP K11": (
            combined["fp_D"],
            combined["retained_non_gt_D"]
        )
    }

    for name, (actual, expected) in checks.items():

        mismatch = int(
            (actual != expected).sum()
        )

        print(f"{name}: {mismatch}/{len(combined)} berbeda")

        if mismatch > 0:
            raise ValueError(
                f"Validasi {name} gagal."
            )

    print("Seluruh validasi TP dan FP berhasil.")


# ============================================================
# KORELASI SPEARMAN
# ============================================================

def calculate_correlations(combined):

    tests = []

    analyses = [
        (
            "K9",
            "gt_retention_pct_B",
            "delta_dice_K9",
            "Retensi GT",
            "Perubahan Dice"
        ),
        (
            "K9",
            "gt_retention_pct_B",
            "delta_fp_K9",
            "Retensi GT",
            "Perubahan FP"
        ),
        (
            "K9",
            "gt_retention_pct_B",
            "delta_fn_K9",
            "Retensi GT",
            "Perubahan FN"
        ),
        (
            "K9",
            "non_gt_retention_pct_B",
            "delta_dice_K9",
            "Retensi non-GT",
            "Perubahan Dice"
        ),
        (
            "K9",
            "non_gt_retention_pct_B",
            "delta_fp_K9",
            "Retensi non-GT",
            "Perubahan FP"
        ),
        (
            "K9",
            "non_gt_retention_pct_B",
            "delta_fn_K9",
            "Retensi non-GT",
            "Perubahan FN"
        ),
        (
            "K11",
            "gt_retention_pct_D",
            "delta_dice_K11",
            "Retensi GT",
            "Perubahan Dice"
        ),
        (
            "K11",
            "gt_retention_pct_D",
            "delta_fp_K11",
            "Retensi GT",
            "Perubahan FP"
        ),
        (
            "K11",
            "gt_retention_pct_D",
            "delta_fn_K11",
            "Retensi GT",
            "Perubahan FN"
        ),
        (
            "K11",
            "non_gt_retention_pct_D",
            "delta_dice_K11",
            "Retensi non-GT",
            "Perubahan Dice"
        ),
        (
            "K11",
            "non_gt_retention_pct_D",
            "delta_fp_K11",
            "Retensi non-GT",
            "Perubahan FP"
        ),
        (
            "K11",
            "non_gt_retention_pct_D",
            "delta_fn_K11",
            "Retensi non-GT",
            "Perubahan FN"
        )
    ]

    for kernel, x_col, y_col, x_label, y_label in analyses:

        data = combined[[x_col, y_col]].replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        if len(data) < 3:
            rho = np.nan
            p_value = np.nan
        elif (
            data[x_col].nunique() < 2
            or data[y_col].nunique() < 2
        ):
            rho = np.nan
            p_value = np.nan
        else:
            result = spearmanr(
                data[x_col],
                data[y_col]
            )

            rho = result.statistic
            p_value = result.pvalue

        tests.append({
            "kernel": kernel,
            "retention_variable": x_label,
            "performance_variable": y_label,
            "x_column": x_col,
            "y_column": y_col,
            "n": len(data),
            "spearman_rho": rho,
            "p_value": p_value
        })

    results = pd.DataFrame(tests)

    # ========================================================
    # KOREKSI HOLM
    # ========================================================

    valid = results["p_value"].notna()

    p_values = results.loc[
        valid, "p_value"
    ].to_numpy()

    m = len(p_values)

    if m > 0:

        order = np.argsort(p_values)
        sorted_p = p_values[order]

        adjusted = np.maximum.accumulate(
            (m - np.arange(m)) * sorted_p
        )

        adjusted = np.minimum(adjusted, 1.0)

        corrected = np.empty(m)
        corrected[order] = adjusted

        results["holm_p_value"] = np.nan
        results.loc[valid, "holm_p_value"] = corrected

    results["significant_holm_005"] = (
        results["holm_p_value"] < 0.05
    )

    return results


# ============================================================
# RINGKASAN PERUBAHAN PERFORMA
# ============================================================

def performance_summary(combined):

    rows = []

    for kernel in ["K9", "K11"]:

        for metric in ["dice", "fp", "fn"]:

            column = f"delta_{metric}_{kernel}"

            values = combined[column]

            rows.append({
                "kernel": kernel,
                "metric": metric,
                "mean_change": values.mean(),
                "median_change": values.median(),
                "increase": int((values > 0).sum()),
                "decrease": int((values < 0).sum()),
                "unchanged": int((values == 0).sum())
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    evaluation, components = load_data()

    retention = calculate_retention(components)

    performance = calculate_performance_changes(
        evaluation
    )

    retention_pivot = create_retention_pivot(
        retention
    )

    combined = performance.merge(
        retention_pivot,
        on="filename",
        how="inner",
        validate="one_to_one"
    )

    if len(combined) != 150:
        raise ValueError(
            f"Jumlah citra gabungan seharusnya 150, "
            f"tetapi ditemukan {len(combined)}."
        )

    validate_data(combined)

    correlations = calculate_correlations(
        combined
    )

    summary = performance_summary(
        combined
    )

    # ========================================================
    # SIMPAN HASIL
    # ========================================================

    retention.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "retensi_komponen_per_citra.csv"
        ),
        index=False
    )

    combined.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "data_gabungan_150_citra.csv"
        ),
        index=False
    )

    correlations.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "korelasi_retensi_performa_150.csv"
        ),
        index=False
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ringkasan_perubahan_performa_150.csv"
        ),
        index=False
    )

    # ========================================================
    # TAMPILKAN HASIL
    # ========================================================

    print("\n" + "=" * 70)
    print("A. RINGKASAN PERUBAHAN PERFORMA")
    print("=" * 70)

    print(
        summary.round(5).to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("B. KORELASI RETENSI KOMPONEN DAN PERFORMA")
    print("=" * 70)

    display_columns = [
        "kernel",
        "retention_variable",
        "performance_variable",
        "n",
        "spearman_rho",
        "p_value",
        "holm_p_value",
        "significant_holm_005"
    ]

    print(
        correlations[display_columns]
        .round(6)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("ANALISIS SELESAI")
    print("=" * 70)

    print(f"Jumlah citra dianalisis: {len(combined)}")
    print(f"Lokasi hasil: {OUTPUT_DIR}")

    print("- retensi_komponen_per_citra.csv")
    print("- data_gabungan_150_citra.csv")
    print("- korelasi_retensi_performa_150.csv")
    print("- ringkasan_perubahan_performa_150.csv")


if __name__ == "__main__":
    main()
