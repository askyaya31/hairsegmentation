
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "05_component_analysis/retensi_150",
    "data_gabungan_150_citra.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "07_statistical_analysis/korelasi_150"
)

PLOT_DIR = os.path.join(
    OUTPUT_DIR,
    "scatter_plots"
)

os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# DAFTAR 12 ANALISIS
# ============================================================

ANALYSES = [
    ("K9", "gt_retention_pct_B", "delta_dice_K9",
     "Retensi GT", "Perubahan Dice"),

    ("K9", "gt_retention_pct_B", "delta_fp_K9",
     "Retensi GT", "Perubahan FP"),

    ("K9", "gt_retention_pct_B", "delta_fn_K9",
     "Retensi GT", "Perubahan FN"),

    ("K9", "non_gt_retention_pct_B", "delta_dice_K9",
     "Retensi non-GT", "Perubahan Dice"),

    ("K9", "non_gt_retention_pct_B", "delta_fp_K9",
     "Retensi non-GT", "Perubahan FP"),

    ("K9", "non_gt_retention_pct_B", "delta_fn_K9",
     "Retensi non-GT", "Perubahan FN"),

    ("K11", "gt_retention_pct_D", "delta_dice_K11",
     "Retensi GT", "Perubahan Dice"),

    ("K11", "gt_retention_pct_D", "delta_fp_K11",
     "Retensi GT", "Perubahan FP"),

    ("K11", "gt_retention_pct_D", "delta_fn_K11",
     "Retensi GT", "Perubahan FN"),

    ("K11", "non_gt_retention_pct_D", "delta_dice_K11",
     "Retensi non-GT", "Perubahan Dice"),

    ("K11", "non_gt_retention_pct_D", "delta_fp_K11",
     "Retensi non-GT", "Perubahan FP"),

    ("K11", "non_gt_retention_pct_D", "delta_fn_K11",
     "Retensi non-GT", "Perubahan FN")
]


# ============================================================
# FUNGSI DASAR
# ============================================================

def spearman_result(x, y):

    if len(x) < 3:
        return np.nan, np.nan

    if pd.Series(x).nunique() < 2:
        return np.nan, np.nan

    if pd.Series(y).nunique() < 2:
        return np.nan, np.nan

    result = spearmanr(x, y)

    return result.statistic, result.pvalue


def holm_correction(p_values):

    p_values = np.asarray(p_values, dtype=float)

    adjusted = np.full(len(p_values), np.nan)

    valid_indices = np.where(
        np.isfinite(p_values)
    )[0]

    if len(valid_indices) == 0:
        return adjusted

    valid_p = p_values[valid_indices]
    order = np.argsort(valid_p)

    sorted_p = valid_p[order]
    m = len(sorted_p)

    sorted_adjusted = np.maximum.accumulate(
        (m - np.arange(m)) * sorted_p
    )

    sorted_adjusted = np.minimum(
        sorted_adjusted, 1.0
    )

    adjusted_valid = np.empty(m)
    adjusted_valid[order] = sorted_adjusted

    adjusted[valid_indices] = adjusted_valid

    return adjusted


def iqr_bounds(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return lower, upper


# ============================================================
# PEMERIKSAAN SETIAP HUBUNGAN
# ============================================================

def inspect_relationship(data, analysis):

    kernel, x_col, y_col, x_label, y_label = analysis

    subset = data[
        ["filename", x_col, y_col]
    ].copy()

    subset = subset.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    subset = subset.reset_index(drop=True)

    x = subset[x_col]
    y = subset[y_col]

    rho, p_value = spearman_result(x, y)

    # --------------------------------------------------------
    # IDENTIFIKASI PENCIILAN DENGAN IQR
    # --------------------------------------------------------

    x_lower, x_upper = iqr_bounds(x)
    y_lower, y_upper = iqr_bounds(y)

    subset["x_outlier_iqr"] = (
        (x < x_lower) | (x > x_upper)
    )

    subset["y_outlier_iqr"] = (
        (y < y_lower) | (y > y_upper)
    )

    subset["iqr_outlier"] = (
        subset["x_outlier_iqr"]
        | subset["y_outlier_iqr"]
    )

    # --------------------------------------------------------
    # LEAVE-ONE-OUT SPEARMAN
    # --------------------------------------------------------

    loo_results = []

    for i in range(len(subset)):

        reduced = subset.drop(index=i)

        loo_rho, loo_p = spearman_result(
            reduced[x_col],
            reduced[y_col]
        )

        if np.isfinite(loo_rho) and np.isfinite(rho):
            rho_change = loo_rho - rho
        else:
            rho_change = np.nan

        loo_results.append({
            "filename": subset.loc[i, "filename"],
            "rho_without_image": loo_rho,
            "p_without_image": loo_p,
            "rho_change": rho_change,
            "absolute_rho_change": abs(rho_change)
            if np.isfinite(rho_change) else np.nan,
            "x_value": subset.loc[i, x_col],
            "y_value": subset.loc[i, y_col],
            "x_outlier_iqr": subset.loc[i, "x_outlier_iqr"],
            "y_outlier_iqr": subset.loc[i, "y_outlier_iqr"],
            "iqr_outlier": subset.loc[i, "iqr_outlier"]
        })

    loo_df = pd.DataFrame(loo_results)

    # --------------------------------------------------------
    # RINGKASAN
    # --------------------------------------------------------

    if len(loo_df):
        max_influence = loo_df.loc[
            loo_df["absolute_rho_change"].idxmax()
        ]
    else:
        max_influence = None

    summary = {
        "kernel": kernel,
        "retention_variable": x_label,
        "performance_variable": y_label,
        "x_column": x_col,
        "y_column": y_col,
        "n": len(subset),
        "spearman_rho": rho,
        "p_value": p_value,
        "iqr_outlier_count": int(
            subset["iqr_outlier"].sum()
        ),
        "max_absolute_loo_change": (
            max_influence["absolute_rho_change"]
            if max_influence is not None else np.nan
        ),
        "most_influential_image": (
            max_influence["filename"]
            if max_influence is not None else None
        ),
        "rho_without_most_influential": (
            max_influence["rho_without_image"]
            if max_influence is not None else np.nan
        )
    }

    # --------------------------------------------------------
    # SCATTER PLOT
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    normal = subset[~subset["iqr_outlier"]]
    outliers = subset[subset["iqr_outlier"]]

    ax.scatter(
        normal[x_col],
        normal[y_col],
        alpha=0.65,
        label="Dalam batas IQR"
    )

    ax.scatter(
        outliers[x_col],
        outliers[y_col],
        marker="x",
        s=65,
        label="Pencilan IQR"
    )

    # Beri label pada titik pencilan
    for _, row in outliers.iterrows():

        ax.annotate(
            row["filename"].replace(".png", ""),
            (row[x_col], row[y_col]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points"
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_title(
        f"{kernel}: {x_label} vs {y_label}\n"
        f"Spearman rho={rho:.4f}, "
        f"p={p_value:.3e}, n={len(subset)}"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    filename = (
        f"{kernel}_"
        f"{x_col}_vs_{y_col}.png"
    )

    fig.savefig(
        os.path.join(PLOT_DIR, filename),
        dpi=200
    )

    plt.close(fig)

    # Tambahkan identitas analisis
    loo_df.insert(0, "kernel", kernel)
    loo_df.insert(1, "x_column", x_col)
    loo_df.insert(2, "y_column", y_col)

    return summary, loo_df


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print("PEMERIKSAAN KORELASI PADA 150 CITRA")
    print("=" * 70)

    print(f"Jumlah citra awal: {len(data)}")

    if len(data) != 150:
        raise ValueError(
            "Jumlah data bukan 150 citra."
        )

    summaries = []
    all_loo = []

    for analysis in ANALYSES:

        summary, loo = inspect_relationship(
            data, analysis
        )

        summaries.append(summary)
        all_loo.append(loo)

    summary_df = pd.DataFrame(summaries)

    # Koreksi Holm dihitung kembali untuk 12 pengujian
    summary_df["holm_p_value"] = holm_correction(
        summary_df["p_value"].to_numpy()
    )

    summary_df["significant_holm_005"] = (
        summary_df["holm_p_value"] < 0.05
    )

    loo_df = pd.concat(
        all_loo,
        ignore_index=True
    )

    # --------------------------------------------------------
    # SIMPAN HASIL
    # --------------------------------------------------------

    summary_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ringkasan_pemeriksaan_korelasi.csv"
        ),
        index=False
    )

    loo_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "hasil_leave_one_out.csv"
        ),
        index=False
    )

    outliers = loo_df[
        loo_df["iqr_outlier"]
    ].copy()

    outliers.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "daftar_pencilan_iqr.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # TAMPILKAN RINGKASAN
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("A. RINGKASAN KORELASI DAN PEMERIKSAAN PENCIILAN")
    print("=" * 70)

    display_columns = [
        "kernel",
        "retention_variable",
        "performance_variable",
        "n",
        "spearman_rho",
        "p_value",
        "holm_p_value",
        "iqr_outlier_count",
        "max_absolute_loo_change",
        "most_influential_image",
        "rho_without_most_influential"
    ]

    print(
        summary_df[display_columns]
        .to_string(
            index=False,
            formatters={
                "spearman_rho": lambda x: f"{x:.5f}",
                "p_value": lambda x: f"{x:.3e}",
                "holm_p_value": lambda x: f"{x:.3e}",
                "max_absolute_loo_change": lambda x: f"{x:.5f}",
                "rho_without_most_influential": lambda x: f"{x:.5f}"
            }
        )
    )

    print("\n" + "=" * 70)
    print("B. DAFTAR PENCIILAN IQR")
    print("=" * 70)

    if outliers.empty:
        print("Tidak ditemukan pencilan berdasarkan aturan IQR.")
    else:
        print(
            outliers[
                [
                    "kernel",
                    "x_column",
                    "y_column",
                    "filename",
                    "x_value",
                    "y_value",
                    "absolute_rho_change"
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("PEMERIKSAAN SELESAI")
    print("=" * 70)

    print(f"Jumlah analisis korelasi: {len(summary_df)}")
    print(f"Jumlah scatter plot: {len(ANALYSES)}")
    print(f"Jumlah baris leave-one-out: {len(loo_df)}")
    print(f"Jumlah temuan pencilan: {len(outliers)}")
    print(f"Lokasi hasil: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
