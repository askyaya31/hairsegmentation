
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "results" / "analisis_kasus"
    / "analisis_weak_strong_testing"
    / "detail_weak_strong_150.csv"
)

OUTPUT_DIR = INPUT_FILE.parent


# ==========================================
# 1. MEMBACA DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

# Menghitung kontribusi weak-only
df["weak_only_TP"] = (
    df["retained_GT_pixels"] - df["strong_TP"]
)

df["weak_only_FP"] = (
    df["retained_FP"] - df["strong_FP"]
)

# Validasi
if (df["weak_only_TP"] < 0).any():
    raise ValueError("Weak-only TP negatif.")

if (df["weak_only_FP"] < 0).any():
    raise ValueError("Weak-only FP negatif.")


# ==========================================
# 2. MEMBENTUK VARIABEL
# ==========================================

df["strong_precision"] = np.where(
    (df["strong_TP"] + df["strong_FP"]) > 0,
    df["strong_TP"] /
    (df["strong_TP"] + df["strong_FP"]),
    np.nan
)

df["weak_only_precision"] = np.where(
    (df["weak_only_TP"] + df["weak_only_FP"]) > 0,
    df["weak_only_TP"] /
    (df["weak_only_TP"] + df["weak_only_FP"]),
    np.nan
)

df["weak_only_FP_share"] = np.where(
    df["retained_FP"] > 0,
    df["weak_only_FP"] / df["retained_FP"],
    np.nan
)

# Pastikan jumlah strong pixels konsisten
df["strong_pixels_check"] = (
    df["strong_TP"] + df["strong_FP"]
)

if not np.array_equal(
    df["strong_pixels"].to_numpy(),
    df["strong_pixels_check"].to_numpy()
):
    raise ValueError("Jumlah strong pixels tidak konsisten.")


# ==========================================
# 3. PASANGAN KORELASI
# ==========================================

correlation_pairs = [
    (
        "strong_pixels",
        "weak_only_TP",
        "Strong pixels vs Weak-only TP"
    ),
    (
        "strong_pixels",
        "weak_only_FP",
        "Strong pixels vs Weak-only FP"
    ),
    (
        "strong_precision",
        "weak_only_precision",
        "Strong precision vs Weak-only precision"
    ),
    (
        "strong_pixels",
        "weak_only_FP_share",
        "Strong pixels vs Proporsi Weak-only FP"
    )
]


# ==========================================
# 4. ANALISIS SPEARMAN
# ==========================================

groups = [
    "SELURUH TESTING",
    "DICE MENINGKAT",
    "DICE MENURUN",
    "DICE TETAP"
]

rows = []

for group in groups:

    if group == "SELURUH TESTING":
        subset = df.copy()
    else:
        subset = df[
            df["kelompok_dice"] == group
        ]

    for x_col, y_col, label in correlation_pairs:

        valid = subset[
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

        rows.append({
            "kelompok": group,
            "hubungan": label,
            "jumlah_citra": len(valid),
            "spearman_rho": rho,
            "p_value": p_value
        })


# ==========================================
# 5. SIMPAN HASIL
# ==========================================

summary = pd.DataFrame(rows)

summary.to_csv(
    OUTPUT_DIR / "analisis_korelasi_spearman.csv",
    index=False
)


# ==========================================
# 6. OUTPUT
# ==========================================

print("=== KORELASI SPEARMAN ===")

print(
    summary.round(5).to_string(index=False)
)

print("\n=== RATA-RATA VARIABEL ===")

print(
    df.groupby("kelompok_dice")[
        [
            "strong_pixels",
            "weak_only_TP",
            "weak_only_FP",
            "strong_precision",
            "weak_only_precision",
            "weak_only_FP_share"
        ]
    ].mean().round(4).to_string()
)

print("\nHasil disimpan:")
print(
    OUTPUT_DIR / "analisis_korelasi_spearman.csv"
)
