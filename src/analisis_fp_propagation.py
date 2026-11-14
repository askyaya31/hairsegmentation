
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "results" / "analisis_kasus"
    / "analisis_weak_strong_testing"
    / "detail_weak_strong_150.csv"
)

OUTPUT_DIR = INPUT_FILE.parent

df = pd.read_csv(INPUT_FILE)

# FP yang dipertahankan melalui weak-only pixels
df["weak_only_FP"] = (
    df["retained_FP"] - df["strong_FP"]
)

# Validasi
if (df["weak_only_FP"] < 0).any():
    raise ValueError("Ditemukan weak-only FP negatif.")

df["strong_precision"] = np.where(
    (df["strong_TP"] + df["strong_FP"]) > 0,
    df["strong_TP"] /
    (df["strong_TP"] + df["strong_FP"]),
    np.nan
)

groups = [
    "DICE MENINGKAT",
    "DICE MENURUN",
    "DICE TETAP"
]

rows = []

for group in groups:
    subset = df[df["kelompok_dice"] == group]

    total_strong_tp = subset["strong_TP"].sum()
    total_strong_fp = subset["strong_FP"].sum()
    total_retained_fp = subset["retained_FP"].sum()
    total_weak_fp = subset["weak_only_FP"].sum()

    total_strong = total_strong_tp + total_strong_fp

    rows.append({
        "kelompok_dice": group,
        "jumlah_citra": len(subset),

        "total_strong_TP": total_strong_tp,
        "total_strong_FP": total_strong_fp,
        "total_retained_FP": total_retained_fp,
        "total_weak_only_FP": total_weak_fp,

        "mean_strong_FP": subset["strong_FP"].mean(),
        "mean_weak_only_FP": subset["weak_only_FP"].mean(),

        "strong_precision_percent": (
            total_strong_tp / total_strong * 100
            if total_strong > 0 else np.nan
        ),

        "weak_only_FP_percent": (
            total_weak_fp / total_retained_fp * 100
            if total_retained_fp > 0 else np.nan
        )
    })

summary = pd.DataFrame(rows)

summary.to_csv(
    OUTPUT_DIR / "analisis_fp_propagation.csv",
    index=False
)

print("=== ANALISIS FALSE POSITIVE ===")
print(summary.round(2).to_string(index=False))

print("\n=== KESELURUHAN TESTING ===")

total_retained_fp = df["retained_FP"].sum()
total_weak_fp = df["weak_only_FP"].sum()
total_strong_tp = df["strong_TP"].sum()
total_strong_fp = df["strong_FP"].sum()

print("Total retained FP:", total_retained_fp)
print("Total strong FP:", total_strong_fp)
print("Total weak-only FP:", total_weak_fp)

print(
    "Persentase weak-only FP:",
    round(total_weak_fp / total_retained_fp * 100, 2),
    "%"
)

print(
    "Precision strong:",
    round(
        total_strong_tp /
        (total_strong_tp + total_strong_fp) * 100,
        2
    ),
    "%"
)

print("\nAnalisis selesai.")
print("Hasil:", OUTPUT_DIR / "analisis_fp_propagation.csv")
