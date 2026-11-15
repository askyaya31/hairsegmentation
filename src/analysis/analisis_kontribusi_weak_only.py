
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "results" / "analisis_kasus"
    / "analisis_weak_strong_testing"
    / "detail_weak_strong_150.csv"
)

OUTPUT_DIR = INPUT_FILE.parent

df = pd.read_csv(INPUT_FILE)

# Kontribusi weak-only terhadap TP dan FP
df["weak_only_TP"] = (
    df["retained_GT_pixels"] - df["strong_TP"]
)

df["weak_only_FP"] = (
    df["retained_FP"] - df["strong_FP"]
)

if (df["weak_only_TP"] < 0).any():
    raise ValueError("Ditemukan weak-only TP negatif.")

if (df["weak_only_FP"] < 0).any():
    raise ValueError("Ditemukan weak-only FP negatif.")

groups = [
    "DICE MENINGKAT",
    "DICE MENURUN",
    "DICE TETAP"
]

rows = []

for group in groups:
    subset = df[df["kelompok_dice"] == group]

    weak_tp = int(subset["weak_only_TP"].sum())
    weak_fp = int(subset["weak_only_FP"].sum())
    strong_tp = int(subset["strong_TP"].sum())
    strong_fp = int(subset["strong_FP"].sum())

    weak_total = weak_tp + weak_fp

    rows.append({
        "kelompok_dice": group,
        "jumlah_citra": len(subset),

        "weak_only_TP": weak_tp,
        "weak_only_FP": weak_fp,
        "weak_only_total": weak_total,

        "strong_TP": strong_tp,
        "strong_FP": strong_fp,

        "weak_only_precision": (
            weak_tp / weak_total * 100
            if weak_total > 0 else np.nan
        ),

        "weak_only_TP_share_percent": (
            weak_tp / (weak_tp + strong_tp) * 100
            if weak_tp + strong_tp > 0 else np.nan
        )
    })

summary = pd.DataFrame(rows)

summary.to_csv(
    OUTPUT_DIR / "analisis_kontribusi_weak_only.csv",
    index=False
)

print("=== KONTRIBUSI WEAK-ONLY ===")
print(summary.round(2).to_string(index=False))

print("\n=== KESELURUHAN TESTING ===")

total_weak_tp = df["weak_only_TP"].sum()
total_weak_fp = df["weak_only_FP"].sum()
total_weak = total_weak_tp + total_weak_fp

print("Weak-only TP:", int(total_weak_tp))
print("Weak-only FP:", int(total_weak_fp))
print("Total weak-only:", int(total_weak))

print(
    "Precision weak-only:",
    round(total_weak_tp / total_weak * 100, 2),
    "%"
)

print("\nAnalisis selesai.")
