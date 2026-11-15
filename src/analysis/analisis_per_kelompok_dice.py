
from pathlib import Path

import numpy as np
import pandas as pd

# Lokasi file
ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT_DIR / "results" / "06_error_analysis" / "analisis_kasus" / "analisis_seluruh_testing_detail.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "06_error_analysis" / "analisis_kasus"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Membaca hasil analisis 150 citra
df = pd.read_csv(INPUT_FILE)

# Menghitung perubahan Dice
df["delta_dice"] = df["hysteresis_dice"] - df["baseline_dice"]

# Mengelompokkan citra berdasarkan perubahan Dice
conditions = [
    df["delta_dice"] > 1e-12,
    df["delta_dice"] < -1e-12
]

choices = ["DICE MENINGKAT", "DICE MENURUN"]

df["kelompok"] = np.select(
    conditions,
    choices,
    default="DICE TETAP"
)

# Urutan kelompok
order = [
    "DICE MENINGKAT",
    "DICE MENURUN",
    "DICE TETAP"
]

# Ringkasan setiap kelompok
summary = (
    df.groupby("kelompok")
    .agg(
        jumlah_citra=("image", "count"),
        rata_dice_baseline=("baseline_dice", "mean"),
        rata_dice_hysteresis=("hysteresis_dice", "mean"),
        rata_perubahan_dice=("delta_dice", "mean"),
        median_perubahan_dice=("delta_dice", "median"),
        rata_perubahan_TP=("delta_TP", "mean"),
        rata_perubahan_FP=("delta_FP", "mean"),
        rata_perubahan_FN=("delta_FN", "mean"),
        median_perubahan_TP=("delta_TP", "median"),
        median_perubahan_FP=("delta_FP", "median"),
        median_perubahan_FN=("delta_FN", "median"),
    )
    .reindex(order)
    .reset_index()
)

# Statistik tambahan: arah perubahan setiap komponen
direction_rows = []

for kelompok in order:
    subset = df[df["kelompok"] == kelompok]

    for komponen in ["TP", "FP", "FN"]:
        kolom = f"delta_{komponen}"

        direction_rows.append({
            "kelompok": kelompok,
            "komponen": komponen,
            "meningkat": int((subset[kolom] > 0).sum()),
            "menurun": int((subset[kolom] < 0).sum()),
            "tetap": int((subset[kolom] == 0).sum()),
        })

direction_summary = pd.DataFrame(direction_rows)

# Menampilkan hasil
print("\n=== JUMLAH CITRA PER KELOMPOK ===")
print(
    df["kelompok"]
    .value_counts()
    .reindex(order, fill_value=0)
    .to_string()
)

print("\n=== RINGKASAN PER KELOMPOK DICE ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n=== ARAH PERUBAHAN TP, FP, FN ===")
print(direction_summary.to_string(index=False))

# Menyimpan hasil
detail_output = OUTPUT_DIR / "analisis_per_kelompok_dice_detail.csv"
summary_output = OUTPUT_DIR / "analisis_per_kelompok_dice_ringkasan.csv"
direction_output = OUTPUT_DIR / "analisis_per_kelompok_dice_komponen.csv"

df.to_csv(detail_output, index=False)
summary.to_csv(summary_output, index=False)
direction_summary.to_csv(direction_output, index=False)

print("\n=== FILE HASIL ===")
print(f"Detail     : {detail_output}")
print(f"Ringkasan  : {summary_output}")
print(f"Komponen   : {direction_output}")
