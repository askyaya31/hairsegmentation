
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT_DIR / "results" / "analisis_kasus" / "analisis_seluruh_testing_detail.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "analisis_kasus"

df = pd.read_csv(INPUT_FILE)

# Mengelompokkan citra berdasarkan perubahan Dice
df["kelompok"] = np.select(
    [
        df["hysteresis_dice"] > df["baseline_dice"],
        df["hysteresis_dice"] < df["baseline_dice"],
    ],
    ["DICE MENINGKAT", "DICE MENURUN"],
    default="DICE TETAP",
)

# Mengelompokkan arah perubahan TP
df["perubahan_TP"] = np.select(
    [df["delta_TP"] > 0, df["delta_TP"] < 0],
    ["MENINGKAT", "MENURUN"],
    default="TETAP",
)

# Mengelompokkan arah perubahan FP
df["perubahan_FP"] = np.select(
    [df["delta_FP"] > 0, df["delta_FP"] < 0],
    ["MENINGKAT", "MENURUN"],
    default="TETAP",
)

print("=== JUMLAH CITRA PER KELOMPOK ===")
print(df["kelompok"].value_counts())

print("\n=== POLA PERUBAHAN TP DAN FP ===")
pola = pd.crosstab(
    [df["kelompok"], df["perubahan_TP"]],
    df["perubahan_FP"],
)
print(pola)

print("\n=== RATA-RATA DAN MEDIAN PERUBAHAN ===")
ringkasan = df.groupby("kelompok").agg(
    jumlah=("image", "count"),
    mean_delta_TP=("delta_TP", "mean"),
    median_delta_TP=("delta_TP", "median"),
    mean_delta_FP=("delta_FP", "mean"),
    median_delta_FP=("delta_FP", "median"),
    mean_delta_FN=("delta_FN", "mean"),
    median_delta_FN=("delta_FN", "median"),
)

print(ringkasan.round(2).to_string())

# Menyimpan hasil
df.to_csv(
    OUTPUT_DIR / "pola_perubahan_per_citra.csv",
    index=False,
)

ringkasan.to_csv(
    OUTPUT_DIR / "ringkasan_pola_perubahan.csv"
)

print("\nFile analisis berhasil disimpan.")
