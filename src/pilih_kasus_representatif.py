
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "results" / "analisis_kasus" / "analisis_seluruh_testing_detail.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "analisis_kasus"

df = pd.read_csv(INPUT_FILE)

# Hitung perubahan Dice
df["delta_dice"] = df["hysteresis_dice"] - df["baseline_dice"]

# Kelompokkan citra
df["kelompok"] = np.select(
    [
        df["delta_dice"] > 1e-12,
        df["delta_dice"] < -1e-12
    ],
    [
        "DICE MENINGKAT",
        "DICE MENURUN"
    ],
    default="DICE TETAP"
)

representatives = []

# Kasus peningkatan dan penurunan:
# pilih citra dengan perubahan Dice paling dekat median kelompok.
for kelompok in ["DICE MENINGKAT", "DICE MENURUN"]:
    subset = df[df["kelompok"] == kelompok].copy()

    median_delta = subset["delta_dice"].median()

    subset["jarak_median"] = (
        subset["delta_dice"] - median_delta
    ).abs()

    selected = subset.sort_values(
        ["jarak_median", "image"]
    ).iloc[0].copy()

    selected["median_delta_kelompok"] = median_delta
    representatives.append(selected)

# Kasus Dice tetap:
# pilih citra dengan perubahan FP paling dekat median kelompok.
subset = df[df["kelompok"] == "DICE TETAP"].copy()

median_fp = subset["delta_FP"].median()

subset["jarak_median"] = (
    subset["delta_FP"] - median_fp
).abs()

selected = subset.sort_values(
    ["jarak_median", "image"]
).iloc[0].copy()

selected["median_delta_kelompok"] = 0.0
representatives.append(selected)

# Simpan hasil
result = pd.DataFrame(representatives)

columns = [
    "image",
    "kelompok",
    "baseline_dice",
    "hysteresis_dice",
    "delta_dice",
    "delta_TP",
    "delta_FP",
    "delta_FN",
    "median_delta_kelompok"
]

result = result[columns]

output_file = OUTPUT_DIR / "kasus_representatif.csv"
result.to_csv(output_file, index=False)

print("\n=== TIGA KASUS REPRESENTATIF ===")
print(result.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

print(f"\nFile disimpan: {output_file}")
