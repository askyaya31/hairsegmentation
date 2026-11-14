
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "results" / "evaluasi_testing" / "hasil_detail.csv"

df = pd.read_csv(INPUT_FILE)

df["dice_difference"] = (
    df["hysteresis_dice"] - df["baseline_dice"]
)

differences = df["dice_difference"].dropna()

improved = (differences > 0).sum()
equal = (differences == 0).sum()
worsened = (differences < 0).sum()

result = wilcoxon(
    df["hysteresis_dice"],
    df["baseline_dice"],
    alternative="two-sided"
)

print("=== ANALISIS PERUBAHAN DICE ===")
print(f"Jumlah citra       : {len(differences)}")
print(f"Dice meningkat     : {improved}")
print(f"Dice sama          : {equal}")
print(f"Dice menurun       : {worsened}")

print("\n=== STATISTIK SELISIH DICE ===")
print(f"Rata-rata selisih  : {differences.mean():.6f}")
print(f"Median selisih     : {differences.median():.6f}")
print(f"Minimum selisih    : {differences.min():.6f}")
print(f"Maksimum selisih   : {differences.max():.6f}")

print("\n=== UJI WILCOXON SIGNED-RANK ===")
print(f"Statistik          : {result.statistic:.4f}")
print(f"P-value            : {result.pvalue:.8f}")

if result.pvalue < 0.05:
    print("\nKesimpulan: terdapat perbedaan Dice yang signifikan secara statistik.")
else:
    print("\nKesimpulan: tidak terdapat perbedaan Dice yang signifikan secara statistik.")
