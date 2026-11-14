import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "results" / "evaluasi_testing" / "hasil_detail.csv"
)

OUTPUT_DIR = ROOT / "results" / "analisis_kasus"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Menghitung perubahan Dice
df["dice_difference"] = (
    df["hysteresis_dice"] - df["baseline_dice"]
)

# Memilih 5 kasus dengan peningkatan terbesar
best_cases = (
    df.sort_values("dice_difference", ascending=False)
    .head(5)
    .copy()
)

best_cases["kategori"] = "PENINGKATAN"

# Memilih 5 kasus dengan penurunan terbesar
worst_cases = (
    df.sort_values("dice_difference", ascending=True)
    .head(5)
    .copy()
)

worst_cases["kategori"] = "PENURUNAN"

# Menggabungkan 10 kasus
selected = pd.concat(
    [best_cases, worst_cases],
    ignore_index=True
)

# Menyimpan hasil
output_file = OUTPUT_DIR / "10_kasus_terpilih.csv"
selected.to_csv(output_file, index=False)

# Menampilkan ringkasan
columns = [
    "image",
    "kategori",
    "baseline_dice",
    "hysteresis_dice",
    "dice_difference"
]

print("=== 5 KASUS PENINGKATAN DICE TERBESAR ===")
print(
    best_cases[columns]
    .to_string(index=False, float_format=lambda x: f"{x:.6f}")
)

print("\n=== 5 KASUS PENURUNAN DICE TERBESAR ===")
print(
    worst_cases[columns]
    .to_string(index=False, float_format=lambda x: f"{x:.6f}")
)

print(f"\nFile tersimpan: {output_file}")
