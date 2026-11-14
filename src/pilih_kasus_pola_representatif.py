
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "results" / "analisis_kasus" / "analisis_pola_kegagalan.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "analisis_kasus"

df = pd.read_csv(INPUT_FILE)

# Pilih kasus yang delta Dice-nya paling dekat dengan median
# setiap pola, serta kasus dengan hasil paling ekstrem.
selected = []

for pola, group in df.groupby("pola"):
    if pd.isna(pola):
        continue

    median_delta = group["delta_dice"].median()

    median_case = group.iloc[
        (group["delta_dice"] - median_delta).abs().argsort()[:1]
    ].copy()
    median_case["jenis_pemilihan"] = "MEDIAN"

    selected.append(median_case)

    # Kasus dengan delta Dice paling rendah
    worst_case = group.nsmallest(1, "delta_dice").copy()
    worst_case["jenis_pemilihan"] = "TERENDAH"

    if worst_case.iloc[0]["image"] != median_case.iloc[0]["image"]:
        selected.append(worst_case)

result = pd.concat(selected).drop_duplicates(subset="image")

columns = [
    "image",
    "pola",
    "jenis_pemilihan",
    "baseline_dice",
    "hysteresis_dice",
    "delta_dice",
    "delta_TP",
    "delta_FP",
    "delta_FN",
]

result = result[columns].sort_values(
    ["pola", "jenis_pemilihan"]
)

print("=== KASUS REPRESENTATIF SETIAP POLA ===")
print(result.to_string(index=False))

output_file = OUTPUT_DIR / "kasus_representatif_setiap_pola.csv"
result.to_csv(output_file, index=False)

print(f"\nFile disimpan: {output_file}")
