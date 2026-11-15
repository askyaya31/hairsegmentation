
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT_DIR / "results" / "06_error_analysis" / "analisis_kasus" / "analisis_seluruh_testing_detail.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "06_error_analysis" / "analisis_kasus"

df = pd.read_csv(INPUT_FILE)

df["delta_dice"] = df["hysteresis_dice"] - df["baseline_dice"]

conditions = [
    (df.delta_TP > 0) & (df.delta_FP > 0),
    (df.delta_TP > 0) & (df.delta_FP < 0),
    (df.delta_TP < 0) & (df.delta_FP > 0),
    (df.delta_TP < 0) & (df.delta_FP < 0),
    (df.delta_TP == 0) & (df.delta_FP > 0),
    (df.delta_TP == 0) & (df.delta_FP == 0),
]

labels = [
    "TP NAIK - FP NAIK",
    "TP NAIK - FP TURUN",
    "TP TURUN - FP NAIK",
    "TP TURUN - FP TURUN",
    "TP TETAP - FP NAIK",
    "TP TETAP - FP TETAP",
]

df["pola"] = pd.Series(
    pd.NA, index=df.index, dtype="object"
)

for condition, label in zip(conditions, labels):
    df.loc[condition, "pola"] = label

print("=== RINGKASAN SELURUH POLA ===")
summary = df.groupby("pola", dropna=False).agg(
    jumlah=("image", "count"),
    mean_delta_dice=("delta_dice", "mean"),
    median_delta_dice=("delta_dice", "median"),
    mean_delta_TP=("delta_TP", "mean"),
    mean_delta_FP=("delta_FP", "mean"),
)

print(summary.round(5).to_string())

print("\n=== TIGA KASUS DENGAN DICE TERENDAH PER POLA ===")

for label in labels:
    subset = df[df["pola"] == label].sort_values("delta_dice")

    if subset.empty:
        continue

    print(f"\n{label}")
    print(
        subset[
            [
                "image",
                "baseline_dice",
                "hysteresis_dice",
                "delta_dice",
                "delta_TP",
                "delta_FP",
                "delta_FN",
            ]
        ].head(3).to_string(index=False)
    )

df.to_csv(
    OUTPUT_DIR / "analisis_pola_kegagalan.csv",
    index=False,
)

summary.to_csv(
    OUTPUT_DIR / "ringkasan_pola_kegagalan.csv"
)

print("\nHasil analisis berhasil disimpan.")
