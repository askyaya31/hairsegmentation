
import pandas as pd
import numpy as np

from pathlib import Path
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "results" / "03_evaluation" / "kernel_terkontrol" / "hasil_detail.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "06_error_analysis" / "korelasi_fp_dice"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISONS = [
    ("A", "B", "Kernel 9: Baseline vs Hysteresis"),
    ("C", "D", "Kernel 11: Baseline vs Hysteresis"),
]

# ============================================================
# 1. BACA DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required = ["filename", "configuration", "fp", "dice"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

df["fp"] = pd.to_numeric(df["fp"], errors="raise")
df["dice"] = pd.to_numeric(df["dice"], errors="raise")
df["configuration"] = df["configuration"].astype(str)

if df.duplicated(["filename", "configuration"]).any():
    raise ValueError("Ditemukan duplikasi citra dan konfigurasi.")

# ============================================================
# 2. ANALISIS SETIAP PASANGAN
# ============================================================

all_details = []
summaries = []

print("=" * 80)
print("ANALISIS KORELASI PERUBAHAN FALSE POSITIVE DAN DICE")
print("=" * 80)

for baseline_config, hyst_config, label in COMPARISONS:

    baseline = (
        df[df["configuration"] == baseline_config]
        [["filename", "fp", "dice"]]
        .rename(columns={
            "fp": "baseline_fp",
            "dice": "baseline_dice"
        })
    )

    hysteresis = (
        df[df["configuration"] == hyst_config]
        [["filename", "fp", "dice"]]
        .rename(columns={
            "fp": "hysteresis_fp",
            "dice": "hysteresis_dice"
        })
    )

    paired = baseline.merge(
        hysteresis,
        on="filename",
        how="inner",
        validate="one_to_one"
    )

    if len(paired) != 150:
        raise ValueError(
            f"{label}: hanya ditemukan {len(paired)} pasangan."
        )

    paired = paired.sort_values("filename").reset_index(drop=True)

    # Perubahan dihitung sebagai Hysteresis - Baseline
    paired["delta_fp"] = (
        paired["hysteresis_fp"] - paired["baseline_fp"]
    )

    paired["delta_dice"] = (
        paired["hysteresis_dice"] - paired["baseline_dice"]
    )

    # Korelasi Spearman
    spearman_r, spearman_p = spearmanr(
        paired["delta_fp"],
        paired["delta_dice"]
    )

    # Korelasi Pearson
    pearson_r, pearson_p = pearsonr(
        paired["delta_fp"],
        paired["delta_dice"]
    )

    # Klasifikasi arah perubahan
    fp_up = paired["delta_fp"] > 0
    fp_down = paired["delta_fp"] < 0
    dice_up = paired["delta_dice"] > 0
    dice_down = paired["delta_dice"] < 0

    both_up = int((fp_up & dice_up).sum())
    fp_up_dice_down = int((fp_up & dice_down).sum())
    fp_down_dice_up = int((fp_down & dice_up).sum())
    both_down = int((fp_down & dice_down).sum())

    unchanged_fp = int((paired["delta_fp"] == 0).sum())
    unchanged_dice = int((paired["delta_dice"] == 0).sum())

    summary = {
        "comparison": label,
        "n": len(paired),

        "mean_delta_fp": paired["delta_fp"].mean(),
        "median_delta_fp": paired["delta_fp"].median(),

        "mean_delta_dice": paired["delta_dice"].mean(),
        "median_delta_dice": paired["delta_dice"].median(),

        "spearman_r": spearman_r,
        "spearman_p": spearman_p,

        "pearson_r": pearson_r,
        "pearson_p": pearson_p,

        "fp_up_dice_up": both_up,
        "fp_up_dice_down": fp_up_dice_down,
        "fp_down_dice_up": fp_down_dice_up,
        "fp_down_dice_down": both_down,

        "unchanged_fp": unchanged_fp,
        "unchanged_dice": unchanged_dice,
    }

    summaries.append(summary)

    paired["comparison"] = label
    all_details.append(paired)

    # ========================================================
    # 3. TAMPILKAN HASIL
    # ========================================================

    print("\n" + "-" * 80)
    print(label)
    print("-" * 80)

    print(f"Jumlah pasangan                  : {len(paired)}")

    print(
        f"Mean perubahan FP               : "
        f"{summary['mean_delta_fp']:+.2f}"
    )

    print(
        f"Median perubahan FP             : "
        f"{summary['median_delta_fp']:+.2f}"
    )

    print(
        f"Mean perubahan Dice             : "
        f"{summary['mean_delta_dice']:+.6f}"
    )

    print(
        f"Median perubahan Dice           : "
        f"{summary['median_delta_dice']:+.6f}"
    )

    print("\nKorelasi:")
    print(f"Spearman r                      : {spearman_r:+.6f}")
    print(f"Spearman p-value                : {spearman_p:.10e}")
    print(f"Pearson r                       : {pearson_r:+.6f}")
    print(f"Pearson p-value                 : {pearson_p:.10e}")

    print("\nPola perubahan per citra:")
    print(f"FP naik, Dice naik              : {both_up}")
    print(f"FP naik, Dice turun             : {fp_up_dice_down}")
    print(f"FP turun, Dice naik             : {fp_down_dice_up}")
    print(f"FP turun, Dice turun            : {both_down}")
    print(f"FP tidak berubah                : {unchanged_fp}")
    print(f"Dice tidak berubah              : {unchanged_dice}")

# ============================================================
# 4. SIMPAN HASIL
# ============================================================

summary_df = pd.DataFrame(summaries)
detail_df = pd.concat(all_details, ignore_index=True)

summary_df.to_csv(
    OUTPUT_DIR / "ringkasan_korelasi.csv",
    index=False
)

detail_df.to_csv(
    OUTPUT_DIR / "detail_perubahan_fp_dice.csv",
    index=False
)

print("\n" + "=" * 80)
print("HASIL DISIMPAN DI:")
print(OUTPUT_DIR)
print("=" * 80)
