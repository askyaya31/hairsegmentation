
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_FILE = ROOT / "results" / "pembagian_dataset.csv"
DETAIL_FILE = (
    ROOT / "results" / "evaluasi_final_testing" / "hasil_detail.csv"
)

OUTPUT_DIR = ROOT / "results" / "analisis_per_kategori"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DETAIL = OUTPUT_DIR / "hasil_detail_per_kategori.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "hasil_ringkasan_per_kategori.csv"


# ============================================================
# MEMUAT DATA
# ============================================================

def load_data():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {SPLIT_FILE}")

    if not DETAIL_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {DETAIL_FILE}")

    split_df = pd.read_csv(SPLIT_FILE)
    detail_df = pd.read_csv(DETAIL_FILE)

    required_split = {"filename", "category", "split"}
    required_detail = {
        "filename",
        "baseline_dice",
        "hysteresis_dice"
    }

    if not required_split.issubset(split_df.columns):
        raise ValueError(
            f"Kolom split tidak lengkap. "
            f"Kolom tersedia: {list(split_df.columns)}"
        )

    if not required_detail.issubset(detail_df.columns):
        raise ValueError(
            f"Kolom hasil evaluasi tidak lengkap. "
            f"Kolom tersedia: {list(detail_df.columns)}"
        )

    # Ambil hanya data pengujian
    testing_df = split_df[
        split_df["split"].astype(str).str.upper() == "PENGUJIAN"
    ].copy()

    # Gabungkan kategori dengan hasil evaluasi
    merged = testing_df.merge(
        detail_df,
        on="filename",
        how="inner",
        validate="one_to_one"
    )

    if len(merged) != len(testing_df):
        raise ValueError(
            "Jumlah data setelah penggabungan tidak sesuai. "
            "Periksa kesesuaian filename."
        )

    if merged[
        ["baseline_dice", "hysteresis_dice"]
    ].isna().any().any():
        raise ValueError("Terdapat nilai Dice yang kosong.")

    if not np.isfinite(
        merged[["baseline_dice", "hysteresis_dice"]].to_numpy()
    ).all():
        raise ValueError("Terdapat nilai Dice yang tidak valid.")

    merged["dice_difference"] = (
        merged["hysteresis_dice"] - merged["baseline_dice"]
    )

    return merged


# ============================================================
# ANALISIS PER KATEGORI
# ============================================================

def analyze_by_category(df):
    results = []

    for category, group in df.groupby("category", sort=True):
        baseline = group["baseline_dice"]
        hysteresis = group["hysteresis_dice"]
        difference = group["dice_difference"]

        results.append({
            "category": category,
            "n_images": len(group),

            "baseline_dice_mean": baseline.mean(),
            "baseline_dice_std": baseline.std(ddof=1),

            "hysteresis_dice_mean": hysteresis.mean(),
            "hysteresis_dice_std": hysteresis.std(ddof=1),

            "mean_difference": difference.mean(),
            "median_difference": difference.median(),

            "dice_increased": int((difference > 0).sum()),
            "dice_decreased": int((difference < 0).sum()),
            "dice_unchanged": int((difference == 0).sum()),
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("ANALISIS PERFORMA BERDASARKAN KATEGORI CITRA")
    print("=" * 65)

    df = load_data()

    print(f"\nJumlah citra pengujian: {len(df)}")

    summary = analyze_by_category(df)

    # Simpan hasil
    df.to_csv(OUTPUT_DETAIL, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    # Tampilkan ringkasan
    print("\nHASIL ANALISIS PER KATEGORI")
    print("=" * 65)

    for _, row in summary.iterrows():
        print(f"\nKategori: {row['category']}")
        print(f"Jumlah citra       : {row['n_images']}")
        print(
            f"Dice Baseline      : "
            f"{row['baseline_dice_mean']:.6f} "
            f"(SD: {row['baseline_dice_std']:.6f})"
        )
        print(
            f"Dice Hysteresis    : "
            f"{row['hysteresis_dice_mean']:.6f} "
            f"(SD: {row['hysteresis_dice_std']:.6f})"
        )
        print(
            f"Rata-rata selisih  : "
            f"{row['mean_difference']:+.6f}"
        )
        print(
            f"Median selisih     : "
            f"{row['median_difference']:+.6f}"
        )
        print(
            f"Dice meningkat    : "
            f"{int(row['dice_increased'])}"
        )
        print(
            f"Dice menurun      : "
            f"{int(row['dice_decreased'])}"
        )
        print(
            f"Dice tetap        : "
            f"{int(row['dice_unchanged'])}"
        )

    print("\n" + "=" * 65)
    print("HASIL DISIMPAN")
    print(f"Detail   : {OUTPUT_DETAIL}")
    print(f"Ringkasan: {OUTPUT_SUMMARY}")
    print("=" * 65)


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SPLIT_FILE = ROOT / "results" / "pembagian_dataset.csv"
DETAIL_FILE = (
    ROOT / "results" / "evaluasi_final_testing" / "hasil_detail.csv"
)

OUTPUT_DIR = ROOT / "results" / "analisis_per_kategori"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DETAIL = OUTPUT_DIR / "hasil_detail_per_kategori.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "hasil_ringkasan_per_kategori.csv"


# ============================================================
# MEMUAT DATA
# ============================================================

def load_data():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {SPLIT_FILE}")

    if not DETAIL_FILE.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {DETAIL_FILE}")

    split_df = pd.read_csv(SPLIT_FILE)
    detail_df = pd.read_csv(DETAIL_FILE)

    required_split = {"filename", "category", "split"}
    required_detail = {
        "filename",
        "baseline_dice",
        "hysteresis_dice"
    }

    if not required_split.issubset(split_df.columns):
        raise ValueError(
            f"Kolom split tidak lengkap. "
            f"Kolom tersedia: {list(split_df.columns)}"
        )

    if not required_detail.issubset(detail_df.columns):
        raise ValueError(
            f"Kolom hasil evaluasi tidak lengkap. "
            f"Kolom tersedia: {list(detail_df.columns)}"
        )

    # Ambil hanya data pengujian
    testing_df = split_df[
        split_df["split"].astype(str).str.upper() == "PENGUJIAN"
    ].copy()

    # Gabungkan kategori dengan hasil evaluasi
    merged = testing_df.merge(
        detail_df,
        on="filename",
        how="inner",
        validate="one_to_one"
    )

    if len(merged) != len(testing_df):
        raise ValueError(
            "Jumlah data setelah penggabungan tidak sesuai. "
            "Periksa kesesuaian filename."
        )

    if merged[
        ["baseline_dice", "hysteresis_dice"]
    ].isna().any().any():
        raise ValueError("Terdapat nilai Dice yang kosong.")

    if not np.isfinite(
        merged[["baseline_dice", "hysteresis_dice"]].to_numpy()
    ).all():
        raise ValueError("Terdapat nilai Dice yang tidak valid.")

    merged["dice_difference"] = (
        merged["hysteresis_dice"] - merged["baseline_dice"]
    )

    return merged


# ============================================================
# ANALISIS PER KATEGORI
# ============================================================

def analyze_by_category(df):
    results = []

    for category, group in df.groupby("category", sort=True):
        baseline = group["baseline_dice"]
        hysteresis = group["hysteresis_dice"]
        difference = group["dice_difference"]

        results.append({
            "category": category,
            "n_images": len(group),

            "baseline_dice_mean": baseline.mean(),
            "baseline_dice_std": baseline.std(ddof=1),

            "hysteresis_dice_mean": hysteresis.mean(),
            "hysteresis_dice_std": hysteresis.std(ddof=1),

            "mean_difference": difference.mean(),
            "median_difference": difference.median(),

            "dice_increased": int((difference > 0).sum()),
            "dice_decreased": int((difference < 0).sum()),
            "dice_unchanged": int((difference == 0).sum()),
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("ANALISIS PERFORMA BERDASARKAN KATEGORI CITRA")
    print("=" * 65)

    df = load_data()

    print(f"\nJumlah citra pengujian: {len(df)}")

    summary = analyze_by_category(df)

    # Simpan hasil
    df.to_csv(OUTPUT_DETAIL, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    # Tampilkan ringkasan
    print("\nHASIL ANALISIS PER KATEGORI")
    print("=" * 65)

    for _, row in summary.iterrows():
        print(f"\nKategori: {row['category']}")
        print(f"Jumlah citra       : {row['n_images']}")
        print(
            f"Dice Baseline      : "
            f"{row['baseline_dice_mean']:.6f} "
            f"(SD: {row['baseline_dice_std']:.6f})"
        )
        print(
            f"Dice Hysteresis    : "
            f"{row['hysteresis_dice_mean']:.6f} "
            f"(SD: {row['hysteresis_dice_std']:.6f})"
        )
        print(
            f"Rata-rata selisih  : "
            f"{row['mean_difference']:+.6f}"
        )
        print(
            f"Median selisih     : "
            f"{row['median_difference']:+.6f}"
        )
        print(
            f"Dice meningkat    : "
            f"{int(row['dice_increased'])}"
        )
        print(
            f"Dice menurun      : "
            f"{int(row['dice_decreased'])}"
        )
        print(
            f"Dice tetap        : "
            f"{int(row['dice_unchanged'])}"
        )

    print("\n" + "=" * 65)
    print("HASIL DISIMPAN")
    print(f"Detail   : {OUTPUT_DETAIL}")
    print(f"Ringkasan: {OUTPUT_SUMMARY}")
    print("=" * 65)


if __name__ == "__main__":
    main()
