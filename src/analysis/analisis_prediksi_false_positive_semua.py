
import pandas as pd
from pathlib import Path

# ============================================================
# ANALISIS JUMLAH PREDIKSI DAN FALSE POSITIVE SELURUH DATA UJI
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "results" / "evaluasi_kernel_terkontrol" / "hasil_detail.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "analisis_prediksi_false_positive_semua"
)

IMAGE_PIXELS = 512 * 512
CONFIGURATIONS = ["A", "B", "C", "D"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. BACA DAN VALIDASI DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "filename",
    "category",
    "configuration",
    "tp",
    "fp",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

df = df.copy()

df["tp"] = pd.to_numeric(df["tp"], errors="raise")
df["fp"] = pd.to_numeric(df["fp"], errors="raise")

df["configuration"] = df["configuration"].astype(str)

if not set(df["configuration"]).issubset(CONFIGURATIONS):
    raise ValueError("Terdapat konfigurasi yang tidak dikenal.")

if df.duplicated(["filename", "configuration"]).any():
    raise ValueError("Ditemukan duplikasi citra pada konfigurasi yang sama.")

# Jumlah piksel prediksi = TP + FP
df["predicted_pixels"] = df["tp"] + df["fp"]

# Persentase area citra
df["predicted_percentage"] = (
    df["predicted_pixels"] / IMAGE_PIXELS
) * 100

df["fp_percentage"] = (
    df["fp"] / IMAGE_PIXELS
) * 100

# ============================================================
# 2. VALIDASI JUMLAH CITRA
# ============================================================

print("=" * 80)
print("ANALISIS PREDIKSI DAN FALSE POSITIVE SELURUH DATA UJI")
print("=" * 80)

print("\nJumlah citra unik:", df["filename"].nunique())
print("\nJumlah citra per konfigurasi:")
print(df.groupby("configuration")["filename"].nunique())

if df["filename"].nunique() != 150:
    raise ValueError("Jumlah citra unik bukan 150. Periksa dataset pengujian.")

for config in CONFIGURATIONS:
    count = df[df["configuration"] == config]["filename"].nunique()
    if count != 150:
        raise ValueError(
            f"Konfigurasi {config} hanya memiliki {count} citra."
        )

# ============================================================
# 3. RINGKASAN SELURUH CITRA
# ============================================================

summary_all = (
    df.groupby("configuration")
    .agg(
        jumlah_citra=("filename", "nunique"),
        total_predicted=("predicted_pixels", "sum"),
        mean_predicted=("predicted_pixels", "mean"),
        median_predicted=("predicted_pixels", "median"),
        total_fp=("fp", "sum"),
        mean_fp=("fp", "mean"),
        median_fp=("fp", "median"),
        std_fp=("fp", "std"),
        mean_predicted_percentage=("predicted_percentage", "mean"),
        mean_fp_percentage=("fp_percentage", "mean"),
    )
    .reindex(CONFIGURATIONS)
    .reset_index()
)

# ============================================================
# 4. RINGKASAN BERDASARKAN KATEGORI
# ============================================================

summary_category = (
    df.groupby(["category", "configuration"], observed=True)
    .agg(
        jumlah_citra=("filename", "nunique"),
        mean_predicted=("predicted_pixels", "mean"),
        mean_fp=("fp", "mean"),
        median_fp=("fp", "median"),
        total_fp=("fp", "sum"),
        mean_predicted_percentage=("predicted_percentage", "mean"),
        mean_fp_percentage=("fp_percentage", "mean"),
    )
    .reset_index()
)

# ============================================================
# 5. PERBANDINGAN RELATIF TERHADAP BASELINE
# ============================================================

comparison = summary_all[
    ["configuration", "mean_predicted", "mean_fp"]
].set_index("configuration")

baseline_a_fp = comparison.loc["A", "mean_fp"]
baseline_c_fp = comparison.loc["C", "mean_fp"]

comparison["fp_vs_A_percent"] = (
    (comparison["mean_fp"] - baseline_a_fp) / baseline_a_fp
) * 100

comparison["fp_vs_C_percent"] = (
    (comparison["mean_fp"] - baseline_c_fp) / baseline_c_fp
) * 100

comparison = comparison.reset_index()

# ============================================================
# 6. TAMPILKAN HASIL
# ============================================================

print("\n" + "=" * 80)
print("RINGKASAN SELURUH 150 CITRA")
print("=" * 80)

print(summary_all.to_string(index=False))

print("\n" + "=" * 80)
print("RINGKASAN BERDASARKAN KATEGORI")
print("=" * 80)

print(
    summary_category.sort_values(
        ["category", "configuration"]
    ).to_string(index=False)
)

print("\n" + "=" * 80)
print("PERBANDINGAN FALSE POSITIVE TERHADAP BASELINE")
print("=" * 80)

print(comparison.to_string(index=False))

# ============================================================
# 7. SIMPAN HASIL
# ============================================================

df.to_csv(
    OUTPUT_DIR / "detail_prediksi_false_positive.csv",
    index=False
)

summary_all.to_csv(
    OUTPUT_DIR / "ringkasan_seluruh_citra.csv",
    index=False
)

summary_category.to_csv(
    OUTPUT_DIR / "ringkasan_per_kategori.csv",
    index=False
)

comparison.to_csv(
    OUTPUT_DIR / "perbandingan_terhadap_baseline.csv",
    index=False
)

print("\nSeluruh hasil disimpan di:")
print(OUTPUT_DIR)
