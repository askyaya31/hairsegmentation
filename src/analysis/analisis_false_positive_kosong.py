import pandas as pd
from pathlib import Path

# ============================================================
# ANALISIS FALSE POSITIVE PADA CITRA DENGAN GROUND TRUTH KOSONG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "results" / "evaluasi_kernel_terkontrol" / "hasil_detail.csv"
OUTPUT_DIR = ROOT / "results" / "analisis_false_positive_kosong"

IMAGE_PIXELS = 512 * 512

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Baca hasil evaluasi
df = pd.read_csv(INPUT_FILE)

required_columns = [
    "filename",
    "category",
    "configuration",
    "fp"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

# Ambil citra dengan kategori KOSONG
df_kosong = df[df["category"] == "KOSONG"].copy()

if df_kosong.empty:
    raise ValueError("Tidak ditemukan citra dengan kategori KOSONG.")

df_kosong["fp"] = pd.to_numeric(df_kosong["fp"], errors="raise")

# Persentase piksel prediksi yang menjadi false positive
df_kosong["fp_percentage"] = (
    df_kosong["fp"] / IMAGE_PIXELS
) * 100

# Urutan konfigurasi
configuration_order = ["A", "B", "C", "D"]

df_kosong["configuration"] = pd.Categorical(
    df_kosong["configuration"],
    categories=configuration_order,
    ordered=True
)

# ============================================================
# RINGKASAN PER KONFIGURASI
# ============================================================

summary = (
    df_kosong
    .groupby("configuration", observed=False)
    .agg(
        jumlah_citra=("filename", "nunique"),
        total_fp=("fp", "sum"),
        mean_fp=("fp", "mean"),
        median_fp=("fp", "median"),
        std_fp=("fp", "std"),
        min_fp=("fp", "min"),
        max_fp=("fp", "max"),
        mean_fp_percentage=("fp_percentage", "mean")
    )
    .reset_index()
)

# ============================================================
# TAMPILKAN HASIL
# ============================================================

print("=" * 75)
print("ANALISIS FALSE POSITIVE PADA KATEGORI KOSONG")
print("=" * 75)

print(f"\nJumlah citra: {df_kosong['filename'].nunique()}")

print("\n--- FALSE POSITIVE PER CITRA ---")
print(
    df_kosong[
        ["filename", "configuration", "fp", "fp_percentage"]
    ].sort_values(["filename", "configuration"]).to_string(index=False)
)

print("\n--- RINGKASAN PER KONFIGURASI ---")
print(summary.to_string(index=False))

# ============================================================
# SIMPAN HASIL
# ============================================================

df_kosong.to_csv(
    OUTPUT_DIR / "false_positive_per_citra.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "ringkasan_false_positive.csv",
    index=False
)

print("\nHasil disimpan di:")
print(OUTPUT_DIR)