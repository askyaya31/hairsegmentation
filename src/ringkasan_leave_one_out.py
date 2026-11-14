
import pandas as pd
import numpy as np

INPUT = "results/pemeriksaan_korelasi_150/hasil_leave_one_out.csv"
OUTPUT = "results/pemeriksaan_korelasi_150/ringkasan_loo_holm.csv"


def holm_correction(p_values):
    """Menghitung adjusted p-value menggunakan metode Holm."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.maximum.accumulate(
        (n - np.arange(n)) * sorted_p
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(n)
    adjusted[order] = adjusted_sorted

    return adjusted


df = pd.read_csv(INPUT)

hasil = []

# Koreksi Holm untuk 12 pengujian pada setiap citra yang dihilangkan
for filename, group in df.groupby("filename"):
    group = group.copy()

    valid = group["p_without_image"].notna()
    group["holm_p_loo"] = np.nan

    if valid.any():
        group.loc[valid, "holm_p_loo"] = holm_correction(
            group.loc[valid, "p_without_image"]
        )

    hasil.append(group)

loo_holm = pd.concat(hasil, ignore_index=True)

# Ringkasan per hubungan
ringkasan = loo_holm.groupby(
    ["kernel", "x_column", "y_column"]
).agg(
    jumlah_pengujian=("p_without_image", "count"),
    signifikan_raw=("p_without_image", lambda x: (x < 0.05).sum()),
    signifikan_holm=("holm_p_loo", lambda x: (x < 0.05).sum()),
    p_holm_min=("holm_p_loo", "min"),
    p_holm_max=("holm_p_loo", "max"),
    rho_min=("rho_without_image", "min"),
    rho_max=("rho_without_image", "max")
).reset_index()

ringkasan.to_csv(OUTPUT, index=False)

print("\nRINGKASAN SIGNIFIKANSI LEAVE-ONE-OUT")
print(ringkasan.to_string(index=False))

print("\nTOTAL")
print("Jumlah pengujian:", len(loo_holm))
print("Signifikan setelah Holm:",
      (loo_holm["holm_p_loo"] < 0.05).sum())
print("Tidak signifikan setelah Holm:",
      (loo_holm["holm_p_loo"] >= 0.05).sum())

print("\nFile tersimpan:", OUTPUT)
