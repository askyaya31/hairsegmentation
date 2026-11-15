import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr


# ============================================================
# KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "results" / "05_component_analysis" / "ground_truth"
INPUT_FILE = INPUT_DIR / "detail_per_komponen.csv"

OUTPUT_DIR = ROOT / "results" / "05_component_analysis" / "seluruh_komponen"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MEMBACA DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "filename",
    "configuration",
    "kernel",
    "component_id",
    "component_size",
    "gt_pixels",
    "non_gt_pixels",
    "strong_pixels",
    "contains_gt",
    "status"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

print("=" * 70)
print("ANALISIS SELURUH KOMPONEN HYSTERESIS")
print("=" * 70)

print(f"Jumlah komponen : {len(df):,}")
print(f"Jumlah citra    : {df['filename'].nunique()}")
print("\nDistribusi konfigurasi:")
print(df["configuration"].value_counts())

print("\nDistribusi status:")
print(df["status"].value_counts())


# ============================================================
# VALIDASI DATA
# ============================================================

assert (
    df["component_size"]
    == df["gt_pixels"] + df["non_gt_pixels"]
).all(), "Ditemukan component_size yang tidak konsisten."

assert (df["strong_pixels"] <= df["component_size"]).all()
assert (df["gt_pixels"] <= df["component_size"]).all()

assert (
    df["contains_gt"]
    == (df["gt_pixels"] > 0)
).all(), "contains_gt tidak konsisten."

assert (
    df["status"]
    == np.where(
        df["strong_pixels"] > 0,
        "DIPERTAHANKAN",
        "DIBUANG"
    )
).all(), "Status tidak konsisten dengan strong_pixels."

print("\nValidasi data: LULUS")


# ============================================================
# FITUR TURUNAN
# ============================================================

df["gt_ratio"] = (
    df["gt_pixels"] / df["component_size"]
)

df["non_gt_ratio"] = (
    df["non_gt_pixels"] / df["component_size"]
)

df["strong_ratio"] = (
    df["strong_pixels"] / df["component_size"]
)

df["has_seed"] = df["strong_pixels"] > 0

df["log_component_size"] = np.log10(
    df["component_size"]
)


# ============================================================
# 1. RINGKASAN KONFIGURASI DAN STATUS
# ============================================================

summary = (
    df.groupby(["configuration", "status"])
    .agg(
        jumlah_komponen=("component_id", "count"),

        mean_size=("component_size", "mean"),
        median_size=("component_size", "median"),
        q1_size=("component_size", lambda x: x.quantile(0.25)),
        q3_size=("component_size", lambda x: x.quantile(0.75)),
        max_size=("component_size", "max"),

        mean_gt_ratio=("gt_ratio", "mean"),
        median_gt_ratio=("gt_ratio", "median"),

        mean_non_gt_ratio=("non_gt_ratio", "mean"),
        median_non_gt_ratio=("non_gt_ratio", "median"),

        total_gt_pixels=("gt_pixels", "sum"),
        total_non_gt_pixels=("non_gt_pixels", "sum"),
        total_strong_pixels=("strong_pixels", "sum")
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT_DIR / "ringkasan_konfigurasi_status.csv",
    index=False
)

print("\n" + "=" * 70)
print("1. RINGKASAN BERDASARKAN KONFIGURASI DAN STATUS")
print("=" * 70)

print(summary.round(4).to_string(index=False))


# ============================================================
# 2. RINGKASAN RETENSI PIXEL
# ============================================================

retention_rows = []

for config, group in df.groupby("configuration"):

    retained = group[group["status"] == "DIPERTAHANKAN"]

    total_components = len(group)
    retained_components = len(retained)

    total_size = group["component_size"].sum()
    retained_size = retained["component_size"].sum()

    total_gt = group["gt_pixels"].sum()
    retained_gt = retained["gt_pixels"].sum()

    total_non_gt = group["non_gt_pixels"].sum()
    retained_non_gt = retained["non_gt_pixels"].sum()

    retention_rows.append({
        "configuration": config,

        "total_components": total_components,
        "retained_components": retained_components,

        "component_retention_pct":
            100 * retained_components / total_components,

        "total_weak_pixels": total_size,
        "retained_pixels": retained_size,

        "pixel_retention_pct":
            100 * retained_size / total_size,

        "total_gt_in_weak": total_gt,
        "retained_gt": retained_gt,

        "gt_retention_pct":
            100 * retained_gt / total_gt if total_gt else np.nan,

        "total_non_gt_in_weak": total_non_gt,
        "retained_non_gt": retained_non_gt,

        "non_gt_retention_pct":
            100 * retained_non_gt / total_non_gt
            if total_non_gt else np.nan
    })

retention = pd.DataFrame(retention_rows)

retention.to_csv(
    OUTPUT_DIR / "ringkasan_retensi_pixel.csv",
    index=False
)

print("\n" + "=" * 70)
print("2. RINGKASAN RETENSI PIXEL")
print("=" * 70)

print(retention.round(4).to_string(index=False))


# ============================================================
# 3. RINGKASAN PER CITRA
# ============================================================

per_image_rows = []

for (filename, config), group in df.groupby(
    ["filename", "configuration"]
):

    retained = group[group["status"] == "DIPERTAHANKAN"]

    total_gt = group["gt_pixels"].sum()
    total_non_gt = group["non_gt_pixels"].sum()

    retained_gt = retained["gt_pixels"].sum()
    retained_non_gt = retained["non_gt_pixels"].sum()

    per_image_rows.append({
        "filename": filename,
        "configuration": config,

        "total_components": len(group),
        "retained_components": len(retained),

        "component_retention_pct":
            100 * len(retained) / len(group),

        "mean_component_size":
            group["component_size"].mean(),

        "median_component_size":
            group["component_size"].median(),

        "retained_gt": retained_gt,
        "total_gt_in_weak": total_gt,

        "gt_retention_pct":
            100 * retained_gt / total_gt
            if total_gt else np.nan,

        "retained_non_gt": retained_non_gt,
        "total_non_gt_in_weak": total_non_gt,

        "non_gt_retention_pct":
            100 * retained_non_gt / total_non_gt
            if total_non_gt else np.nan
    })

per_image = pd.DataFrame(per_image_rows)

per_image.to_csv(
    OUTPUT_DIR / "ringkasan_per_citra.csv",
    index=False
)

print("\n" + "=" * 70)
print("3. RINGKASAN PER CITRA")
print("=" * 70)

print(per_image.round(4).to_string(index=False))


# ============================================================
# 4. KORELASI KARAKTERISTIK KOMPONEN
# ============================================================

correlation_rows = []

correlation_pairs = [
    ("component_size", "gt_ratio"),
    ("component_size", "non_gt_ratio"),
    ("component_size", "strong_pixels"),
    ("strong_pixels", "non_gt_ratio"),
    ("strong_pixels", "gt_ratio")
]

for config, group in df.groupby("configuration"):

    for x_col, y_col in correlation_pairs:

        valid = group[[x_col, y_col]].dropna()

        rho, p_value = spearmanr(
            valid[x_col],
            valid[y_col]
        )

        correlation_rows.append({
            "configuration": config,
            "x": x_col,
            "y": y_col,
            "n": len(valid),
            "spearman_rho": rho,
            "p_value": p_value
        })

correlations = pd.DataFrame(correlation_rows)

correlations.to_csv(
    OUTPUT_DIR / "korelasi_karakteristik_komponen.csv",
    index=False
)

print("\n" + "=" * 70)
print("4. KORELASI SPEARMAN")
print("=" * 70)

print(correlations.round(5).to_string(index=False))


# ============================================================
# 5. DISTRIBUSI UKURAN KOMPONEN
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, config in zip(axes, ["B", "D"]):

    subset = df[df["configuration"] == config]

    for status in ["DIBUANG", "DIPERTAHANKAN"]:

        values = subset.loc[
            subset["status"] == status,
            "component_size"
        ]

        ax.hist(
            np.log10(values),
            bins=40,
            alpha=0.6,
            label=status
        )

    ax.set_title(f"Konfigurasi {config}")
    ax.set_xlabel("log10(Ukuran komponen)")
    ax.set_ylabel("Jumlah komponen")
    ax.legend()

fig.suptitle("Distribusi Ukuran Komponen")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "distribusi_ukuran_komponen.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 6. DISTRIBUSI GT RATIO
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, config in zip(axes, ["B", "D"]):

    subset = df[df["configuration"] == config]

    data = [
        subset.loc[
            subset["status"] == status,
            "gt_ratio"
        ].values
        for status in ["DIBUANG", "DIPERTAHANKAN"]
    ]

    ax.boxplot(
        data,
        tick_labels=["Dibuang", "Dipertahankan"],
        showfliers=False
    )

    ax.set_title(f"Konfigurasi {config}")
    ax.set_ylabel("Proporsi GT")
    ax.set_ylim(-0.05, 1.05)

fig.suptitle("Distribusi Proporsi GT per Komponen")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "distribusi_gt_ratio.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 7. DISTRIBUSI NON-GT RATIO
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, config in zip(axes, ["B", "D"]):

    subset = df[df["configuration"] == config]

    data = [
        subset.loc[
            subset["status"] == status,
            "non_gt_ratio"
        ].values
        for status in ["DIBUANG", "DIPERTAHANKAN"]
    ]

    ax.boxplot(
        data,
        tick_labels=["Dibuang", "Dipertahankan"],
        showfliers=False
    )

    ax.set_title(f"Konfigurasi {config}")
    ax.set_ylabel("Proporsi non-GT")
    ax.set_ylim(-0.05, 1.05)

fig.suptitle("Distribusi Proporsi Non-GT per Komponen")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "distribusi_non_gt_ratio.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 8. HUBUNGAN UKURAN DAN PROPORSI GT
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, config in zip(axes, ["B", "D"]):

    subset = df[df["configuration"] == config]

    for status in ["DIBUANG", "DIPERTAHANKAN"]:

        part = subset[subset["status"] == status]

        ax.scatter(
            part["component_size"],
            part["gt_ratio"],
            s=8,
            alpha=0.3,
            label=status
        )

    ax.set_xscale("log")
    ax.set_title(f"Konfigurasi {config}")
    ax.set_xlabel("Ukuran komponen (log)")
    ax.set_ylabel("Proporsi GT")
    ax.legend()

fig.suptitle("Hubungan Ukuran Komponen dan Proporsi GT")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "ukuran_vs_gt_ratio.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# SELESAI
# ============================================================

print("\n" + "=" * 70)
print("ANALISIS SELESAI")
print("=" * 70)

print("Lokasi hasil:")
print(OUTPUT_DIR)

print("\nFile yang dihasilkan:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print("-", path.name)
