
import os
import pandas as pd

# ============================================================
# ANALISIS PER KATEGORI - EMPAT KONFIGURASI TERKONTROL
# ============================================================

INPUT_FILE = (
    "results/03_evaluation/kernel_terkontrol/hasil_detail.csv"
)

OUTPUT_DIR = (
    "results/06_error_analysis/per_kategori_kernel"
)

METRICS = ["precision", "recall", "dice", "iou"]

CONFIG_ORDER = ["A", "B", "C", "D"]


def main():
    print("=" * 70)
    print("ANALISIS PER KATEGORI - KERNEL TERKONTROL")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "filename",
        "category",
        "configuration",
        *METRICS
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing}"
        )

    # Pastikan setiap citra hanya memiliki satu hasil
    # untuk masing-masing konfigurasi.
    duplicates = df.duplicated(
        subset=["filename", "configuration"]
    )

    if duplicates.any():
        raise ValueError(
            "Ditemukan duplikasi kombinasi filename "
            "dan configuration."
        )

    # Ringkasan per kategori dan konfigurasi
    summary = (
        df.groupby(
            ["category", "configuration"],
            observed=True
        )
        .agg(
            n=("filename", "count"),

            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),

            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),

            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),

            iou_mean=("iou", "mean"),
            iou_std=("iou", "std")
        )
        .reset_index()
    )

    # Mengurutkan konfigurasi A, B, C, D
    summary["configuration"] = pd.Categorical(
        summary["configuration"],
        categories=CONFIG_ORDER,
        ordered=True
    )

    summary = summary.sort_values(
        ["category", "configuration"]
    )

    summary["configuration"] = (
        summary["configuration"].astype(str)
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "hasil_ringkasan_per_kategori.csv"
    )

    summary.to_csv(
        output_file,
        index=False,
        float_format="%.6f"
    )

    # Ringkasan jumlah citra per kategori
    category_counts = (
        df.groupby("category", observed=True)["filename"]
        .nunique()
        .reset_index(name="jumlah_citra")
    )

    count_file = os.path.join(
        OUTPUT_DIR,
        "jumlah_citra_per_kategori.csv"
    )

    category_counts.to_csv(
        count_file,
        index=False
    )

    # Tampilkan hasil
    print("\nJUMLAH CITRA PER KATEGORI")
    print("-" * 70)
    print(category_counts.to_string(index=False))

    print("\nHASIL RINGKASAN PER KATEGORI")
    print("-" * 70)

    for category in summary["category"].unique():
        subset = summary[
            summary["category"] == category
        ]

        print(f"\nKategori: {category}")
        print(
            subset[
                [
                    "configuration",
                    "n",
                    "precision_mean",
                    "recall_mean",
                    "dice_mean",
                    "iou_mean"
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    print("\n" + "=" * 70)
    print("ANALISIS SELESAI")
    print("=" * 70)
    print(f"Ringkasan : {output_file}")
    print(f"Jumlah    : {count_file}")


if __name__ == "__main__":
    main()
