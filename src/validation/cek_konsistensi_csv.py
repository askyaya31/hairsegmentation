
import pandas as pd
import numpy as np

from pathlib import Path


# ============================================================
# KONFIGURASI DIREKTORI
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

COMPONENT_FILE = (
    ROOT / "results" /
    "05_component_analysis" /
    "komponen_150_gray" /
    "ringkasan_per_citra.csv"
)

EVALUATION_FILE = (
    ROOT
    / "results"
    / "03_evaluation"
    / "kernel_terkontrol"
    / "hasil_detail.csv"
)

CONFIGS = ["B", "D"]


# ============================================================
# PEMERIKSAAN FILE
# ============================================================

def check_files():
    print("=" * 65)
    print("PEMERIKSAAN KONSISTENSI CSV")
    print("=" * 65)

    for file_path in [COMPONENT_FILE, EVALUATION_FILE]:
        print(f"\nFile: {file_path}")
        print("Ditemukan:", file_path.exists())

        if not file_path.exists():
            raise FileNotFoundError(
                f"File tidak ditemukan: {file_path}"
            )


# ============================================================
# PEMERIKSAAN KONSISTENSI
# ============================================================

def check_consistency(components, evaluation, config):

    comp = components[
        components["configuration"] == config
    ].copy()

    ev = evaluation[
        evaluation["configuration"] == config
    ].copy()

    print("\n" + "=" * 65)
    print(f"KONFIGURASI {config}")
    print("=" * 65)

    print("Jumlah data komponen:", len(comp))
    print("Jumlah data evaluasi:", len(ev))

    # --------------------------------------------------------
    # Periksa duplikasi filename
    # --------------------------------------------------------

    duplicate_comp = comp["filename"].duplicated().sum()
    duplicate_eval = ev["filename"].duplicated().sum()

    print("\nDuplikasi filename:")
    print("Komponen:", duplicate_comp)
    print("Evaluasi:", duplicate_eval)

    # --------------------------------------------------------
    # Gabungkan berdasarkan filename
    # --------------------------------------------------------

    merged = comp.merge(
        ev,
        on="filename",
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    print("\nStatus penggabungan:")
    print(merged["_merge"].value_counts().to_string())

    matched = merged[
        merged["_merge"] == "both"
    ].copy()

    print("\nJumlah pasangan yang cocok:", len(matched))

    if matched.empty:
        print("Tidak ada pasangan data yang cocok.")
        return

    # --------------------------------------------------------
    # Bandingkan TP dan retained GT
    # --------------------------------------------------------

    tp_diff = (
        matched["retained_gt"] - matched["tp"]
    )

    tp_mismatch = (tp_diff != 0).sum()

    # --------------------------------------------------------
    # Bandingkan FP dan retained non-GT
    # --------------------------------------------------------

    fp_diff = (
        matched["retained_non_gt"] - matched["fp"]
    )

    fp_mismatch = (fp_diff != 0).sum()

    # --------------------------------------------------------
    # Bandingkan FN dengan GT total - retained GT
    # --------------------------------------------------------

    expected_fn = (
        matched["total_gt_pixels"] -
        matched["retained_gt"]
    )

    fn_diff = expected_fn - matched["fn"]

    fn_mismatch = (fn_diff != 0).sum()

    # --------------------------------------------------------
    # Tampilkan hasil
    # --------------------------------------------------------

    print("\nHASIL PERBANDINGAN")
    print("-" * 45)

    print(
        f"TP berbeda              : "
        f"{tp_mismatch}/{len(matched)}"
    )

    print(
        f"FP berbeda              : "
        f"{fp_mismatch}/{len(matched)}"
    )

    print(
        f"FN berbeda              : "
        f"{fn_mismatch}/{len(matched)}"
    )

    print("\nSelisih maksimum absolut:")
    print("TP:", tp_diff.abs().max())
    print("FP:", fp_diff.abs().max())
    print("FN:", fn_diff.abs().max())

    # --------------------------------------------------------
    # Simpan detail perbandingan
    # --------------------------------------------------------

    comparison = matched[
        [
            "filename",
            "retained_gt",
            "tp",
            "retained_non_gt",
            "fp",
            "total_gt_pixels",
            "fn"
        ]
    ].copy()

    comparison["selisih_tp"] = tp_diff
    comparison["selisih_fp"] = fp_diff
    comparison["selisih_fn"] = fn_diff

    output_dir = (
        ROOT / "results" / "09_validation" / "konsistensi_csv"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir /
        f"perbandingan_{config}.csv"
    )

    comparison.to_csv(
        output_file,
        index=False
    )

    print("\nFile perbandingan disimpan:")
    print(output_file)

    # --------------------------------------------------------
    # Kesimpulan
    # --------------------------------------------------------

    if (
        tp_mismatch == 0
        and fp_mismatch == 0
        and fn_mismatch == 0
        and len(matched) == len(comp) == len(ev)
    ):
        print("\nKESIMPULAN: SELURUH DATA KONSISTEN.")
    else:
        print("\nKESIMPULAN: DITEMUKAN KETIDAKCOCOKAN DATA.")


# ============================================================
# MAIN
# ============================================================

def main():

    check_files()

    components = pd.read_csv(COMPONENT_FILE)
    evaluation = pd.read_csv(EVALUATION_FILE)

    print("\nJumlah baris CSV komponen:", len(components))
    print("Jumlah baris CSV evaluasi:", len(evaluation))

    for config in CONFIGS:
        check_consistency(
            components,
            evaluation,
            config
        )

    print("\n" + "=" * 65)
    print("PEMERIKSAAN SELESAI")
    print("=" * 65)


if __name__ == "__main__":
    main()
