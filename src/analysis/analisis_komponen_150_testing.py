
import cv2
import numpy as np
import pandas as pd
import time

from pathlib import Path
from src.core.black_tophat import multi_directional_black_tophat


# ============================================================
# KONFIGURASI DIREKTORI
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

SPLIT_FILE = ROOT / "results" / "pembagian_dataset.csv"

# Folder baru agar hasil lama tidak tertimpa.
OUTPUT_DIR = (
    ROOT / "results" /
    "05_component_analysis" /
    "komponen_150_gray"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_FILE = OUTPUT_DIR / "detail_per_komponen.csv"
SUMMARY_FILE = OUTPUT_DIR / "ringkasan_per_citra.csv"


# ============================================================
# KONFIGURASI EKSPERIMEN
# ============================================================

CONFIGURATIONS = {
    "B": {
        "kernel": 9,
        "low": 10,
        "high": 30
    },
    "D": {
        "kernel": 11,
        "low": 15,
        "high": 40
    }
}


# ============================================================
# FUNGSI HYSTERESIS
# ============================================================

def hysteresis_components(response, low, high):

    weak = (response >= low).astype(np.uint8)
    strong = (response >= high).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(
        weak,
        connectivity=8
    )

    return weak, strong, num_labels, labels


# ============================================================
# MEMUAT DATA PENGUJIAN
# ============================================================

def load_testing_data():

    split_df = pd.read_csv(SPLIT_FILE)

    testing_df = split_df[
        split_df["split"].astype(str).str.upper() == "PENGUJIAN"
    ].copy()

    if len(testing_df) != 150:
        raise ValueError(
            f"Jumlah citra pengujian harus 150, "
            f"tetapi ditemukan {len(testing_df)}."
        )

    if testing_df["filename"].duplicated().any():
        raise ValueError("Ditemukan filename duplikat.")

    return testing_df


# ============================================================
# VALIDASI DIREKTORI DAN FILE
# ============================================================

def validate_files(testing_df):

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Folder citra tidak ditemukan: {IMAGE_DIR}"
        )

    if not MASK_DIR.exists():
        raise FileNotFoundError(
            f"Folder mask tidak ditemukan: {MASK_DIR}"
        )

    missing_images = []
    missing_masks = []

    for filename in testing_df["filename"]:

        if not (IMAGE_DIR / filename).exists():
            missing_images.append(filename)

        if not (MASK_DIR / filename).exists():
            missing_masks.append(filename)

    if missing_images or missing_masks:

        print("\nFile tidak ditemukan.")

        if missing_images:
            print("Citra:", missing_images[:10])

        if missing_masks:
            print("Mask:", missing_masks[:10])

        raise FileNotFoundError(
            "Periksa nama file dan direktori dataset."
        )


# ============================================================
# ANALISIS SATU CITRA DAN SATU KONFIGURASI
# ============================================================

def analyze_one_configuration(
    filename,
    image,
    gt_mask,
    configuration,
    params
):

    kernel = params["kernel"]
    low = params["low"]
    high = params["high"]

    # --------------------------------------------------------
    # BLACK TOP-HAT
    # --------------------------------------------------------

    response = multi_directional_black_tophat(
        image,
        length=kernel
    )

    # --------------------------------------------------------
    # WEAK MASK, STRONG MASK, CONNECTED COMPONENTS
    # --------------------------------------------------------

    weak, strong, num_labels, labels = hysteresis_components(
        response,
        low,
        high
    )

    # --------------------------------------------------------
    # STATISTIK PER KOMPONEN
    # --------------------------------------------------------

    component_sizes = np.bincount(
        labels.ravel(),
        minlength=num_labels
    )

    gt_counts = np.bincount(
        labels[gt_mask],
        minlength=num_labels
    )

    strong_counts = np.bincount(
        labels[strong == 1],
        minlength=num_labels
    )

    # Label 0 merupakan background.
    component_ids = np.arange(1, num_labels)

    sizes = component_sizes[1:]
    gt_pixels = gt_counts[1:]
    strong_pixels = strong_counts[1:]

    non_gt_pixels = sizes - gt_pixels

    # Komponen dipertahankan jika memiliki minimal
    # satu piksel strong.
    retained_flags = strong_pixels > 0

    # --------------------------------------------------------
    # DETAIL PER KOMPONEN
    # --------------------------------------------------------

    detail_df = pd.DataFrame({
        "filename": filename,
        "configuration": configuration,
        "kernel": kernel,
        "component_id": component_ids,
        "component_size": sizes,
        "gt_pixels": gt_pixels,
        "non_gt_pixels": non_gt_pixels,
        "strong_pixels": strong_pixels,
        "contains_gt": gt_pixels > 0,
        "status": np.where(
            retained_flags,
            "DIPERTAHANKAN",
            "DIBUANG"
        )
    })

    # --------------------------------------------------------
    # RINGKASAN SATU CITRA
    # --------------------------------------------------------

    total_components = len(component_ids)

    retained_components = int(
        retained_flags.sum()
    )

    total_weak_pixels = int(weak.sum())

    total_gt = int(gt_mask.sum())

    total_gt_in_weak = int(gt_pixels.sum())

    total_non_gt_in_weak = int(non_gt_pixels.sum())

    retained_gt = int(
        gt_pixels[retained_flags].sum()
    )

    retained_non_gt = int(
        non_gt_pixels[retained_flags].sum()
    )

    retained_pixels = retained_gt + retained_non_gt

    gt_outside_weak = total_gt - total_gt_in_weak

    summary = {
        "filename": filename,
        "configuration": configuration,
        "kernel": kernel,
        "low_threshold": low,
        "high_threshold": high,

        "total_components": total_components,
        "retained_components": retained_components,

        "component_retention_pct":
            100 * retained_components / total_components
            if total_components else 0,

        "total_weak_pixels": total_weak_pixels,
        "retained_pixels": retained_pixels,

        "pixel_retention_pct":
            100 * retained_pixels / total_weak_pixels
            if total_weak_pixels else 0,

        "total_gt_pixels": total_gt,
        "total_gt_in_weak": total_gt_in_weak,
        "gt_outside_weak": gt_outside_weak,

        "retained_gt": retained_gt,

        "gt_retention_pct":
            100 * retained_gt / total_gt_in_weak
            if total_gt_in_weak else np.nan,

        "total_non_gt_in_weak": total_non_gt_in_weak,
        "retained_non_gt": retained_non_gt,

        "non_gt_retention_pct":
            100 * retained_non_gt / total_non_gt_in_weak
            if total_non_gt_in_weak else np.nan
    }

    return detail_df, summary


# ============================================================
# PROSES SELURUH CITRA
# ============================================================

def main():

    print("=" * 70)
    print("ANALISIS KOMPONEN HYSTERESIS - 150 CITRA PENGUJIAN")
    print("GRAYSCALE: IMREAD COLOR + CVT COLOR")
    print("=" * 70)

    testing_df = load_testing_data()

    validate_files(testing_df)

    print("\nJumlah citra pengujian:", len(testing_df))
    print("Jumlah konfigurasi:", len(CONFIGURATIONS))
    print(
        "Total kombinasi:",
        len(testing_df) * len(CONFIGURATIONS)
    )

    print("\nKonfigurasi:")

    for name, params in CONFIGURATIONS.items():
        print(
            f"{name}: K{params['kernel']}, "
            f"low={params['low']}, high={params['high']}"
        )

    # --------------------------------------------------------
    # RESET OUTPUT BARU
    # --------------------------------------------------------

    if DETAIL_FILE.exists():
        DETAIL_FILE.unlink()

    all_summaries = []

    start_time = time.time()

    # --------------------------------------------------------
    # LOOP CITRA
    # --------------------------------------------------------

    for image_index, row in enumerate(
        testing_df.itertuples(index=False),
        start=1
    ):

        filename = row.filename

        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / filename

        # ----------------------------------------------------
        # BACA CITRA DALAM FORMAT BGR
        # SAMA DENGAN EVALUASI KERNEL TERKONTROL
        # ----------------------------------------------------

        color_image = cv2.imread(
            str(image_path)
        )

        if color_image is None:
            raise ValueError(
                f"Gagal membaca citra: {image_path}"
            )

        # Konversi BGR ke grayscale.
        image = cv2.cvtColor(
            color_image,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # BACA GROUND TRUTH
        # ----------------------------------------------------

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:
            raise ValueError(
                f"Gagal membaca mask: {mask_path}"
            )

        if image.shape != mask.shape:
            raise ValueError(
                f"Ukuran citra dan mask berbeda: {filename}"
            )

        # Foreground ground truth adalah piksel > 0.
        gt_mask = mask > 0

        # ----------------------------------------------------
        # PROSES KONFIGURASI B DAN D
        # ----------------------------------------------------

        for configuration, params in CONFIGURATIONS.items():

            detail_df, summary = analyze_one_configuration(
                filename=filename,
                image=image,
                gt_mask=gt_mask,
                configuration=configuration,
                params=params
            )

            # Simpan detail secara bertahap.
            detail_df.to_csv(
                DETAIL_FILE,
                mode="a",
                header=not DETAIL_FILE.exists(),
                index=False
            )

            all_summaries.append(summary)

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if image_index % 10 == 0 or image_index == 1:

            elapsed = time.time() - start_time

            print(
                f"[{image_index:3d}/150] "
                f"{filename} | "
                f"{elapsed:.1f} detik"
            )

    # ========================================================
    # SIMPAN RINGKASAN
    # ========================================================

    summary_df = pd.DataFrame(all_summaries)

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("ANALISIS SELESAI")
    print("=" * 70)

    print("Jumlah citra:", len(testing_df))
    print("Jumlah konfigurasi:", len(CONFIGURATIONS))
    print("Jumlah baris ringkasan:", len(summary_df))

    print(
        "Jumlah komponen:",
        f"{sum(summary_df['total_components']):,}"
    )

    print("Waktu proses:", f"{elapsed:.2f} detik")

    print("\nLokasi hasil:")
    print(OUTPUT_DIR)

    print("\nFile yang dihasilkan:")
    print("-", DETAIL_FILE.name)
    print("-", SUMMARY_FILE.name)

    # ========================================================
    # AGREGASI SELURUH CITRA
    # ========================================================

    aggregate = summary_df.groupby("configuration").agg(
        total_components=("total_components", "sum"),
        retained_components=("retained_components", "sum"),
        total_weak_pixels=("total_weak_pixels", "sum"),
        retained_pixels=("retained_pixels", "sum"),
        total_gt_in_weak=("total_gt_in_weak", "sum"),
        retained_gt=("retained_gt", "sum"),
        total_non_gt_in_weak=("total_non_gt_in_weak", "sum"),
        retained_non_gt=("retained_non_gt", "sum"),
        gt_outside_weak=("gt_outside_weak", "sum")
    )

    aggregate["component_retention_pct"] = (
        aggregate["retained_components"] /
        aggregate["total_components"] * 100
    )

    aggregate["pixel_retention_pct"] = (
        aggregate["retained_pixels"] /
        aggregate["total_weak_pixels"] * 100
    )

    aggregate["gt_retention_pct"] = (
        aggregate["retained_gt"] /
        aggregate["total_gt_in_weak"] * 100
    )

    aggregate["non_gt_retention_pct"] = (
        aggregate["retained_non_gt"] /
        aggregate["total_non_gt_in_weak"] * 100
    )

    print("\nRINGKASAN AGREGAT:")
    print(aggregate.round(3).to_string())


if __name__ == "__main__":
    main()
