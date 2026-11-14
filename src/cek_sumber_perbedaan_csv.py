
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from black_tophat import multi_directional_black_tophat


# ============================================================
# DIREKTORI
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

COMPONENT_FILE = (
    ROOT / "results" /
    "analisis_komponen_150_testing" /
    "ringkasan_per_citra.csv"
)

EVALUATION_FILE = (
    ROOT / "results" /
    "evaluasi_kernel_terkontrol" /
    "hasil_detail.csv"
)

OUTPUT_DIR = ROOT / "results" / "cek_sumber_perbedaan_csv"

CONFIGS = {
    "B": {"kernel": 9, "low": 10, "high": 30},
    "D": {"kernel": 11, "low": 15, "high": 40}
}


# ============================================================
# HITUNG RINGKASAN KOMPONEN
# ============================================================

def calculate_component_summary(gray, gt, config):

    params = CONFIGS[config]

    response = multi_directional_black_tophat(
        gray,
        length=params["kernel"]
    )

    weak = (response >= params["low"]).astype(np.uint8)
    strong = (response >= params["high"]).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(
        weak,
        connectivity=8
    )

    component_sizes = np.bincount(
        labels.ravel(),
        minlength=num_labels
    )

    gt_counts = np.bincount(
        labels.ravel()[gt.ravel()],
        minlength=num_labels
    )

    strong_counts = np.bincount(
        labels.ravel()[strong.ravel() == 1],
        minlength=num_labels
    )

    # Abaikan label 0 karena merupakan background
    sizes = component_sizes[1:]
    gt_pixels = gt_counts[1:]
    strong_pixels = strong_counts[1:]

    non_gt_pixels = sizes - gt_pixels

    retained = strong_pixels > 0

    total_components = len(sizes)
    retained_components = int(retained.sum())

    total_weak_pixels = int(sizes.sum())
    retained_pixels = int(sizes[retained].sum())

    total_gt_pixels = int(gt.sum())
    total_gt_in_weak = int(gt_pixels.sum())

    retained_gt = int(gt_pixels[retained].sum())
    retained_non_gt = int(non_gt_pixels[retained].sum())

    total_non_gt_in_weak = int(non_gt_pixels.sum())

    return {
        "total_components": total_components,
        "retained_components": retained_components,
        "total_weak_pixels": total_weak_pixels,
        "retained_pixels": retained_pixels,
        "total_gt_pixels": total_gt_pixels,
        "total_gt_in_weak": total_gt_in_weak,
        "gt_outside_weak": total_gt_pixels - total_gt_in_weak,
        "retained_gt": retained_gt,
        "total_non_gt_in_weak": total_non_gt_in_weak,
        "retained_non_gt": retained_non_gt,
    }


# ============================================================
# PEMERIKSAAN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    components = pd.read_csv(COMPONENT_FILE)
    evaluation = pd.read_csv(EVALUATION_FILE)

    results = []

    numeric_columns = [
        "total_components",
        "retained_components",
        "total_weak_pixels",
        "retained_pixels",
        "total_gt_pixels",
        "total_gt_in_weak",
        "gt_outside_weak",
        "retained_gt",
        "total_non_gt_in_weak",
        "retained_non_gt"
    ]

    for row in components.itertuples(index=False):

        filename = row.filename
        config = row.configuration

        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / filename

        color = cv2.imread(str(image_path))
        gray_direct = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if color is None or gray_direct is None or mask is None:
            raise ValueError(f"Gagal membaca {filename}")

        gray_converted = cv2.cvtColor(
            color,
            cv2.COLOR_BGR2GRAY
        )

        gt = mask > 0

        # Hitung menggunakan kedua jalur grayscale
        direct = calculate_component_summary(
            gray_direct, gt, config
        )

        converted = calculate_component_summary(
            gray_converted, gt, config
        )

        # Ambil data CSV komponen lama
        old = row._asdict()

        result = {
            "filename": filename,
            "configuration": config,
        }

        for column in numeric_columns:

            old_value = old[column]

            result[f"old_{column}"] = old_value
            result[f"direct_{column}"] = direct[column]
            result[f"converted_{column}"] = converted[column]

            result[f"direct_match_{column}"] = (
                old_value == direct[column]
            )

            result[f"converted_match_{column}"] = (
                old_value == converted[column]
            )

        results.append(result)

    detail = pd.DataFrame(results)

    detail.to_csv(
        OUTPUT_DIR / "detail_pemeriksaan.csv",
        index=False
    )

    print("=" * 65)
    print("HASIL PEMERIKSAAN SUMBER PERBEDAAN")
    print("=" * 65)

    for config in CONFIGS:

        subset = detail[
            detail["configuration"] == config
        ]

        print(f"\nKONFIGURASI {config}")
        print("-" * 45)

        for source in ["direct", "converted"]:

            columns = [
                f"{source}_match_{column}"
                for column in numeric_columns
            ]

            all_match = subset[columns].all(axis=1)

            print(
                f"Seluruh kolom cocok dengan {source}: "
                f"{all_match.sum()}/{len(subset)} citra"
            )

            for column in numeric_columns:

                match_count = subset[
                    f"{source}_match_{column}"
                ].sum()

                print(
                    f"  {column}: "
                    f"{match_count}/{len(subset)}"
                )

    print("\nFile detail disimpan di:")
    print(OUTPUT_DIR / "detail_pemeriksaan.csv")

    print("\nPEMERIKSAAN SELESAI")


if __name__ == "__main__":
    main()
