
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from src.core.black_tophat import multi_directional_black_tophat

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

OUTPUT_DIR = (
    ROOT
    / "results"
    / "analisis_komponen_ground_truth"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ENAM CITRA KASUS EKSTREM
# ============================================================

SELECTED_IMAGES = [
    "ISIC_0009945.png",
    "ISIC_0016019.png",
    "ISIC_0000456.png",
    "ISIC_0014968.png",
    "ISIC_0014806.png",
    "ISIC_0015418.png",
]


# ============================================================
# PARAMETER HYSTERESIS
# ============================================================

CONFIGURATIONS = [
    {
        "configuration": "B",
        "kernel": 9,
        "low": 10,
        "high": 30
    },
    {
        "configuration": "D",
        "kernel": 11,
        "low": 15,
        "high": 40
    }
]


# ============================================================
# ANALISIS KOMPONEN TERHUBUNG
# ============================================================

def analyze_components(response, ground_truth, config):

    low = config["low"]
    high = config["high"]

    weak = (response >= low).astype(np.uint8)
    strong = (response >= high).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(
        weak,
        connectivity=8
    )

    strong_labels = np.unique(
        labels[strong == 1]
    )

    valid_labels = strong_labels[
        strong_labels != 0
    ]

    retained_label_set = set(
        valid_labels.tolist()
    )

    # --------------------------------------------------------
    # IDENTIFIKASI STATUS SETIAP KOMPONEN
    # --------------------------------------------------------

    component_records = []

    total_retained_gt = 0
    total_retained_non_gt = 0

    total_discarded_gt = 0
    total_discarded_non_gt = 0

    retained_components_with_gt = 0
    retained_components_without_gt = 0

    discarded_components_with_gt = 0
    discarded_components_without_gt = 0

    for label_id in range(1, num_labels):

        component_mask = (
            labels == label_id
        )

        component_size = int(
            component_mask.sum()
        )

        gt_pixels = int(
            np.logical_and(
                component_mask,
                ground_truth
            ).sum()
        )

        non_gt_pixels = (
            component_size - gt_pixels
        )

        strong_pixels = int(
            np.logical_and(
                component_mask,
                strong
            ).sum()
        )

        has_gt = gt_pixels > 0

        retained = (
            label_id in retained_label_set
        )

        if retained:

            status = "DIPERTAHANKAN"

            total_retained_gt += gt_pixels
            total_retained_non_gt += non_gt_pixels

            if has_gt:
                retained_components_with_gt += 1
            else:
                retained_components_without_gt += 1

        else:

            status = "DIBUANG"

            total_discarded_gt += gt_pixels
            total_discarded_non_gt += non_gt_pixels

            if has_gt:
                discarded_components_with_gt += 1
            else:
                discarded_components_without_gt += 1

        component_records.append({
            "filename": config["filename"],
            "configuration": config["configuration"],
            "kernel": config["kernel"],
            "component_id": label_id,
            "component_size": component_size,
            "gt_pixels": gt_pixels,
            "non_gt_pixels": non_gt_pixels,
            "strong_pixels": strong_pixels,
            "contains_gt": has_gt,
            "status": status
        })

    # --------------------------------------------------------
    # PIXEL GT DI LUAR WEAK MASK
    # --------------------------------------------------------

    gt_outside_weak = int(
        np.logical_and(
            ground_truth,
            weak == 0
        ).sum()
    )

    # --------------------------------------------------------
    # RINGKASAN PIXEL
    # --------------------------------------------------------

    gt_total = int(ground_truth.sum())

    weak_pixels = int(weak.sum())
    strong_pixels = int(strong.sum())

    retained_pixels = (
        total_retained_gt
        + total_retained_non_gt
    )

    discarded_pixels = (
        total_discarded_gt
        + total_discarded_non_gt
    )

    retained_prediction = (
        np.isin(labels, valid_labels).astype(np.uint8)
    )

    summary = {
        "filename": config["filename"],
        "configuration": config["configuration"],
        "kernel": config["kernel"],
        "low_threshold": low,
        "high_threshold": high,

        "total_gt_pixels": gt_total,

        "total_weak_components": num_labels - 1,
        "retained_components": len(valid_labels),
        "discarded_components": (
            num_labels - 1 - len(valid_labels)
        ),

        "retained_components_with_gt":
            retained_components_with_gt,

        "retained_components_without_gt":
            retained_components_without_gt,

        "discarded_components_with_gt":
            discarded_components_with_gt,

        "discarded_components_without_gt":
            discarded_components_without_gt,

        "weak_pixels": weak_pixels,
        "strong_pixels": strong_pixels,

        "retained_pixels": retained_pixels,
        "discarded_pixels": discarded_pixels,

        "retained_gt_pixels": total_retained_gt,
        "retained_non_gt_pixels": total_retained_non_gt,

        "discarded_gt_pixels": total_discarded_gt,
        "discarded_non_gt_pixels": total_discarded_non_gt,

        "gt_outside_weak_pixels": gt_outside_weak,

        "gt_coverage_in_weak": (
            (gt_total - gt_outside_weak) / gt_total
            if gt_total > 0 else np.nan
        ),

        "gt_retention_ratio": (
            total_retained_gt / gt_total
            if gt_total > 0 else np.nan
        ),

        "non_gt_fraction_retained": (
            total_retained_non_gt / retained_pixels
            if retained_pixels > 0 else np.nan
        ),

        "prediction_gt_fraction": (
            total_retained_gt / retained_pixels
            if retained_pixels > 0 else np.nan
        )
    }

    return component_records, summary


# ============================================================
# MAIN
# ============================================================

def main():

    all_component_records = []
    all_summary_records = []

    print("=" * 70)
    print("ANALISIS KOMPONEN TERHADAP GROUND TRUTH")
    print("=" * 70)

    print("Jumlah citra:", len(SELECTED_IMAGES))
    print("Konfigurasi: B (K9) dan D (K11)")
    print()

    for filename in SELECTED_IMAGES:

        image_path = IMAGE_DIR / filename
        mask_path = MASK_DIR / filename

        image = cv2.imread(str(image_path))

        ground_truth = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(image_path)

        if ground_truth is None:
            raise FileNotFoundError(mask_path)

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        ground_truth = (
            ground_truth > 0
        ).astype(np.uint8)

        print(f"\nCitra: {filename}")

        for config in CONFIGURATIONS:

            kernel = config["kernel"]

            response = multi_directional_black_tophat(
                gray,
                length=kernel
            )

            current_config = {
                **config,
                "filename": filename
            }

            component_records, summary = (
                analyze_components(
                    response,
                    ground_truth,
                    current_config
                )
            )

            all_component_records.extend(
                component_records
            )

            all_summary_records.append(summary)

            print(
                f"K{kernel} | "
                f"Komponen: {summary['total_weak_components']} | "
                f"Lolos: {summary['retained_components']} | "
                f"GT lolos: {summary['retained_gt_pixels']} | "
                f"GT dibuang: {summary['discarded_gt_pixels']} | "
                f"GT di luar weak: {summary['gt_outside_weak_pixels']}"
            )

    # ========================================================
    # SIMPAN HASIL PER KOMPONEN
    # ========================================================

    component_df = pd.DataFrame(
        all_component_records
    )

    component_path = (
        OUTPUT_DIR / "detail_per_komponen.csv"
    )

    component_df.to_csv(
        component_path,
        index=False
    )

    # ========================================================
    # SIMPAN RINGKASAN PER CITRA
    # ========================================================

    summary_df = pd.DataFrame(
        all_summary_records
    )

    summary_path = (
        OUTPUT_DIR / "ringkasan_komponen_gt.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # ========================================================
    # TAMPILKAN RINGKASAN
    # ========================================================

    display_columns = [
        "filename",
        "configuration",
        "retained_components_with_gt",
        "retained_components_without_gt",
        "discarded_components_with_gt",
        "discarded_components_without_gt",
        "retained_gt_pixels",
        "retained_non_gt_pixels",
        "discarded_gt_pixels",
        "discarded_non_gt_pixels",
        "gt_outside_weak_pixels"
    ]

    print("\n" + "=" * 70)
    print("RINGKASAN ANALISIS KOMPONEN")
    print("=" * 70)

    print(
        summary_df[display_columns].to_string(
            index=False
        )
    )

    print("\nFile hasil:")
    print("Detail komponen :", component_path)
    print("Ringkasan       :", summary_path)

    print("\nAnalisis selesai.")


if __name__ == "__main__":
    main()
