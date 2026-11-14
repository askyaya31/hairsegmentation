import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from black_tophat import multi_directional_black_tophat

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"
SPLIT_FILE = ROOT / "results" / "pembagian_dataset.csv"

CONFIGS = {
    "B": {"kernel": 9, "low": 10, "high": 30},
    "D": {"kernel": 11, "low": 15, "high": 40}
}


def main():
    split = pd.read_csv(SPLIT_FILE)
    testing = split[
        split["split"].str.upper() == "PENGUJIAN"
    ]

    mismatch_gray = 0
    mismatch_tp = 0
    mismatch_fp = 0
    mismatch_prediction = 0
    total = 0

    for row in testing.itertuples(index=False):
        filename = row.filename

        color = cv2.imread(str(IMAGE_DIR / filename))
        gray_direct = cv2.imread(
            str(IMAGE_DIR / filename),
            cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.imread(
            str(MASK_DIR / filename),
            cv2.IMREAD_GRAYSCALE
        )

        if color is None or gray_direct is None or mask is None:
            raise ValueError(f"Gagal membaca {filename}")

        gray_converted = cv2.cvtColor(
            color, cv2.COLOR_BGR2GRAY
        )

        same_gray = np.array_equal(
            gray_direct, gray_converted
        )

        if not same_gray:
            mismatch_gray += 1

        gt = mask > 0

        for name, params in CONFIGS.items():
            response = multi_directional_black_tophat(
                gray_converted,
                length=params["kernel"]
            )

            weak = (
                response >= params["low"]
            ).astype(np.uint8)

            strong = (
                response >= params["high"]
            ).astype(np.uint8)

            num_labels, labels = cv2.connectedComponents(
                weak, connectivity=8
            )

            strong_labels = np.unique(
                labels[strong == 1]
            )

            valid_labels = strong_labels[
                strong_labels != 0
            ]

            prediction = np.isin(
                labels, valid_labels
            )

            # Metrik dari prediksi akhir
            tp = np.logical_and(prediction, gt).sum()
            fp = np.logical_and(prediction, ~gt).sum()

            # Perhitungan berdasarkan komponen
            gt_counts = np.bincount(
                labels.ravel()[gt.ravel()],
                minlength=num_labels
            )

            component_sizes = np.bincount(
                labels.ravel(),
                minlength=num_labels
            )

            non_gt_counts = component_sizes - gt_counts

            retained_gt = gt_counts[valid_labels].sum()
            retained_non_gt = non_gt_counts[valid_labels].sum()

            if tp != retained_gt:
                mismatch_tp += 1

            if fp != retained_non_gt:
                mismatch_fp += 1

            # Pastikan prediksi sama dengan gabungan komponen
            if not np.array_equal(
                prediction,
                np.isin(labels, valid_labels)
            ):
                mismatch_prediction += 1

            total += 1

    print("=" * 55)
    print("HASIL PEMERIKSAAN KONSISTENSI")
    print("=" * 55)
    print("Total kombinasi:", total)
    print("Grayscale berbeda:", mismatch_gray)
    print("TP berbeda:", mismatch_tp)
    print("FP berbeda:", mismatch_fp)
    print("Prediksi berbeda:", mismatch_prediction)


if __name__ == "__main__":
    main()
