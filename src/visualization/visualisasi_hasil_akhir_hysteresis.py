
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from src.core.black_tophat import multi_directional_black_tophat


# ============================================================
# KONFIGURASI DIREKTORI
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

OUTPUT_DIR = ROOT / "results" / "08_visualizations" / "hasil_akhir_hysteresis"
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
# PARAMETER EKSPERIMEN
# ============================================================

CONFIGURATIONS = {
    "A": {
        "method": "Baseline",
        "kernel": 9,
        "threshold": 20
    },
    "B": {
        "method": "Hysteresis",
        "kernel": 9,
        "low": 10,
        "high": 30
    },
    "C": {
        "method": "Baseline",
        "kernel": 11,
        "threshold": 20
    },
    "D": {
        "method": "Hysteresis",
        "kernel": 11,
        "low": 15,
        "high": 40
    }
}


# ============================================================
# FUNGSI HYSTERESIS
# Mengikuti implementasi evaluasi utama
# ============================================================

def hysteresis_threshold(response, low, high):

    weak = (response >= low).astype(np.uint8)
    strong = (response >= high).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(
        weak,
        connectivity=8
    )

    strong_labels = np.unique(labels[strong == 1])

    valid_labels = strong_labels[strong_labels != 0]

    result = np.isin(
        labels,
        valid_labels
    ).astype(np.uint8)

    # Statistik komponen
    total_components = num_labels - 1
    retained_components = len(valid_labels)
    discarded_components = (
        total_components - retained_components
    )

    weak_pixels = int(weak.sum())
    strong_pixels = int(strong.sum())
    retained_pixels = int(result.sum())
    discarded_pixels = weak_pixels - retained_pixels

    statistics = {
        "total_weak_components": total_components,
        "retained_components": retained_components,
        "discarded_components": discarded_components,
        "weak_pixels": weak_pixels,
        "strong_pixels": strong_pixels,
        "retained_pixels": retained_pixels,
        "discarded_pixels": discarded_pixels
    }

    return result, weak, strong, statistics


# ============================================================
# FUNGSI BASELINE
# ============================================================

def baseline_threshold(response, threshold):

    return (
        response >= threshold
    ).astype(np.uint8)


# ============================================================
# FUNGSI METRIK
# ============================================================

def calculate_metrics(prediction, ground_truth):

    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else np.nan
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else np.nan
    )

    dice = (
        1.0
        if 2 * tp + fp + fn == 0
        else (2 * tp) / (2 * tp + fp + fn)
    )

    iou = (
        1.0
        if tp + fp + fn == 0
        else tp / (tp + fp + fn)
    )

    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn)
    }


# ============================================================
# VISUALISASI FP DAN FN
# ============================================================

def create_error_map(prediction, ground_truth):

    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    # RGB:
    # Putih = background
    # Abu-abu = True Positive
    # Merah = False Positive
    # Biru = False Negative

    error_map = np.ones(
        (*pred.shape, 3),
        dtype=np.uint8
    ) * 255

    error_map[tp] = [180, 180, 180]
    error_map[fp] = [255, 0, 0]
    error_map[fn] = [0, 80, 255]

    return error_map


# ============================================================
# MEMBUAT SATU VISUALISASI
# ============================================================

def visualize_image(filename):

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

    original = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    ground_truth = (
        ground_truth > 0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # HITUNG RESPONS BTH
    # --------------------------------------------------------

    responses = {}

    for kernel in [9, 11]:

        responses[kernel] = multi_directional_black_tophat(
            gray,
            length=kernel
        )

    # --------------------------------------------------------
    # HASIL EMPAT KONFIGURASI
    # --------------------------------------------------------

    predictions = {}
    component_statistics = {}

    for config_id, config in CONFIGURATIONS.items():

        response = responses[config["kernel"]]

        if config["method"] == "Baseline":

            prediction = baseline_threshold(
                response,
                config["threshold"]
            )

        else:

            prediction, weak, strong, statistics = (
                hysteresis_threshold(
                    response,
                    config["low"],
                    config["high"]
                )
            )

            component_statistics[config_id] = statistics

        predictions[config_id] = prediction

    # --------------------------------------------------------
    # BUAT ERROR MAP
    # --------------------------------------------------------

    error_k9 = create_error_map(
        predictions["B"],
        ground_truth
    )

    error_k11 = create_error_map(
        predictions["D"],
        ground_truth
    )

    # --------------------------------------------------------
    # SIAPKAN PANEL
    # --------------------------------------------------------

    panels = [
        (original, "Original", None),
        (ground_truth, "Ground Truth", "gray"),
        (predictions["A"], "Baseline K9 (T20)", "gray"),
        (predictions["B"], "Hysteresis K9 (10/30)", "gray"),
        (predictions["C"], "Baseline K11 (T20)", "gray"),
        (predictions["D"], "Hysteresis K11 (15/40)", "gray"),
        (error_k9, "Error Map K9", None),
        (error_k11, "Error Map K11", None)
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 9)
    )

    axes = axes.flatten()

    for ax, (data, title, cmap) in zip(axes, panels):

        if cmap is None:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)

        ax.set_title(
            title,
            fontsize=11,
            fontweight="bold"
        )

        ax.axis("off")

    fig.suptitle(
        f"Hasil Segmentasi Hysteresis - {filename}",
        fontsize=14,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.015,
        "Error Map: Abu-abu = TP | Merah = FP | Biru = FN",
        ha="center",
        fontsize=10
    )

    plt.tight_layout(
        rect=[0, 0.04, 1, 0.94]
    )

    output_path = OUTPUT_DIR / (
        Path(filename).stem + "_hasil_akhir.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # SIMPAN METRIK DAN STATISTIK
    # --------------------------------------------------------

    records = []

    for config_id, config in CONFIGURATIONS.items():

        metrics = calculate_metrics(
            predictions[config_id],
            ground_truth
        )

        record = {
            "filename": filename,
            "configuration": config_id,
            "method": config["method"],
            "kernel": config["kernel"],
            **metrics
        }

        if config_id in component_statistics:
            record.update(
                component_statistics[config_id]
            )

        records.append(record)

    print(f"[OK] {filename}")

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VISUALISASI HASIL AKHIR HYSTERESIS")
    print("=" * 70)

    print(f"Jumlah citra: {len(SELECTED_IMAGES)}")
    print(f"Output      : {OUTPUT_DIR}")
    print()

    all_records = []

    for filename in SELECTED_IMAGES:

        records = visualize_image(filename)

        all_records.extend(records)

    # --------------------------------------------------------
    # SIMPAN CSV
    # --------------------------------------------------------

    result_df = pd.DataFrame(all_records)

    csv_path = OUTPUT_DIR / "statistik_hasil_akhir.csv"

    result_df.to_csv(
        csv_path,
        index=False
    )

    print()
    print("=" * 70)
    print("RINGKASAN KOMPONEN HYSTERESIS")
    print("=" * 70)

    component_columns = [
        "filename",
        "configuration",
        "total_weak_components",
        "retained_components",
        "discarded_components",
        "weak_pixels",
        "strong_pixels",
        "retained_pixels",
        "discarded_pixels"
    ]

    print(
        result_df[
            result_df["configuration"].isin(["B", "D"])
        ][component_columns].to_string(index=False)
    )

    print()
    print("Visualisasi selesai.")
    print("Folder output :", OUTPUT_DIR)
    print("Statistik CSV :", csv_path)


if __name__ == "__main__":
    main()
