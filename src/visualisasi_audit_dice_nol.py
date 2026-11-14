
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path

from evaluasi_final_testing import (
    IMAGE_DIR,
    MASK_DIR,
    BASELINE_KERNEL,
    BASELINE_THRESHOLD,
    HYSTERESIS_KERNEL,
    LOW_THRESHOLD,
    HIGH_THRESHOLD,
    hysteresis_threshold
)

from black_tophat import multi_directional_black_tophat


# ============================================================
# KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

AUDIT_FILE = (
    ROOT / "results" / "audit_dice_nol"
    / "hasil_audit_dice_nol.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "visualisasi_audit_dice_nol"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Citra yang akan divisualisasikan
TARGET_IMAGES = [
    "ISIC_0013713.png",
    "ISIC_0012879.png",
    "ISIC_0013320.png",
    "ISIC_0013601.png",
    "ISIC_0012089.png",
    "ISIC_0013554.png",
    "ISIC_0012966.png"
]


# ============================================================
# VISUALISASI OVERLAY
# ============================================================

def create_overlay(image, prediction, ground_truth):
    """
    Hijau  : True Positive
    Merah  : False Positive
    Biru   : False Negative
    """

    overlay = image.copy()

    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt

    overlay[tp] = (
        0.5 * overlay[tp] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)

    overlay[fp] = (
        0.5 * overlay[fp] + 0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    overlay[fn] = (
        0.5 * overlay[fn] + 0.5 * np.array([0, 0, 255])
    ).astype(np.uint8)

    return overlay


# ============================================================
# VISUALISASI SATU CITRA
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

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gt = ground_truth > 0

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_response = multi_directional_black_tophat(
        gray,
        length=BASELINE_KERNEL
    )

    baseline_prediction = (
        baseline_response >= BASELINE_THRESHOLD
    ).astype(np.uint8)

    # --------------------------------------------------------
    # HYSTERESIS
    # --------------------------------------------------------

    hysteresis_response = multi_directional_black_tophat(
        gray,
        length=HYSTERESIS_KERNEL
    )

    hysteresis_prediction = hysteresis_threshold(
        hysteresis_response,
        low=LOW_THRESHOLD,
        high=HIGH_THRESHOLD
    )

    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    baseline_overlay = create_overlay(
        image_rgb,
        baseline_prediction,
        gt
    )

    hysteresis_overlay = create_overlay(
        image_rgb,
        hysteresis_prediction,
        gt
    )

    # --------------------------------------------------------
    # INFORMASI PIXEL
    # --------------------------------------------------------

    baseline_tp = np.logical_and(
        baseline_prediction > 0, gt
    ).sum()

    baseline_fp = np.logical_and(
        baseline_prediction > 0, ~gt
    ).sum()

    baseline_fn = np.logical_and(
        baseline_prediction == 0, gt
    ).sum()

    hyst_tp = np.logical_and(
        hysteresis_prediction > 0, gt
    ).sum()

    hyst_fp = np.logical_and(
        hysteresis_prediction > 0, ~gt
    ).sum()

    hyst_fn = np.logical_and(
        hysteresis_prediction == 0, gt
    ).sum()

    # Dice dengan konvensi yang sama seperti evaluasi
    def dice_score(tp, fp, fn):
        denominator = 2 * tp + fp + fn
        return 1.0 if denominator == 0 else 2 * tp / denominator

    baseline_dice = dice_score(
        baseline_tp, baseline_fp, baseline_fn
    )

    hysteresis_dice = dice_score(
        hyst_tp, hyst_fp, hyst_fn
    )

    # --------------------------------------------------------
    # GAMBAR
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 10)
    )

    fig.suptitle(
        f"Audit Segmentasi Rambut: {filename}",
        fontsize=15,
        fontweight="bold"
    )

    panels = [
        (image_rgb, "Citra Asli", None),
        (gt, "Ground Truth", "gray"),
        (baseline_prediction, "Prediksi Baseline", "gray"),
        (hysteresis_prediction, "Prediksi Hysteresis", "gray"),
        (baseline_overlay, "Overlay Baseline", None),
        (hysteresis_overlay, "Overlay Hysteresis", None)
    ]

    for ax, (data, title, cmap) in zip(
        axes.flat, panels
    ):
        if cmap:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(data)

        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # Informasi metrik pada bagian bawah
    info = (
        f"Baseline: TP={baseline_tp}, FP={baseline_fp}, "
        f"FN={baseline_fn}, Dice={baseline_dice:.6f}\n"
        f"Hysteresis: TP={hyst_tp}, FP={hyst_fp}, "
        f"FN={hyst_fn}, Dice={hysteresis_dice:.6f}\n"
        "Overlay: Hijau=TP | Merah=FP | Biru=FN"
    )

    fig.text(
        0.5, 0.025,
        info,
        ha="center",
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0.09, 1, 0.93])

    output_path = OUTPUT_DIR / (
        Path(filename).stem + "_audit.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Selesai: {filename}")
    print(f"  Baseline Dice   : {baseline_dice:.6f}")
    print(f"  Hysteresis Dice : {hysteresis_dice:.6f}")
    print(f"  Disimpan        : {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    if not AUDIT_FILE.exists():
        raise FileNotFoundError(AUDIT_FILE)

    audit_df = pd.read_csv(AUDIT_FILE)

    available = set(audit_df["filename"].astype(str))

    print("=" * 65)
    print("VISUALISASI AUDIT DICE NOL")
    print("=" * 65)

    print(f"Jumlah target: {len(TARGET_IMAGES)}")

    for filename in TARGET_IMAGES:

        if filename not in available:
            print(f"Peringatan: {filename} tidak ditemukan di audit CSV")
            continue

        visualize_image(filename)

    print("\n" + "=" * 65)
    print("VISUALISASI SELESAI")
    print(f"Lokasi: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
