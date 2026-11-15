
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path
from src.core.black_tophat import multi_directional_black_tophat


# ============================================================
# KONFIGURASI
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

CSV_PATH = (
    ROOT
    / "results"
    / "03_evaluation"
    / "kernel_terkontrol"
    / "hasil_detail.csv"
)

OUTPUT_DIR = ROOT / "results" / "08_visualizations" / "respons_bth_ekstrem"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def select_extreme_cases(df):

    wide = df.pivot(
        index="filename",
        columns="configuration",
        values=["dice", "fp"]
    )

    # Ratakan nama kolom hasil pivot terlebih dahulu
    wide.columns = [
        f"{metric}_{config}"
        for metric, config in wide.columns
    ]

    # Hitung perubahan metrik
    wide["delta_dice_k9"] = (
        wide["dice_B"] - wide["dice_A"]
    )

    wide["delta_dice_k11"] = (
        wide["dice_D"] - wide["dice_C"]
    )

    wide["delta_fp_k9"] = (
        wide["fp_B"] - wide["fp_A"]
    )

    wide["delta_fp_k11"] = (
        wide["fp_D"] - wide["fp_C"]
    )

    wide = wide.reset_index()

    categories = (
        df.groupby("filename")["category"].first()
    )

    wide["category"] = wide["filename"].map(categories)

    # Kriteria pemilihan kasus
    criteria = [
        ("K9_DICE_NAIK", "delta_dice_k9", False),
        ("K9_DICE_TURUN", "delta_dice_k9", True),
        ("K9_FP_NAIK", "delta_fp_k9", False),
        ("K11_DICE_NAIK", "delta_dice_k11", False),
        ("K11_DICE_TURUN", "delta_dice_k11", True),
        ("K11_FP_NAIK", "delta_fp_k11", False),
    ]

    selected = []
    used = set()

    for label, column, ascending in criteria:

        candidates = wide.sort_values(
            column,
            ascending=ascending
        )

        for _, row in candidates.iterrows():

            filename = row["filename"]

            # Hindari citra yang sama terpilih berulang
            if filename not in used:

                record = row.to_dict()
                record["selection"] = label

                selected.append(record)
                used.add(filename)

                break

    return pd.DataFrame(selected)


def visualize_image(filename, selection, metrics):

    image_path = IMAGE_DIR / filename
    mask_path = MASK_DIR / filename

    if not image_path.exists():
        print(f"[SKIP] Citra tidak ditemukan: {image_path}")
        return

    if not mask_path.exists():
        print(f"[SKIP] Mask tidak ditemukan: {mask_path}")
        return

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"[SKIP] Gagal membaca: {filename}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Respons Black Top-Hat
    response_k9 = multi_directional_black_tophat(
        gray,
        length=9
    )

    response_k11 = multi_directional_black_tophat(
        gray,
        length=11
    )

    # K9: low=10, high=30
    weak_only_k9 = (
        (response_k9 >= 10) &
        (response_k9 < 30)
    ).astype(np.uint8) * 255

    strong_k9 = (
        response_k9 >= 30
    ).astype(np.uint8) * 255

    # K11: low=15, high=40
    weak_only_k11 = (
        (response_k11 >= 15) &
        (response_k11 < 40)
    ).astype(np.uint8) * 255

    strong_k11 = (
        response_k11 >= 40
    ).astype(np.uint8) * 255

    gt = (mask > 0).astype(np.uint8) * 255

    # ========================================================
    # PLOT
    # ========================================================

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 10)
    )

    fig.suptitle(
        f"{filename} | {selection}\n"
        f"ΔDice K9={metrics['delta_dice_k9']:+.4f} | "
        f"ΔFP K9={metrics['delta_fp_k9']:+.0f} | "
        f"ΔDice K11={metrics['delta_dice_k11']:+.4f} | "
        f"ΔFP K11={metrics['delta_fp_k11']:+.0f}",
        fontsize=12,
        fontweight="bold"
    )

    panels = [
        (rgb, "Original", None),
        (gt, "Ground Truth", "gray"),
        (response_k9, "BTH K9", "inferno"),
        (weak_only_k9, "Weak-only K9 (10-29)", "gray"),
        (strong_k9, "Strong K9 (>=30)", "gray"),
        (response_k11, "BTH K11", "inferno"),
        (weak_only_k11, "Weak-only K11 (15-39)", "gray"),
        (strong_k11, "Strong K11 (>=40)", "gray"),
    ]

    for ax, (data, title, cmap) in zip(
        axes.ravel(),
        panels
    ):

        if title.startswith("BTH"):
            ax.imshow(
                data,
                cmap=cmap,
                vmin=0,
                vmax=255
            )

        elif cmap is None:
            ax.imshow(data)

        else:
            ax.imshow(
                data,
                cmap=cmap,
                vmin=0,
                vmax=255
            )

        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.91])

    output_path = OUTPUT_DIR / (
        Path(filename).stem + "_respons_bth.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"[OK] {output_path.name}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("VISUALISASI RESPONS BLACK TOP-HAT KASUS EKSTREM")
    print("=" * 65)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV tidak ditemukan: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    required = {
        "filename",
        "category",
        "configuration",
        "dice",
        "fp"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Kolom CSV tidak lengkap: {missing}"
        )

    selected = select_extreme_cases(df)

    selected.to_csv(
        OUTPUT_DIR / "daftar_kasus_terpilih.csv",
        index=False
    )

    print("\nDAFTAR KASUS TERPILIH")
    print("-" * 65)

    for _, row in selected.iterrows():

        print(
            f"{row['selection']:18s} | "
            f"{row['filename']} | "
            f"Kategori: {row['category']}"
        )

    print("\nMEMBUAT VISUALISASI")
    print("-" * 65)

    for _, row in selected.iterrows():

        visualize_image(
            filename=row["filename"],
            selection=row["selection"],
            metrics=row
        )

    print("\n" + "=" * 65)
    print("SELESAI")
    print(f"Folder output: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
