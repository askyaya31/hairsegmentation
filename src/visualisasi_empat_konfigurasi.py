
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path
from black_tophat import multi_directional_black_tophat


# ============================================================
# DIREKTORI
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

DETAIL_FILE = (
    ROOT / "results" / "evaluasi_kernel_terkontrol"
    / "hasil_detail.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "visualisasi_empat_konfigurasi"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PARAMETER KONFIGURASI
# ============================================================

CONFIGURATIONS = [
    {
        "configuration": "A",
        "title": "A: Baseline K9",
        "kernel": 9,
        "method": "Baseline",
        "single_threshold": 20
    },
    {
        "configuration": "B",
        "title": "B: Hysteresis K9",
        "kernel": 9,
        "method": "Hysteresis",
        "low_threshold": 10,
        "high_threshold": 30
    },
    {
        "configuration": "C",
        "title": "C: Baseline K11",
        "kernel": 11,
        "method": "Baseline",
        "single_threshold": 20
    },
    {
        "configuration": "D",
        "title": "D: Hysteresis K11",
        "kernel": 11,
        "method": "Hysteresis",
        "low_threshold": 15,
        "high_threshold": 40
    }
]


# ============================================================
# HYSTERESIS THRESHOLDING
# Implementasi identik dengan evaluasi
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

    return result


# ============================================================
# MEMBUAT PREDIKSI
# ============================================================

def generate_prediction(response, config):

    if config["method"] == "Baseline":

        prediction = (
            response >= config["single_threshold"]
        ).astype(np.uint8)

    else:

        prediction = hysteresis_threshold(
            response,
            config["low_threshold"],
            config["high_threshold"]
        )

    return prediction


# ============================================================
# MEMILIH CITRA REPRESENTATIF
# ============================================================

def select_representative_images(detail_df):

    # Ambil hasil konfigurasi B
    hyst_df = detail_df[
        detail_df["configuration"] == "B"
    ].copy()

    if len(hyst_df) != 150:
        raise ValueError(
            "Jumlah hasil konfigurasi B bukan 150."
        )

    selected = []

    # --------------------------------------------------------
    # NORMAL: Dice terendah, median, tertinggi
    # --------------------------------------------------------

    normal = hyst_df[
        hyst_df["category"] == "NORMAL"
    ].sort_values(
        ["dice", "filename"]
    ).reset_index(drop=True)

    normal_indices = [
        0,
        len(normal) // 2,
        len(normal) - 1
    ]

    for idx, label in zip(
        normal_indices,
        ["NORMAL_RENDAH", "NORMAL_MEDIAN", "NORMAL_TINGGI"]
    ):
        row = normal.iloc[idx].copy()
        row["selection_reason"] = label
        selected.append(row)

    # --------------------------------------------------------
    # SANGAT_BESAR: Dice terendah dan tertinggi
    # --------------------------------------------------------

    large = hyst_df[
        hyst_df["category"] == "SANGAT_BESAR"
    ].sort_values(
        ["dice", "filename"]
    )

    for idx, (_, row) in enumerate(large.iterrows()):
        if idx == 0 or idx == len(large) - 1:
            row = row.copy()
            row["selection_reason"] = (
                "SANGAT_BESAR_RENDAH"
                if idx == 0
                else "SANGAT_BESAR_TINGGI"
            )
            selected.append(row)

    # --------------------------------------------------------
    # SANGAT_SEDIKIT: seluruh citra
    # --------------------------------------------------------

    sparse = hyst_df[
        hyst_df["category"] == "SANGAT_SEDIKIT"
    ].sort_values("filename")

    for _, row in sparse.iterrows():
        row = row.copy()
        row["selection_reason"] = "SANGAT_SEDIKIT_SEMUA"
        selected.append(row)

    # --------------------------------------------------------
    # KOSONG: pilih prediksi B dengan piksel paling sedikit
    # dan paling banyak
    # --------------------------------------------------------

    empty = hyst_df[
        hyst_df["category"] == "KOSONG"
    ].copy()

    empty_counts = []

    for _, row in empty.iterrows():

        image_path = IMAGE_DIR / row["filename"]

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(image_path)

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        response = multi_directional_black_tophat(
            gray,
            length=9
        )

        prediction = hysteresis_threshold(
            response,
            10,
            30
        )

        empty_counts.append(
            int(prediction.sum())
        )

    empty["prediction_pixels_B"] = empty_counts

    empty = empty.sort_values(
        ["prediction_pixels_B", "filename"]
    ).reset_index(drop=True)

    empty_indices = [0, len(empty) - 1]

    for idx in empty_indices:

        row = empty.iloc[idx].copy()
        row["selection_reason"] = (
            "KOSONG_FP_RENDAH"
            if idx == 0
            else "KOSONG_FP_TINGGI"
        )

        selected.append(row)

    selected_df = pd.DataFrame(selected)

    selected_path = OUTPUT_DIR / "citra_terpilih.csv"

    selected_df.to_csv(
        selected_path,
        index=False
    )

    return selected_df


# ============================================================
# MEMPROSES SATU CITRA
# ============================================================

def process_image(filename):

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

    # Hitung BTH untuk kernel 9 dan 11
    responses = {}

    for kernel in [9, 11]:

        responses[kernel] = (
            multi_directional_black_tophat(
                gray,
                length=kernel
            )
        )

    predictions = {}

    for config in CONFIGURATIONS:

        response = responses[config["kernel"]]

        predictions[config["configuration"]] = (
            generate_prediction(response, config)
        )

    return (
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        ground_truth,
        predictions
    )


# ============================================================
# VISUALISASI SATU CITRA
# ============================================================

def visualize_single(row):

    filename = row["filename"]
    category = row["category"]

    image, gt, predictions = process_image(filename)

    panels = [
        (image, "Original", True),
        (gt, "Ground Truth", False),
        (predictions["A"], "A: Baseline K9", False),
        (predictions["B"], "B: Hysteresis K9", False),
        (predictions["C"], "C: Baseline K11", False),
        (predictions["D"], "D: Hysteresis K11", False)
    ]

    fig, axes = plt.subplots(
        1,
        6,
        figsize=(20, 4)
    )

    for ax, (data, title, is_rgb) in zip(axes, panels):

        if is_rgb:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap="gray", vmin=0, vmax=1)

        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{filename} | {category} | "
        f"Dice B = {row['dice']:.4f}",
        fontsize=12
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / f"{Path(filename).stem}_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ============================================================
# CONTACT SHEET SEMBILAN CITRA
# ============================================================

def create_contact_sheet(selected_df):

    n = len(selected_df)

    fig, axes = plt.subplots(
        n,
        6,
        figsize=(19, 3.2 * n),
        squeeze=False
    )

    column_titles = [
        "Original",
        "Ground Truth",
        "A: Baseline K9",
        "B: Hysteresis K9",
        "C: Baseline K11",
        "D: Hysteresis K11"
    ]

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(
            title,
            fontsize=10,
            fontweight="bold"
        )

    for row_idx, (_, row) in enumerate(
        selected_df.iterrows()
    ):

        image, gt, predictions = process_image(
            row["filename"]
        )

        panels = [
            (image, True),
            (gt, False),
            (predictions["A"], False),
            (predictions["B"], False),
            (predictions["C"], False),
            (predictions["D"], False)
        ]

        for col_idx, (data, is_rgb) in enumerate(panels):

            ax = axes[row_idx, col_idx]

            if is_rgb:
                ax.imshow(data)
            else:
                ax.imshow(
                    data,
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )

            ax.axis("off")

        axes[row_idx, 0].set_ylabel(
            f"{row['selection_reason']}\n"
            f"{row['filename']}\n"
            f"Dice B: {row['dice']:.4f}",
            fontsize=8,
            rotation=0,
            labelpad=100,
            va="center"
        )

    fig.suptitle(
        "Visual Comparison of Four Controlled Configurations",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0.08, 0, 1, 0.98])

    output_path = OUTPUT_DIR / "contact_sheet_9_citra.png"

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VISUALISASI EMPAT KONFIGURASI TERKONTROL")
    print("=" * 70)

    detail_df = pd.read_csv(DETAIL_FILE)

    selected_df = select_representative_images(
        detail_df
    )

    print("\nCITRA YANG DIPILIH")
    print("-" * 70)

    print(
        selected_df[
            [
                "filename",
                "category",
                "configuration",
                "dice",
                "selection_reason"
            ]
        ].to_string(index=False)
    )

    print("\nMembuat visualisasi individual...")

    for idx, (_, row) in enumerate(
        selected_df.iterrows(),
        start=1
    ):

        output_path = visualize_single(row)

        print(
            f"[{idx}/{len(selected_df)}] "
            f"{row['filename']} selesai"
        )

    print("\nMembuat contact sheet...")

    contact_path = create_contact_sheet(
        selected_df
    )

    print("\n" + "=" * 70)
    print("VISUALISASI SELESAI")
    print("=" * 70)

    print(f"Folder output : {OUTPUT_DIR}")
    print(f"Contact sheet : {contact_path}")
    print(
        f"Jumlah citra  : {len(selected_df)}"
    )


if __name__ == "__main__":
    main()
