import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from black_tophat import multi_directional_black_tophat
ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

INPUT_FILE = (
    ROOT / "results"
    / "evaluasi_kernel_terkontrol"
    / "hasil_detail.csv"
)

OUTPUT_DIR = ROOT / "results" / "visualisasi_kasus_ekstrem"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# KONFIGURASI EKSPERIMEN
# ============================================================

CONFIGURATIONS = {
    "A": {
        "kernel": 9,
        "method": "Baseline",
        "threshold": 20
    },
    "B": {
        "kernel": 9,
        "method": "Hysteresis",
        "low": 10,
        "high": 30
    },
    "C": {
        "kernel": 11,
        "method": "Baseline",
        "threshold": 20
    },
    "D": {
        "kernel": 11,
        "method": "Hysteresis",
        "low": 15,
        "high": 40
    }
}

N_SELECTED = 5

PANEL_NAMES = [
    "Original",
    "Ground Truth",
    "A: K9 Baseline",
    "B: K9 Hysteresis",
    "C: K11 Baseline",
    "D: K11 Hysteresis"
]


# ============================================================
# FUNGSI HYSTERESIS THRESHOLDING
# Sama dengan evaluasi_kernel_terkontrol.py
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
# FUNGSI EVALUASI METRIK
# Sama dengan evaluasi_kernel_terkontrol.py
# ============================================================

def calculate_metrics(prediction, ground_truth):

    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    dice = (
        1.0
        if 2 * tp + fp + fn == 0
        else (2 * tp) / (2 * tp + fp + fn)
    )

    return {
        "dice": float(dice),
        "fp": int(fp)
    }


# ============================================================
# MEMBUAT PREDIKSI EMPAT KONFIGURASI
# ============================================================

def generate_predictions(gray):

    responses = {}

    for kernel in [9, 11]:

        responses[kernel] = multi_directional_black_tophat(
            gray,
            length=kernel
        )

    predictions = {}

    for config_name, config in CONFIGURATIONS.items():

        response = responses[config["kernel"]]

        if config["method"] == "Baseline":

            prediction = (
                response >= config["threshold"]
            ).astype(np.uint8)

        else:

            prediction = hysteresis_threshold(
                response,
                config["low"],
                config["high"]
            )

        predictions[config_name] = prediction

    return predictions


# ============================================================
# MEMILIH 5 CITRA BERDASARKAN PERUBAHAN DICE DAN FP
# ============================================================

def select_extreme_cases(detail_df):

    wide = detail_df.pivot(
        index=["filename", "category"],
        columns="configuration",
        values=["dice", "fp"]
    )

    wide.columns = [
        f"{metric}_{config}"
        for metric, config in wide.columns
    ]

    wide = wide.reset_index()

    # Perubahan dihitung Hysteresis - Baseline
    wide["delta_dice_k9"] = (
        wide["dice_B"] - wide["dice_A"]
    )

    wide["delta_fp_k9"] = (
        wide["fp_B"] - wide["fp_A"]
    )

    wide["delta_dice_k11"] = (
        wide["dice_D"] - wide["dice_C"]
    )

    wide["delta_fp_k11"] = (
        wide["fp_D"] - wide["fp_C"]
    )

    selections = {}

    for kernel in [9, 11]:

        dice_col = f"delta_dice_k{kernel}"
        fp_col = f"delta_fp_k{kernel}"

        # Peningkatan Dice terbesar
        selections[f"K{kernel}_dice_naik"] = (
            wide.nlargest(N_SELECTED, dice_col).copy()
        )

        # Penurunan Dice terbesar
        selections[f"K{kernel}_dice_turun"] = (
            wide.nsmallest(N_SELECTED, dice_col).copy()
        )

        # Peningkatan FP terbesar
        selections[f"K{kernel}_fp_naik"] = (
            wide.nlargest(N_SELECTED, fp_col).copy()
        )

    return wide, selections


# ============================================================
# MEMBUAT CONTACT SHEET
# ============================================================

def create_contact_sheet(
    selected_df,
    title,
    delta_dice_column,
    delta_fp_column,
    prediction_cache
):

    n = len(selected_df)

    fig, axes = plt.subplots(
        n,
        6,
        figsize=(19, 3.4 * n),
        squeeze=False
    )

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold",
        y=0.995
    )

    for row_index, (_, row) in enumerate(
        selected_df.iterrows()
    ):

        filename = row["filename"]
        category = row["category"]

        image_rgb, ground_truth, predictions = (
            prediction_cache[filename]
        )

        panels = [
            image_rgb,
            ground_truth,
            predictions["A"],
            predictions["B"],
            predictions["C"],
            predictions["D"]
        ]

        for col_index, panel in enumerate(panels):

            ax = axes[row_index, col_index]

            if col_index == 0:
                ax.imshow(panel)

            else:
                ax.imshow(
                    panel,
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )

            ax.axis("off")

            if row_index == 0:
                ax.set_title(
                    PANEL_NAMES[col_index],
                    fontsize=10,
                    fontweight="bold"
                )

        delta_dice = row[delta_dice_column]
        delta_fp = row[delta_fp_column]

        axes[row_index, 0].set_ylabel(
            (
                f"{filename}\n"
                f"{category}\n"
                f"Delta Dice: {delta_dice:+.4f}\n"
                f"Delta FP: {delta_fp:+,.0f}"
            ),
            rotation=0,
            labelpad=105,
            fontsize=8,
            va="center"
        )

    plt.tight_layout(rect=[0.08, 0, 1, 0.97])

    output_path = OUTPUT_DIR / f"{title.replace(' ', '_')}.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Contact sheet disimpan: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("VISUALISASI KASUS EKSTREM HYSTERESIS")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. Baca hasil pengujian
    # --------------------------------------------------------

    detail_df = pd.read_csv(INPUT_FILE)

    if detail_df["filename"].nunique() != 150:
        raise ValueError("Jumlah citra pengujian bukan 150.")

    for config in ["A", "B", "C", "D"]:

        count = detail_df[
            detail_df["configuration"] == config
        ]["filename"].nunique()

        if count != 150:
            raise ValueError(
                f"Konfigurasi {config}: jumlah citra {count}."
            )

    print("\nData pengujian berhasil dibaca.")

    # --------------------------------------------------------
    # 2. Pilih kasus ekstrem
    # --------------------------------------------------------

    wide, selections = select_extreme_cases(detail_df)

    # Simpan semua hasil perubahan per citra
    wide.to_csv(
        OUTPUT_DIR / "seluruh_perubahan_per_citra.csv",
        index=False
    )

    selection_summary = []

    for group_name, selected in selections.items():

        selected_copy = selected.copy()
        selected_copy.insert(0, "selection", group_name)

        selection_summary.append(selected_copy)

        print(f"\n{group_name}")
        print(
            selected[
                [
                    "filename",
                    "category",
                    "delta_dice_k9",
                    "delta_fp_k9",
                    "delta_dice_k11",
                    "delta_fp_k11"
                ]
            ].to_string(index=False)
        )

    selection_df = pd.concat(
        selection_summary,
        ignore_index=True
    )

    selection_df.to_csv(
        OUTPUT_DIR / "daftar_kasus_terpilih.csv",
        index=False
    )

    # --------------------------------------------------------
    # 3. Gabungkan filename unik
    # --------------------------------------------------------

    selected_filenames = sorted(
        selection_df["filename"].unique()
    )

    print("\nJumlah citra unik terpilih:", len(selected_filenames))

    # --------------------------------------------------------
    # 4. Siapkan lookup hasil evaluasi
    # --------------------------------------------------------

    lookup = detail_df.set_index(
        ["filename", "configuration"]
    )

    # --------------------------------------------------------
    # 5. Hitung ulang prediksi untuk citra terpilih
    # --------------------------------------------------------

    prediction_cache = {}

    print("\nMemproses prediksi citra terpilih...")

    for index, filename in enumerate(selected_filenames, start=1):

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

        predictions = generate_predictions(gray)

        # ----------------------------------------------------
        # Validasi hasil hitung ulang dengan hasil evaluasi
        # ----------------------------------------------------

        for config_name in CONFIGURATIONS:

            metrics = calculate_metrics(
                predictions[config_name],
                ground_truth
            )

            stored = lookup.loc[
                (filename, config_name)
            ]

            dice_match = np.isclose(
                metrics["dice"],
                stored["dice"],
                atol=1e-10
            )

            fp_match = (
                metrics["fp"] == int(stored["fp"])
            )

            if not dice_match or not fp_match:

                raise ValueError(
                    f"Hasil tidak cocok untuk {filename}, "
                    f"konfigurasi {config_name}.\n"
                    f"Dice hitung ulang: {metrics['dice']}, "
                    f"Dice tersimpan: {stored['dice']}\n"
                    f"FP hitung ulang: {metrics['fp']}, "
                    f"FP tersimpan: {stored['fp']}"
                )

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        prediction_cache[filename] = (
            image_rgb,
            ground_truth,
            predictions
        )

        print(
            f"[{index}/{len(selected_filenames)}] "
            f"{filename} selesai dan tervalidasi."
        )

    # --------------------------------------------------------
    # 6. Buat enam contact sheet
    # --------------------------------------------------------

    sheet_configurations = [
        (
            "K9_Dice_Naik_Terbesar",
            "K9_dice_naik",
            "delta_dice_k9",
            "delta_fp_k9"
        ),
        (
            "K9_Dice_Turun_Terbesar",
            "K9_dice_turun",
            "delta_dice_k9",
            "delta_fp_k9"
        ),
        (
            "K9_FP_Naik_Terbesar",
            "K9_fp_naik",
            "delta_dice_k9",
            "delta_fp_k9"
        ),
        (
            "K11_Dice_Naik_Terbesar",
            "K11_dice_naik",
            "delta_dice_k11",
            "delta_fp_k11"
        ),
        (
            "K11_Dice_Turun_Terbesar",
            "K11_dice_turun",
            "delta_dice_k11",
            "delta_fp_k11"
        ),
        (
            "K11_FP_Naik_Terbesar",
            "K11_fp_naik",
            "delta_dice_k11",
            "delta_fp_k11"
        )
    ]

    for title, selection_key, dice_col, fp_col in sheet_configurations:

        create_contact_sheet(
            selections[selection_key],
            title,
            dice_col,
            fp_col,
            prediction_cache
        )

    # --------------------------------------------------------
    # 7. Selesai
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("VISUALISASI SELESAI")
    print("=" * 75)

    print("Jumlah citra unik :", len(selected_filenames))
    print("Jumlah contact sheet :", len(sheet_configurations))
    print("Direktori hasil :", OUTPUT_DIR)


if __name__ == "__main__":
    main()
