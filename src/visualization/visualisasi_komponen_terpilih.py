
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from src.core.black_tophat import multi_directional_black_tophat


# ============================================================
# DIREKTORI
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

OUTPUT_DIR = (
    ROOT
    / "results"
    / "visualisasi_komponen_terpilih"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# KOMPONEN YANG DIPERIKSA
# ============================================================

CASES = [
    {
        "filename": "ISIC_0000456.png",
        "configuration": "D",
        "kernel": 11,
        "low": 15,
        "high": 40,
        "component_id": 16,
        "description": "GT-dominan tanpa strong seed"
    },
    {
        "filename": "ISIC_0009945.png",
        "configuration": "D",
        "kernel": 11,
        "low": 15,
        "high": 40,
        "component_id": 3,
        "description": "Komponen GT-dominan yang dibuang"
    },
    {
        "filename": "ISIC_0014806.png",
        "configuration": "B",
        "kernel": 9,
        "low": 10,
        "high": 30,
        "component_id": 6,
        "description": "Komponen besar campuran"
    },
    {
        "filename": "ISIC_0016019.png",
        "configuration": "B",
        "kernel": 9,
        "low": 10,
        "high": 30,
        "component_id": 1,
        "description": "Komponen besar campuran"
    }
]


# ============================================================
# WARNA DAN OVERLAY
# ============================================================

def overlay_mask(image, mask, color, alpha=0.65):

    result = image.copy()

    mask_bool = mask.astype(bool)

    color = np.array(
        color,
        dtype=np.float32
    )

    result[mask_bool] = (
        (1 - alpha) * result[mask_bool]
        + alpha * color
    ).astype(np.uint8)

    return result


def create_comparison_overlay(
    image,
    component_mask,
    ground_truth
):

    result = image.copy()

    # Komponen dan GT sama-sama aktif
    intersection = (
        component_mask & ground_truth
    )

    # Komponen di luar GT
    component_only = (
        component_mask & ~ground_truth
    )

    # GT di luar komponen
    gt_only = (
        ground_truth & ~component_mask
    )

    # Hijau: irisan komponen dan GT
    result[intersection] = (
        255, 255, 0
    )

    # Merah: komponen non-GT
    result[component_only] = (
        255, 60, 60
    )

    # Biru: GT yang tidak masuk komponen
    result[gt_only] = (
        60, 130, 255
    )

    return result


# ============================================================
# PROSES SETIAP KASUS
# ============================================================

def process_case(case):

    filename = case["filename"]

    image_path = IMAGE_DIR / filename
    mask_path = MASK_DIR / filename

    image_bgr = cv2.imread(
        str(image_path)
    )

    gt_gray = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image_bgr is None:
        raise FileNotFoundError(image_path)

    if gt_gray is None:
        raise FileNotFoundError(mask_path)

    image = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    ground_truth = gt_gray > 0

    # --------------------------------------------------------
    # BTH
    # --------------------------------------------------------

    response = multi_directional_black_tophat(
        gray,
        length=case["kernel"]
    )

    weak = response >= case["low"]
    strong = response >= case["high"]

    # --------------------------------------------------------
    # CONNECTED COMPONENTS
    # --------------------------------------------------------

    num_labels, labels = cv2.connectedComponents(
        weak.astype(np.uint8),
        connectivity=8
    )

    component_id = case["component_id"]

    if component_id <= 0 or component_id >= num_labels:
        raise ValueError(
            f"Komponen {component_id} tidak ditemukan "
            f"pada {filename}, konfigurasi "
            f"{case['configuration']}. "
            f"Jumlah komponen: {num_labels - 1}"
        )

    component = labels == component_id

    component_strong = (
        component & strong
    )

    # --------------------------------------------------------
    # STATISTIK KOMPONEN
    # --------------------------------------------------------

    component_size = int(
        component.sum()
    )

    gt_pixels = int(
        (component & ground_truth).sum()
    )

    non_gt_pixels = (
        component_size - gt_pixels
    )

    strong_pixels = int(
        component_strong.sum()
    )

    gt_ratio = (
        gt_pixels / component_size
        if component_size > 0 else 0
    )

    status = (
        "DIPERTAHANKAN"
        if strong_pixels > 0
        else "DIBUANG"
    )

    # --------------------------------------------------------
    # PANEL 1: ORIGINAL
    # --------------------------------------------------------

    original_panel = image

    # --------------------------------------------------------
    # PANEL 2: GROUND TRUTH
    # --------------------------------------------------------

    gt_panel = overlay_mask(
        image,
        ground_truth,
        (0, 255, 0),
        alpha=0.75
    )

    # --------------------------------------------------------
    # PANEL 3: KOMPONEN TERPILIH
    # --------------------------------------------------------

    component_panel = overlay_mask(
        image,
        component,
        (255, 200, 0),
        alpha=0.85
    )

    # --------------------------------------------------------
    # PANEL 4: PERBANDINGAN KOMPONEN DAN GT
    # --------------------------------------------------------

    comparison_panel = create_comparison_overlay(
        image,
        component,
        ground_truth
    )

    # --------------------------------------------------------
    # PANEL 5: WEAK MASK DAN STRONG SEED
    # --------------------------------------------------------

    weak_panel = overlay_mask(
        image,
        weak,
        (150, 150, 150),
        alpha=0.55
    )

    weak_panel = overlay_mask(
        weak_panel,
        strong,
        (255, 100, 0),
        alpha=0.95
    )

    # --------------------------------------------------------
    # PANEL 6: KOMPONEN DAN STRONG SEED
    # --------------------------------------------------------

    seed_panel = overlay_mask(
        image,
        component,
        (0, 200, 255),
        alpha=0.8
    )

    seed_panel = overlay_mask(
        seed_panel,
        component_strong,
        (255, 230, 0),
        alpha=1.0
    )

    # --------------------------------------------------------
    # VISUALISASI
    # --------------------------------------------------------

    panels = [
        ("1. Original", original_panel),
        ("2. Ground Truth (hijau)", gt_panel),
        ("3. Komponen terpilih (kuning)", component_panel),
        (
            "4. Perbandingan GT dan komponen",
            comparison_panel
        ),
        (
            "5. Weak mask (abu-abu), strong seed (oranye)",
            weak_panel
        ),
        (
            "6. Komponen (cyan), strong seed (kuning)",
            seed_panel
        )
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 11)
    )

    fig.suptitle(
        f"{filename} | Konfigurasi {case['configuration']} "
        f"| K{case['kernel']} "
        f"| Komponen #{component_id}\n"
        f"{case['description']}\n"
        f"Size={component_size:,} | "
        f"GT={gt_pixels:,} | "
        f"Non-GT={non_gt_pixels:,} | "
        f"Strong={strong_pixels:,} | "
        f"GT ratio={gt_ratio:.3f} | "
        f"{status}",
        fontsize=13,
        fontweight="bold",
        y=0.99
    )

    for ax, (title, panel) in zip(
        axes.flat,
        panels
    ):

        ax.imshow(panel)
        ax.set_title(
            title,
            fontsize=10,
            pad=7
        )
        ax.axis("off")

    plt.tight_layout(
        rect=[0, 0, 1, 0.91]
    )

    output_name = (
        f"{Path(filename).stem}_"
        f"{case['configuration']}_"
        f"component_{component_id}.png"
    )

    output_path = OUTPUT_DIR / output_name

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------------
    # SIMPAN STATISTIK
    # --------------------------------------------------------

    return {
        "filename": filename,
        "configuration": case["configuration"],
        "kernel": case["kernel"],
        "low": case["low"],
        "high": case["high"],
        "component_id": component_id,
        "component_size": component_size,
        "gt_pixels": gt_pixels,
        "non_gt_pixels": non_gt_pixels,
        "strong_pixels": strong_pixels,
        "gt_ratio": gt_ratio,
        "status": status,
        "output_image": output_name
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("VISUALISASI KOMPONEN TERPILIH")
    print("=" * 70)

    results = []

    for case in CASES:

        print(
            f"\nMemproses {case['filename']} "
            f"| {case['configuration']} "
            f"| Komponen #{case['component_id']}"
        )

        result = process_case(case)

        results.append(result)

        print(
            f"Size={result['component_size']:,} | "
            f"GT={result['gt_pixels']:,} | "
            f"Non-GT={result['non_gt_pixels']:,} | "
            f"Strong={result['strong_pixels']:,} | "
            f"Status={result['status']}"
        )

    result_df = pd.DataFrame(results)

    csv_path = (
        OUTPUT_DIR / "statistik_komponen_terpilih.csv"
    )

    result_df.to_csv(
        csv_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("SELESAI")
    print("=" * 70)

    print("Direktori:", OUTPUT_DIR)
    print("Statistik:", csv_path)


if __name__ == "__main__":
    main()
