
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# 1. KONFIGURASI
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

CASE_FILE = (
    ROOT / "results" / "analisis_kasus"
    / "kasus_representatif_setiap_pola.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "analisis_kasus"
    / "peta_weak_strong"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_LENGTH = 15
ANGLES = [0, 30, 60, 90, 120, 150]

LOW = 15
HIGH = 40


# ==========================================
# 2. BLACK TOP-HAT
# ==========================================

def create_line_kernel(length, angle):
    kernel = np.zeros((length, length), dtype=np.uint8)

    center = (length // 2, length // 2)
    radians = np.deg2rad(angle)

    dx = int((length // 2) * np.cos(radians))
    dy = int((length // 2) * np.sin(radians))

    pt1 = (center[0] - dx, center[1] - dy)
    pt2 = (center[0] + dx, center[1] + dy)

    cv2.line(kernel, pt1, pt2, 1, thickness=1)

    return kernel


def multi_directional_black_tophat(gray):
    responses = []

    for angle in ANGLES:
        kernel = create_line_kernel(KERNEL_LENGTH, angle)

        closing = cv2.morphologyEx(
            gray, cv2.MORPH_CLOSE, kernel
        )

        response = cv2.subtract(closing, gray)
        responses.append(response)

    return np.maximum.reduce(responses)


# ==========================================
# 3. PENCARIAN FILE
# ==========================================

def find_image(directory, image_name):
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        path = directory / f"{image_name}{ext}"

        if path.exists():
            return path

    raise FileNotFoundError(
        f"File tidak ditemukan: {image_name} di {directory}"
    )


# ==========================================
# 4. ANALISIS SATU KASUS
# ==========================================

def analyze_case(row):

    name = row["image"]

    image_path = find_image(IMAGE_DIR, name)
    mask_path = find_image(MASK_DIR, name)

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError(f"Gagal membaca citra atau mask: {name}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gt = mask > 127

    # Respons BTH
    response = multi_directional_black_tophat(gray)

    # Weak dan strong
    weak = response >= LOW
    strong = response >= HIGH

    # Connected components pada weak mask
    num_labels, labels = cv2.connectedComponents(
        weak.astype(np.uint8),
        connectivity=8
    )

    # Komponen yang memiliki minimal satu piksel strong
    strong_labels = np.unique(labels[strong])
    strong_labels = strong_labels[strong_labels != 0]

    retained = np.isin(labels, strong_labels)
    discarded = weak & ~retained

    # Piksel di bawah low threshold
    below_low = ~weak

    # ======================================
    # STATISTIK GROUND TRUTH
    # ======================================

    gt_total = int(gt.sum())

    # Tiga kategori piksel GT
    retained_gt = int((retained & gt).sum())
    discarded_gt = int((discarded & gt).sum())
    below_low_gt = int((below_low & gt).sum())

    total_classified = (
        retained_gt + discarded_gt + below_low_gt
    )

    if total_classified != gt_total:
        raise ValueError(
            f"Klasifikasi GT tidak sesuai pada {name}: "
            f"{total_classified} != {gt_total}"
        )

    def percentage(value):
        return (
            round(value / gt_total * 100, 2)
            if gt_total > 0 else 0.0
        )

    # ======================================
    # STATISTIK KESELURUHAN
    # ======================================

    summary = {
        "image": name,

        "gt_pixels": gt_total,

        "weak_pixels": int(weak.sum()),
        "strong_pixels": int(strong.sum()),

        "strong_TP": int((strong & gt).sum()),
        "strong_FP": int((strong & ~gt).sum()),

        "total_components": num_labels - 1,
        "retained_components": len(strong_labels),
        "discarded_components": (
            num_labels - 1 - len(strong_labels)
        ),

        "retained_pixels": int(retained.sum()),
        "retained_TP": retained_gt,
        "retained_FP": int((retained & ~gt).sum()),

        "retained_GT_pixels": retained_gt,
        "weak_discarded_GT_pixels": discarded_gt,
        "below_low_GT_pixels": below_low_gt,

        "retained_GT_percent": percentage(retained_gt),
        "weak_discarded_GT_percent": percentage(discarded_gt),
        "below_low_GT_percent": percentage(below_low_gt),

        "total_classified_GT": total_classified,
        "classification_valid": total_classified == gt_total,
    }

    # ======================================
    # STATISTIK KOMPONEN
    # ======================================

    component_rows = []

    for label in range(1, num_labels):

        component = labels == label
        component_response = response[component]

        area = int(component.sum())
        gt_pixels = int((component & gt).sum())
        strong_pixels = int((component & strong).sum())

        component_rows.append({
            "image": name,
            "label": label,
            "area": area,
            "strong_pixels": strong_pixels,
            "gt_pixels": gt_pixels,
            "non_gt_pixels": area - gt_pixels,
            "retained": bool(label in strong_labels),
            "mean_response": float(component_response.mean()),
            "max_response": int(component_response.max()),
        })

    # ======================================
    # VISUALISASI
    # ======================================

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Weak-only: cyan
    overlay[weak & ~strong] = [0, 255, 255]

    # Strong: merah
    overlay[strong] = [255, 0, 0]

    # Error map
    error = np.zeros((*gt.shape, 3), dtype=np.uint8)

    error[retained & gt] = [255, 255, 255]
    error[retained & ~gt] = [255, 0, 0]
    error[~retained & gt] = [0, 100, 255]

    panels = [
        (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "Citra Asli",
            None
        ),
        (gt, "Ground Truth", "gray"),
        (response, "Respons BTH (0-255)", "response"),
        (weak, f"Weak (>= {LOW})", "binary"),
        (strong, f"Strong (>= {HIGH})", "binary"),
        (overlay, "Weak-only: Cyan | Strong: Merah", None),
        (retained, "Komponen Dipertahankan", "binary"),
        (error, "Error Map Hysteresis", None),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    for ax, (data, title, display_type) in zip(
        axes.flat, panels
    ):

        if display_type == "response":
            ax.imshow(data, cmap="gray", vmin=0, vmax=255)

        elif display_type == "binary":
            ax.imshow(data, cmap="gray", vmin=0, vmax=1)

        elif display_type == "gray":
            ax.imshow(data, cmap="gray")

        else:
            ax.imshow(data)

        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{name} | {row['pola']}\n"
        f"TP: {row['delta_TP']:+.0f} | "
        f"FP: {row['delta_FP']:+.0f} | "
        f"Delta Dice: {row['delta_dice']:+.4f}",
        fontsize=13
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = OUTPUT_DIR / f"{name}_weak_strong.png"

    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Visualisasi: {output_path}")

    return summary, component_rows


# ==========================================
# 5. MAIN
# ==========================================

def main():

    df = pd.read_csv(CASE_FILE)

    # Lima kasus representatif yang dipilih berdasarkan median
    df = df[df["jenis_pemilihan"] == "MEDIAN"].copy()

    print(f"Jumlah kasus: {len(df)}")

    summaries = []
    all_components = []

    for _, row in df.iterrows():

        summary, components = analyze_case(row)

        summaries.append(summary)
        all_components.extend(components)

    # Simpan ringkasan
    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        OUTPUT_DIR / "ringkasan_weak_strong.csv",
        index=False
    )

    # Simpan detail komponen
    pd.DataFrame(all_components).to_csv(
        OUTPUT_DIR / "detail_komponen_weak_strong.csv",
        index=False
    )

    print("\n=== RINGKASAN WEAK DAN STRONG ===")

    print(summary_df.to_string(index=False))

    print("\n=== VALIDASI KLASIFIKASI GT ===")

    print(
        summary_df[
            [
                "image",
                "gt_pixels",
                "total_classified_GT",
                "classification_valid"
            ]
        ].to_string(index=False)
    )

    print("\nSemua analisis selesai.")
    print(f"Folder hasil: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
