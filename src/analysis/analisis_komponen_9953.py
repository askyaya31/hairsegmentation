
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "dataset" / "dermoscopic_image"
MASK_DIR = ROOT / "dataset" / "hair_mask"

OUTPUT_DIR = ROOT / "results" / "analisis_kasus" / "komponen_9953"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_ID = "ISIC_0009953"
LOW = 15
HIGH = 40
BASELINE_THRESHOLD = 20

KERNEL_LENGTH = 15
ANGLES = [0, 30, 60, 90, 120, 150]


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
            gray,
            cv2.MORPH_CLOSE,
            kernel
        )

        response = cv2.subtract(closing, gray)
        responses.append(response)

    return np.maximum.reduce(responses)


def find_image(folder, image_id):
    for extension in [".jpg", ".jpeg", ".png", ".bmp"]:
        path = folder / (image_id + extension)
        if path.exists():
            return path

    raise FileNotFoundError(f"File {image_id} tidak ditemukan di {folder}")


def main():
    image_path = find_image(IMAGE_DIR, IMAGE_ID)
    mask_path = find_image(MASK_DIR, IMAGE_ID)

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError("Gagal membaca citra atau ground truth.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    response = multi_directional_black_tophat(gray)

    gt = mask > 127
    weak = response >= LOW
    strong = response >= HIGH

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        weak.astype(np.uint8),
        connectivity=8
    )

    baseline = response >= BASELINE_THRESHOLD
    hysteresis = np.zeros_like(weak)

    rows = []

    for label in range(1, num_labels):
        component = labels == label

        area = int(component.sum())
        strong_count = int(np.logical_and(component, strong).sum())

        retained = strong_count > 0

        if retained:
            hysteresis[component] = True

        tp = int(np.logical_and(component, gt).sum())
        fp = int(np.logical_and(component, ~gt).sum())

        baseline_tp = int(
            np.logical_and(
                np.logical_and(component, baseline), gt
            ).sum()
        )

        baseline_fp = int(
            np.logical_and(
                np.logical_and(component, baseline), ~gt
            ).sum()
        )

        values = response[component]

        rows.append({
            "label": label,
            "area": area,
            "strong_pixels": strong_count,
            "retained": retained,
            "gt_pixels": tp,
            "non_gt_pixels": fp,
            "mean_response": float(values.mean()),
            "max_response": int(values.max()),
            "baseline_TP": baseline_tp,
            "baseline_FP": baseline_fp,
            "hysteresis_TP": tp if retained else 0,
            "hysteresis_FP": fp if retained else 0
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_DIR / "detail_komponen.csv",
        index=False
    )

    retained_df = df[df["retained"] == True].copy()

    print("=== RINGKASAN KOMPONEN ===")
    print(f"Total komponen weak : {len(df)}")
    print(f"Komponen dipertahankan: {len(retained_df)}")
    print(f"Komponen dibuang     : {(~df['retained']).sum()}")

    print("\n=== TOTAL PIKSEL ===")
    print(f"Baseline TP : {int(df['baseline_TP'].sum())}")
    print(f"Baseline FP : {int(df['baseline_FP'].sum())}")
    print(f"Hysteresis TP: {int(retained_df['hysteresis_TP'].sum())}")
    print(f"Hysteresis FP: {int(retained_df['hysteresis_FP'].sum())}")

    print("\n=== 15 KOMPONEN DENGAN FP TERBESAR ===")
    columns = [
        "label", "area", "strong_pixels",
        "gt_pixels", "non_gt_pixels",
        "mean_response", "max_response"
    ]

    print(
        retained_df.sort_values(
            "non_gt_pixels", ascending=False
        )[columns].head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    print("\nDetail disimpan di:")
    print(OUTPUT_DIR / "detail_komponen.csv")


if __name__ == "__main__":
    main()
