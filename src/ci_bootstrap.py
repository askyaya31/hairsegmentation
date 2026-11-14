
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULT_FILE = (
    ROOT / "results" / "evaluasi_final_testing" / "hasil_detail.csv"
)

N_BOOTSTRAP = 20_000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42


def main():

    if not RESULT_FILE.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {RESULT_FILE}"
        )

    df = pd.read_csv(RESULT_FILE)

    required_columns = [
        "baseline_dice",
        "hysteresis_dice"
    ]

    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            "Kolom baseline_dice atau hysteresis_dice tidak ditemukan."
        )

    if df[required_columns].isna().any().any():
        raise ValueError("Terdapat nilai kosong pada data Dice.")

    baseline = df["baseline_dice"].to_numpy(dtype=float)
    hysteresis = df["hysteresis_dice"].to_numpy(dtype=float)

    if len(baseline) != len(hysteresis):
        raise ValueError("Jumlah pasangan data tidak sama.")

    if len(baseline) != 150:
        raise ValueError(
            f"Jumlah citra seharusnya 150, ditemukan {len(baseline)}."
        )

    if not np.isfinite(baseline).all() or not np.isfinite(hysteresis).all():
        raise ValueError("Ditemukan nilai Dice yang tidak valid.")

    # Selisih Dice berpasangan
    difference = hysteresis - baseline

    observed_difference = difference.mean()

    # Paired Bootstrap
    rng = np.random.default_rng(RANDOM_SEED)

    n = len(difference)

    bootstrap_means = np.empty(N_BOOTSTRAP)

    for i in range(N_BOOTSTRAP):

        # Indeks yang sama digunakan untuk kedua metode
        indices = rng.integers(0, n, size=n)

        bootstrap_means[i] = difference[indices].mean()

    # Percentile Confidence Interval
    alpha = 1 - CONFIDENCE_LEVEL

    lower_ci = np.percentile(
        bootstrap_means,
        100 * alpha / 2
    )

    upper_ci = np.percentile(
        bootstrap_means,
        100 * (1 - alpha / 2)
    )

    bootstrap_se = bootstrap_means.std(ddof=1)

    print("=" * 65)
    print("PAIRED BOOTSTRAP CONFIDENCE INTERVAL")
    print("=" * 65)

    print(f"Jumlah citra          : {n}")
    print(f"Jumlah bootstrap      : {N_BOOTSTRAP}")
    print(f"Random seed           : {RANDOM_SEED}")

    print("\nSelisih Dice:")
    print(f"Mean baseline         : {baseline.mean():.6f}")
    print(f"Mean hysteresis       : {hysteresis.mean():.6f}")
    print(f"Mean selisih Dice     : {observed_difference:.6f}")

    print("\nHasil Bootstrap:")
    print(f"Bootstrap SE          : {bootstrap_se:.6f}")
    print(
        f"95% CI                : "
        f"[{lower_ci:.6f}, {upper_ci:.6f}]"
    )

    print("\nInterpretasi:")

    if lower_ci > 0:
        print("Seluruh interval berada di atas nol.")
        print("Hysteresis memiliki peningkatan Dice yang konsisten.")
    elif upper_ci < 0:
        print("Seluruh interval berada di bawah nol.")
        print("Hysteresis memiliki Dice yang lebih rendah.")
    else:
        print("Interval mencakup nol.")
        print("Peningkatan Dice belum dapat dipastikan berdasarkan CI ini.")

    # Simpan hasil
    output_dir = ROOT / "results" / "evaluasi_final_testing"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([{
        "n_images": n,
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "mean_difference": observed_difference,
        "bootstrap_se": bootstrap_se,
        "ci_lower": lower_ci,
        "ci_upper": upper_ci,
        "confidence_level": CONFIDENCE_LEVEL
    }])

    output_file = output_dir / "hasil_bootstrap_ci.csv"
    summary.to_csv(output_file, index=False)

    print(f"\nHasil disimpan di:\n{output_file}")


if __name__ == "__main__":
    main()
