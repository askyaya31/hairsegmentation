
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import wilcoxon, rankdata


# ============================================================
# DIREKTORI
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "results"
    / "03_evaluation"
    / "kernel_terkontrol"
    / "hasil_detail.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "03_evaluation"
    / "kernel_terkontrol"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PARAMETER BOOTSTRAP
# ============================================================

N_BOOTSTRAP = 20000
RANDOM_SEED = 42
ALPHA = 0.05


# ============================================================
# KOREKSI HOLM
# ============================================================

def holm_correction(p_values):

    p_values = np.asarray(p_values, dtype=float)

    n = len(p_values)
    order = np.argsort(p_values)

    adjusted = np.empty(n)

    previous = 0.0

    for rank, index in enumerate(order):

        value = (n - rank) * p_values[index]

        adjusted[index] = min(
            1.0,
            max(previous, value)
        )

        previous = adjusted[index]

    return adjusted


# ============================================================
# RANK-BISERIAL CORRELATION
# ============================================================

def calculate_rank_biserial(differences):

    differences = np.asarray(differences)

    nonzero = differences[differences != 0]

    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero))

    positive_sum = ranks[nonzero > 0].sum()
    negative_sum = ranks[nonzero < 0].sum()

    denominator = positive_sum + negative_sum

    if denominator == 0:
        return 0.0

    return (
        positive_sum - negative_sum
    ) / denominator


# ============================================================
# PAIRED BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def paired_bootstrap(differences):

    differences = np.asarray(differences)

    rng = np.random.default_rng(RANDOM_SEED)

    n = len(differences)

    bootstrap_means = np.empty(N_BOOTSTRAP)

    for i in range(N_BOOTSTRAP):

        sample = rng.choice(
            differences,
            size=n,
            replace=True
        )

        bootstrap_means[i] = sample.mean()

    ci_lower, ci_upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    bootstrap_se = bootstrap_means.std(ddof=1)

    return {
        "bootstrap_se": bootstrap_se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }


# ============================================================
# UJI SATU PASANGAN KONFIGURASI
# ============================================================

def evaluate_pair(
    detail_df,
    config_a,
    config_b,
    comparison_name
):

    data_a = detail_df[
        detail_df["configuration"] == config_a
    ][["filename", "dice"]].copy()

    data_b = detail_df[
        detail_df["configuration"] == config_b
    ][["filename", "dice"]].copy()

    paired = data_a.merge(
        data_b,
        on="filename",
        suffixes=("_a", "_b"),
        validate="one_to_one"
    )

    paired = paired.sort_values("filename")

    if len(paired) != 150:
        raise ValueError(
            f"Jumlah pasangan {config_a}-{config_b} "
            f"tidak sesuai: {len(paired)}"
        )

    dice_a = paired["dice_a"].to_numpy()
    dice_b = paired["dice_b"].to_numpy()

    differences = dice_b - dice_a

    # Wilcoxon Signed-Rank Test
    test = wilcoxon(
        dice_b,
        dice_a,
        alternative="two-sided",
        zero_method="wilcox"
    )

    # Bootstrap CI
    bootstrap = paired_bootstrap(differences)

    # Effect size
    rank_biserial = calculate_rank_biserial(
        differences
    )

    return {
        "comparison": comparison_name,
        "config_a": config_a,
        "config_b": config_b,
        "n": len(paired),

        "dice_a_mean": dice_a.mean(),
        "dice_b_mean": dice_b.mean(),

        "mean_difference": differences.mean(),
        "median_difference": np.median(differences),

        "increase": int((differences > 0).sum()),
        "decrease": int((differences < 0).sum()),
        "unchanged": int((differences == 0).sum()),

        "wilcoxon_statistic": test.statistic,
        "p_value": test.pvalue,

        "rank_biserial": rank_biserial,

        **bootstrap
    }


# ============================================================
# MAIN
# ============================================================

def main():

    detail_df = pd.read_csv(INPUT_FILE)

    results = []

    print("=" * 70)
    print("UJI SIGNIFIKANSI KERNEL TERKONTROL")
    print("=" * 70)

    comparisons = [
        ("A", "B", "Kernel 9: Baseline vs Hysteresis"),
        ("C", "D", "Kernel 11: Baseline vs Hysteresis"),
        ("B", "D", "Hysteresis: Kernel 9 vs Kernel 11")
    ]
    for config_a, config_b, name in comparisons:

        result = evaluate_pair(
            detail_df,
            config_a,
            config_b,
            name
        )

        results.append(result)

    # Koreksi Holm untuk dua p-value
    p_values = [
        result["p_value"]
        for result in results
    ]

    adjusted_p = holm_correction(p_values)

    for result, p_adj in zip(results, adjusted_p):

        result["p_value_holm"] = p_adj

        result["significant"] = (
            p_adj < ALPHA
        )

    results_df = pd.DataFrame(results)

    output_file = (
        OUTPUT_DIR
        / "hasil_uji_signifikansi_kernel.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # TAMPILKAN HASIL
    # ========================================================

    for result in results:

        print("\n" + "-" * 70)
        print(result["comparison"])
        print("-" * 70)

        print(
            f"Jumlah pasangan       : {result['n']}"
        )

        print(
            f"Mean Dice A           : "
            f"{result['dice_a_mean']:.6f}"
        )

        print(
            f"Mean Dice B           : "
            f"{result['dice_b_mean']:.6f}"
        )

        print(
            f"Mean selisih Dice     : "
            f"{result['mean_difference']:+.6f}"
        )

        print(
            f"Median selisih Dice   : "
            f"{result['median_difference']:+.6f}"
        )

        print(
            f"Meningkat             : "
            f"{result['increase']}"
        )

        print(
            f"Menurun               : "
            f"{result['decrease']}"
        )

        print(
            f"Tetap                 : "
            f"{result['unchanged']}"
        )

        print(
            f"Wilcoxon statistic    : "
            f"{result['wilcoxon_statistic']:.4f}"
        )

        print(
            f"P-value               : "
            f"{result['p_value']:.10e}"
        )

        print(
            f"P-value Holm          : "
            f"{result['p_value_holm']:.10e}"
        )

        print(
            f"Rank-biserial         : "
            f"{result['rank_biserial']:.6f}"
        )

        print(
            f"Bootstrap SE          : "
            f"{result['bootstrap_se']:.6f}"
        )

        print(
            f"95% CI                : "
            f"[{result['ci_lower']:.6f}, "
            f"{result['ci_upper']:.6f}]"
        )

        print(
            f"Signifikan (Holm)     : "
            f"{result['significant']}"
        )

    print("\n" + "=" * 70)
    print("HASIL DISIMPAN")
    print(output_file)
    print("=" * 70)


if __name__ == "__main__":
    main()
