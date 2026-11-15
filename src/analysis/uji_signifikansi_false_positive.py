
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import wilcoxon, rankdata

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "results" / "evaluasi_kernel_terkontrol" / "hasil_detail.csv"
)

OUTPUT_DIR = (
    ROOT / "results" / "uji_signifikansi_false_positive"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 20000
RANDOM_SEED = 42
IMAGE_PIXELS = 512 * 512

COMPARISONS = [
    ("A", "B", "Kernel 9: Baseline vs Hysteresis"),
    ("C", "D", "Kernel 11: Baseline vs Hysteresis"),
]

# ============================================================
# 1. BACA DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required = ["filename", "configuration", "fp"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Kolom tidak ditemukan: {missing}")

df["fp"] = pd.to_numeric(df["fp"], errors="raise")
df["configuration"] = df["configuration"].astype(str)

if df.duplicated(["filename", "configuration"]).any():
    raise ValueError("Ditemukan duplikasi filename dan configuration.")

# ============================================================
# 2. FUNGSI RANK-BISERIAL CORRELATION
# ============================================================

def calculate_rank_biserial(differences):
    """
    Efek berpasangan:
    positif = konfigurasi kedua memiliki FP lebih tinggi.
    """
    differences = np.asarray(differences, dtype=float)

    nonzero = differences[differences != 0]

    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero))

    positive_rank_sum = ranks[nonzero > 0].sum()
    negative_rank_sum = ranks[nonzero < 0].sum()

    denominator = positive_rank_sum + negative_rank_sum

    return (positive_rank_sum - negative_rank_sum) / denominator


# ============================================================
# 3. FUNGSI PAIRED BOOTSTRAP
# ============================================================

def paired_bootstrap(differences, n_bootstrap, seed):
    rng = np.random.default_rng(seed)

    differences = np.asarray(differences, dtype=float)
    n = len(differences)

    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sampled = rng.choice(differences, size=n, replace=True)
        bootstrap_means[i] = np.mean(sampled)

    ci_lower, ci_upper = np.percentile(
        bootstrap_means, [2.5, 97.5]
    )

    bootstrap_se = np.std(bootstrap_means, ddof=1)

    return bootstrap_se, ci_lower, ci_upper


# ============================================================
# 4. UJI SETIAP PASANGAN
# ============================================================

results = []
detail_results = []

print("=" * 75)
print("UJI SIGNIFIKANSI FALSE POSITIVE")
print("=" * 75)

for baseline_config, hyst_config, label in COMPARISONS:

    baseline = (
        df[df["configuration"] == baseline_config]
        [["filename", "fp"]]
        .rename(columns={"fp": "baseline_fp"})
    )

    hysteresis = (
        df[df["configuration"] == hyst_config]
        [["filename", "fp"]]
        .rename(columns={"fp": "hysteresis_fp"})
    )

    paired = baseline.merge(
        hysteresis,
        on="filename",
        how="inner",
        validate="one_to_one"
    )

    if len(paired) != 150:
        raise ValueError(
            f"{label}: jumlah pasangan tidak sama dengan 150."
        )

    paired = paired.sort_values("filename").reset_index(drop=True)

    # Selisih positif berarti Hysteresis menghasilkan FP lebih banyak
    differences = (
        paired["hysteresis_fp"] - paired["baseline_fp"]
    ).to_numpy(dtype=float)

    baseline_values = paired["baseline_fp"].to_numpy(dtype=float)
    hyst_values = paired["hysteresis_fp"].to_numpy(dtype=float)

    # Wilcoxon Signed-Rank Test
    statistic, p_value = wilcoxon(
        hyst_values,
        baseline_values,
        alternative="two-sided",
        zero_method="wilcox"
    )

    # Paired bootstrap
    bootstrap_se, ci_lower, ci_upper = paired_bootstrap(
        differences,
        N_BOOTSTRAP,
        RANDOM_SEED
    )

    # Statistik deskriptif
    increased = int(np.sum(differences > 0))
    decreased = int(np.sum(differences < 0))
    unchanged = int(np.sum(differences == 0))

    mean_difference = float(np.mean(differences))
    median_difference = float(np.median(differences))

    mean_baseline = float(np.mean(baseline_values))
    mean_hysteresis = float(np.mean(hyst_values))

    relative_change = (
        (mean_hysteresis - mean_baseline) / mean_baseline * 100
        if mean_baseline != 0 else np.nan
    )

    # Rank-biserial correlation
    rank_biserial = calculate_rank_biserial(differences)

    results.append({
        "comparison": label,
        "baseline": baseline_config,
        "hysteresis": hyst_config,
        "n": len(paired),
        "mean_fp_baseline": mean_baseline,
        "mean_fp_hysteresis": mean_hysteresis,
        "mean_difference": mean_difference,
        "median_difference": median_difference,
        "relative_change_percent": relative_change,
        "increased": increased,
        "decreased": decreased,
        "unchanged": unchanged,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "rank_biserial": rank_biserial,
        "bootstrap_se": bootstrap_se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_lower_percentage": ci_lower / IMAGE_PIXELS * 100,
        "ci_upper_percentage": ci_upper / IMAGE_PIXELS * 100,
    })

    paired["comparison"] = label
    paired["fp_difference"] = differences
    detail_results.append(paired)

# ============================================================
# 5. KOREKSI HOLM
# ============================================================

results_df = pd.DataFrame(results)

p_values = results_df["p_value"].to_numpy()
m = len(p_values)

order = np.argsort(p_values)
adjusted = np.empty(m)

running_max = 0.0

for rank, index in enumerate(order):
    adjusted_value = (m - rank) * p_values[index]
    running_max = max(running_max, adjusted_value)
    adjusted[index] = min(running_max, 1.0)

results_df["p_value_holm"] = adjusted
results_df["signifikan_holm"] = (
    results_df["p_value_holm"] < 0.05
)

# ============================================================
# 6. TAMPILKAN HASIL
# ============================================================

for _, row in results_df.iterrows():

    print("\n" + "-" * 75)
    print(row["comparison"])
    print("-" * 75)

    print(f"Jumlah pasangan       : {row['n']}")
    print(f"Mean FP Baseline      : {row['mean_fp_baseline']:.2f}")
    print(f"Mean FP Hysteresis    : {row['mean_fp_hysteresis']:.2f}")
    print(f"Mean selisih FP       : {row['mean_difference']:+.2f}")
    print(f"Median selisih FP     : {row['median_difference']:+.2f}")

    print(
        f"Perubahan relatif     : "
        f"{row['relative_change_percent']:+.2f}%"
    )

    print(f"FP meningkat          : {int(row['increased'])}")
    print(f"FP menurun            : {int(row['decreased'])}")
    print(f"Tidak berubah         : {int(row['unchanged'])}")

    print(f"Wilcoxon statistic    : {row['wilcoxon_statistic']:.2f}")
    print(f"p-value               : {row['p_value']:.10e}")
    print(f"p-value Holm          : {row['p_value_holm']:.10e}")

    print(f"Rank-biserial         : {row['rank_biserial']:.6f}")

    print(f"Bootstrap SE          : {row['bootstrap_se']:.2f}")
    print(
        f"95% CI mean difference: "
        f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"
    )

    print(
        f"95% CI area (%)      : "
        f"[{row['ci_lower_percentage']:.4f}%, "
        f"{row['ci_upper_percentage']:.4f}%]"
    )

    print(f"Signifikan (Holm)     : {row['signifikan_holm']}")

# ============================================================
# 7. SIMPAN HASIL
# ============================================================

detail_df = pd.concat(detail_results, ignore_index=True)

results_df.to_csv(
    OUTPUT_DIR / "hasil_uji_signifikansi_fp.csv",
    index=False
)

detail_df.to_csv(
    OUTPUT_DIR / "detail_selisih_fp.csv",
    index=False
)

print("\n" + "=" * 75)
print("HASIL DISIMPAN DI:")
print(OUTPUT_DIR)
print("=" * 75)
