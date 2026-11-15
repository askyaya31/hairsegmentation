
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]

RESULT_FILE = (
    ROOT / "results" / "evaluasi_final_testing" / "hasil_detail.csv"
)


def main():

    df = pd.read_csv(RESULT_FILE)

    required_columns = ["baseline_dice", "hysteresis_dice"]

    if not all(col in df.columns for col in required_columns):
        raise ValueError("Kolom Dice tidak ditemukan dalam file CSV.")

    if df[required_columns].isna().any().any():
        raise ValueError("Terdapat nilai kosong pada data Dice.")

    baseline = df["baseline_dice"].to_numpy()
    hysteresis = df["hysteresis_dice"].to_numpy()

    if len(baseline) != len(hysteresis):
        raise ValueError("Jumlah pasangan data tidak sama.")

    difference = hysteresis - baseline

    # Wilcoxon Signed-Rank Test
    statistic, p_value = wilcoxon(
        hysteresis,
        baseline,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto"
    )

    # Effect size: matched-pairs rank-biserial correlation
    nonzero = difference[difference != 0]

    from scipy.stats import rankdata

    ranks = rankdata(np.abs(nonzero))

    positive_rank = ranks[nonzero > 0].sum()
    negative_rank = ranks[nonzero < 0].sum()

    rank_biserial = (
        (positive_rank - negative_rank) /
        (positive_rank + negative_rank)
    )

    alpha = 0.05

    print("=" * 65)
    print("UJI SIGNIFIKANSI WILCOXON SIGNED-RANK TEST")
    print("=" * 65)

    print(f"Jumlah citra       : {len(df)}")
    print(f"Dice baseline      : {baseline.mean():.6f}")
    print(f"Dice hysteresis    : {hysteresis.mean():.6f}")

    print(f"Mean selisih Dice  : {difference.mean():.6f}")
    print(f"Median selisih     : {np.median(difference):.6f}")

    print(f"Dice meningkat     : {(difference > 0).sum()}")
    print(f"Dice menurun       : {(difference < 0).sum()}")
    print(f"Dice tetap         : {(difference == 0).sum()}")

    print("\nHasil Wilcoxon:")
    print(f"Statistic          : {statistic}")
    print(f"P-value            : {p_value:.10g}")
    print(f"Alpha              : {alpha}")

    print("\nEffect Size:")
    print(f"Rank-biserial      : {rank_biserial:.6f}")

    print("\nKeputusan:")

    if p_value < alpha:
        print("Tolak H0.")
        print("Terdapat perbedaan Dice yang signifikan secara statistik.")
    else:
        print("Gagal menolak H0.")
        print("Belum terdapat bukti perbedaan Dice yang signifikan.")

    if difference.mean() > 0:
        print("Rata-rata Dice Hysteresis lebih tinggi.")
    elif difference.mean() < 0:
        print("Rata-rata Dice Hysteresis lebih rendah.")
    else:
        print("Rata-rata Dice kedua metode sama.")

    print("\nFile sumber:")
    print(RESULT_FILE)


if __name__ == "__main__":
    main()
