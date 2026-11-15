
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_FILE = ROOT / "results" / "pembagian_dataset.csv"

df = pd.read_csv(CSV_FILE)

exploratory_ids = {
    "ISIC_0000042",
    "ISIC_0000043",
    "ISIC_0000004",
    "ISIC_0000022",
    "ISIC_0000032"
}

# Sesuaikan nama kolom berdasarkan isi CSV
print("Kolom CSV:", df.columns.tolist())

filename_col = "filename"
split_col = "split"

df["image_id"] = df[filename_col].apply(
    lambda x: Path(str(x)).stem
)

result = df[df["image_id"].isin(exploratory_ids)]

print("\n=== PEMERIKSAAN DATA EKSPLORASI ===")
print(result[["image_id", split_col]].to_string(index=False))

if (result[split_col].str.upper() == "PENGUJIAN").any():
    print("\nPERINGATAN: Ada citra eksplorasi yang masuk data pengujian.")
else:
    print("\nAMAN: Semua citra eksplorasi berada di data tuning.")
