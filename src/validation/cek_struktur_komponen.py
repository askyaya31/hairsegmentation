
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Cari file berdasarkan nama di dalam folder results
files = list((ROOT / "results").rglob("detail_per_komponen.csv"))

if not files:
    print("File detail_per_komponen.csv tidak ditemukan.")
    raise SystemExit

csv_path = files[0]
df = pd.read_csv(csv_path)

print("=" * 60)
print("LOKASI FILE:", csv_path)
print("=" * 60)

print("\nJUMLAH BARIS DAN KOLOM:")
print(df.shape)

print("\nNAMA KOLOM:")
print(df.columns.tolist())

print("\nTIPE DATA:")
print(df.dtypes)

print("\nCONTOH 5 BARIS:")
print(df.head().to_string(index=False))

print("\nNILAI UNIK KOLOM KATEGORIS:")
for col in df.select_dtypes(include=["object", "category"]).columns:
    print(f"\n{col}:")
    print(df[col].value_counts().head(15).to_dict())
