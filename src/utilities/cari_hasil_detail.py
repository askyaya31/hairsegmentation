
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

files = list(ROOT.rglob("hasil_detail.csv"))

print("=" * 70)
print("DAFTAR FILE hasil_detail.csv")
print("=" * 70)

if not files:
    print("Tidak ditemukan.")
else:
    for i, path in enumerate(files, 1):
        print(f"\n{i}. {path.relative_to(ROOT)}")

print(f"\nTotal ditemukan: {len(files)}")
