
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

CATEGORIES = {
    "core": ["black_tophat.py"],
    "pipeline": [
        "split_dataset.py",
        "evaluasi_baseline.py",
        "evaluasi_final_testing.py",
        "evaluasi_hysteresis.py",
        "evaluasi_kernel_terkontrol.py",
        "evaluasi_testing.py",
        "tuning_parameter_kernel.py",
        "eksperimen_variasi_kernel.py",
    ],
}

for folder in ["core", "pipeline", "analysis", "visualization",
               "validation", "utilities", "dataset"]:
    (SRC / folder).mkdir(parents=True, exist_ok=True)

def category(filename):
    for folder, names in CATEGORIES.items():
        if filename in names:
            return folder
    if filename.startswith("visualisasi_"):
        return "visualization"
    if filename.startswith("cek_"):
        return "validation"
    if filename.startswith("pilih_kasus"):
        return "utilities"
    if filename.startswith(("analisis_", "uji_signifikansi",
                            "ci_bootstrap", "audit_dice",
                            "ringkasan_", "pemeriksaan_")):
        return "analysis"
    return "utilities"

# Perbaiki path dan import sebelum memindahkan file
files = list(SRC.glob("*.py"))

for file in files:
    content = file.read_text(encoding="utf-8")

    content = content.replace(
        "Path(__file__).resolve().parent.parent",
        "Path(__file__).resolve().parents[2]"
    )
    content = content.replace(
        "Path(__file__).resolve().parent",
        "Path(__file__).resolve().parents[1]"
    )
    content = content.replace(
        "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
        "os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"
    )
    content = content.replace(
        "os.path.dirname(os.path.abspath(__file__))",
        "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))"
    )
    content = content.replace(
        "from black_tophat import",
        "from src.core.black_tophat import"
    )
    content = content.replace(
        "import black_tophat",
        "import src.core.black_tophat as black_tophat"
    )
    content = content.replace(
        "from evaluasi_final_testing import",
        "from src.pipeline.evaluasi_final_testing import"
    )

    file.write_text(content, encoding="utf-8")

# Pindahkan seluruh script yang masih berada di src
moved = []

for file in list(SRC.glob("*.py")):
    if file.name == "__init__.py":
        continue

    folder = category(file.name)
    destination = SRC / folder / file.name

    if destination.exists():
        raise FileExistsError(f"File tujuan sudah ada: {destination}")

    shutil.move(str(file), str(destination))
    moved.append(f"{file.name} -> {folder}/")

# Buat package Python
for folder in ["", "core", "pipeline", "analysis",
               "visualization", "validation", "utilities", "dataset"]:
    init_file = SRC / folder / "__init__.py"
    init_file.touch(exist_ok=True)

print(f"\nMIGRASI SELESAI: {len(moved)} file dipindahkan.")
for item in moved:
    print(item)
