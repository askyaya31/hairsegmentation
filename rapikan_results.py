
from pathlib import Path
import shutil
import re

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SRC = ROOT / "src"

# Folder tujuan untuk setiap kelompok hasil
MAPPING = {
    "analisis_kasus": "06_error_analysis/analisis_kasus",
    "analisis_korelasi_fp_dice": "06_error_analysis/korelasi_fp_dice",
    "analisis_overlap_prediksi": "06_error_analysis/overlap_prediksi",
    "analisis_per_kategori": "06_error_analysis/per_kategori",
    "analisis_per_kategori_kernel_terkontrol": "06_error_analysis/per_kategori_kernel",
    "analisis_prediksi_false_positive_semua": "06_error_analysis/false_positive",
    "audit_dice_nol": "06_error_analysis/audit_dice_nol",

    "analisis_komponen_150_testing": "05_component_analysis/komponen_150_testing",
    "analisis_komponen_150_testing_gray_converted": "05_component_analysis/komponen_150_gray",
    "analisis_komponen_ground_truth": "05_component_analysis/ground_truth",
    "analisis_retensi_komponen_150": "05_component_analysis/retensi_150",
    "analisis_seluruh_komponen": "05_component_analysis/seluruh_komponen",

    "black_tophat": "02_preprocessing/black_tophat",

    "evaluasi_baseline": "03_evaluation/baseline",
    "evaluasi_baseline_tuning": "03_evaluation/baseline_tuning",
    "evaluasi_final_testing": "03_evaluation/final_testing",
    "evaluasi_hysteresis_tuning": "03_evaluation/hysteresis_tuning",
    "evaluasi_kernel_terkontrol": "03_evaluation/kernel_terkontrol",
    "evaluasi_testing": "03_evaluation/testing",

    "tuning_parameter_kernel": "04_tuning/parameter_kernel",
    "eksperimen_variasi_kernel": "09_kernel_experiments/variasi_kernel",

    "pemeriksaan_korelasi_150": "07_statistical_analysis/korelasi_150",
    "uji_signifikansi_false_positive": "07_statistical_analysis/false_positive",

    "cek_konsistensi_csv": "09_validation/konsistensi_csv",
    "cek_sumber_perbedaan_csv": "09_validation/sumber_perbedaan_csv",

    "visualisasi_audit_dice_nol": "08_visualizations/audit_dice_nol",
    "visualisasi_empat_konfigurasi": "08_visualizations/empat_konfigurasi",
    "visualisasi_hasil_akhir_hysteresis": "08_visualizations/hasil_akhir_hysteresis",
    "visualisasi_kasus_ekstrem": "08_visualizations/kasus_ekstrem",
    "visualisasi_komponen_terpilih": "08_visualizations/komponen_terpilih",
    "visualisasi_respons_bth_ekstrem": "08_visualizations/respons_bth_ekstrem",
}

# Buat seluruh folder kategori
for folder in [
    "01_dataset", "02_preprocessing", "03_evaluation",
    "04_tuning", "05_component_analysis", "06_error_analysis",
    "07_statistical_analysis", "08_visualizations",
    "09_kernel_experiments", "09_validation"
]:
    (RESULTS / folder).mkdir(parents=True, exist_ok=True)

# Perbarui referensi path pada script Python
updated = 0

for script in SRC.rglob("*.py"):
    content = script.read_text(encoding="utf-8")
    original = content

    for old, new in MAPPING.items():
        category, subfolder = new.split("/", 1)

        pattern = re.compile(
            r'(["\'])results\1(\s*/\s*)(["\'])'
            + re.escape(old) + r'\3'
        )

        content = pattern.sub(
            lambda m: (
                m.group(1) + "results" + m.group(1)
                + m.group(2) + m.group(3) + category + m.group(3)
                + m.group(2) + m.group(3) + subfolder + m.group(3)
            ),
            content
        )

        content = content.replace(
            "results/" + old,
            "results/" + new
        )
        content = content.replace(
            "results\\" + old,
            "results\\" + new.replace("/", "\\")
        )

    if content != original:
        script.write_text(content, encoding="utf-8")
        updated += 1

# Pindahkan folder hasil
moved = 0

for old, new in MAPPING.items():
    source = RESULTS / old
    destination = RESULTS / new

    if not source.exists():
        continue

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)
        shutil.rmtree(source)
    else:
        shutil.move(str(source), str(destination))
    moved += 1
    print(f"{old} -> {new}")

# Pindahkan CSV analisis dataset
csv = RESULTS / "analisis_dataset.csv"
if csv.exists():
    shutil.move(str(csv), str(RESULTS / "01_dataset" / csv.name))
    print("analisis_dataset.csv -> 01_dataset/")

print("\nMIGRASI RESULTS SELESAI")
print(f"Folder dipindahkan: {moved}")
print(f"Script diperbarui: {updated}")
