# Segmentasi Rambut pada Citra Dermoskopi

Proyek pengolahan citra digital untuk melakukan segmentasi rambut pada citra dermoskopi menggunakan metode **Multi-Directional Black Top-Hat**, serta membandingkan metode **Single Thresholding** dan **Hysteresis Thresholding**.

## 1. Gambaran Umum

Proyek ini bertujuan untuk mengkaji penerapan metode pengolahan citra berbasis morfologi dalam mendeteksi dan melakukan segmentasi rambut pada citra dermoskopi.

Metode utama yang digunakan adalah Multi-Directional Black Top-Hat (BTH) untuk memperkuat respons struktur gelap dan memanjang yang berpotensi merupakan rambut.

Hasil BTH kemudian diubah menjadi mask biner menggunakan dua pendekatan:

- **Single Thresholding:** Metode pembanding yang menggunakan satu nilai threshold.
- **Hysteresis Thresholding:** Metode yang menggunakan dua nilai threshold, yaitu low dan high, serta Connected Components untuk menentukan komponen yang dipertahankan.

Kedua metode dibandingkan berdasarkan Dice Score, Intersection over Union (IoU), Precision, Recall, dan waktu pemrosesan.

## 2. Metode yang Digunakan

### 2.1 Multi-Directional Black Top-Hat

Black Top-Hat merupakan operasi morfologi yang diperoleh dari selisih hasil morphological closing dengan citra grayscale asli.

\[
BTH(I)=Closing(I)-I
\]

Operasi ini digunakan untuk memperkuat struktur gelap yang ukurannya lebih kecil dibandingkan structuring element.

Dalam proyek ini, digunakan line-shaped structuring element dengan enam orientasi:

`0°, 30°, 60°, 90°, 120°, 150°`

Hasil dari keenam orientasi tersebut digabungkan menggunakan nilai maksimum pada setiap piksel untuk menghasilkan response map akhir.

### 2.2 Single Thresholding

Single Thresholding mengubah response map menjadi mask biner menggunakan satu nilai threshold.

Piksel dengan nilai respons lebih besar atau sama dengan threshold dikategorikan sebagai foreground, sedangkan piksel lainnya menjadi background.

### 2.3 Hysteresis Thresholding

Hysteresis Thresholding menggunakan dua nilai threshold:

- **Low threshold:** Menentukan piksel yang termasuk dalam weak mask.
- **High threshold:** Menentukan piksel yang termasuk dalam strong mask.

Selanjutnya, weak mask dibagi menjadi beberapa komponen menggunakan 8-connected Components.

Komponen akan dipertahankan apabila memiliki minimal satu piksel yang termasuk dalam strong mask. Komponen tanpa piksel strong akan dibuang.

Pendekatan ini digunakan untuk menguji apakah pemilihan komponen berdasarkan keberadaan piksel strong dapat meningkatkan hasil segmentasi dibandingkan Single Thresholding.

## 3. Dataset

Penelitian ini menggunakan dataset publik:

**A Skin Lesion Hair Mask Dataset with Fine-Grained Annotations (2023)**

- Jumlah citra: 500
- Resolusi citra: 512 × 512 piksel
- Data: Citra dermoskopi beserta hair mask.
- DOI: [10.1016/j.dib.2023.109249](https://doi.org/10.1016/j.dib.2023.109249)
- Sumber dataset: [Mendeley Data](https://data.mendeley.com/datasets/j5ywpd2p27/1)

Dataset tidak disertakan dalam repository ini dan perlu diunduh secara terpisah.

### Struktur Dataset

Setelah diunduh, susun dataset dengan struktur berikut:

```text
dataset/
├── dermoscopic_image/
├── hair_mask/
├── overlay/
└── additional_materials/
```

Setiap citra dermoskopi harus memiliki hair mask yang sesuai dengan nama file yang sama.

## 4. Persiapan Dataset

Sebelum eksperimen dilakukan, dataset dianalisis untuk memeriksa:

- Ketersediaan citra dan mask.
- Kesesuaian dimensi citra dan mask.
- Nilai piksel pada mask.
- Persentase area rambut pada setiap mask.
- Citra dengan area rambut kosong, sangat sedikit, atau sangat besar.

Selanjutnya, 500 citra dibagi menjadi dua subset:

| Subset | Jumlah Citra | Tujuan |
|---|---:|---|
| Tuning | 350 | Pemilihan parameter |
| Testing | 150 | Evaluasi akhir |

Pembagian menggunakan stratified sampling dengan `random_state=42`. Lima citra eksplorasi yang telah ditentukan sebelumnya dimasukkan ke subset tuning.

Hasil analisis dataset dan pembagian data disimpan dalam file CSV di direktori `results/`.

## 5. Konfigurasi Eksperimen

Empat konfigurasi diuji menggunakan 150 citra testing yang sama.

| Konfigurasi | Kernel Length | Metode | Parameter |
|---|---:|---|---|
| A | 9 | Single Thresholding | Threshold = 20 |
| B | 9 | Hysteresis Thresholding | Low = 10, High = 30 |
| C | 11 | Single Thresholding | Threshold = 20 |
| D | 11 | Hysteresis Thresholding | Low = 15, High = 40 |

Konfigurasi dipilih berdasarkan hasil parameter tuning pada subset tuning.

## 6. Evaluation Metrics

Evaluasi hasil segmentasi menggunakan beberapa metrik:

- **Dice Score:** Mengukur tingkat kesamaan antara hasil prediksi dan Ground Truth.
- **Intersection over Union (IoU):** Mengukur perbandingan area irisan terhadap area gabungan hasil prediksi dan Ground Truth.
- **Precision:** Mengukur proporsi piksel hasil prediksi yang sesuai dengan Ground Truth.
- **Recall:** Mengukur proporsi piksel rambut pada Ground Truth yang berhasil terdeteksi.
- **Processing Time:** Mengukur waktu yang dibutuhkan untuk menjalankan proses BTH dan thresholding.

## 7. Analisis Tambahan

Selain evaluasi performa, dilakukan beberapa analisis lanjutan untuk memahami karakteristik hasil segmentasi:

- **Wilcoxon Signed-Rank Test:** Menguji perbedaan hasil Dice Score pada data berpasangan.
- **Bootstrap Confidence Interval:** Mengestimasi rentang ketidakpastian hasil pengukuran.
- **Rank-Biserial Correlation:** Mengukur besar efek perbedaan hasil.
- **Holm Correction:** Mengoreksi nilai signifikansi pada pengujian berganda.
- **Spearman Correlation:** Menganalisis hubungan antara retensi komponen dan perubahan performa.
- **Leave-One-Out Analysis:** Menguji kestabilan hasil korelasi dengan mengeluarkan satu citra secara bergantian.
- **Connected Component Analysis:** Menganalisis komponen yang dipertahankan dan dibuang oleh Hysteresis Thresholding.

## 8. Struktur Repository

```text
HairSegmentation/
├── dataset/
│   ├── dermoscopic_image/
│   ├── hair_mask/
│   ├── overlay/
│   └── additional_materials/
│
├── src/
│   ├── black_tophat.py
│   ├── analisis_dataset.py
│   ├── split_dataset.py
│   ├── tuning_parameter_kernel.py
│   ├── evaluasi_kernel_terkontrol.py
│   ├── uji_signifikansi_kernel_terkontrol.py
│   ├── analisis_komponen_150_testing.py
│   ├── analisis_retensi_komponen_150.py
│   └── ...
│
├── results/
│   ├── analisis_dataset.csv
│   ├── pembagian_dataset.csv
│   ├── tuning_parameter_kernel/
│   ├── evaluasi_kernel_terkontrol/
│   ├── analisis_komponen_150_testing_gray_converted/
│   ├── analisis_retensi_komponen_150/
│   └── ...
│
├── main.py
├── requirements.txt
└── README.md
```

Folder `dataset/` tidak disertakan dalam GitHub. File CSV hasil eksperimen dan analisis disertakan sebagai dokumentasi hasil penelitian.

## 9. Instalasi

### Kebutuhan

- Python 3.10
- Library Python yang tercantum dalam `requirements.txt`

Instal seluruh library dengan perintah:

```bash
python -m pip install -r requirements.txt
```

### Persiapan Dataset

Unduh dataset dari sumber yang telah dicantumkan, kemudian letakkan citra dan mask pada direktori masing-masing.

Pastikan setiap citra memiliki mask dengan nama file yang sama.

## 10. Menjalankan Eksperimen

Seluruh perintah dijalankan dari direktori utama proyek.

### Tahap 1: Analisis dan Pembagian Dataset

```bash
python src/analisis_dataset.py
python src/split_dataset.py
```

### Tahap 2: Parameter Tuning

```bash
python src/tuning_parameter_kernel.py
```

### Tahap 3: Evaluasi pada Data Testing

```bash
python src/evaluasi_kernel_terkontrol.py
```

### Tahap 4: Uji Signifikansi Statistik

```bash
python src/uji_signifikansi_kernel_terkontrol.py
```

### Tahap 5: Analisis Komponen

```bash
python src/analisis_komponen_150_testing.py
```

### Tahap 6: Analisis Retensi Komponen

```bash
python src/analisis_retensi_komponen_150.py
```

Script analisis tambahan dan visualisasi tersedia di direktori `src/`. Jalankan setelah file masukan yang dibutuhkan tersedia.

## 11. Hasil Eksperimen

Hasil evaluasi pada 150 citra testing menunjukkan bahwa Hysteresis Thresholding menghasilkan peningkatan rata-rata Dice Score pada kedua kernel yang diuji.

| Konfigurasi | Dice Score |
|---|---:|
| A: Baseline K9 | 0.397228 |
| B: Hysteresis K9 | 0.438264 |
| C: Baseline K11 | 0.402083 |
| D: Hysteresis K11 | 0.421373 |

Beberapa temuan dari hasil pengujian:

- Pada Kernel 9, Hysteresis Thresholding meningkatkan Dice Score dan Recall, tetapi juga menghasilkan peningkatan False Positive yang cukup besar.
- Pada Kernel 11, Hysteresis Thresholding meningkatkan Dice Score dan Recall, sedangkan perubahan False Positive tidak signifikan secara statistik.
- Perbandingan kedua konfigurasi Hysteresis menunjukkan bahwa hasil Kernel 9 lebih tinggi daripada Kernel 11 berdasarkan Dice Score.

Hasil tersebut menunjukkan adanya trade-off antara peningkatan deteksi rambut dan bertambahnya piksel prediksi yang tidak sesuai dengan Ground Truth. Oleh karena itu, Hysteresis Thresholding tidak selalu memberikan peningkatan pada seluruh metrik evaluasi.

## 12. Catatan

- Parameter tuning dilakukan pada 350 citra tuning.
- Evaluasi akhir dilakukan pada 150 citra testing.
- Data testing tidak digunakan dalam pemilihan parameter.
- Seluruh konfigurasi evaluasi menggunakan citra testing yang sama.
- Metode yang digunakan berbasis pengolahan citra klasik dan operasi morfologi.
- Dataset tidak disertakan dalam repository karena perlu diunduh secara terpisah.