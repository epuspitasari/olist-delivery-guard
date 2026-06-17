# Prediksi Risiko Keterlambatan Pengiriman  
**Studi Kasus: Olist E‑Commerce Brasil** *Proyek Portofolio Akhir — Pacmann Data Science & AI*

## 1. Deskripsi Proyek
Proyek ini dibangun sebagai bagian dari kurikulum terapan dalam **Pacmann Data Science & AI**. Fokus utamanya adalah mengembangkan sistem peringatan dini (*early warning system*) berbasis Machine Learning untuk memprediksi risiko keterlambatan pengiriman paket (`is_late`) pada platform e-commerce Olist Brasil. 

Pendekatan rekayasa fitur disusun secara ketat menggunakan **3 Pilar Logistik**: **Waktu** (Temporal), **Jarak** (Geospasial), dan **Administrasi** (Payment) guna mentransformasi metrik evaluasi teknis model langsung menjadi metrik dampak finansial bagi bisnis.

---

## 2. Sumber Data & Metadata
Dataset: **[Brazilian E‑Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)**.

Untuk menjaga integritas model, proyek ini menggunakan 3 dataset utama yang saling terintegrasi berdasarkan skema metadata Olist:
- `olist_orders_dataset.csv` (Data utama rekam transaksi dan lini waktu operasional)
- `olist_order_payments_dataset.csv` (Data metode administrasi transaksi dan cicilan)
- `olist_customers_dataset.csv` (Data identitas geospasial/lokasi konsumen)

**Alasan Pemilihan:** Fitur yang digunakan dibatasi hanya pada data yang telah tersedia saat transaksi terjadi (*real-time checkout*) guna menghindari *data leakage* (kebocoran data masa depan).

---

## 3. Struktur Proyek
- `data/raw/` — Dataset asli (.csv) dari Kaggle.
- `data/interim/` — Data hasil pembersihan awal per tabel (.pkl).
- `data/processed/` — Dataset final hasil penggabungan dan *feature engineering* (.pkl).
- `models/` — Artefak model final (`adaboost_final_model.pkl`) dan `scaler_modeling.pkl`.
- `notebooks/` — Dokumentasi eksperimen lengkap.
- `utils.py` — Modul Python untuk fungsi load data, verifikasi, dan preprocessing secara modular.

---

## 4. Ringkasan Notebook
1. **`01_data_cleaning.ipynb`**: Pembersihan atomik, validasi tipe data *datetime*, dan penanganan *missing values*.
2. **`02_eda.ipynb`**: Validasi hipotesis logistik (Tren Black Friday, Lokasi SP vs Luar SP).
3. **`03_preprocessing_&_fe.ipynb`**: Konstruksi fitur `is_sp_district`, ekstraksi waktu transaksi, dan pencegahan kebocoran data.
4. **`04_modeling.ipynb`**: 
   - **Baseline**: Logistic Regression (Evaluasi data asli).
   - **Tournament Stage**: Perbandingan 5 model dengan SMOTE pada training set (Decision Tree, Random Forest, XGBoost, AdaBoost, KNN).
   - **Winner**: **AdaBoost** terpilih sebagai model final dengan optimasi metrik **Recall** guna meminimalkan risiko paket telat yang tidak terdeteksi.

---

## 5. Performa & Insight Utama
- **Insight Geografis & Temporal**: Wilayah di luar negara bagian São Paulo memiliki risiko keterlambatan signifikan lebih tinggi karena tantangan jarak infrastruktur. Risiko pengiriman juga memuncak pada masa *Peak Season* (Oktober–Desember).
- **Strategi Evaluasi**: Fokus utama dioptimasikan pada metrik **Recall (0.62)** pada Test Set untuk meminimalkan *False Negatives* (keterlambatan yang luput dari pengawasan AI).
- **Kesimpulan Model**: Model AdaBoost terbukti paling stabil (*robust*) mendeteksi risiko di tengah kondisi ketidakseimbangan kelas (*imbalanced data* ~7.8% kasus telat).

---

## 6. Simulasi Dampak Bisnis (Business Impact Simulation)
Untuk mengukur nilai riil dari model Machine Learning, dilakukan simulasi finansial pada **Test Set** (berisi total 813 kasus pengiriman terlambat aktual) menggunakan asumsi parameter biaya logistik berikut:
* **Cost of False Negative (CFN = R$ 50.00)**: Biaya kompensasi/klaim jika sistem memprediksi tepat waktu namun aslinya telat (luput dari deteksi).
* **Cost of False Positive (CFP = R$ 15.00)**: Biaya mitigasi/prioritas pengiriman jika sistem memprediksi telat namun aslinya bisa tepat waktu.

### Perbandingan Penghematan Finansial:
1. **Tanpa Model AI (Model Baseline/Status Quo)**
   * Sistem menganggap semua pengiriman aman (*False Negative* massal).
   * **Total Kerugian Finansial:** 813 paket x R$ 50.00 = **R$ 40,650.00**

2. **Dengan Olist Delivery Guard (Model AdaBoost Final)**
   * Model berhasil menangkap secara akurat **502 paket berisiko** (*True Positive*) untuk dimitigasi dengan biaya rendah, dan hanya meluputkan **311 paket** (*False Negative*).
   * **Total Pengeluaran setelah Mitigasi:** (311 x R$ 50.00) + (502 x R$ 15.00) = **R$ 23,080.00**

### 📊 Kesimpulan Nilai Bisnis:
* **Total Biaya yang Diselamatkan:** R$ 40,650.00 - R$ 23,080.00 = **R$ 17,570.00** *(Catatan: Jika didasarkan pada penyelamatan kerugian murni dari porsi data bocor yang berhasil diintervensi, total kas operasional yang diamankan bernilai R$ 25,100.00 sebelum dikurangi pengeluaran biaya operasional mitigasi).*
* **Efisiensi Finansial Operasional:** Penggunaan model ini sukses memberikan efisiensi biaya logistik sebesar **61.75%** bagi Olist Brasil.

---

## 7. Cara Menjalankan Aplikasi
1. Pastikan seluruh dataset mentah (.csv) tersimpan di folder `data/raw/`.
2. Instal semua pustaka dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt