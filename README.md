# Prediksi Risiko Keterlambatan Pengiriman  
Studi Kasus: Olist E‑Commerce Brasil

## 1. Deskripsi Proyek
Proyek ini membangun sistem peringatan dini (*early warning system*) untuk memprediksi risiko keterlambatan pengiriman (`is_late`) pada platform Olist Brasil menggunakan pendekatan **3 Pilar Logistik**: **Waktu** (Temporal), **Jarak** (Geospasial), dan **Administrasi** (Payment).

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
   - **Winner**: **AdaBoost** terpilih sebagai model final dengan optimasi metrik **Recall**.

---

## 5. Performa & Insight Utama
- **Insight Geografis**: Wilayah di luar negara bagian São Paulo memiliki risiko keterlambatan signifikan lebih tinggi karena tantangan jarak infrastruktur.
- **Strategi Evaluasi**: Fokus utama dioptimasikan pada metrik **Recall (0.62)** pada Test Set untuk meminimalkan *False Negatives* (keterlambatan yang luput dari sistem AI).
- **Kesimpulan**: Model AdaBoost terbukti paling stabil (*robust*) mendeteksi risiko di tengah kondisi ketidakseimbangan kelas (*imbalanced data* ~7.8% kasus telat).

---

## 6. Cara Menjalankan Aplikasi
1. Pastikan seluruh dataset mentah (.csv) tersimpan di folder `data/raw/`.
2. Instal semua pustaka dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt