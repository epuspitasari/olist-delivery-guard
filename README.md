# Prediksi Risiko Keterlambatan Pengiriman  
Studi Kasus: Olist E‑Commerce Brasil

## 1. Deskripsi Proyek
Proyek ini membangun sistem peringatan dini (*early warning system*) untuk memprediksi risiko keterlambatan pengiriman (`is_late`) pada platform Olist Brasil menggunakan pendekatan **3 Pilar Logistik**: **Waktu** (Temporal), **Jarak** (Geospasial), dan **Administrasi** (Payment).

---

## 2. Sumber Data & Metadata
Dataset: **[Brazilian E‑Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)**.

Untuk menjaga integritas model, proyek ini menggunakan 3 dataset mentah utama:
- `olist_orders_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_customers_dataset.csv`

**Alasan Pemilihan:** Fitur yang digunakan hanya data yang tersedia saat transaksi terjadi guna menghindari *data leakage* dari informasi pasca-pengiriman.

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
   - **Baseline**: Logistic Regression.
   - **Tournament Stage**: Perbandingan 5 model (Decision Tree, Random Forest, XGBoost, AdaBoost, KNN).
   - **Winner**: **AdaBoost** terpilih sebagai model final dengan optimasi metrik **Recall**.

---

## 5. Performa & Insight Utama
- **Insight Geografis**: Wilayah di luar São Paulo memiliki risiko keterlambatan signifikan lebih tinggi.
- **Strategi Evaluasi**: Fokus pada **Recall (0.64)** untuk meminimalkan *False Negatives* (keterlambatan yang tidak terdeteksi).
- **Kesimpulan**: Model AdaBoost terbukti stabil dalam mendeteksi risiko di tengah kondisi data yang tidak seimbang (*imbalanced*).

---

## 6. Cara Menjalankan
1. Pastikan dataset mentah tersimpan di folder `data/raw/`.
2. Instal semua pustaka sekaligus:
   ```bash
   pip install -r requirements.txt
---

## 6. Cara Menjalankan
1. Letakkan dataset mentah di `data/raw/`.
2. Instal pustaka yang diperlukan: `pip install -r requirements.txt` dll.
3. Jalankan notebook secara berurutan (01-04).
4. Jalankan aplikasi demo: `streamlit run app.py`.

## 7. Library Utama
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `joblib`, `streamlit`.
