import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Olist Delivery Guard",
    page_icon="🚚",
    layout="wide"
)

# --- 2. FUNGSI UNTUK MEMUAT MODEL (BACKEND) ---
@st.cache_resource
def load_artifacts():
    # Pastikan path ini sesuai dengan struktur folder kamu
    model_path = 'models/adaboost_final_model.pkl'
    scaler_path = 'models/scaler_modeling.pkl'
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Load model dan scaler
try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Gagal memuat file model atau scaler. Pastikan ada folder 'models' berisi file .pkl. Error: {e}")
    st.stop()

# --- 3. TAMPILAN FRONTEND ---
st.title("🚚 Olist Delivery Guard")
st.markdown("""
Aplikasi AI untuk memprediksi risiko keterlambatan pengiriman pesanan pada ekosistem Olist.
Sistem ini menganalisis pola **Waktu**, **Lokasi**, dan **Nilai Transaksi** untuk memberikan peringatan dini.
""")

# Ilustrasi gambar (menggunakan URL agar praktis)
st.image("https://images.unsplash.com/photo-1580674285054-bed31e145f59?q=80&w=1200&auto=format&fit=crop", 
         caption="Optimasi Logistik & Rantai Pasok Olist", use_column_width=True)

st.divider()

# --- 4. FORM INPUT USER ---
st.subheader("🔍 Input Detail Transaksi")

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Pilar Waktu**")
        month = st.selectbox("Bulan Pembelian", range(1, 13), help="Bulan saat customer melakukan checkout")
        day_of_week = st.selectbox("Hari Pembelian", 
                                   options=[0, 1, 2, 3, 4, 5, 6], 
                                   format_func=lambda x: ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][x])
        
    with col2:
        st.info("**Pilar Lokasi & Jarak**")
        is_sp = st.radio("Tujuan ke Distrik São Paulo (SP)?", ["Ya", "Tidak"])
        is_sp_val = 1 if is_sp == "Ya" else 0
        
    with col3:
        st.info("**Pilar Administrasi**")
        payment_val = st.number_input("Nilai Pembayaran (BRL)", min_value=1.0, value=100.0)
        installments = st.slider("Jumlah Cicilan", 1, 24, 1)
        pay_type = st.selectbox("Metode Pembayaran", ["Credit Card", "Debit Card", "Voucher", "Boleto"])

    # Tombol Prediksi
    submitted = st.form_submit_button("Analisis Risiko Keterlambatan")

# --- 5. LOGIKA PREDIKSI & OUTPUT (BACKEND) ---
if submitted:
    # A. Mapping input agar sesuai dengan urutan fitur saat training (9 Fitur)
    # Urutan fitur harus eksak: 
    # [month, dayofweek, is_sp, installments, value, pay_debit, pay_notdef, pay_voucher]
    # (Catatan: pay_credit_card biasanya dibuang saat drop_first=True)
    
    input_data = {
        'order_purchase_month': month,
        'order_purchase_dayofweek': day_of_week,
        'is_sp_district': is_sp_val,
        'payment_installments': installments,
        'payment_value': payment_val,
        'payment_type_debit_card': 1 if pay_type == "Debit Card" else 0,
        'payment_type_not_defined': 0,
        'payment_type_voucher': 1 if pay_type == "Voucher" else 0
    }
    
    # B. Konversi ke DataFrame
    df_predict = pd.DataFrame([input_data])
    
    # C. Scaling Data (Sangat Penting: Samakan dengan skala saat training)
    df_predict_scaled = scaler.transform(df_predict)
    
    # D. Eksekusi Model
    prediction = model.predict(df_predict_scaled)[0]
    probability = model.predict_proba(df_predict_scaled)[0][1]

    # --- 6. TAMPILKAN HASIL ---
    st.subheader("📊 Hasil Analisis AI")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if prediction == 1:
            st.error("### STATUS: BERISIKO TERLAMBAT")
            st.write(f"Sistem mendeteksi probabilitas keterlambatan sebesar **{probability*100:.2f}%**.")
        else:
            st.success("### STATUS: TEPAT WAKTU")
            st.write(f"Sistem memprediksi pesanan akan sampai sesuai jadwal (Risiko: {probability*100:.2f}%).")
            
    with res_col2:
        # Penjelasan singkat berbasis Feature Importance
        st.write("**Faktor Berpengaruh:**")
        st.write(f"- Analisis didominasi oleh faktor **Bulan ke-{month}**.")
        if is_sp_val == 1:
            st.write("- Lokasi di dalam São Paulo cenderung mempercepat pengiriman.")
        else:
            st.write("- Lokasi di luar pusat SP meningkatkan risiko durasi kirim.")

st.divider()
st.caption("© 2024 Proyek Data Science Olist - Ety Puspitasari")