import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Olist Delivery Guard",
    page_icon="🚚",
    layout="wide"
)

# --- 2. FUNGSI UNTUK MEMUAT MODEL ---
@st.cache_resource
def load_artifacts():
    # Mengacu pada folder 'models' sesuai struktur project kamu
    model_path = 'models/adaboost_final_model.pkl'
    scaler_path = 'models/scaler_modeling.pkl'
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Gagal memuat file model. Pastikan folder 'models' tersedia. Error: {e}")
    st.stop()

# --- 3. TAMPILKAN FRONTEND ---
st.title("🚚 Olist Delivery Guard")
st.markdown("""
Aplikasi AI ini membantu memprediksi apakah sebuah pesanan di **Olist E-commerce** berisiko terlambat.
""")

st.divider()

# --- 4. FORM INPUT USER ---
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📅 Pilar Waktu")
        month = st.selectbox("Bulan Pembelian", range(1, 13))
        day_of_week = st.selectbox("Hari Pembelian", 
                                   options=[0, 1, 2, 3, 4, 5, 6], 
                                   format_func=lambda x: ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][x])
        
    with col2:
        st.markdown("### 📍 Pilar Jarak")
        is_sp = st.radio("Tujuan ke Distrik São Paulo (SP)?", ["Ya", "Tidak"])
        is_sp_val = 1 if is_sp == "Ya" else 0
        
    with col3:
        st.markdown("### 💳 Pilar Administrasi")
        payment_val = st.number_input("Nilai Pembayaran (BRL)", min_value=1.0, value=100.0)
        installments = st.slider("Jumlah Cicilan", 1, 24, 1)
        # Menambahkan 'Boleto' sebagai pilihan di UI
        pay_type = st.selectbox("Metode Pembayaran", ["Credit Card", "Boleto", "Debit Card", "Voucher"])

    submitted = st.form_submit_button("Analisis Risiko Keterlambatan")

# --- 5. LOGIKA PREDIKSI ---
if submitted:
    # Kita bungkus semua proses di bawah spinner ini
    with st.spinner('Sedang menganalisis data logistik...'):
        # A. Buat dictionary awal
        input_data = {
            'order_purchase_month': month,
            'order_purchase_dayofweek': day_of_week,
            'is_sp_district': is_sp_val,
            'payment_installments': installments,
            'payment_value': payment_val,
            'payment_type_credit_card': 1 if pay_type == "Credit Card" else 0,
            'payment_type_debit_card': 1 if pay_type == "Debit Card" else 0,
            'payment_type_not_defined': 0,
            'payment_type_voucher': 1 if pay_type == "Voucher" else 0
        }
        
        # B. Konversi ke DataFrame
        df_predict = pd.DataFrame([input_data])
        
        try:
            # C. Sinkronisasi urutan fitur
            fitur_seharusnya = scaler.feature_names_in_.tolist()
            
            # D. Reindex
            df_predict = df_predict.reindex(columns=fitur_seharusnya, fill_value=0)
            
            # E. Scaling Data
            df_predict_scaled = scaler.transform(df_predict)
            
            # F. Eksekusi Model
            prediction = model.predict(df_predict_scaled)[0]
            probability = model.predict_proba(df_predict_scaled)[0][1]

            # --- 6. TAMPILKAN HASIL ---
            st.subheader("📊 Hasil Analisis AI")
            if prediction == 1:
                st.error(f"### STATUS: BERISIKO TERLAMBAT ({probability*100:.2f}%)")
                st.write("Saran: Segera koordinasi dengan tim gudang untuk prioritas packing.")
            else:
                st.success(f"### STATUS: TEPAT WAKTU (Risiko: {probability*100:.2f}%)")
                st.write("Pesanan ini berada dalam zona aman logistik.")

        except Exception as e:
            st.error(f"⚠️ Kesalahan Teknis: {e}")