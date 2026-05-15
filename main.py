import streamlit as st
import requests

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Olist Delivery Guard (API Mode)",
    page_icon="🚚",
    layout="wide"
)

# --- 2. TAMPILKAN FRONTEND ---
st.title("🚚 Olist Delivery Guard")
st.markdown("Mode: **Frontend Terpisah (Menggunakan API)**")

st.divider()

# --- 3. FORM INPUT USER ---
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
        
    with col3:
        st.markdown("### 💳 Pilar Administrasi")
        payment_val = st.number_input("Nilai Pembayaran (BRL)", min_value=1.0, value=100.0)
        installments = st.slider("Jumlah Cicilan", 1, 24, 1)
        pay_type = st.selectbox("Metode Pembayaran", ["Credit Card", "Boleto", "Debit Card", "Voucher"])

    submitted = st.form_submit_button("Analisis Risiko Keterlambatan")

# --- 4. LOGIKA PENGIRIMAN KE API ---
if submitted:
    # Kita tambahkan spinner supaya user tahu aplikasi sedang menunggu jawaban dari API
    with st.spinner('Menghubungi server AI untuk analisis...'):
        # Membungkus data menjadi 'payload' (paket data)
        payload = {
            "month": int(month),
            "day_of_week": int(day_of_week),
            "is_sp": 1 if is_sp == "Ya" else 0,
            "installments": int(installments),
            "payment_value": float(payment_val),
            "pay_type": pay_type
        }
        
        try:
            # Mengirim data ke API yang sedang jalan di localhost:8000
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            
            if response.status_code == 200:
                res = response.json()
                st.subheader("📊 Hasil Analisis AI (via API)")
                
                # Menampilkan hasil berdasarkan jawaban dari api.py
                if res["status"] == "Late":
                    st.error(f"### STATUS: BERISIKO TERLAMBAT ({res['probability']*100:.2f}%)")
                    st.write("Saran: Segera koordinasi dengan tim gudang.")
                else:
                    st.success(f"### STATUS: TEPAT WAKTU (Risiko: {res['probability']*100:.2f}%)")
                    st.write("Pesanan berada dalam zona aman.")
            else:
                st.error(f"⚠️ API merespon dengan kesalahan (Status: {response.status_code})")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Koneksi Gagal! Pastikan `api.py` sudah jalan di terminal dengan perintah: `uvicorn api:app --reload`")