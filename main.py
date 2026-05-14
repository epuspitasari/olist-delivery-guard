import streamlit as st
import requests

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Olist Frontend (API Mode)",
    page_icon="🚚",
    layout="centered"
)

# --- 2. TAMPILAN FRONTEND ---
st.title("🚚 Olist Delivery Guard")
st.subheader("Mode: API + Frontend (Manual)")

st.markdown("""
Aplikasi ini berjalan dengan mengirimkan data ke **Backend API (api.py)**. 
Pastikan server Uvicorn sudah aktif sebelum menekan tombol prediksi.
""")

# Ilustrasi agar menarik
st.image("https://images.unsplash.com/photo-1566576721346-d4a3b4eaad5b?q=80&w=1000&auto=format&fit=crop", 
         caption="Sistem Logistik Olist", use_column_width=True)

st.divider()

# --- 3. FORM INPUT USER ---
with st.form("delivery_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Waktu & Lokasi**")
        month = st.selectbox("Bulan Pembelian", range(1, 13))
        day_of_week = st.selectbox("Hari (0=Senin, 6=Minggu)", range(7))
        is_sp = st.selectbox("Lokasi ke Distrik SP?", ["Ya", "Tidak"])
        
    with col2:
        st.info("**Detail Pembayaran**")
        payment_val = st.number_input("Nilai Bayar (BRL)", min_value=1.0, value=100.0)
        installments = st.number_input("Jumlah Cicilan", min_value=1, max_value=24, value=1)
        pay_type = st.selectbox("Metode Pembayaran", ["Credit Card", "Debit Card", "Voucher", "Boleto"])
    
    submit = st.form_submit_button("Cek Risiko Keterlambatan")

# --- 4. LOGIKA PENGIRIMAN DATA KE API ---
if submit:
    # Siapkan data (Payload) sesuai struktur 'DeliveryInput' di api.py
    payload = {
        "month": int(month),
        "day_of_week": int(day_of_week),
        "is_sp": 1 if is_sp == "Ya" else 0,
        "installments": int(installments),
        "payment_value": float(payment_val),
        "pay_type": pay_type
    }
    
    st.write("🔍 *Mengirim data ke API...*")
    
    try:
        # Menembak endpoint API (pastikan api.py jalan di port 8000)
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            res_data = response.json()
            
            st.divider()
            # Menampilkan hasil dari API
            if res_data["status"] == "Late":
                st.error(f"### ⚠️ HASIL: BERISIKO TERLAMBAT")
                st.write(f"Probabilitas Risiko: **{res_data['probability']*100:.2f}%**")
                st.warning("Rekomendasi: Lakukan percepatan pada proses pengemasan.")
            else:
                st.success(f"### ✅ HASIL: AMAN / TEPAT WAKTU")
                st.write(f"Probabilitas Terlambat: **{res_data['probability']*100:.2f}%**")
        else:
            st.error(f"Terjadi kesalahan pada API. Status Code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("❌ **Koneksi Gagal!**")
        st.write("Pastikan kamu sudah menjalankan `api.py` dengan perintah: `uvicorn api:app --reload` di terminal.")

st.divider()
st.caption("© 2024 Project Olist - Mode Terpisah (API-Frontend)")