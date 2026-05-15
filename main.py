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
    with st.spinner('Menghubungi server AI untuk analisis...'):
        payload = {
            "month": int(month),
            "day_of_week": int(day_of_week),
            "is_sp": 1 if is_sp == "Ya" else 0,
            "installments": int(installments),
            "payment_value": float(payment_val),
            "pay_type": pay_type
        }
        
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            
            if response.status_code == 200:
                res = response.json()
                st.subheader("📊 Hasil Analisis AI (via API)")
                
                if res["status"] == "Late":
                    st.error(f"### STATUS: BERISIKO TERLAMBAT ({res['probability']*100:.2f}%)")
                    st.write("Saran: Segera koordinasi dengan tim gudang.")
                else:
                    st.success(f"### STATUS: TEPAT WAKTU (Risiko: {res['probability']*100:.2f}%)")
                    st.write("Pesanan berada dalam zona aman.")
                
                # Menampilkan Insight otomatis dari API (jika ada)
                if "insights" in res:
                    st.info(f"💡 **AI Insight:** {res['insights']}")
                    
            else:
                st.error(f"⚠️ API merespon dengan kesalahan (Status: {response.status_code})")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Koneksi Gagal! Pastikan `api.py` sudah jalan di terminal.")

st.divider()

# --- 5. INFORMASI TAMBAHAN (INSIGHTS) ---
with st.expander("ℹ️ Mengapa bulan atau lokasi tertentu berisiko tinggi?"):
    st.markdown("""
    Berdasarkan pola data historis e-commerce di Brazil:
    * **Oktober - Desember (Bulan 10-12):** Periode *peak season* (Black Friday & Natal). Volume paket yang membeludak sering menyebabkan keterlambatan logistik.
    * **Januari - Februari (Bulan 1-2):** Efek sisa pengiriman akhir tahun dan adanya musim liburan serta Karnaval di Brazil yang mempengaruhi operasional kurir.
    * **Lokasi (São Paulo):** Meskipun SP adalah pusat logistik, pengiriman ke luar wilayah ini menempuh jarak jauh dengan infrastruktur jalan yang menantang, sehingga risiko keterlambatan meningkat.
    """)