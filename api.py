from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import numpy as np

# Inisialisasi API
app = FastAPI()

# Load model dan scaler
try:
    model = joblib.load('models/adaboost_final_model.pkl')
    scaler = joblib.load('models/scaler_modeling.pkl')
except Exception as e:
    print(f"Error loading artifacts: {e}")

# Struktur data yang dikirim dari Streamlit
class DeliveryInput(BaseModel):
    month: int
    day_of_week: int
    is_sp: int
    installments: int
    payment_value: float
    pay_type: str

@app.post("/predict")
def get_prediction(data: DeliveryInput):
    # Logika 9 Fitur (Tanpa Boleto secara eksplisit)
    # Jika user pilih 'Boleto', maka semua payment_type_* di bawah akan bernilai 0
    input_dict = {
        'order_purchase_month': data.month,
        'order_purchase_dayofweek': data.day_of_week,
        'is_sp_district': data.is_sp,
        'payment_installments': data.installments,
        'payment_value': data.payment_value,
        'payment_type_credit_card': 1 if data.pay_type == "Credit Card" else 0,
        'payment_type_debit_card': 1 if data.pay_type == "Debit Card" else 0,
        'payment_type_not_defined': 0,
        'payment_type_voucher': 1 if data.pay_type == "Voucher" else 0
    }
    
    df_input = pd.DataFrame([input_dict])
    
    try:
        # --- PENYELAMAT: SINKRONISASI FITUR ---
        # Mengambil urutan 9 kolom asli dari Scaler
        fitur_seharusnya = scaler.feature_names_in_.tolist()
        
        # Memaksa df_input mengikuti urutan fitur_seharusnya
        df_input = df_input.reindex(columns=fitur_seharusnya, fill_value=0)
        
        # Preprocessing & Prediksi
        df_scaled = scaler.transform(df_input)
        prediction = int(model.predict(df_scaled)[0])
        probability = float(model.predict_proba(df_scaled)[0][1])
        
        return {
            "status": "Late" if prediction == 1 else "On Time",
            "probability": probability,
            "message": "Prediksi berhasil dihitung"
        }
    except Exception as e:
        return {"error": str(e)}

# Jalankan dengan: uvicorn nama_file_api:app --reload