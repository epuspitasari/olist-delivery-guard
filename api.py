from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# Inisialisasi API
app = FastAPI()

# Load model dan scaler dari folder models
model = joblib.load('models/adaboost_final_model.pkl')
scaler = joblib.load('models/scaler_modeling.pkl')

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
    # Mapping ke 9 fitur yang sinkron dengan NB4
    input_dict = {
        'order_purchase_month': data.month,
        'order_purchase_dayofweek': data.day_of_week,
        'is_sp_district': data.is_sp,
        'payment_installments': data.installments,
        'payment_value': data.payment_value,
        'payment_type_debit_card': 1 if data.pay_type == "Debit Card" else 0,
        'payment_type_not_defined': 0,
        'payment_type_voucher': 1 if data.pay_type == "Voucher" else 0
    }
    
    # Preprocessing
    df_input = pd.DataFrame([input_dict])
    df_scaled = scaler.transform(df_input)
    
    # Prediksi
    prediction = int(model.predict(df_scaled)[0])
    probability = float(model.predict_proba(df_scaled)[0][1])
    
    return {
        "status": "Late" if prediction == 1 else "On Time",
        "probability": probability
    }