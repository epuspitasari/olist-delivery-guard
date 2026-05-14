import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. FUNGSI LOAD & MERGE
# ==========================================

def load_all_data(base_path="../data/raw/"):
    """Memuat data orders, payments, dan customers format CSV dengan nama file Olist."""
    import pandas as pd
    import os
    
    # Nama file disesuaikan persis dengan screenshot kamu
    orders = pd.read_csv(os.path.join(base_path, "olist_orders_dataset.csv"))
    payments = pd.read_csv(os.path.join(base_path, "olist_order_payments_dataset.csv"))
    customers = pd.read_csv(os.path.join(base_path, "olist_customers_dataset.csv"))
    
    return orders, payments, customers

def verify_final_data(df):
    """Mengecek info dasar dataset hasil merge."""
    print("--- Verifikasi Data Akhir ---")
    print(f"Total Baris: {df.shape[0]}")
    print(f"Total Kolom: {df.shape[1]}")
    print("\nKolom yang tersedia:")
    print(df.columns.tolist())

def check_missing_post_merge(df):
    """Mengecek apakah ada data kosong setelah proses merge."""
    missing_data = df.isnull().sum()
    total_missing = missing_data.sum()
    print("\n--- Pengecekan Missing Values ---")
    if total_missing == 0:
        print("Mantap! Tidak ada data kosong.")
    else:
        print(missing_data[missing_data > 0])

def load_and_merge_data(order_path, payment_path, customer_path):
    """Menggabungkan data orders, payments, dan customers dengan Inner Join."""
    df_orders = pd.read_pickle(order_path)
    df_payments = pd.read_pickle(payment_path)
    df_customers = pd.read_pickle(customer_path)
    
    df_merged = pd.merge(df_orders, df_customers, on='customer_id', how='inner')
    df_final = pd.merge(df_merged, df_payments, on='order_id', how='inner')
    
    print(f"--- Merging (Inner) Selesai: {df_final.shape[0]} baris & {df_final.shape[1]} kolom ---")
    return df_final

def load_processed_data(file_path):
    """Memuat data tunggal format .pkl atau .csv."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} tidak ditemukan!")
        return None
    return pd.read_pickle(file_path) if file_path.endswith('.pkl') else pd.read_csv(file_path)

# ==========================================
# 2. FEATURE ENGINEERING (NB 3)
# ==========================================

def create_features(df):
    """
    Melakukan transformasi data mentah menjadi fitur prediktor logistik.
    1. Deduplikasi Payment | 2. Drop Non-Delivered | 3. Hitung delivery_time & is_late
    """
    df_new = df.copy()
    
    # 1. Penanganan Duplikasi Administrasi
    df_new = df_new.sort_values('payment_value', ascending=False).drop_duplicates('order_id')
    
    # 2. Pembersihan Data Non-Logistik
    df_new = df_new.dropna(subset=['order_delivered_customer_date'])
    
    # 3. Konstruksi Fitur
    date_columns = ['order_delivered_customer_date', 'order_purchase_timestamp', 'order_estimated_delivery_date']
    for col in date_columns:
        df_new[col] = pd.to_datetime(df_new[col])
    
    df_new['delivery_time'] = (df_new['order_delivered_customer_date'] - df_new['order_purchase_timestamp']).dt.days
    df_new['is_late'] = (df_new['order_delivered_customer_date'] > df_new['order_estimated_delivery_date']).astype(int)
    
    late_pct = (df_new['is_late'].sum() / len(df_new)) * 100
    print(f"--- Feature Engineering Selesai: {len(df_new)} baris | Late: {late_pct:.2f}% ---")
    return df_new

# ==========================================
# 3. PREPROCESSING & CLEANING (UNTUK NB 4)
# ==========================================

def clean_logistics_outliers(df):
    """
    Menghapus outliers pengiriman berdasarkan temuan di NB 3 (0-60 hari).
    Juga melakukan dropna untuk memastikan data bersih total.
    """
    initial_shape = df.shape[0]
    # Menggunakan batas 60 hari sesuai hasil analisis visual NB 3
    df_clean = df[(df['delivery_time'] >= 0) & (df['delivery_time'] <= 60)].copy()
    df_clean = df_clean.dropna()
    
    print(f"--- Outliers & Null Dibuang: {initial_shape - df_clean.shape[0]} baris ---")
    return df_clean

def prepare_modeling_data(df):
    """
    One-Hot Encoding & Pemilihan Fitur Akhir.
    SINKRON DENGAN 3 PILAR LOGISTIK (NB 3).
    """
    df_model = df.copy()

    # --- 1. PENGECEKAN KOLOM TARGET (KUNCI PERBAIKAN) ---
    # Jika is_late tidak ada, kita buatkan berdasarkan logika delivery_time
    if 'is_late' not in df_model.columns:
        print("Catatan: Kolom 'is_late' tidak ditemukan, membuat kolom berdasarkan estimasi...")
        # Pastikan kolom waktu sudah bertipe datetime
        date_cols = ['order_delivered_customer_date', 'order_estimated_delivery_date', 'order_purchase_timestamp']
        for col in date_cols:
            if col in df_model.columns:
                df_model[col] = pd.to_datetime(df_model[col])
        
        # Logika: Jika sampai ke pelanggan lewat dari estimasi
        df_model['is_late'] = (df_model['order_delivered_customer_date'] > df_model['order_estimated_delivery_date']).astype(int)

    # --- 2. KONSTRUKSI FITUR 3 PILAR ---
    if 'order_purchase_month' not in df_model.columns:
        df_model['order_purchase_timestamp'] = pd.to_datetime(df_model['order_purchase_timestamp'])
        df_model['order_purchase_month'] = df_model['order_purchase_timestamp'].dt.month
        df_model['order_purchase_dayofweek'] = df_model['order_purchase_timestamp'].dt.dayofweek

    if 'is_sp_district' not in df_model.columns:
        if 'customer_state' in df_model.columns:
            df_model['is_sp_district'] = df_model['customer_state'].apply(lambda x: 1 if x == 'SP' else 0)
        else:
            # Default ke 0 jika customer_state tidak ada
            df_model['is_sp_district'] = 0

    # --- 3. PEMILIHAN FITUR FINAL ---
    features = [
        'order_purchase_month', 
        'order_purchase_dayofweek', 
        'is_sp_district',
        'payment_type', 
        'payment_installments', 
        'payment_value', 
        'is_late'  # Target harus ada di sini
    ]
    
    # Filter hanya kolom yang benar-benar ada di dataframe
    existing_cols = [col for col in features if col in df_model.columns]
    df_model = df_model[existing_cols].copy()
    
    # --- 4. ENCODING & CLEANING ---
    if 'payment_type' in df_model.columns:
        df_model = pd.get_dummies(df_model, columns=['payment_type'], drop_first=True)
    
    # Pastikan target tetap integer
    if 'is_late' in df_model.columns:
        df_model['is_late'] = df_model['is_late'].astype(int)
    
    print(f"--- Modeling Data Siap: {df_model.shape[1]} kolom dihasilkan ---")
    return df_model

# ==========================================
# 4. MODELING & EVALUASI
# ==========================================

def split_data_logistics(df, target_col='is_late', test_size=0.1, val_size=0.1):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split Train+Val dan Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Split Train dan Val
    val_adj = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_adj, random_state=42, stratify=y_train_val)
    
    print(f"--- Split Selesai: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)}) ---")
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_model_artifact(model, filename="../models/logistic_regression_v1.pkl"):
    """Menyimpan model ke folder models (Nanti dipakai di NB 4)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"--- Model Berhasil Disimpan: {filename} ---")